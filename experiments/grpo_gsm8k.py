"""GRPO fine-tuning of Gemma-3-1B-IT on GSM8K.

Adapted from https://github.com/google/tunix/blob/main/examples/grpo_gemma.ipynb.
All hyperparameters are exposed via GRPOTrainConfig so this can be driven by
hyperoptax for automated tuning.

Usage:
    uv run python experiments/grpo_gsm8k.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import grain
import jax
import jax.numpy as jnp
import optax
import qwix
import tensorflow_datasets as tfds
from flax import nnx
from huggingface_hub import snapshot_download
from orbax import checkpoint as ocp
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

# ---------------------------------------------------------------------------
# Prompt templates (fixed — not hyperparameters)
# ---------------------------------------------------------------------------

_REASONING_START = "<reasoning>"
_REASONING_END = "</reasoning>"
_SOLUTION_START = "<answer>"
_SOLUTION_END = "</answer>"

_SYSTEM_PROMPT = (
    f"You are given a problem. First, think about the problem "
    f"and provide your reasoning. Place it between {_REASONING_START} and "
    f"{_REASONING_END}. Then, provide the final answer (i.e., just one numerical "
    f"value) between {_SOLUTION_START} and {_SOLUTION_END}."
)

_TEMPLATE = (
    "<start_of_turn>user\n"
    "{system_prompt}\n\n"
    "{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

_MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{_REASONING_START}.+?{_REASONING_END}.*?"
    rf"{_SOLUTION_START}(.+?){_SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

_MATCH_NUMBERS = re.compile(
    rf"{_SOLUTION_START}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOTrainConfig:
    # --- model ---
    model_id: str = "google/gemma-3-1b-it"
    tokenizer_path: str = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

    # --- data ---
    train_data_dir: str = "./data/train"
    test_data_dir: str = "./data/test"
    data_source: Literal["tfds", "kaggle"] = "tfds"
    train_fraction: float = 0.9
    num_batches: int = 3738
    num_test_batches: int = 64
    num_epochs: int = 1

    # --- LoRA ---
    rank: int = 64
    alpha: float = 64.0

    # --- GRPO algorithm ---
    num_generations: int = 2   # G in the paper
    num_iterations: int = 1    # μ in the paper
    beta: float = 0.08         # KL penalty coefficient
    epsilon: float = 0.2       # policy clipping

    # --- generation during training ---
    max_prompt_length: int = 256
    total_generation_steps: int = 768
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50

    # --- optimizer ---
    learning_rate: float = 3e-6
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    warmup_fraction: float = 0.1  # fraction of max_steps used for linear warmup

    # --- training loop ---
    train_micro_batch_size: int = 1
    eval_every_n_steps: int = 64

    # --- checkpointing / logging ---
    save_interval_steps: int = 500
    max_to_keep: int = 4
    ckpt_dir: str = "/tmp/grpo_ckpts"
    log_dir: str = "/tmp/grpo_logs"

    # --- hardware ---
    mesh_counts: tuple[int, int] | None = None  # auto-detected if None

    # --- derived (set in __post_init__) ---
    max_steps: int = field(init=False)
    warmup_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.max_steps = int(
            self.num_batches * self.num_iterations * self.train_fraction * self.num_epochs
        )
        self.warmup_steps = int(self.warmup_fraction * self.max_steps)

    def _resolve_mesh_counts(self) -> tuple[int, int]:
        if self.mesh_counts is not None:
            return self.mesh_counts
        n = len(jax.devices())
        if n == 8:
            return (1, 4)
        return (1, 1)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def match_format_exactly(prompts, completions, **kwargs):
    return [0 if _MATCH_FORMAT.search(r) is None else 3.0 for r in completions]


def match_format_approximately(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0.0
        score += 0.5 if completion.count(_REASONING_START) == 1 else -0.5
        score += 0.5 if completion.find(_REASONING_START) == 0 else -0.5
        score += 0.5 if completion.count(_REASONING_END) == 1 else -0.5
        score += 0.5 if completion.count(_SOLUTION_START) == 1 else -0.5
        score += 0.5 if completion.count(_SOLUTION_END) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    extracted = [
        m.group(1) if r is not None and (m := _MATCH_FORMAT.search(r)) is not None else None
        for r in completions
    ]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        score = 0.0
        if guess == true_answer:
            score = 3.0
        elif guess.strip() == true_answer.strip():
            score = 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score = 0.5
                elif 0.8 <= ratio <= 1.2:
                    score = 0.25
                else:
                    score = -1.0
            except Exception:
                score = -0.5
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    extracted = [
        m.group(1) if (m := _MATCH_NUMBERS.search(r)) is not None else None
        for r in completions
    ]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            scores.append(1.5 if float(guess.strip()) == float(true_answer.strip()) else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def _download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    import kagglehub
    os.makedirs(target_dir, exist_ok=True)
    src = Path(kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a"))
    dst = Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def _build_dataset(data_dir: str, split: str, source: str) -> grain.MapDataset:
    os.makedirs(data_dir, exist_ok=True)

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k  # noqa: F401
        raw = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
    elif source == "kaggle":
        kaggle_dir = _download_kaggle_dataset(data_dir)
        csv_path = os.path.join(kaggle_dir, f"main_{split}.csv")
        raw = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw.append({"question": row["question"], "answer": row["answer"]})
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    def _as_text(v):
        return v if isinstance(v, str) else v.decode("utf-8")

    return (
        grain.MapDataset.source(raw)
        .shuffle(seed=42)
        .map(
            lambda x: {
                "prompts": _TEMPLATE.format(
                    system_prompt=_SYSTEM_PROMPT,
                    question=_as_text(x["question"]),
                ),
                "question": _as_text(x["question"]),
                "answer": _extract_hash_answer(_as_text(x["answer"])),
            }
        )
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _generate(questions, sampler, eos_tokens, total_generation_steps, temperature, top_k, top_p, seed=None):
    if isinstance(questions, str):
        questions = [questions]
        single = True
    else:
        single = False

    input_batch = [
        _TEMPLATE.format(system_prompt=_SYSTEM_PROMPT, question=q)
        for q in questions
    ]
    out = sampler(
        input_strings=input_batch,
        max_generation_steps=total_generation_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=eos_tokens,
    )
    return out.text[0] if single else out.text


def _evaluate(
    dataset,
    sampler,
    eos_tokens: list[int],
    total_generation_steps: int,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> dict[str, float]:
    corr = partially_corr = corr_format = total = 0

    for batch in tqdm(dataset, desc="eval"):
        answers = batch["answer"]
        questions = batch["question"]
        responses = _generate(
            questions, sampler, eos_tokens, total_generation_steps,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=0,
        )

        for response, answer in zip(responses, answers):
            extracted = (
                m.group(1) if (m := _MATCH_NUMBERS.search(response)) is not None else "-1e9"
            )
            try:
                if float(extracted.strip()) == float(answer.strip()):
                    corr += 1
                ratio = float(extracted.strip()) / float(answer.strip())
                if 0.9 <= ratio <= 1.1:
                    partially_corr += 1
            except Exception:
                pass
            if _MATCH_FORMAT.search(response) is not None:
                corr_format += 1
            total += 1

    if total == 0:
        return {"accuracy": 0.0, "partial_accuracy": 0.0, "format_accuracy": 0.0}

    return {
        "accuracy": corr / total * 100,
        "partial_accuracy": partially_corr / total * 100,
        "format_accuracy": corr_format / total * 100,
    }


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(config: GRPOTrainConfig) -> dict[str, float]:
    """Run GRPO fine-tuning and return evaluation metrics."""

    mesh_counts = config._resolve_mesh_counts()
    mesh = jax.make_mesh(
        mesh_counts,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

    # --- load base model ---
    print(f"Downloading {config.model_id} from Hugging Face...")
    local_model_path = snapshot_download(
        repo_id=config.model_id, ignore_patterns=["*.pth"]
    )

    eos_tokens: list[int] = []
    gen_cfg_path = os.path.join(local_model_path, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        with open(gen_cfg_path) as f:
            eos_tokens = json.load(f).get("eos_token_id", [])

    if "gemma-3-270m" in config.model_id:
        model_config = gemma_lib.ModelConfig.gemma3_270m()
    elif "gemma-3-1b" in config.model_id:
        model_config = gemma_lib.ModelConfig.gemma3_1b_it()
    else:
        raise ValueError(f"Unsupported model: {config.model_id}")

    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )

    # --- apply LoRA ---
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
        ),
        rank=config.rank,
        alpha=config.alpha,
    )
    model_input = base_model.get_model_input()
    lora_policy = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)
    with mesh:
        state = nnx.state(lora_policy)
        pspecs = nnx.get_partition_spec(state)
        nnx.update(lora_policy, jax.lax.with_sharding_constraint(state, pspecs))

    # --- tokenizer ---
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=config.tokenizer_path)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())

    # --- datasets ---
    raw_dataset = _build_dataset(
        config.train_data_dir, "train", config.data_source
    ).batch(config.train_micro_batch_size)[:config.num_batches]

    if config.train_fraction >= 1.0:
        train_dataset = raw_dataset.repeat(config.num_epochs)
        val_dataset = None
    else:
        split_idx = int(len(raw_dataset) * config.train_fraction)
        train_dataset = raw_dataset[:split_idx].repeat(config.num_epochs)
        val_dataset = raw_dataset[split_idx:].repeat(config.num_epochs)

    test_dataset = _build_dataset(
        config.test_data_dir, "test", config.data_source
    ).batch(config.train_micro_batch_size)[:config.num_test_batches]

    # --- optimizer ---
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=config.max_grad_norm),
        optax.adamw(
            learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.learning_rate,
                warmup_steps=config.warmup_steps,
                decay_steps=config.max_steps,
                end_value=0.0,
            ),
            b1=config.b1,
            b2=config.b2,
            weight_decay=config.weight_decay,
        ),
    )

    # --- training configuration ---
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=config.save_interval_steps,
        max_to_keep=config.max_to_keep,
    )
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=config.log_dir,
        flush_every_n_steps=20,
    )

    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=config.eval_every_n_steps,
            max_steps=config.max_steps,
            mini_batch_size=config.train_micro_batch_size,
            train_micro_batch_size=config.train_micro_batch_size,
            metrics_logging_options=metrics_logging_options,
            checkpoint_root_directory=config.ckpt_dir,
            checkpointing_options=checkpointing_options,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=config.total_generation_steps,
            max_prompt_length=config.max_prompt_length,
            kv_cache_size=config.max_prompt_length + config.total_generation_steps + 256,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            eos_tokens=eos_tokens,
        ),
    )

    grpo_config = GRPOConfig(
        num_generations=config.num_generations,
        num_iterations=config.num_iterations,
        beta=config.beta,
        epsilon=config.epsilon,
    )

    # --- build cluster + trainer ---
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora_policy,
        reference=base_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        algo_config=grpo_config,
    )

    # --- train ---
    grpo_trainer.train(train_dataset, val_dataset)

    # --- load best checkpoint ---
    trained_ckpt_path = os.path.join(
        config.ckpt_dir, "actor", str(config.max_steps), "model_params"
    )
    if os.path.exists(trained_ckpt_path):
        abs_params = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            nnx.state(lora_policy, nnx.LoRAParam),
        )
        checkpointer = ocp.StandardCheckpointer()
        trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)
        nnx.update(
            lora_policy,
            jax.tree.map(lambda a, b: b, nnx.state(lora_policy, nnx.LoRAParam), trained_lora_params),
        )

    # --- evaluate ---
    sampler = sampler_lib.Sampler(
        transformer=lora_policy,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=config.max_prompt_length + config.total_generation_steps + 256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    metrics = _evaluate(
        test_dataset,
        sampler,
        eos_tokens=eos_tokens,
        total_generation_steps=config.total_generation_steps,
        temperature=None,  # greedy
        top_k=1,
        top_p=None,
    )
    print(f"Evaluation results: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train(GRPOTrainConfig())
    print(results)
