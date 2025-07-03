<img src="logo.png" alt="Hyperoptax Logo" style="width:80%;"/>

# Hyperoptax: Parallel hyperparameter tuning with JAX

[![PyPI version](https://img.shields.io/pypi/v/hyperoptax)](https://pypi.org/project/hyperoptax)
![CI status](https://github.com/TheodoreWolf/hyperoptax/actions/workflows/test.yml/badge.svg?branch=main)

## Introduction

Hyperoptax is a lightweight toolbox for parallel hyperparameter optimisation of pure JAX functions. It provides a concise API that lets you wrap any JAX-compatible loss or evaluation function and search across spaces __in parallel__  – all while staying in pure JAX. 

## Installation

```bash
pip install hyperoptax
```

If you want to use the notebooks:
```bash
pip install hyperoptax[notebooks]
```

If you do not yet have JAX installed, pick the right wheel for your accelerator:

```bash
# CPU-only
pip install --upgrade "jax[cpu]"
# or GPU/TPU – see the official JAX installation guide
```
## In a nutshell
Hyperoptax offers a simple API to wrap pure JAX functions for hyperparameter search and making use of parallelisation (vmap or pmap). See the [notebooks](https://github.com/TheodoreWolf/hyperoptax/tree/main/notebooks) for more examples.
```python
from hyperoptax.bayes import BayesOptimiser
from hyperoptax.spaces import LogSpace, LinearSpace

@jax.jit
def train_nn(learning_rate, final_lr_pct):
    ...
    return val_loss

search_space = {"learning_rate": LogSpace(1e-5,1e-1, 100),
                "final_lr_pct": LinearSpace(0.01, 0.5, 100)}

search = BayesOptimiser(search_space, train_nn)
best_params = search.optimise(n_iterations=100, 
                              n_parallel=10, 
                              maximise=False,
                              pmap=True
                              )
```
## The Sharp Bits

Since we are working in pure JAX the same [sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) apply. Some consequences of this for hyperoptax:
1. Parameters that change the length of an evaluation (e.g: epochs, generations...) can't be optimised in parallel.
2. Neural network structures can't be optimised in parallel either.
3. Strings can't be used as hyperparameters.

## Contributing

We welcome pull requests! To get started:

1. Open an issue describing the bug or feature.
2. Fork the repository and create a feature branch (`git checkout -b my-feature`).
3. Install dependencies:

```bash
pip install -e .
```

4. Run the test suite:

```bash
python -m unittest discover -s tests
```

5. Format your code with `ruff`.
6. Submit a pull request.

## Citation

If you use Hyperoptax in academic work, please cite:

```bibtex
@misc{hyperoptax2025,
  author = {Theo Wolf},
  title = {{Hyperoptax}: Parallel hyperparameter tuning with JAX},
  year = {2025},
  url = {https://github.com/TheodoreWolf/hyperoptax}
}
```