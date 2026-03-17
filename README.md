<img src="./assets/logo_transparent.png" alt="Hyperoptax Logo" style="width:80%;"/>

# Hyperoptax: Parallel hyperparameter tuning with JAX

[![PyPI version](https://img.shields.io/pypi/v/hyperoptax)](https://pypi.org/project/hyperoptax)
![CI status](https://github.com/TheodoreWolf/hyperoptax/actions/workflows/test.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/TheodoreWolf/hyperoptax/graph/badge.svg?token=Y582MZ25GG)](https://codecov.io/gh/TheodoreWolf/hyperoptax)

>[!WARNING]
> Hyperoptax is still a WIP and the API is subject to change. There are _many_ rough edges to smooth out. It is recommended to download specific versions or to download from source if you want to use it in a large scale project.

## ⛰️ Introduction

Hyperoptax is a lightweight toolbox for parallel hyperparameter optimization of pure JAX functions. It provides a concise API that lets you wrap any JAX-compatible loss or evaluation function and search across spaces __in parallel__  – all while staying in pure JAX.

## 🏗️ Installation

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
## 🥜 In a nutshell

All optimizers follow the same stateless pattern: `Optimizer.init` returns a `(state, optimizer)` pair, and `optimizer.optimize` runs the search loop. Your objective function must have the signature `fn(key, params) -> scalar`.

```python
import jax
from hyperoptax.bayesian import BayesianSearch
from hyperoptax.spaces import LogSpace, LinearSpace

def train_nn(key, params):
    learning_rate = params["learning_rate"]
    final_lr_pct = params["final_lr_pct"]
    ...
    return val_loss  # scalar, lower is better

search_space = {
    "learning_rate": LogSpace(1e-5, 1e-1),
    "final_lr_pct": LinearSpace(0.01, 0.5),
}

state, optimizer = BayesianSearch.init(
    search_space,
    n_max=100,       # observation buffer size (= number of iterations)
    n_parallel=4,    # Kriging Believer batch size
    maximize=False,
)

state, (params_hist, results_hist) = optimizer.optimize(
    state, jax.random.PRNGKey(0), train_nn
)

# Retrieve best result
print(optimizer.best_result(state))
print(optimizer.best_params(state))
```

Other available optimizers:

```python
from hyperoptax.random import RandomSearch
from hyperoptax.grid import GridSearch
from hyperoptax.spaces import DiscreteSpace

# Random search
state, optimizer = RandomSearch.init(search_space, n_parallel=8)
state, history = optimizer.optimize(state, jax.random.PRNGKey(0), train_nn, n_iterations=50)

# Grid search (DiscreteSpace only)
grid_space = {"lr": DiscreteSpace([1e-4, 1e-3, 1e-2]), "dropout": DiscreteSpace([0.1, 0.3, 0.5])}
state, optimizer = GridSearch.init(grid_space)
state, history = optimizer.optimize(state, jax.random.PRNGKey(0), train_nn, n_iterations=9)
```

## 💪 Hyperoptax in action
<img src="./assets/gp_animation.gif" alt="BayesOpt animation" style="width:80%;"/>

## 🔪 The Sharp Bits

Since we are working in pure JAX the same [sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) apply. Some consequences of this for hyperoptax:
1. Parameters that change the length of an evaluation (e.g: epochs, generations...) can't be optimized in parallel.
2. Neural network structures can't be optimized in parallel either.
3. Strings can't be used as hyperparameters.

## 🫂 Contributing

We welcome pull requests! To get started:

1. Open an issue describing the bug or feature.
2. Fork the repository and create a feature branch (`git checkout -b my-feature`).
3. Install dependencies:

```bash
pip install -e ".[all]"
```

4. Run the test suite:

```bash
pytest
```
5. Ensure the notebooks still work.
6. Format your code with `ruff`.
7. Submit a pull request.

## Roadmap
I'm developing this both as a passion project and for my work in my PhD. I have a few ideas on where to go with this library:
- Callbacks!
- Some clumpiness in the acquisition function; there is literature that can help.
- Reduce redundant kernel recomputation — currently the full K matrix is rebuilt each iteration when only the new row/column is needed.
- Length scale tuning currently uses a fixed Adam step count; smarter convergence criteria could help.

## 📝 Citation

If you use Hyperoptax in academic work, please cite:

```bibtex
@misc{hyperoptax,
  author = {Theo Wolf},
  title = {{Hyperoptax}: Parallel hyperparameter tuning with JAX},
  year = {2025},
  url = {https://github.com/TheodoreWolf/hyperoptax}
}
```
