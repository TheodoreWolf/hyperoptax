Notebooks
=========

The repository contains executable studies rather than embedding notebook
outputs in the documentation build:

* `Search examples <https://github.com/TheodoreWolf/hyperoptax/blob/main/notebooks/search.ipynb>`_:
  grid search and Bayesian optimization.
* `Bayesian optimization design study <https://github.com/TheodoreWolf/hyperoptax/blob/main/notebooks/bo_design_study.ipynb>`_:
  acquisition functions and their hyperparameters.
* `High-dimensional behavior <https://github.com/TheodoreWolf/hyperoptax/blob/main/notebooks/high_dimensional.ipynb>`_:
  Bayesian versus random search from 10 to 1,000 dimensions.
* `Performance comparison <https://github.com/TheodoreWolf/hyperoptax/blob/main/notebooks/performance.ipynb>`_:
  comparisons with other optimization libraries and batch sizes.
* `RL hyperparameters <https://github.com/TheodoreWolf/hyperoptax/blob/main/notebooks/rl_hparams.ipynb>`_:
  tuning Rejax reinforcement-learning runs.
* `Bayesian optimization visualization <https://github.com/TheodoreWolf/hyperoptax/blob/main/notebooks/visualization.ipynb>`_:
  GP posterior and acquisition behavior.

Install the optional dependencies before running them:

.. code-block:: console

   uv pip install -e ".[notebooks]"
