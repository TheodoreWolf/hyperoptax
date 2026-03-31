.. Hyperoptax documentation master file, created by
   sphinx-quickstart on Tue Jul 15 22:39:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <div class="wip-warning">
   🚧 WORK IN PROGRESS - This documentation is currently under development 🚧
   </div>

Hyperoptax Documentation
========================

Welcome to Hyperoptax - a lightweight toolbox for parallel hyperparameter optimization of pure JAX functions.

Hyperoptax provides a concise API that lets you wrap any JAX-compatible loss or evaluation function and search across parameter spaces **in parallel** - all while staying in pure JAX.

Quick Start
-----------

.. code-block:: python

   import jax
   from hyperoptax import BayesianSearch, LogSpace, LinearSpace

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
       n_max=100,
       maximize=False,
   )
   state, (params_hist, results_hist) = optimizer.optimize(
       state, jax.random.PRNGKey(0), train_nn
   )
   print(optimizer.best_params(state))

.. toctree::
   :maxdepth: 2
   :titlesonly:

   api/optimizers
   api/spaces
   api/kernels
   api/acquisition

