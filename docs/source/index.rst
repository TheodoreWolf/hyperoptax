.. Hyperoptax documentation master file, created by
   sphinx-quickstart on Tue Jul 15 22:39:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hyperoptax Documentation
========================

Welcome to Hyperoptax - a lightweight toolbox for parallel hyperparameter optimization of pure JAX functions.

Hyperoptax provides a concise API that lets you wrap any JAX-compatible loss or evaluation function and search across parameter spaces **in parallel** – all while staying in pure JAX.

Quick Start
-----------

.. code-block:: python

   from hyperoptax.bayesian import BayesianOptimizer
   from hyperoptax.spaces import LogSpace, LinearSpace

   @jax.jit
   def train_nn(learning_rate, final_lr_pct):
       ...
       return val_loss

   search_space = {"learning_rate": LogSpace(1e-5,1e-1, 100),
                   "final_lr_pct": LinearSpace(0.01, 0.5, 100)}

   search = BayesianOptimizer(search_space, train_nn)
   best_params = search.optimize(n_iterations=100, 
                                 n_parallel=10, 
                                 maximize=False)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/optimizers
   api/spaces
   api/kernels
   api/acquisition

