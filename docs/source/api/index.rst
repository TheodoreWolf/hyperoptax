API Overview
============

Hyperoptax provides several key modules for hyperparameter optimization:

Core Modules
------------

.. autosummary::
   :toctree: _autosummary
   :caption: Core API

   hyperoptax.base
   hyperoptax.bayesian
   hyperoptax.grid
   hyperoptax.spaces
   hyperoptax.kernels
   hyperoptax.acquisition

Quick Reference
---------------

**Optimizers**
  - :class:`~hyperoptax.bayesian.BayesianOptimizer` - Bayesian optimization with Gaussian processes
  - :class:`~hyperoptax.grid.GridSearch` - Grid search and random search
  - :class:`~hyperoptax.base.BaseOptimizer` - Base class for all optimizers

**Search Spaces**
  - :class:`~hyperoptax.spaces.LinearSpace` - Linear parameter space
  - :class:`~hyperoptax.spaces.LogSpace` - Logarithmic parameter space

**Kernels**
  - :class:`~hyperoptax.kernels.RBF` - Radial basis function kernel
  - :class:`~hyperoptax.kernels.Matern` - Matern kernel

**Acquisition Functions**
  - :class:`~hyperoptax.acquisition.UCB` - Upper confidence bound
  - :class:`~hyperoptax.acquisition.EI` - Expected improvement 