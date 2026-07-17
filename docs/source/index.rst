Hyperoptax
==========

Hyperoptax is a functional hyperparameter-optimization library for JAX. It
supports PyTree search spaces, parallel candidate evaluation, Python and
``jax.lax.scan`` loops, and random, grid, and Gaussian-process Bayesian search.

Quick start
-----------

Every objective has signature ``objective(key, params) -> scalar``. Optimizers
are initialized as an immutable state PyTree plus a stateless optimizer object.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from hyperoptax import BayesianSearch, LinearSpace, LogSpace


   def objective(key, params):
       noise = 0.01 * jax.random.normal(key)
       return (
           (jnp.log10(params["learning_rate"]) + 3.0) ** 2
           + (params["dropout"] - 0.2) ** 2
           + noise
       )


   space = {
       "learning_rate": LogSpace(1e-5, 1e-1),
       "dropout": LinearSpace(0.0, 0.5),
   }
   state, optimizer = BayesianSearch.init(
       space,
       n_max=80,
       n_parallel=4,
       maximize=False,
   )
   state, history = optimizer.optimize_scan(
       state,
       jax.random.PRNGKey(0),
       objective,
       n_iterations=20,
   )

   print(optimizer.best_result(state))
   print(optimizer.best_params(state))

The example performs 80 evaluations: 20 iterations with four candidates per
iteration. See :doc:`usage` for buffer semantics, optimizer selection, and the
lower-level ask/tell-style API.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   usage
   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/optimizers
   api/spaces
   api/acquisition
   api/kernels
