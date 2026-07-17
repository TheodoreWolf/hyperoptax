Kernels
=======

Bayesian search uses a Matérn kernel with ``nu=0.5`` by default. RBF and Matérn
kernels accept scalar initial length scales; Bayesian search maintains and tunes
per-dimension ARD length scales in optimizer state.

.. code-block:: python

   from hyperoptax import BayesianSearch, Matern, RBF

   state, optimizer = BayesianSearch.init(
       space,
       n_max=80,
       kernel=Matern(length_scale=1.0, nu=1.5),
   )

   rbf = RBF(length_scale=0.5)

The supported Matérn smoothness values are ``0.5``, ``1.5``, ``2.5``, and
``float("inf")``. The infinite-smoothness case is equivalent to RBF.

API
---

.. automodule:: hyperoptax.kernels
   :members:
   :show-inheritance:
