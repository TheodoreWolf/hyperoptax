Acquisition and hallucination strategies
========================================

Bayesian search scores candidate points with an acquisition function.
Probability of improvement is the default; expected improvement and upper
confidence bound are also available.

Parallel Bayesian batches are selected sequentially. After each selection, a
hallucination strategy supplies a temporary objective value so the following
slot explores a different part of the posterior. Posterior sampling is the
default; mean, UCB, and constant strategies are available.

.. code-block:: python

   from hyperoptax import BayesianSearch, EI, MeanHallucination, UCB

   state, optimizer = BayesianSearch.init(
       space,
       n_max=80,
       n_parallel=4,
       acquisition=EI(xi=0.01),
       hallucination=MeanHallucination(),
   )

   # UCB is an alternative acquisition strategy.
   state, optimizer = BayesianSearch.init(
       space,
       n_max=80,
       acquisition=UCB(kappa=2.0),
   )

API
---

.. automodule:: hyperoptax.acquisition
   :members:
   :show-inheritance:
