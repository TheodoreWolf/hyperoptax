Optimizers
==========

All optimizers implement the functional interface defined by
:class:`hyperoptax.base.Optimizer`: initialize state, request a batch, evaluate
it, then update state with the results.

Base interface
--------------

.. automodule:: hyperoptax.base
   :members:
   :show-inheritance:

Random search
-------------

.. automodule:: hyperoptax.random
   :members:
   :show-inheritance:

Grid search
-----------

.. automodule:: hyperoptax.grid
   :members:
   :show-inheritance:

Bayesian search
---------------

.. automodule:: hyperoptax.bayesian
   :members:
   :show-inheritance:
