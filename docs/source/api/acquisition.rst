Acquisition Functions
====================

Acquisition functions determine which points to evaluate next in Bayesian optimization.

.. automodule:: hyperoptax.acquisition
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Upper Confidence Bound (UCB)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax.acquisition import UCB
   
   # Create UCB acquisition function
   acq = UCB(kappa=2.0)

Expected Improvement (EI)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax.acquisition import EI
   
   # Create EI acquisition function
   acq = EI(xi=0.01) 