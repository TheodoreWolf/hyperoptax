Parameter Spaces
================

Parameter spaces define the search domains for hyperparameter optimization.

.. automodule:: hyperoptax.spaces
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating a Linear Space
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax import LinearSpace

   dropout_space = LinearSpace(0.0, 0.5)

Creating a Logarithmic Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax import LogSpace

   lr_space = LogSpace(1e-5, 1e-1)

Creating a Discrete Space
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax import DiscreteSpace

   optimizer_space = DiscreteSpace(["adam", "sgd", "rmsprop"])
   lr_space = DiscreteSpace([1e-4, 1e-3, 1e-2])