Parameter Spaces
================

Parameter spaces define the search domains for hyperparameter optimization.

.. automodule:: hyperoptax.spaces
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating Linear Space
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax.spaces import LinearSpace
   
   # Create a linear space from 0.01 to 1.0 with 100 points
   lr_space = LinearSpace(0.01, 1.0, 100)

Creating Logarithmic Space
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax.spaces import LogSpace
   
   # Create a log space from 1e-5 to 1e-1 with 50 points
   lr_space = LogSpace(1e-5, 1e-1, 50) 