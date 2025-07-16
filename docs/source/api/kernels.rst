Kernels
=======

Kernels are used by the Gaussian process in Bayesian optimization to model function similarity.

.. automodule:: hyperoptax.kernels
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

RBF Kernel
~~~~~~~~~~~

.. code-block:: python

   from hyperoptax.kernels import RBF
   
   # Create RBF kernel with length scale 1.0
   kernel = RBF(length_scale=1.0)

Matern Kernel
~~~~~~~~~~~~~

.. code-block:: python

   from hyperoptax.kernels import Matern
   
   # Create Matern kernel with custom parameters
   kernel = Matern(length_scale=1.0, nu=2.5) 