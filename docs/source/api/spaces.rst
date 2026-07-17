Search spaces
=============

Search spaces are JAX PyTree leaves and may be nested in dicts, lists, tuples,
or other PyTree containers. Candidate batches preserve the same structure and
add a leading ``n_parallel`` axis to every leaf.

.. code-block:: python

   import jax.numpy as jnp

   from hyperoptax import (
       DiscreteSpace,
       LinearSpace,
       LogSpace,
       QLinearSpace,
       QLogSpace,
   )

   space = {
       "optimizer": {
           "learning_rate": LogSpace(1e-5, 1e-1),
           "weight_decay": LinearSpace(0.0, 0.1),
       },
       "depth": QLinearSpace(2, 8, datatype=jnp.int32),
       "width": QLogSpace(32, 1024, datatype=jnp.int32),
       "batch_size": DiscreteSpace((32, 64, 128, 256)),
   }

``DiscreteSpace`` values must be numeric. Encode string categories as numbers
and map them back to labels outside the vmapped objective when needed.

API
---

.. automodule:: hyperoptax.spaces
   :members:
   :exclude-members: datatype
   :show-inheritance:
