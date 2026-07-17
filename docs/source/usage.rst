Using Hyperoptax
================

Installation
------------

.. code-block:: console

   uv pip install hyperoptax

Hyperoptax requires Python 3.10 or newer. Install accelerator-specific JAX
wheels by following the `JAX installation guide
<https://docs.jax.dev/en/latest/installation.html>`_.

Choosing an optimizer
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Optimizer
     - Best suited to
     - Constraints
   * - :class:`~hyperoptax.random.RandomSearch`
     - Cheap baseline and broad exploration
     - Stateless; does not retain observations.
   * - :class:`~hyperoptax.grid.GridSearch`
     - Small, explicitly enumerated grids
     - Every leaf must be a numeric
       :class:`~hyperoptax.spaces.DiscreteSpace`. The grid size should be
       divisible by ``n_parallel``.
   * - :class:`~hyperoptax.bayesian.BayesianSearch`
     - Expensive, smooth objectives
     - Stores observations in a fixed ``n_max`` buffer; GP work grows with the
       buffer size.

Iterations, batches, and buffers
--------------------------------

``n_parallel`` is the number of configurations evaluated in each optimizer
iteration. Therefore ``n_iterations * n_parallel`` is the number of objective
evaluations.

Bayesian search uses fixed-size buffers so its state has static JAX shapes.
``n_max`` is the number of observations that fit in the buffer, not the number
of iterations. When ``n_iterations`` is omitted, Bayesian search runs until the
buffer is full. Keep ``n_max`` divisible by ``n_parallel`` so the last batch
fits exactly.

Grid search builds the full Cartesian product during ``init``. If the grid size
is not divisible by ``n_parallel``, its incomplete final batch is omitted.

Loop APIs
---------

:meth:`~hyperoptax.base.Optimizer.optimize` runs the outer loop in Python. It
returns ``params_history`` and ``results_history`` as lists indexed by
iteration. The parameter PyTrees have a leading ``n_parallel`` axis and each
result array has shape ``(n_parallel,)``.

:meth:`~hyperoptax.base.Optimizer.optimize_scan` uses ``jax.lax.scan`` and
requires a JAX-traceable objective. It returns stacked parameter leaves with
shape ``(n_iterations, n_parallel, ...)`` and results with shape
``(n_iterations, n_parallel)``.

Both APIs evaluate the batch with ``jax.vmap``. The objective must accept one
unbatched configuration and return a scalar.

External evaluation loop
------------------------

Use :meth:`~hyperoptax.base.Optimizer.get_next_params` and
:meth:`~hyperoptax.base.Optimizer.update_state` directly when a job scheduler,
service, or other external system evaluates configurations.

.. code-block:: python

   import jax

   key = jax.random.PRNGKey(0)
   previous_params = None
   previous_results = None

   for _ in range(20):
       key, ask_key, eval_key, tell_key = jax.random.split(key, 4)
       params = optimizer.get_next_params(
           state, ask_key, previous_params, previous_results
       )
       eval_keys = jax.random.split(eval_key, optimizer.n_parallel)
       results = jax.vmap(objective)(eval_keys, params)
       state = optimizer.update_state(state, tell_key, results, params)
       previous_params, previous_results = params, results

JAX constraints
---------------

* Search-space leaves and objective results must be representable as JAX
  arrays. String categories are not supported; encode them numerically.
* Data-dependent Python control flow, variable-shaped outputs, and side effects
  are not compatible with ``optimize_scan``.
* Parameters that change evaluation shape or duration may not batch cleanly.
  Use ``n_parallel=1`` when evaluations cannot be synchronized.
