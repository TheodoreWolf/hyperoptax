import inspect
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from hyperoptax.base import BaseOptimiser
from hyperoptax.spaces import BaseSpace


class GridSearch(BaseOptimiser):
    def __init__(self, domain: dict[str, BaseSpace], f: Callable, n_parallel: int = 10):
        super().__init__(domain, f, n_parallel)

        n_args = len(inspect.signature(f).parameters)
        n_points = np.prod([len(domain[k]) for k in domain])
        assert n_args == len(domain), (
            f"Function must have the same number of arguments as the domain, "
            f"got {n_args} arguments and {len(domain)} domains."
        )
        grid = jnp.array(jnp.meshgrid(*[space.array for space in domain.values()]))
        self.domain = grid.reshape(n_args, n_points).T

    def optimise(self, n_iterations: int = -1):
        """Evaluate the objective on ``n_iterations`` points of the grid.

        The evaluation is performed in batches of ``self.n_parallel`` elements so
        that we can still JIT-compile the inner loop without triggering the
        *dynamic slice* error observed when using NumPy-style indexing with a
        traced value.  We therefore rely on ``jax.lax.dynamic_slice`` whose
        indices can be dynamic while the slice *sizes* remain static.
        """

        # Select the portion of the grid we want to evaluate
        if n_iterations == -1:
            domain = self.domain
            n_iterations = domain.shape[0]
        else:
            domain = self.domain[:n_iterations]

        # Helper that vectorises the objective over the first (point) axis
        vmap_f = jax.vmap(self.f, in_axes=(0,) * domain.shape[1])

        # Number of batches we need to cover all requested iterations
        n_batches = (n_iterations + self.n_parallel - 1) // self.n_parallel

        n_dims = domain.shape[1]  # static – number of arguments of f

        def _inner_loop(start_idx, _):
            """Evaluate a single batch starting at ``start_idx``."""
            # Ensure we stay within bounds. The clamp keeps the slice valid even
            # when the last batch is not full (extra rows are discarded later).
            start_idx = jnp.minimum(start_idx, n_iterations - self.n_parallel)

            batch = jax.lax.dynamic_slice(
                domain,
                (start_idx, 0),
                (self.n_parallel, n_dims),
            )

            batch_results = vmap_f(*batch.T)
            return start_idx + self.n_parallel, batch_results

        # Scan over all batches
        _, batch_results = jax.lax.scan(_inner_loop, 0, None, length=n_batches)

        # Flatten and truncate the padded tail (if any)
        results = jnp.concatenate(batch_results, axis=0)[:n_iterations]

        # Identify (potentially multiple) maxima
        max_idxs = jnp.where(results == results.max())[0]
        return domain[max_idxs]

    # TODO: pmap support
    # TODO: handle multiple maxima properly
    # TODO: add support for minimisation


class RandomSearch(GridSearch):
    def __init__(
        self,
        domain: dict[str, BaseSpace],
        f: Callable,
        n_parallel: int = 10,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        super().__init__(domain, f, n_parallel)
        idxs = jax.random.choice(
            key, self.domain.shape[0], (self.domain.shape[0],), replace=False
        )
        self.domain = self.domain[idxs]
