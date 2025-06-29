import logging
from typing import Callable

import jax
import jax.numpy as jnp

from hyperoptax.base import BaseOptimiser
from hyperoptax.spaces import BaseSpace

logger = logging.getLogger(__name__)


class GridSearch(BaseOptimiser):
    # TODO: dicts are not ordered, we might have some parameters that get swapped
    def __init__(self, domain: dict[str, BaseSpace], f: Callable):
        super().__init__(domain, f)

    def optimise(
        self,
        n_iterations: int = -1,
        n_parallel: int = 10,
        pmap: bool = False,
    ):
        # Select the portion of the grid we want to evaluate
        if n_iterations == -1:
            domain = self.domain
            n_iterations = domain.shape[0]
        else:
            domain = self.domain[:n_iterations]

        if pmap:
            n_devices = jax.device_count()
            map_f = jax.pmap(self.f, in_axes=(0,) * domain.shape[1])
            logger.warning(
                f"Using pmap with {n_devices} devices, "
                f"but {n_parallel} parallel evaluations was requested."
                f"Overriding n_parallel from {n_parallel} to {n_devices}."
            )
            n_parallel = n_devices
        else:
            map_f = jax.vmap(self.f, in_axes=(0,) * domain.shape[1])

        # Number of batches we need to cover all requested iterations
        n_batches = (n_iterations + n_parallel - 1) // n_parallel

        n_dims = domain.shape[1]  # static – number of arguments of f

        def _inner_loop(start_idx, _):
            """Evaluate a single batch starting at ``start_idx``."""
            # Ensure we stay within bounds. The clamp keeps the slice valid even
            # when the last batch is not full (extra rows are discarded later).
            start_idx = jnp.minimum(start_idx, n_iterations - n_parallel)

            batch = jax.lax.dynamic_slice(
                domain,
                (start_idx, 0),
                (n_parallel, n_dims),
            )

            batch_results = map_f(*batch.T)
            return start_idx + n_parallel, batch_results

        # Scan over all batches
        _, batch_results = jax.lax.scan(_inner_loop, 0, None, length=n_batches)

        # Flatten and truncate the padded tail (if any)
        results = jnp.concatenate(batch_results, axis=0)[:n_iterations]

        # Identify (potentially multiple) maxima
        max_idxs = jnp.where(results == results.max())[0]
        # save the results for later use
        self.results = results
        return domain[max_idxs]

    # TODO: handle multiple maxima properly
    # TODO: add support for minimisation


class RandomSearch(GridSearch):
    def __init__(
        self,
        domain: dict[str, BaseSpace],
        f: Callable,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        super().__init__(domain, f)
        idxs = jax.random.choice(
            key, self.domain.shape[0], (self.domain.shape[0],), replace=False
        )
        self.domain = self.domain[idxs]
