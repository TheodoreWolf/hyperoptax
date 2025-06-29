import logging
from typing import Callable

import jax
import jax.numpy as jnp

from hyperoptax.base import BaseOptimiser
from hyperoptax.spaces import BaseSpace

logger = logging.getLogger(__name__)


class GridSearch(BaseOptimiser):
    def __init__(self, domain: dict[str, BaseSpace], f: Callable, pmap: bool = False):
        super().__init__(domain, f)

    def search(
        self,
        n_iterations: int = -1,
        n_parallel: int = 10,
    ):
        # Select the portion of the grid we want to evaluate
        if n_iterations == -1:
            domain = self.domain
            n_iterations = domain.shape[0]
        else:
            domain = self.domain[:n_iterations]

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

            batch_results = self.map_f(*batch.T)
            return start_idx + n_parallel, batch_results

        # Scan over all batches of parameters
        _, batch_results = jax.lax.scan(_inner_loop, 0, None, length=n_batches)

        # Flatten and truncate the padded tail (if any)
        results = jnp.concatenate(batch_results, axis=0)[:n_iterations]

        return results

    def optimise(
        self,
        n_iterations: int = -1,
        n_parallel: int = 10,
        jit: bool = False,
        maximise: bool = True,
        pmap: bool = False,
        save_results: bool = False,
    ):
        if pmap:
            n_devices = jax.device_count()
            self.map_f = jax.pmap(self.f, in_axes=(0,) * self.domain.shape[1])
            logger.warning(
                f"Using pmap with {n_devices} devices, "
                f"but {n_parallel} parallel evaluations was requested."
                f"Overriding n_parallel from {n_parallel} to {n_devices}."
            )
            n_parallel = n_devices
        else:
            self.map_f = jax.vmap(self.f, in_axes=(0,) * self.domain.shape[1])
        if jit:
            results = jax.jit(self.search, static_argnums=(0, 1))(
                n_iterations, n_parallel
            )
        else:
            results = self.search(n_iterations, n_parallel)
        if save_results:
            self.results = results
        # Identify (potentially multiple) maxima
        if maximise:
            max_idxs = jnp.where(results == results.max())[0]
        else:
            max_idxs = jnp.where(results == results.min())[0]
        return self.domain[max_idxs].flatten()


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
