import logging
from typing import Callable

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from hyperoptax.base import BaseOptimizer
from hyperoptax.spaces import BaseSpace

logger = logging.getLogger(__name__)


class GridSearch(BaseOptimizer):
    def __init__(
        self,
        domain: dict[str, BaseSpace],
        f: Callable,
        random_search: bool = False,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        super().__init__(domain, f)
        if random_search:
            idxs = jax.random.choice(
                key, self.domain.shape[0], (self.domain.shape[0],), replace=False
            )
            self.domain = self.domain[idxs]

    def search(
        self,
        n_iterations: int,
        n_parallel: int,
    ):
        domain = self.domain[:n_iterations]

        # Number of batches we need to cover all requested iterations
        n_batches = (n_iterations + n_parallel - 1) // n_parallel

        n_dims = domain.shape[1]  # static – number of arguments of f

        @scan_tqdm(n_batches)
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
        _, batch_results = jax.lax.scan(
            _inner_loop, 0, jnp.arange(n_batches), length=n_batches
        )

        # Flatten and truncate the padded tail (if any)
        results = jnp.concatenate(batch_results, axis=0)[:n_iterations]

        return self.domain[:n_iterations], results
