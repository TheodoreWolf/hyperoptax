import inspect
import logging
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# TODO: use existing results if they exist
# TODO: add support for keys
# TODO: implement callback/wandb logging
class BaseOptimizer:
    def __init__(
        self,
        domain: dict[str, jax.Array],
        f: Callable,
        callback: Callable = lambda x: None,
    ):
        self.f = f
        self.callback = callback
        self.results = None

        n_args = len(inspect.signature(f).parameters)
        n_points = np.prod([len(domain[k]) for k in domain])
        if n_points > 1e6:
            # TODO: what do if the matrix is too large?
            logger.warning(
                f"Creating a {n_points}x{n_args} grid, this may be too large!"
            )

        assert n_args == len(domain), (
            f"Function must have the same number of arguments as the domain, "
            f"got {n_args} arguments and {len(domain)} domains."
        )
        grid = jnp.array(jnp.meshgrid(*[space.array for space in domain.values()]))
        self.domain = grid.reshape(n_args, n_points).T

    def optimize(
        self,
        n_iterations: int = -1,
        n_vmap: int = 1,
        n_pmap: int = 1,
        maximize: bool = True,
        jit: bool = False,
    ):
        if n_iterations == -1:
            n_iterations = self.domain.shape[0]

        if maximize:
            self.map_f = jax.vmap(self.f, in_axes=(0,) * self.domain.shape[1])
        else:
            self.map_f = jax.vmap(
                lambda *args: -self.f(*args), in_axes=(0,) * self.domain.shape[1]
            )

        if n_pmap > 1:
            assert n_iterations % n_pmap == 0, (
                "n_iterations must be divisible by n_pmap"
            )
            assert n_pmap == jax.local_device_count(), (
                "n_pmap must be equal to the number of devices"
            )
            # TODO: fix this for the bayesian optimizer
            domains = jnp.array(jnp.array_split(self.domain[:n_iterations], n_pmap))
            n_iterations = n_iterations // n_pmap
            X_seen, y_seen = jax.pmap(
                partial(self.search, n_iterations=n_iterations, n_vmap=n_vmap),
            )(domain=domains)

        # mostly for debugging purposes
        elif jit:
            X_seen, y_seen = jax.jit(self.search, static_argnums=(0, 1))(
                n_iterations, n_vmap
            )
        else:
            X_seen, y_seen = self.search(n_iterations, n_vmap)

        max_idxs = jnp.where(y_seen == y_seen.max())

        if not maximize:
            y_seen = -y_seen

        self.results = (X_seen, y_seen)

        return X_seen[max_idxs].squeeze()

    def search(self, n_iterations: int, n_parallel: int):
        raise NotImplementedError

    @property
    def max(self) -> dict[str, jax.Array]:
        assert self.results is not None, "No results found, run optimize first."
        return {
            "target": self.results[1].max(),
            "params": self.results[0][self.results[1].argmax()].flatten(),
        }

    @property
    def min(self) -> dict[str, jax.Array]:
        assert self.results is not None, "No results found, run optimize first."
        return {
            "target": self.results[1].min(),
            "params": self.results[0][self.results[1].argmin()].flatten(),
        }

    def shard_domain(self, n_iterations: int, n_shards: int):
        raise NotImplementedError
