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
        results = []
        # If n_iterations is -1 (default), use the whole domain
        if n_iterations == -1:
            domain = self.domain
        else:
            domain = self.domain[:n_iterations]
        # could use jax.lax.fori_loop here for pure jax
        for i in range(0, domain.shape[0], self.n_parallel):
            results.append(
                jax.vmap(self.f, in_axes=(0,) * domain.shape[1])(
                    *(domain[i : i + self.n_parallel].T)
                )
            )
        results = jnp.concatenate(results, axis=0)
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
