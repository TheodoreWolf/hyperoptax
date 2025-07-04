from abc import ABC, abstractmethod
import inspect
import logging
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    def __init__(self, domain: dict[str, jax.Array], f: Callable):
        self.f = f
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

    @abstractmethod
    def optimize(
        self,
        n_iterations: int,
        n_parallel: int,
        jit: bool = False,
        maximise: bool = True,
        pmap: bool = False,
        save_results: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def search(self, n_iterations: int, n_parallel: int):
        raise NotImplementedError
