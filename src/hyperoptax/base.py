from abc import ABC, abstractmethod
import inspect
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp


# TODO: make sure dictionary is ordered
class BaseOptimiser(ABC):
    def __init__(self, domain: dict[str, jax.Array], f: Callable):
        self.f = f
        n_args = len(inspect.signature(f).parameters)
        n_points = np.prod([len(domain[k]) for k in domain])
        assert n_args == len(domain), (
            f"Function must have the same number of arguments as the domain, "
            f"got {n_args} arguments and {len(domain)} domains."
        )
        grid = jnp.array(jnp.meshgrid(*[space.array for space in domain.values()]))
        self.domain = grid.reshape(n_args, n_points).T

    @abstractmethod
    def optimise(
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

# TODO: add progress bar https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/