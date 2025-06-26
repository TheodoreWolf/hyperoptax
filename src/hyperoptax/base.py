from abc import ABC, abstractmethod
from typing import Callable

import jax


class BaseOptimiser(ABC):
    def __init__(self, domain: dict[str, jax.Array], f: Callable, n_parallel: int):
        self.f = f
        self.n_parallel = n_parallel

    @abstractmethod
    def optimise(self, n_iterations: int):
        pass

