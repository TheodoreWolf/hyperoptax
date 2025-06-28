from abc import ABC, abstractmethod
from typing import Callable

import jax


class BaseOptimiser(ABC):
    def __init__(self, domain: dict[str, jax.Array], f: Callable):
        self.f = f

    @abstractmethod
    def optimise(self, n_iterations: int, n_parallel: int):
        NotImplementedError

