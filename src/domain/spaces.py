from abc import ABC, abstractmethod
import jax

class Space(ABC):
    @abstractmethod
    def sample(self, key: jax.random.KeyArray) -> jax.Array:
        pass

    @abstractmethod
    def contains(self, x: jax.Array) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

