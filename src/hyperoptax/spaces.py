from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class BaseSpace:
    start: float
    end: float
    n_points: int

    def __len__(self) -> int:
        return self.n_points
    
    @property
    def array(self) -> jax.Array:
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> jax.Array:
        return self.array[idx]
    
    def __iter__(self):
        return iter(self.array)


@dataclass
class LinearSpace(BaseSpace):
    name: str = "linear_space"

    @property
    def array(self) -> jax.Array:
        return jnp.linspace(self.start, self.end, self.n_points)


@dataclass
class LogSpace(BaseSpace):
    log_base: float = 10
    name: str = "log_space"

    @property
    def array(self) -> jax.Array:
        log_space = jnp.linspace(
            self.log(self.start), self.log(self.end), self.n_points
        )
        return self.log_base**log_space

    def log(self, x: float) -> float:
        # conersion of log base
        return jnp.log(x) / jnp.log(self.log_base)

# TODO; quantise version of each
# TODO: add distribution versions