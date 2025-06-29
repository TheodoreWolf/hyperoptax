from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class BaseSpace:
    start: float | int
    end: float | int
    n_points: float | int

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
class ArbitrarySpace:
    values: list[int | float]
    name: str = "arbitrary_space"

    def __post_init__(self):
        assert self.array.ndim == 1, (
            "I don't support arrays that aren't one dimensional (yet), "
            "try entering each dimension as a separate space."
        )
        self.start = jnp.min(self.array)
        self.end = jnp.max(self.array)
        self.n_points = len(self.array)

    @property
    def array(self) -> jax.Array:
        return jnp.array(self.values)


@dataclass
class LinearSpace(BaseSpace):
    name: str = "linear_space"

    @property
    def array(self) -> jax.Array:
        return jnp.linspace(self.start, self.end, self.n_points)


@dataclass
class LogSpace(BaseSpace):
    log_base: float | int = 10
    name: str = "log_space"

    def __post_init__(self):
        # JAX silently converts negative numbers to nan
        assert self.start > 0 and self.end > 0 and self.log_base > 0, (
            "Log space must be positive and have a positive log base."
        )

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
