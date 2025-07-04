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
    base: float | int = 10
    name: str = "log_space"

    def __post_init__(self):
        # JAX silently converts negative numbers to nan
        assert self.start > 0 and self.end > 0 and self.base > 0, (
            "Log space must be positive and have a positive log base."
        )

    @property
    def array(self) -> jax.Array:
        log_space = jnp.linspace(
            self.log(self.start), self.log(self.end), self.n_points
        )
        return self.base**log_space

    def log(self, x: float) -> float:
        # conersion of log base
        return jnp.log(x) / jnp.log(self.base)


@dataclass
class ExpSpace(LogSpace):
    base: float | int = 10
    name: str = "exp_space"

    def __post_init__(self):
        # JAX silently converts negative numbers to nan
        assert self.base > 0, "Base must be positive."

    @property
    def array(self) -> jax.Array:
        return self.log(
            jnp.linspace(self.base**self.start, self.base**self.end, self.n_points)
        )


@dataclass
class QuantizedLinearSpace:
    start: int | float
    end: int | float
    quantization_factor: int | float
    name: str = "quantized_space"

    def __post_init__(self):
        self.n_points = jnp.int32(
            (self.end - self.start) / self.quantization_factor + 1
        )

    @property
    def array(self) -> jax.Array:
        return jnp.linspace(self.start, self.end, self.n_points)


# class QuantizedLogSpace(QuantizedLinearSpace):
#     base: float | int = 10
#     name: str = "quantized_log_space"

#     @property
#     def array(self) -> jax.Array:
#         arr = jnp.log(super().array) / jnp.log(self.base)
#         return self.base**arr


# TODO: add distribution versions
# TODO: add support for nested spaces with pytrees
