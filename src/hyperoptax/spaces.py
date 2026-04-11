from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


# transformation between logs
def log_transform(x: float, base: float) -> float:
    return jnp.log(x) / jnp.log(base)


@dataclass(frozen=True)
class Space(ABC):
    """Abstract base class for hyperparameter search spaces."""

    @abstractmethod
    def sample(self, key: jax.random.PRNGKey) -> jax.Array:
        raise NotImplementedError

    def transform(self, value):
        return value


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearSpace(Space):
    """Uniform continuous space over ``[lower_bound, upper_bound]``.

    Attributes:
        lower_bound: Inclusive lower bound of the interval.
        upper_bound: Exclusive upper bound of the interval.
    """

    lower_bound: float = field(metadata=dict(static=True))
    upper_bound: float = field(metadata=dict(static=True))

    def __post_init__(self):
        assert self.lower_bound < self.upper_bound, (
            "lower_bound is greater or equal to upper_bound."
        )

    def sample(self, key: jax.random.PRNGKey) -> float:
        return self.transform(
            jax.random.uniform(
                key, shape=(1,), minval=self.lower_bound, maxval=self.upper_bound
            )
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteSpace(Space):
    """Discrete space over a fixed set of values.

    Samples uniformly from ``values``. ``transform`` snaps any continuous
    value to the nearest element, which is useful when discrete candidates
    are generated via continuous optimization (e.g. in ``BayesianSearch``).

    Attributes:
        values: Tuple of candidate values to sample from.
    """

    values: tuple = field(metadata=dict(static=True))

    @property
    def lower_bound(self) -> float:
        return float(min(self.values))

    @property
    def upper_bound(self) -> float:
        return float(max(self.values))

    def sample(self, key: jax.random.PRNGKey) -> float:
        return self.transform(
            jax.random.choice(key, jnp.array(self.values), shape=(1,))
        )

    def transform(self, value) -> jax.Array:
        vals = jnp.array(self.values)
        value = jnp.asarray(value)
        flat = jnp.ravel(value)
        snapped = vals[jnp.argmin(jnp.abs(vals[:, None] - flat[None, :]), axis=0)]
        return snapped.reshape(value.shape)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LogSpace(LinearSpace):
    """Log-uniform continuous space over ``[lower_bound, upper_bound]``.

    Samples uniformly in log space so that each order of magnitude receives
    equal probability mass. Useful for learning rates and other scale
    parameters that span several orders of magnitude.

    Attributes:
        lower_bound: Inclusive lower bound (in original scale, e.g. ``1e-5``).
        upper_bound: Exclusive upper bound (in original scale, e.g. ``1e-1``).
        base: Logarithm base (default ``10``). Must be greater than 1.
    """

    base: float = field(default=10, metadata=dict(static=True))

    def __post_init__(self):
        super().__post_init__()
        assert self.base > 1, "Log base must be greater than 1"

    def sample(self, key: jax.random.PRNGKey) -> jax.Array:
        return self.transform(
            self.base
            ** jax.random.uniform(
                key,
                shape=(1,),
                minval=log_transform(self.lower_bound, self.base),
                maxval=log_transform(self.upper_bound, self.base),
            )
        )


# TODO: maybe use something more robust than astype?
# TODO: can we do something with mixins? Currently hitting some ordering problems
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QLinearSpace(LinearSpace):
    """Quantized (integer) variant of :class:`LinearSpace`.

    Samples uniformly from ``[lower_bound, upper_bound]`` and rounds to the
    nearest integer. Use this for discrete integer hyperparameters with a
    uniform prior (e.g. number of layers, batch size).

    Attributes:
        lower_bound: Inclusive lower bound.
        upper_bound: Exclusive upper bound.
        datatype: Integer dtype used after rounding (default ``jnp.int32``).
    """

    datatype: type = field(default=jnp.int32, metadata=dict(static=True))

    def transform(self, value) -> jax.Array:
        return jnp.round(value).astype(self.datatype)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QLogSpace(LogSpace):
    """Quantized (integer) variant of :class:`LogSpace`.

    Samples in log space and rounds to the nearest integer. Use this for
    integer hyperparameters whose scale spans orders of magnitude
    (e.g. number of hidden units, number of warmup steps).
    """

    datatype: type = field(default=jnp.int32, metadata=dict(static=True))

    def transform(self, value) -> jax.Array:
        return jnp.round(value).astype(self.datatype)
