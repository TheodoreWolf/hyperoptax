import jax
import jax.numpy as jnp
from flax import struct


# transformation between logs
def log_transform(x: float, base: float) -> float:
    return jnp.log(x) / jnp.log(base)


class Space(struct.PyTreeNode):
    pass

    def sample(self, key: jax.random.PRNGKey) -> jax.Array:
        raise NotImplementedError

    def transform(self, value):
        return value


class LinearSpace(Space):
    lower_bound: float = struct.field(pytree_node=False)
    upper_bound: float = struct.field(pytree_node=False)

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


class DiscreteSpace(Space):
    values: tuple = struct.field(pytree_node=False)

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


class LogSpace(LinearSpace):
    base: float = struct.field(pytree_node=False, default=10)

    def __post_init__(
        self,
    ):
        super().__post_init__()
        assert self.base > 1, "Log base must be greater than 1"

    def sample(self, key: jax.random.PRNGKey) -> float:
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
class QLinearSpace(LinearSpace):
    datatype: type = struct.field(pytree_node=False, default=jnp.int32)

    def transform(self, value) -> jax.Array:
        return jnp.round(value).astype(self.datatype)


class QLogSpace(LogSpace, QLinearSpace):
    pass
