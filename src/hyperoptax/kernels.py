from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


def cdist(x: jax.Array, y: jax.Array) -> jax.Array:
    # jax compatible cdist https://github.com/jax-ml/jax/discussions/15862
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


class BaseKernel(ABC):
    @abstractmethod
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        raise NotImplementedError


# TODO: add basic operations between kernels
# TODO: add Matern kernel

class RBF(BaseKernel):
    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.exp(-(cdist(x, y) ** 2) / (2 * self.length_scale**2))

    def diag(self, x: jax.Array) -> jax.Array:
        return jnp.ones(x.shape[0]) * self.length_scale**2
