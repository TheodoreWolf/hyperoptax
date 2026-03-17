from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


def cdist(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pairwise Euclidean distance (``cdist``) between two 2-D arrays.

    Parameters
    ----------
    x, y : jax.Array
        Arrays with shape ``(N, D)`` and ``(M, D)``, respectively.

    Returns
    -------
    jax.Array
        A distance matrix of shape ``(N, M)``.
    """
    # jax compatible cdist https://github.com/jax-ml/jax/discussions/15862
    # Use double-where trick to avoid inf gradients at zero distance (sqrt'(0) = inf).
    d2 = jnp.sum((x[:, None] - y[None, :]) ** 2, -1)
    safe_d2 = jnp.where(d2 == 0, jnp.ones_like(d2), d2)
    return jnp.where(d2 == 0, jnp.zeros_like(d2), jnp.sqrt(safe_d2))


class BaseKernel(ABC):
    """Abstract base class for positive-definite kernels."""

    @abstractmethod
    def __call__(self, x: jax.Array, y: jax.Array, length_scale=None) -> jax.Array:
        raise NotImplementedError


# TODO: add basic operations between kernels
class RBF(BaseKernel):
    """Radial basis function (RBF) / squared-exponential kernel."""

    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def __call__(self, x: jax.Array, y: jax.Array, length_scale=None) -> jax.Array:
        ls = self.length_scale if length_scale is None else length_scale
        return jnp.exp(-(cdist(x, y) ** 2) / (2 * ls**2))


class Matern(BaseKernel):
    """Matern kernel family.

    Parameters
    ----------
    length_scale : float, default = 1.0
        Characteristic length scale.
    nu : float, default = 2.5
        Controls smoothness (``nu`` ∈ {0.5, 1.5, 2.5, ∞}).
    """

    def __init__(self, length_scale: float = 1.0, nu: float = 2.5):
        self.length_scale = length_scale
        self.nu = nu  # controls smoothness of the kernel, lower is less smooth

    def __call__(self, x: jax.Array, y: jax.Array, length_scale=None) -> jax.Array:
        ls = self.length_scale if length_scale is None else length_scale
        dists = cdist(x / ls, y / ls)
        if self.nu == 0.5:
            return jnp.exp(-dists)
        elif self.nu == 1.5:
            K = jnp.sqrt(3) * dists
            return (1 + K) * jnp.exp(-K)
        elif self.nu == 2.5:
            K = jnp.sqrt(5) * dists
            return (1 + K + K**2 / 3) * jnp.exp(-K)
        elif self.nu == jnp.inf:  # RBF kernel
            return jnp.exp(-(dists**2) / 2)
        else:
            raise ValueError(f"Matern kernel with nu={self.nu} is not supported.")
