import jax
import jax.numpy as jnp


class BaseAquisition:
    def __init__(self):
        pass

    def __call__(self, mean: jax.Array, std: jax.Array):
        raise NotImplementedError

    def get_argmax(
        self, mean: jax.Array, std: jax.Array, X: jax.Array, seen_idx: jax.Array
    ):
        raise NotImplementedError


class UCB(BaseAquisition):
    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def __call__(self, mean: jax.Array, std: jax.Array):
        return mean + self.kappa * std

    def get_argmax(self, mean: jax.Array, std: jax.Array, seen_idx: jax.Array):
        idx = jnp.argsort(self(mean, std))
        idx = idx[~jnp.isin(idx, seen_idx)]
        return idx[-1]

    def get_max(
        self, mean: jax.Array, std: jax.Array, X: jax.Array, seen_idx: jax.Array
    ):
        return X[self.get_argmax(mean, std, seen_idx)]


# TODO: More aquisition functions
