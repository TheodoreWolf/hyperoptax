import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


class BaseAcquisition:
    def __init__(self):
        pass

    def __call__(self, mean: jax.Array, std: jax.Array):
        raise NotImplementedError

    def get_argmax(
        self, mean: jax.Array, std: jax.Array, seen_idx: jax.Array, n_points: int = 1
    ):
        # Acquisition values for all points
        acq_vals = self(mean, std)  # shape (N,)

        # Boolean mask of points that have already been evaluated.
        idxs = jnp.arange(acq_vals.shape[0])
        seen_mask = jnp.isin(idxs, seen_idx)

        # Replace acquisition values of seen points with -inf so they are never selected
        masked_acq = jnp.where(seen_mask, -jnp.inf, acq_vals)

        return jnp.argsort(masked_acq)[-n_points:]

    def get_max(
        self, mean: jax.Array, std: jax.Array, X: jax.Array, seen_idx: jax.Array
    ):
        return X[self.get_argmax(mean, std, seen_idx)]


class UCB(BaseAcquisition):
    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def __call__(self, mean: jax.Array, std: jax.Array):
        return mean + self.kappa * std


class EI(BaseAcquisition):
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array):
        y_max = jnp.max(mean)
        a = mean - self.xi - y_max
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


# TODO: ConstantLiar as detailed in https://hal.science/hal-00260579v1/document
