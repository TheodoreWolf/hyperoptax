import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


class BaseAcquisition:
    def __init__(self, stochastic_multiplier: int = 1):
        self.stochastic_multiplier = stochastic_multiplier

    def __call__(self, mean: jax.Array, std: jax.Array):
        raise NotImplementedError

    def sort_acq_vals(self, mean: jax.Array, std: jax.Array, seen_idx: jax.Array):
        # Acquisition values for all points
        acq_vals = self(mean, std)  # shape (N,)

        # Boolean mask of points that have already been evaluated.
        idxs = jnp.arange(acq_vals.shape[0])
        seen_mask = jnp.isin(idxs, seen_idx)

        # Replace acquisition values of seen points with -inf so they are never selected
        masked_acq = jnp.where(seen_mask, -jnp.inf, acq_vals)

        return jnp.argsort(masked_acq)

    def get_argmax(
        self, mean: jax.Array, std: jax.Array, seen_idx: jax.Array, n_points: int = 1
    ):
        return self.sort_acq_vals(mean, std, seen_idx)[-n_points:]

    def get_stochastic_argmax(
        self,
        mean: jax.Array,
        std: jax.Array,
        seen_idx: jax.Array,
        n_points: int,
        key: jax.random.PRNGKey,
    ):
        # We sample points randomly the top n_points * stochastic_multiplier
        # to avoid selecting points that are very close to each other.
        sample_idx = jax.random.choice(
            key,
            jnp.arange(n_points * self.stochastic_multiplier),
            (n_points,),
            replace=False,
        )
        return self.sort_acq_vals(mean, std, seen_idx)[::-1][sample_idx]

    def get_max(
        self, mean: jax.Array, std: jax.Array, X: jax.Array, seen_idx: jax.Array
    ):
        return X[self.get_argmax(mean, std, seen_idx)]


class UCB(BaseAcquisition):
    def __init__(self, kappa: float = 2.0, stochastic_multiplier: int = 2):
        super().__init__(stochastic_multiplier)
        self.kappa = kappa

    def __call__(self, mean: jax.Array, std: jax.Array):
        return mean + self.kappa * std


class EI(BaseAcquisition):
    def __init__(self, xi: float = 0.01, stochastic_multiplier: int = 2):
        super().__init__(stochastic_multiplier)
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array):
        y_max = jnp.max(mean)
        a = mean - self.xi - y_max
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


# TODO: ConstantLiar as detailed in https://hal.science/hal-00260579v1/document
