import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


class BaseAcquisition:
    """Base class for acquisition functions."""

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        """
        Compute the acquisition value for a given mean and standard deviation.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            y_max: Optional pre-computed reference value (e.g. best observed mean).
                   Used by EI to ensure consistency when evaluating a single point.

        Returns:
            (N,): The acquisition value for the given mean and standard deviation.
        """
        raise NotImplementedError

    def _sort_acq_vals(self, mean: jax.Array, std: jax.Array, seen_mask: jax.Array):
        """
        Sort the acquisition values for a given mean and standard deviation.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            seen_mask (N,): Boolean mask, True for points already evaluated.

        Returns:
            (N,): The indices of the points sorted by acquisition value.
        """
        acq_vals = self(mean, std)  # shape (N,)
        masked_acq = jnp.where(seen_mask, -jnp.inf, acq_vals)
        return jnp.argsort(masked_acq)

    def get_argmax(
        self, mean: jax.Array, std: jax.Array, seen_mask: jax.Array, n_points: int = 1
    ):
        return self._sort_acq_vals(mean, std, seen_mask)[-n_points:]


class UCB(BaseAcquisition):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        return mean + self.kappa * std


class EI(BaseAcquisition):
    """Expected Improvement acquisition function."""

    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        # y_max should be the best observed function value. Falling back to
        # max(mean) is an approximation suitable for standalone use only —
        # always pass y_max explicitly when observations are available.
        _y_max = jnp.max(mean) if y_max is None else y_max
        a = mean - self.xi - _y_max
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


class PI(BaseAcquisition):
    """Probability of Improvement acquisition function."""

    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        # y_max should be the best observed function value. Falling back to
        # max(mean) is an approximation suitable for standalone use only —
        # always pass y_max explicitly when observations are available.
        _y_max = jnp.max(mean) if y_max is None else y_max
        z = (mean - self.xi - _y_max) / std
        return norm.cdf(z)


class BaseHallucination:
    """Base class for Kriging Believer hallucination strategies."""

    def __call__(
        self,
        mean: jax.Array,
        std: jax.Array,
        key: jax.Array,
        y_max: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError


class MeanHallucination(BaseHallucination):
    """Classical Kriging Believer: hallucinate with GP posterior mean."""

    def __call__(self, mean, std, key, y_max):
        return mean[0]


class SampleHallucination(BaseHallucination):
    """Randomized Kriging Believer (RKB, arXiv 2603.01470): hallucinate with a posterior sample."""

    def __call__(self, mean, std, key, y_max):
        return mean[0] + std[0] * jax.random.normal(key)


class UCBHallucination(BaseHallucination):
    """Optimistic hallucination: mean + kappa * std."""

    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def __call__(self, mean, std, key, y_max):
        return mean[0] + self.kappa * std[0]


class ConstantHallucination(BaseHallucination):
    """Ginsbourger et al. 2010: hallucinate with y_max or a fixed constant.

    If value is None, uses the current best observed value (y_max).
    Otherwise uses the fixed value regardless of observations.
    """

    def __init__(self, value: float | None = None):
        self.value = value

    def __call__(self, mean, std, key, y_max):
        if self.value is None:
            return y_max
        return jnp.asarray(self.value, dtype=mean.dtype)
