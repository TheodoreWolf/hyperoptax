import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


class BaseAcquisition:
    """
    Base class for acquisition functions.

    Args:
        n_samples: The number of points to sample
        randomly to avoid selecting points that are very close to each other.
    """

    def __init__(self, n_samples: int = 1):
        self.n_samples = n_samples

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

    def sort_acq_vals(self, mean: jax.Array, std: jax.Array, seen_mask: jax.Array):
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
        """
        Get the indices of the points with the highest acquisition values.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            seen_mask (N,): Boolean mask, True for points already evaluated.
            n_points (int): The number of points to select.

        Returns:
            (n_points,): The indices of the points with the highest acquisition values.
        """
        return self.sort_acq_vals(mean, std, seen_mask)[-n_points:]

    def get_stochastic_argmax(
        self,
        mean: jax.Array,
        std: jax.Array,
        seen_mask: jax.Array,
        n_points: int,
        key: jax.random.PRNGKey,
    ):
        """
        Get a random sample of indices of points with high acquisition values.
        This is to avoid picking points that are very close to each other.
        When n_samples is 1, this method is equivalent to get_argmax.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            seen_mask (N,): Boolean mask, True for points already evaluated.
            n_points (int): The number of points to select.
            key (jax.random.PRNGKey): The random key to use for sampling.

        Returns:
            (n_points,): A random sample of indices of points with high
            acquisition values.

        """
        # We sample points randomly from the top n_points * n_samples
        # to avoid selecting points that are very close to each other.
        sample_idx = jax.random.choice(
            key,
            jnp.arange(n_points * self.n_samples),
            (n_points,),
            replace=False,
        )
        return self.sort_acq_vals(mean, std, seen_mask)[::-1][sample_idx]

    def get_max(
        self, mean: jax.Array, std: jax.Array, X: jax.Array, seen_mask: jax.Array
    ):
        """
        Get the points with the highest acquisition values.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            X (N, D): The points to evaluate.
            seen_mask (N,): Boolean mask, True for points already evaluated.

        Returns:
            (n_points, D): The points with the highest acquisition values.
        """
        return X[self.get_argmax(mean, std, seen_mask)]


class UCB(BaseAcquisition):
    """
    Upper Confidence Bound acquisition function.
    """

    def __init__(self, kappa: float = 2.0, n_samples: int = 2):
        super().__init__(n_samples)
        self.kappa = kappa

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        return mean + self.kappa * std


class EI(BaseAcquisition):
    """
    Expected Improvement acquisition function.
    """

    def __init__(self, xi: float = 0.01, n_samples: int = 2):
        super().__init__(n_samples)
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        _y_max = jnp.max(mean) if y_max is None else y_max
        a = mean - self.xi - _y_max
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


class PI(BaseAcquisition):
    """
    Probability of Improvement acquisition function.
    """

    def __init__(self, xi: float = 0.01, n_samples: int = 2):
        super().__init__(n_samples)
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array, y_max=None):
        _y_max = jnp.max(mean) if y_max is None else y_max
        z = (mean - self.xi - _y_max) / std
        return norm.cdf(z)


# TODO: ConstantLiar as detailed in https://hal.science/hal-00260579v1/document
