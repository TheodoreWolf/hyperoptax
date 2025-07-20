import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


class BaseAcquisition:
    """
    Base class for acquisition functions.

    Args:
        stochastic_multiplier: The number of points to sample
        randomly to avoid selecting points that are very close to each other.
    """

    def __init__(self, stochastic_multiplier: int = 1):
        self.stochastic_multiplier = stochastic_multiplier

    def __call__(self, mean: jax.Array, std: jax.Array):
        """
        Compute the acquisition value for a given mean and standard deviation.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.

        Returns:
            (N,): The acquisition value for the given mean and standard deviation.
        """
        raise NotImplementedError

    def sort_acq_vals(self, mean: jax.Array, std: jax.Array, seen_idx: jax.Array):
        """
        Sort the acquisition values for a given mean and standard deviation.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            seen_idx (M,): The indices of the points that have already been evaluated.

        Returns:
            (N,): The indices of the points sorted by acquisition value.
        """
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
        """
        Get the indices of the points with the highest acquisition values.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            seen_idx (M,): The indices of the points that have already been evaluated.
            n_points (int): The number of points to select.

        Returns:
            (n_points,): The indices of the points with the highest acquisition values.
        """
        return self.sort_acq_vals(mean, std, seen_idx)[-n_points:]

    def get_stochastic_argmax(
        self,
        mean: jax.Array,
        std: jax.Array,
        seen_idx: jax.Array,
        n_points: int,
        key: jax.random.PRNGKey,
    ):
        """
        Get a random sample of indices of points with high acquisition values.
        This is to avoid picking points that are very close to each other.
        When stochastic_multiplier is 1, this method is equivalent to get_argmax.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            seen_idx (M,): The indices of the points that have already been evaluated.
            n_points (int): The number of points to select.
            key (jax.random.PRNGKey): The random key to use for sampling.

        Returns:
            (n_points,): A random sample of indices of points with high
            acquisition values.

        """
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
        """
        Get the points with the highest acquisition values.

        Args:
            mean (N,): The mean of the Gaussian process.
            std (N,): The standard deviation of the Gaussian process.
            X (N, D): The points to evaluate.
            seen_idx (M,): The indices of the points that have already been evaluated.

        Returns:
            (n_points, D): The points with the highest acquisition values.
        """
        return X[self.get_argmax(mean, std, seen_idx)]


class UCB(BaseAcquisition):
    """
    Upper Confidence Bound acquisition function.
    """

    def __init__(self, kappa: float = 2.0, stochastic_multiplier: int = 2):
        super().__init__(stochastic_multiplier)
        self.kappa = kappa

    def __call__(self, mean: jax.Array, std: jax.Array):
        return mean + self.kappa * std


class EI(BaseAcquisition):
    """
    Expected Improvement acquisition function.
    """

    def __init__(self, xi: float = 0.01, stochastic_multiplier: int = 2):
        super().__init__(stochastic_multiplier)
        self.xi = xi

    def __call__(self, mean: jax.Array, std: jax.Array):
        y_max = jnp.max(mean)
        a = mean - self.xi - y_max
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


# TODO: ConstantLiar as detailed in https://hal.science/hal-00260579v1/document
