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

    def get_argmax(
        self, mean: jax.Array, std: jax.Array, seen_idx: jax.Array, n_points: int = 1
    ):
        """Return the index that maximises the acquisition value while
        excluding indices present in *seen_idx*.

        The implementation avoids dynamic boolean indexing (which is not
        supported under `jax.jit`) by replacing the acquisition values of
        *seen* points with ``-inf`` and then computing ``argmax`` in a
        single, shape-stable operation.
        """

        # Acquisition values for all points.
        acq_vals = self(mean, std)  # shape (N,)

        # Boolean mask of points that have already been evaluated.
        idxs = jnp.arange(acq_vals.shape[0])
        seen_mask = jnp.isin(idxs, seen_idx)

        # Replace acquisition values of seen points with -inf so they are never selected.
        masked_acq = jnp.where(seen_mask, -jnp.inf, acq_vals)

        return jnp.argsort(
            masked_acq,
        )[-n_points:]

    def get_max(
        self, mean: jax.Array, std: jax.Array, X: jax.Array, seen_idx: jax.Array
    ):
        return X[self.get_argmax(mean, std, seen_idx)]


# TODO: More aquisition functions
