import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from hyperoptax.acquisition import UCB, BaseAcquisition
from hyperoptax.base import BaseOptimizer
from hyperoptax.kernels import BaseKernel, Matern
from hyperoptax.spaces import BaseSpace

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        domain: dict[str, BaseSpace],
        f: Callable,
        kernel: BaseKernel = Matern(length_scale=1.0, nu=2.5),
        acquisition: BaseAcquisition = UCB(kappa=2.0),
        jitter: float = 1e-6,
    ):
        super().__init__(domain, f)

        self.kernel = kernel
        self.acquisition = acquisition
        self.jitter = jitter  # has to be quite high to avoid numerical issues

    # TODO:for pmap, we should have a shared y_seen and X_seen array across GPUs.
    def search(
        self,
        n_iterations: int,
        n_vmap: int,
        key: jax.random.PRNGKey,
        domain: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array]:
        del domain  # unused
        if n_iterations >= self.domain.shape[0]:
            logger.warning(
                f"n_iterations={n_iterations} is greater or equal to the number of "
                f"points in the domain={self.domain.shape[0]},"
                "this will result in a full grid search."
            )

        # Number of batches we need to cover all requested iterations
        n_batches = (n_iterations + n_vmap - 1) // n_vmap
        n_batches -= 1  # because we do the first batch separately
        idx = jax.random.choice(
            key,
            jnp.arange(len(self.domain)),
            (n_vmap,),
        )
        # Because jax.lax.fori_loop doesn't support dynamic slicing and sizes,
        # we abuse the fact that GPs can handle duplicate points,
        # we can therefore create the array and dynamically replace
        # the values during the loop.
        X_seen = jnp.zeros((n_iterations, self.domain.shape[1]))
        X_seen = X_seen.at[:n_vmap].set(self.domain[idx])
        X_seen = X_seen.at[n_vmap:].set(self.domain[idx[0]])
        results = self.map_f(*X_seen[:n_vmap].T)

        y_seen = jnp.zeros(n_iterations)
        y_seen = y_seen.at[:n_vmap].set(results)
        y_seen = y_seen.at[n_vmap:].set(results[0])

        seen_idx = jnp.zeros(n_iterations)
        seen_idx = seen_idx.at[:n_vmap].set(idx)
        seen_idx = seen_idx.at[n_vmap:].set(idx[0])

        # @loop_tqdm(n_batches)
        def _inner_loop(i, carry):
            X_seen, y_seen, seen_idx, key = carry
            key, subkey = jax.random.split(key)

            mean, std = self.fit_gp(X_seen, y_seen)
            # can potentially sample points that are very close to each other
            candidate_idxs = self.acquisition.get_stochastic_argmax(
                mean, std, seen_idx, n_points=n_vmap, key=subkey
            )

            candidate_points = self.domain[candidate_idxs]
            results = self.map_f(*candidate_points.T)
            X_seen = jax.lax.dynamic_update_slice(
                X_seen, candidate_points, (n_vmap + i * n_vmap, 0)
            )

            y_seen = jax.lax.dynamic_update_slice(
                y_seen, results, (n_vmap + i * n_vmap,)
            )
            seen_idx = jax.lax.dynamic_update_slice(
                seen_idx,
                candidate_idxs.astype(jnp.float32),
                (n_vmap + i * n_vmap,),
            )

            return X_seen, y_seen, seen_idx, key

        (X_seen, y_seen, seen_idx, _) = jax.lax.fori_loop(
            0, n_batches, _inner_loop, (X_seen, y_seen, seen_idx, key)
        )
        return X_seen, y_seen

    def fit_gp(self, X: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Fit a Gaussian process to the data.

        Args:
            X (N, D): The points that have been evaluated.
            y (N,): The values of the points that have been evaluated.

        Returns:
            (N,): The mean of the Gaussian process.
            (N,): The standard deviation of the Gaussian process.
        """
        X_test = self.domain

        # we calculated our posterior distribution conditioned on data
        K = self.kernel(X, X)
        K = K + jnp.eye(K.shape[0]) * self.jitter
        L = jsp.linalg.cholesky(K, lower=True)
        w = jsp.linalg.cho_solve((L, True), self.sanitize_and_normalize(y))

        K_trans = self.kernel(X_test, X)
        y_mean = K_trans @ w
        V = jsp.linalg.solve_triangular(L, K_trans.T, lower=True)
        y_var = self.kernel.diag(X_test)
        # hack to avoid doing the whole matrix multiplication
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py#L475
        y_var -= jnp.einsum("ij,ji->i", V.T, V)

        return y_mean, jnp.sqrt(jnp.clip(y_var, 0))

    def sanitize_and_normalize(self, y_seen: jax.Array):
        """
        Sanitize the values of the points that have been evaluated.
        This is to avoid numerical issues.

        Args:
            y_seen (N,): The values of the points that have been evaluated.

        Returns:
            (N,): The sanitized values of the points that have been evaluated.
        """
        y_seen = jnp.where(jnp.isnan(y_seen), jnp.min(y_seen), y_seen)
        y_seen = (y_seen - y_seen.mean()) / (y_seen.std() + 1e-10)
        return y_seen
