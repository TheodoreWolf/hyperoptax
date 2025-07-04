from typing import Callable
import logging
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax_tqdm import loop_tqdm

from hyperoptax.base import BaseOptimizer
from hyperoptax.kernels import BaseKernel, Matern
from hyperoptax.spaces import BaseSpace
from hyperoptax.aquisition import BaseAquisition, UCB

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        domain: dict[str, BaseSpace],
        f: Callable,
        kernel: BaseKernel = Matern(length_scale=1.0, nu=2.5),
        aquisition: BaseAquisition = UCB(kappa=2.0),
        jitter: float = 1e-6,
    ):
        super().__init__(domain, f)

        self.kernel = kernel
        self.aquisition = aquisition
        self.jitter = jitter  # has to be quite high to avoid numerical issues

    def search(
        self,
        n_iterations: int,
        n_parallel: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> tuple[jax.Array, jax.Array]:
        if n_iterations >= self.domain.shape[0]:
            logger.warning(
                f"n_iterations={n_iterations} is greater or equal to the number of "
                f"points in the domain={self.domain.shape[0]},"
                "this will result in a full grid search."
            )
        # Number of batches we need to cover all requested iterations
        n_batches = (n_iterations + n_parallel - 1) // n_parallel
        n_batches -= 1  # because we do the first batch separately
        idx = jax.random.choice(
            key,
            jnp.arange(len(self.domain)),
            (n_parallel,),
        )
        # Because jax.lax.fori_loop doesn't support dynamic slicing and sizes,
        # we abuse the fact that GPs can handle duplicate points,
        # we can therefore create the array and dynamically replace the values during the loop.
        X_seen = jnp.zeros((n_iterations, self.domain.shape[1]))
        X_seen = X_seen.at[:n_parallel].set(self.domain[idx])
        X_seen = X_seen.at[n_parallel:].set(self.domain[idx[0]])
        results = self.map_f(*X_seen[:n_parallel].T)

        y_seen = jnp.zeros(n_iterations)
        y_seen = y_seen.at[:n_parallel].set(results)
        y_seen = y_seen.at[n_parallel:].set(results[0])

        seen_idx = jnp.zeros(n_iterations)
        seen_idx = seen_idx.at[:n_parallel].set(idx)
        seen_idx = seen_idx.at[n_parallel:].set(idx[0])

        @loop_tqdm(n_batches)
        def _inner_loop(i, carry):
            X_seen, y_seen, seen_idx = carry

            mean, std = self.fit_gp(X_seen, y_seen)
            # can potentially sample points that are very close to each other
            candidate_idxs = self.aquisition.get_argmax(
                mean, std, seen_idx, n_points=n_parallel
            )

            candidate_points = self.domain[candidate_idxs]
            results = self.map_f(*candidate_points.T)
            X_seen = jax.lax.dynamic_update_slice(
                X_seen, candidate_points, (n_parallel + i * n_parallel, 0)
            )

            y_seen = jax.lax.dynamic_update_slice(
                y_seen, results, (n_parallel + i * n_parallel,)
            )
            seen_idx = jax.lax.dynamic_update_slice(
                seen_idx,
                candidate_idxs.astype(jnp.float32),
                (n_parallel + i * n_parallel,),
            )

            return X_seen, y_seen, seen_idx

        (X_seen, y_seen, seen_idx) = jax.lax.fori_loop(
            0, n_batches, _inner_loop, (X_seen, y_seen, seen_idx)
        )
        return X_seen, y_seen

    def fit_gp(self, X: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        X_test = self.domain

        # we calculated our posterior distribution conditioned on data
        K = self.kernel(X, X)
        K = K + jnp.eye(K.shape[0]) * self.jitter
        L = jsp.linalg.cholesky(K, lower=True)
        w = jsp.linalg.cho_solve((L, True), y)

        K_trans = self.kernel(X_test, X)
        y_mean = K_trans @ w
        V = jsp.linalg.solve_triangular(L, K_trans.T, lower=True)
        y_var = self.kernel.diag(X_test)
        # hack to avoid doing the whole matrix multiplication
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py#L475
        y_var -= jnp.einsum("ij,ji->i", V.T, V)

        # TODO: clip to 0
        return y_mean, jnp.sqrt(y_var)

    # TODO: not used yet
    def sanitize_and_normalize(self, y_seen: jax.Array):
        # TODO: remove nans and infs and replace with... something?
        y_seen = y_seen.at[jnp.isnan(y_seen)].set(jnp.nan)
        y_seen = (y_seen - y_seen.mean()) / (y_seen.std() + 1e-10)
        return y_seen
