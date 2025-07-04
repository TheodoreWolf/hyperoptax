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
        n_iterations: int = -1,
        n_parallel: int = 10,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
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

        @loop_tqdm(n_iterations // n_parallel)
        def _inner_loop(i, carry):
            X_seen, y_seen, seen_idx = carry

            mean, std = self.fit_gp(X_seen, y_seen)
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
            0, n_iterations // n_parallel, _inner_loop, (X_seen, y_seen, seen_idx)
        )
        return X_seen, y_seen

    # TODO: ensure that -1 is handled correctly
    def optimize(
        self,
        n_iterations: int,
        n_parallel: int,
        jit: bool = False,
        maximise: bool = True,
        pmap: bool = False,
        save_results: bool = True,
    ):
        # TODO: pmap is not supported yet: can't use jax.pmap in the search function
        logger.warning("pmap is not supported yet: defaulting to vmap instead")
        # if pmap:
        #     n_devices = jax.device_count()
        #     self.map_f = jax.pmap(self.f, in_axes=(0,) * self.domain.shape[1])
        #     if n_devices != n_parallel:
        #         logger.warning(
        #             f"Using pmap with {n_devices} devices, "
        #             f"but {n_parallel} parallel evaluations was requested."
        #             f"Overriding n_parallel from {n_parallel} to {n_devices}."
        #         )
        #         n_parallel = n_devices
        # else:
        self.map_f = jax.vmap(self.f, in_axes=(0,) * self.domain.shape[1])

        if jit:
            X_seen, y_seen = jax.jit(self.search, static_argnums=(0, 1))(
                n_iterations, n_parallel
            )
        else:
            X_seen, y_seen = self.search(n_iterations, n_parallel)
        if save_results:
            self.results = X_seen, y_seen
        if maximise:
            max_idx = jnp.where(y_seen == y_seen.max())
        else:
            max_idx = jnp.where(y_seen == y_seen.min())

        return X_seen[max_idx].flatten()

    def fit_gp(self, X: jax.Array, y: jax.Array):
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

        return y_mean, jnp.sqrt(jnp.abs(y_var))
