from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from hyperoptax.base import BaseOptimiser
from hyperoptax.kernels import BaseKernel, RBF
from hyperoptax.spaces import BaseSpace
from hyperoptax.aquisition import BaseAquisition, UCB


class BayesOptimiser(BaseOptimiser):
    def __init__(
        self,
        domain: dict[str, BaseSpace],
        f: Callable,
        kernel: BaseKernel = RBF(length_scale=1.0),
        aquisition: BaseAquisition = UCB(kappa=2.0),
    ):
        super().__init__(domain, f)
        self.kernel = kernel
        self.aquisition = aquisition
        self.jitter = 1e-12

    def optimise(
        self,
        n_iterations: int = -1,
        n_parallel: int = 10,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        # at iter 1: randomly sample n_parallel points
        # fit GP
        # aquire next point
        # fill sample with worse result
        # iteratte n_parallel times
        # iterate until n_iterations
        # sample n_parallel points from domain
        map_f = jax.vmap(self.f, in_axes=(0,) * self.domain.shape[1])
        seen_idx = jax.random.choice(
            key,
            jnp.arange(len(self.domain)),
            (n_parallel,),
        )
        X_seen = self.domain[seen_idx]
        y_seen = map_f(*X_seen.T)

        for i in range(n_iterations - n_parallel):
            # fit GP
            mean, covariance = self.fit_gp(X_seen, y_seen)
            candidate_idx = self.aquisition.get_argmax(
                mean, jnp.sqrt(jnp.diag(covariance)), seen_idx
            )
            candidate_point = self.domain[candidate_idx]
            result = self.f(*candidate_point)
            y_seen = jnp.concatenate([y_seen, result.reshape(1)])
            X_seen = jnp.concatenate([X_seen, candidate_point.reshape(1, -1)])
            seen_idx = jnp.concatenate([seen_idx, candidate_idx.reshape(1)])

        max_idx = jnp.where(y_seen == y_seen.max())
        return X_seen[max_idx]

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
        y_var -= jnp.einsum("ij,ji->i", V.T, V)

        return y_mean, jnp.sqrt(jnp.abs(y_var))
    
