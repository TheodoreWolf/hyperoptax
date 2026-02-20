import logging
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from hyperoptax.acquisition import UCB, BaseAcquisition
from hyperoptax.base import BaseOptimizer
from hyperoptax.kernels import BaseKernel, Matern
from hyperoptax.spaces import ContinuousSpace

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        domain: dict[str, ContinuousSpace],
        f: Callable,
        kernel: Optional[BaseKernel] = None,
        acquisition: Optional[BaseAcquisition] = None,
        jitter: float = 1e-6,
        n_vals: int = 100_000,
    ):
        self.f = f

        self.kernel = kernel or Matern(length_scale=1.0, nu=2.5)
        self.acquisition = acquisition or UCB(kappa=2.0)
        self.jitter = jitter  # has to be quite high to avoid numerical issues
        self.domain = DomainWrapper(domain)
        self.n_vals = n_vals

    # TODO:for pmap, we should have a shared y_seen and X_seen array across GPUs.
    def search(
        self,
        n_iterations: int,
        n_vmap: int,
        key: jax.random.PRNGKey,
        domain: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array]:
        # get initial domain
        domain = self.domain.sample(self.n_vals, key)
        X = self.domain.make_grid(self.n_vals, key)

        # Number of batches we need to cover all requested iterations
        n_batches = (n_iterations + n_vmap - 1) // n_vmap
        n_batches -= 1  # because we do the first batch separately
        idx = jax.random.choice(
            key,
            jnp.arange(
                self.n_vals,
            ),
            (n_vmap,),
        )
        candidate_points = jax.tree.map(lambda x: x[idx], domain)
        # Because jax.lax.fori_loop doesn't support dynamic slicing and sizes,
        # we abuse the fact that GPs can handle duplicate points,
        # we can therefore create the array and dynamically replace
        # the values during the loop.
        X_seen = jnp.zeros((n_iterations, X.shape[1]))
        X_seen = X_seen.at[:n_vmap].set(X[idx])
        X_seen = X_seen.at[n_vmap:].set(X[idx[0]])
        results = self.map_f(**candidate_points)

        y_seen = jnp.zeros(n_iterations)
        y_seen = y_seen.at[:n_vmap].set(results)
        y_seen = y_seen.at[n_vmap:].set(results[0])

        seen_idx = jnp.zeros(n_iterations)
        seen_idx = seen_idx.at[:n_vmap].set(idx)
        seen_idx = seen_idx.at[n_vmap:].set(idx[0])

        # @loop_tqdm(n_batches)
        def _inner_loop(i, carry):
            X_seen, y_seen, seen_idx, key = carry
            key, domain_key, acq_key = jax.random.split(key, 3)
            # sample new domain
            domain = self.domain.sample(self.n_vals, domain_key)
            grid = jnp.array(jax.tree.flatten(domain)[0]).T

            mean, std = self.fit_gp(grid, X_seen, y_seen)
            # can potentially sample points that are very close to each other
            candidate_idxs = self.acquisition.get_stochastic_argmax(
                mean, std, seen_idx, n_points=n_vmap, key=acq_key
            )

            candidate_points = jax.tree.map(lambda x: x[candidate_idxs], domain)
            results = self.map_f(**candidate_points)
            X_seen = jax.lax.dynamic_update_slice(
                X_seen, grid[candidate_idxs], (n_vmap + i * n_vmap, 0)
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

    def fit_gp(
        self, X_test: jax.Array, X: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """
        Fit a Gaussian process to the data.

        Args:
            X (N, D): The points that have been evaluated.
            y (N,): The values of the points that have been evaluated.

        Returns:
            (N,): The mean of the Gaussian process.
            (N,): The standard deviation of the Gaussian process.
        """
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

    def optimize(
        self,
        n_iterations: int = -1,
        n_vmap: int = 1,
        n_pmap: int = 1,
        maximize: bool = True,
        jit: bool = False,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        """
        Optimize the function.
        Note: pmap doesn't work as expected for the Bayesian optimizer... yet.

        Args:
            n_iterations (int): The number of iterations to run.
            n_vmap (int): The number of points to evaluate in parallel on the
                same device.
            n_pmap (int): The number of points to evaluate in parallel on different
                devices.
            maximize (bool): Whether to maximize or minimize the function.
            jit (bool): Whether to jit the function.
            key (jax.random.PRNGKey): The random key to use for sampling.

        """
        if n_iterations == -1:
            n_iterations = self.domain.shape[0]

        if maximize:
            self.map_f = jax.vmap(self.f)
        else:
            self.map_f = jax.vmap(lambda **kwargs: -self.f(**kwargs))

        if n_pmap > 1:
            assert n_iterations % n_pmap == 0, (
                "n_iterations must be divisible by n_pmap"
            )
            assert n_pmap == jax.local_device_count(), (
                "n_pmap must be equal to the number of devices"
            )
            # TODO: fix this for the bayesian optimizer
            domains = jnp.array(jnp.array_split(self.domain[:n_iterations], n_pmap))
            n_iterations = n_iterations // n_pmap
            X_seen, y_seen = jax.pmap(
                partial(self.search, n_iterations=n_iterations, n_vmap=n_vmap, key=key),
            )(domain=domains)

        # mostly for debugging purposes
        elif jit:
            X_seen, y_seen = jax.jit(self.search, static_argnums=(0, 1))(
                n_iterations, n_vmap, key
            )
        else:
            X_seen, y_seen = self.search(n_iterations, n_vmap, key)

        max_idxs = jnp.where(y_seen == y_seen.max())

        if not maximize:
            y_seen = -y_seen

        self.results = (X_seen, y_seen)

        return X_seen[max_idxs].squeeze()


# Create matching tree of keys
def create_keys_tree(tree, base_key):
    flat_tree, tree_def = jax.tree.flatten(tree)
    num_leaves = len(flat_tree)
    keys = jax.random.split(base_key, num_leaves)
    return jax.tree.unflatten(tree_def, keys)


class DomainWrapper:
    def __init__(self, spaces: dict[str | ContinuousSpace]):
        self.spaces = spaces

    def sample(self, n_vals, key):
        keys = create_keys_tree(self.spaces, key)
        tree_grid = jax.tree.map(
            lambda space, key: space.sample(n_vals, key), self.spaces, keys
        )
        return tree_grid

    def make_grid(self, n_vals, key):
        tree_grid = self.sample(n_vals, key)
        grid, _ = jax.tree.flatten(tree_grid)
        return jnp.array(grid).T

    @property
    def shape(
        self,
    ):
        return (None, len(self.spaces))


def test_domain():
    dummy = {"a": ContinuousSpace(0, 1), "b": ContinuousSpace(2, 3)}
    domain = DomainWrapper(dummy)
    key = jax.random.PRNGKey(0)
    a = domain.make_grid(100, key)
    assert a.shape == (100, 2)


import matplotlib.pyplot as plt


class TestBayes:
    def setup_method(self):
        self.high_dim_domain = {
            "x": ContinuousSpace(-1, 1),
            "y": ContinuousSpace(-1, 1),
            "z": ContinuousSpace(-1, 1),
            "w": ContinuousSpace(-1, 1),
        }
        self.low_dim_domain = {
            "x": ContinuousSpace(-1, 1),
        }
        self.high_dim_function = lambda x, y, z, w: -(x**2) - (y**2) - (z**2) - (w**2)
        self.low_dim_function = lambda x: -(x**2)

    def test_bayes_optimizer_improve_in_high_dim(self):
        # make function where optimum is in the center of high dimensional domain
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_vmap=10)
        print(result)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0]))

    def test_bayes_optimizer_jit(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_vmap=10, jit=True)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0]))

    def test_bayes_optimizer_when_n_parallel_is_1(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_vmap=1)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0]))

    def test_bayes_optimizer_when_n_parallel_not_multiple_of_n_iterations(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_vmap=13)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0]))

    def test_bayes_optimizer_when_n_iterations_is_minus_1(self):
        bayes = BayesianOptimizer(self.low_dim_domain, self.low_dim_function)
        result = bayes.optimize(n_iterations=-1, n_vmap=2)
        assert jnp.allclose(result, jnp.array([0.0]))

    def test_optimizer_when_maximize_is_false(self):
        def minus_f(x, y, z, w):
            return -self.high_dim_function(x, y, z, w)

        bayes = BayesianOptimizer(self.high_dim_domain, minus_f)
        result = bayes.optimize(n_iterations=100, n_vmap=1, maximize=False)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0]))

    def test_bayes_optimizer_with_pmap(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=400, n_vmap=4, n_pmap=4)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0]))



