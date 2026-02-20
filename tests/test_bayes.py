import jax.numpy as jnp
import pytest

from hyperoptax.bayesian import BayesianOptimizer
from hyperoptax.spaces import LinearSpace


def high_dim_domain():
    domain = {
        "x": LinearSpace(-1, 1, 11),
        "y": LinearSpace(-1, 1, 11),
        "z": LinearSpace(-1, 1, 11),
        "w": LinearSpace(-1, 1, 11),
    }

    return domain


def high_dim_fn(x, y, z, w):
    return -(x**2) - (y**2) - (z**2) - (w**2)


def low_dim_domain():
    domain = {"x": LinearSpace(-1, 1, 11)}
    return domain


def low_dim_fn(x):
    return -(x**2)


high_dim = (high_dim_domain(), high_dim_fn, jnp.array([0.0, 0.0, 0.0, 0.0]))
low_dim = (low_dim_domain(), low_dim_fn, jnp.array([0.0]))


class TestBayes:
    @pytest.mark.parametrize(
        "domain, function, result",
        [high_dim, low_dim],
    )
    def test_bayes_optimizer_high_dim(self, domain, function, result):
        # make function where optimum is in the center of high dimensional domain
        bayes = BayesianOptimizer(domain, function)
        result = bayes.optimize(n_iterations=100, n_vmap=10)
        assert jnp.allclose(result, result)

    @pytest.mark.parametrize(
        "domain, function, result",
        [high_dim, low_dim],
    )
    def test_bayes_optimizer_jit(self, domain, function, result):
        bayes = BayesianOptimizer(domain, function)
        result = bayes.optimize(n_iterations=100, n_vmap=10, jit=True)
        assert jnp.allclose(result, result)

    @pytest.mark.parametrize(
        "domain, function, result",
        [high_dim, low_dim],
    )
    def test_bayes_optimizer_when_n_parallel_is_1(self, domain, function, result):
        bayes = BayesianOptimizer(domain, function)
        result = bayes.optimize(n_iterations=100, n_vmap=1)
        assert jnp.allclose(result, result)

    @pytest.mark.parametrize(
        "domain, function, result",
        [high_dim],
    )
    def test_bayes_optimizer_when_n_parallel_not_multiple_of_n_iterations(
        self, domain, function, result
    ):
        bayes = BayesianOptimizer(domain, function)
        result = bayes.optimize(n_iterations=100, n_vmap=13)
        assert jnp.allclose(result, result)

    @pytest.mark.parametrize(
        "domain, function, result",
        [low_dim],
    )
    def test_bayes_optimizer_when_n_iterations_is_minus_1(
        self, domain, function, result
    ):
        bayes = BayesianOptimizer(domain, function)
        result = bayes.optimize(n_iterations=-1, n_vmap=2)
        assert jnp.allclose(result, result)

    @pytest.mark.parametrize(
        "domain, minus_function, result",
        (
            (
                high_dim_domain(),
                lambda x, y, z, w: -high_dim_fn(x, y, z, w),
                jnp.array([0.0, 0.0, 0.0, 0.0]),
            ),
            (low_dim_domain(), lambda x: -low_dim_fn(x), jnp.array([0.0])),
        ),
    )
    def test_optimizer_when_maximize_is_false(self, domain, minus_function, result):
        bayes = BayesianOptimizer(domain, minus_function)
        result = bayes.optimize(n_iterations=100, n_vmap=1, maximize=False)
        assert jnp.allclose(result, result)

    @pytest.mark.parametrize(
        "domain, function, result",
        [high_dim],
    )
    def test_bayes_optimizer_with_pmap(self, domain, function, result):
        bayes = BayesianOptimizer(domain, function)
        result = bayes.optimize(n_iterations=400, n_vmap=4, n_pmap=4)
        assert jnp.allclose(result, result)
