import unittest

import jax.numpy as jnp

from hyperoptax.bayesian import BayesianOptimizer
from hyperoptax.spaces import LinearSpace


class TestBayes(unittest.TestCase):
    def setUp(self):
        self.high_dim_domain = {
            "x": LinearSpace(-1, 1, 11),
            "y": LinearSpace(-1, 1, 11),
            "z": LinearSpace(-1, 1, 11),
            "w": LinearSpace(-1, 1, 11),
        }
        self.low_dim_domain = {
            "x": LinearSpace(-1, 1, 11),
        }
        self.high_dim_function = lambda x, y, z, w: -(x**2) - (y**2) - (z**2) - (w**2)
        self.low_dim_function = lambda x: -(x**2)

    def test_bayes_optimizer_improve_in_high_dim(self):
        # make function where optimum is in the center of high dimensional domain
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_parallel=10)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))

    def test_bayes_optimizer_jit(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_parallel=10, jit=True)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))

    def test_bayes_optimizer_when_n_parallel_is_1(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_parallel=1)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))

    def test_bayes_optimizer_when_n_parallel_not_multiple_of_n_iterations(self):
        bayes = BayesianOptimizer(self.high_dim_domain, self.high_dim_function)
        result = bayes.optimize(n_iterations=100, n_parallel=13)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))

    def test_bayes_optimizer_when_n_iterations_is_minus_1(self):
        bayes = BayesianOptimizer(self.low_dim_domain, self.low_dim_function)
        result = bayes.optimize(n_iterations=-1, n_parallel=2)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0])))

    def test_BayesianOptimizer_when_minimize_is_false(self):
        def minus_f(x, y, z, w):
            return -self.high_dim_function(x, y, z, w)

        bayes = BayesianOptimizer(self.high_dim_domain, minus_f)
        result = bayes.optimize(n_iterations=100, n_parallel=1, maximize=False)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))
