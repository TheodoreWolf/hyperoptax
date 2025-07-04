import unittest

import jax.numpy as jnp

from hyperoptax.bayesian import BayesianOptimizer
from hyperoptax.spaces import LinearSpace, LogSpace


class TestBayes(unittest.TestCase):
    def test_bayes_optimizer(self):
        def f(x, y):
            return x**2 - y**2

        domain = {
            "x": LogSpace(1e-4, 1e-2, 10),
            "y": LinearSpace(0.01, 0.99, 10),
        }
        bayes = BayesianOptimizer(domain, f)
        result = bayes.optimize(n_iterations=100, n_parallel=10)
        self.assertTrue(jnp.allclose(result, jnp.array([0.01, 0.01])))

    def test_bayes_optimizer_improve_in_high_dim(self):
        # make function where optimum is in the center of high dimensional domain
        def f(x, y, z, w):
            return -(x**2) - (y**2) - (z**2) - (w**2)

        domain = {
            "x": LinearSpace(-1, 1, 11),
            "y": LinearSpace(-1, 1, 11),
            "z": LinearSpace(-1, 1, 11),
            "w": LinearSpace(-1, 1, 11),
        }
        bayes = BayesianOptimizer(domain, f)
        result = bayes.optimize(n_iterations=100, n_parallel=10)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))

    def test_bayes_optimizer_jit(self):
        def f(x, y, z, w):
            return -(x**2) - (y**2) - (z**2) - (w**2)

        domain = {
            "x": LinearSpace(-1, 1, 11),
            "y": LinearSpace(-1, 1, 11),
            "z": LinearSpace(-1, 1, 11),
            "w": LinearSpace(-1, 1, 11),
        }
        bayes = BayesianOptimizer(domain, f)
        result = bayes.optimize(n_iterations=100, n_parallel=10, jit=True)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 0.0, 0.0])))
