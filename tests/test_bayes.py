import unittest

import jax.numpy as jnp

from hyperoptax.bayes import BayesOptimiser
from hyperoptax.spaces import LinearSpace, LogSpace


class TestBayes(unittest.TestCase):
    def test_bayes_optimiser(self):
        def f(x, y):
            return x**2 - y**2
        domain = {
            "x": LogSpace(1e-4, 1e-2, 10),
            "y": LinearSpace(0.01, 0.99, 10),
        }
        bayes = BayesOptimiser(domain, f)
        result = bayes.optimise(n_iterations=100, n_parallel=10)
        self.assertTrue(jnp.allclose(result, jnp.array([0.01, 0.01])))
