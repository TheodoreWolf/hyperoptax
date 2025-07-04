import unittest

import jax.numpy as jnp

from hyperoptax.base import BaseOptimizer
from hyperoptax.spaces import LinearSpace


class TestBaseOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = BaseOptimizer(
            domain={"x": LinearSpace(0, 1, 10)}, f=lambda x: x
        )

    def test_when_no_results_are_found(self):
        with self.assertRaises(AssertionError):
            self.optimizer.max
        with self.assertRaises(AssertionError):
            self.optimizer.min

    def test_max(self):
        # manually set the results
        self.optimizer.results = (
            self.optimizer.domain,
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )
        self.assertEqual(self.optimizer.max["target"], 10)

    def test_min(self):
        # manually set the results
        self.optimizer.results = (
            self.optimizer.domain,
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )
        self.assertEqual(self.optimizer.min["target"], 1)
