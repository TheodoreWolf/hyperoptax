import unittest

import jax.numpy as jnp

from hyperoptax.spaces import LinearSpace, LogSpace, ArbitrarySpace


class TestArbitrarySpace(unittest.TestCase):
    def test_setup(self):
        space = ArbitrarySpace(values=[0, 2, 5, 10])
        self.assertEqual(space.start, 0)
        self.assertEqual(space.end, 10)
        self.assertEqual(space.n_points, 4)


class TestLinearSpace(unittest.TestCase):
    def test_array(self):
        space = LinearSpace(0, 1, 11)
        expected = jnp.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.assertTrue(jnp.allclose(space.array, expected))

    def test_len(self):
        space = LinearSpace(0, 1, 11)
        self.assertEqual(len(space), 11)


class TestLogSpace(unittest.TestCase):
    def test_array_log_base_10(self):
        space = LogSpace(1e-4, 1e-1, 4)
        expected = jnp.array([1e-4, 1e-3, 1e-2, 1e-1])
        self.assertTrue(jnp.allclose(space.array, expected))

    def test_array_log_base_2(self):
        space = LogSpace(32, 256, 4, 2)
        expected = jnp.array([32, 64, 128, 256])
        self.assertTrue(jnp.allclose(space.array, expected))

    def test_len(self):
        space = LogSpace(1e-4, 1e-2, 10)
        self.assertEqual(len(space), 10)
