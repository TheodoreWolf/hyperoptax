import unittest

import jax.numpy as jnp

from hyperoptax.kernels import RBF


class TestRBF(unittest.TestCase):
    def test_rbf_with_same_points(self):
        kernel = RBF(length_scale=1.0)
        x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        # correlation matrix should be all ones
        self.assertTrue(jnp.allclose(kernel(x, x), jnp.full((3, 3), 1.0)))

    def test_rbf_with_different_points(self):
        kernel = RBF(length_scale=1.0)
        x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
        # correlation matrix should be all exp(-1)
        self.assertTrue(jnp.allclose(kernel(x, y), jnp.full((3, 3), jnp.exp(-1))))

    def test_rbf_with_different_data_sizes(self):
        kernel = RBF(length_scale=1.0)
        x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
        # correlation matrix should be all exp(-1) with shape (4, 3)
        self.assertTrue(jnp.allclose(kernel(x, y), jnp.full((4, 3), jnp.exp(-1))))

    def test_rbf_with_different_length_scales(self):
        kernel = RBF(length_scale=2.0)
        x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
        # correlation matrix should be all exp(-1/4) with shape (3, 3)
        self.assertTrue(jnp.allclose(kernel(x, y), jnp.full((3, 3), jnp.exp(-1 / 4))))