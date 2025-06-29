import unittest

import jax
import jax.numpy as jnp

from hyperoptax.aquisition import UCB


class TestUCB(unittest.TestCase):
    def test_get_argmax(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_idx = jnp.array([])

        max_val = ucb.get_max(mean, std, X, seen_idx)
        self.assertTrue(jnp.allclose(max_val, jnp.array([2.0, 2.0])))

    def test_get_argmax_with_seen_idx(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_idx = jnp.array([0])

        max_val = ucb.get_max(mean, std, X, seen_idx)
        self.assertTrue(jnp.allclose(max_val, jnp.array([0.0, 0.0])))
    
    def test_get_argmax_jitted(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_idx = jnp.array([0])

        max_val = jax.jit(ucb.get_max)(mean, std, X, seen_idx)
        self.assertTrue(jnp.allclose(max_val, jnp.array([0.0, 0.0])))