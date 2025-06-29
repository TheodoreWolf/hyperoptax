import unittest

import jax
import jax.numpy as jnp

from hyperoptax.grid_search import GridSearch, RandomSearch
from hyperoptax.spaces import LinearSpace


class TestGridSearch(unittest.TestCase):
    def test_grid_search(self):
        def f(x):
            return -(x**2) + 10

        domain = {"x": LinearSpace(-1, 0, 1000)}
        grid_search = GridSearch(domain, f)
        result = grid_search.optimise()
        self.assertTrue(jnp.allclose(result, jnp.array([0])))

    def test_nd_grid_search(self):
        def f(x, y):
            # 2d function with max at (0, 0)
            return -(x**2 + y**2) + 10

        linspace1 = LinearSpace(0, 2, 10)
        linspace2 = LinearSpace(-2, 0, 100)
        domain = {"x": linspace1, "y": linspace2}
        grid_search = GridSearch(domain, f)
        result = grid_search.optimise()
        self.assertTrue(jnp.allclose(result, jnp.array([0, 0])))

    def test_mismatched_domain_and_function(self):
        def f(x, y):
            return -(x**2 + y**2) + 10

        domain = {"x": LinearSpace(-1, 1, 1000)}
        with self.assertRaises(AssertionError):
            GridSearch(domain, f)

    def test_n_parallel(self):
        def f(x):
            return -(x**2) + 10

        domain = {"x": LinearSpace(-1, 0, 1000)}
        grid_search = GridSearch(domain, f)
        result = grid_search.optimise(n_parallel=10)
        self.assertTrue(jnp.allclose(result, jnp.array([0])))

    def test_jit(self):
        def f(x):
            return -(x**2) + 10

        domain = {"x": LinearSpace(-1, 0, 1000)}
        grid_search = GridSearch(domain, f)
        result = grid_search.optimise(n_iterations=1000, n_parallel=10, jit=True)
        self.assertTrue(jnp.allclose(result, jnp.array([0])))


class TestRandomSearch(unittest.TestCase):
    def test_domain_is_shuffled(self):
        def f(x):
            return -(x**2) + 10

        domain = {"x": LinearSpace(-1, 0, 1000)}
        random_search = RandomSearch(domain, f)
        self.assertEqual(random_search.domain.shape[0], len(domain["x"]))
        self.assertFalse(jnp.allclose(random_search.domain, domain["x"].array))
