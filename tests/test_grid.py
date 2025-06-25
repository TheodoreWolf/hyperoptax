import inspect
import unittest
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


class GridSearch:
    def __init__(self, domain: dict[str, jax.Array], f: Callable):
        self.domain = domain
        n_args = len(inspect.signature(f).parameters)
        # get number of points in all domains
        n_points = np.prod([len(domain[k]) for k in domain])

        # check that the resultant dimension is the product of the domain dimensions
        # domain is a 2d array of shape (n_args, n_points)
        grid = jnp.array(jnp.meshgrid(*domain.values()))
        try:
            self.domain = grid.reshape(n_args, n_points).T
        except TypeError as e:
            raise TypeError(
                f"Tried to reshape the domain into a 2d array of shape ({n_args}, {n_points}),"
                f"but I have domain.size = {grid.size}"
            ) from e
        self.f = f

    def get_max(self):
        results = jax.vmap(self.f, in_axes=(0,) * self.domain.shape[1])(
            *(self.domain.T)
        )
        return self.domain[results.argmax()]

    def get_min(self):
        results = jax.vmap(self.f, in_axes=(0,) * self.domain.shape[1])(
            *(self.domain.T)
        )
        return self.domain[results.argmin()]


class TestGridSearch(unittest.TestCase):
    def test_grid_search(self):
        def f(x):
            return -(x**2) + 10

        domain = {"x": jnp.linspace(-1, 1, 100000)}
        # Simple test: apply f to the domain
        grid_search = GridSearch(domain, f)
        result = grid_search.get_max()
        # make test more robust
        self.assertAlmostEqual(result, 0, places=2)

    def test_nd_grid_search(self):
        def f(x, y):
            # 2d function with max at (0, 0)
            return -(x**2 + y**2) + 10

        linspace1 = jnp.linspace(0, 2, 10)
        linspace2 = jnp.linspace(-2, 0, 100)
        domain = {"x": linspace1, "y": linspace2}
        grid_search = GridSearch(domain, f)
        result = grid_search.get_max()
        self.assertTrue(jnp.allclose(result, jnp.array([0, 0])))

    def test_mismatched_domain_and_function(self):
        def f(x, y):
            return -(x**2 + y**2) + 10

        domain = {"x": jnp.linspace(-1, 1, 1000)}
        with self.assertRaises(TypeError):
            GridSearch(domain, f)
