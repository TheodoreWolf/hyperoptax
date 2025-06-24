import jax
import unittest


def f(x):
    return -(x**2)


class TestGridSearch(unittest.TestCase):
    def test_grid_search(self):
        domain = jax.numpy.linspace(-10, 10, 100)
        # Simple test: apply f to the domain
        result = jax.vmap(f)(domain)
        self.assertEqual(result.shape, (100,))

    def test_random_search(self):
        domain = jax.numpy.linspace(-10, 10, 100)
        # Simple test: apply f to the domain
        result = jax.vmap(f)(domain)
        self.assertEqual(result.shape, (100,))


if __name__ == "__main__":
    unittest.main()
