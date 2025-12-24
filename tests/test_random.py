import jax

from hyperoptax.random import RandomSearch
from hyperoptax import spaces as sp


def test_random_search():
    def _func(x: float) -> float:
        return x**2

    space = {"x": sp.LinearSpace(0, 1)}
    key = jax.random.PRNGKey(0)
    state = RandomSearch.init(space)
    state, results = RandomSearch.optimize(state, key, _func, 100, 10)
    assert results.shape == (1000, 1)
    assert results.min() < 0.1
    assert results.max() > 0.9
