import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.random import GridSearch, RandomSearch, update_state


def test_random_search():
    def _func(x: float) -> float:
        return x**2

    space = {"x": sp.LinearSpace(0, 1)}
    key = jax.random.PRNGKey(0)
    state = RandomSearch.init(space)
    state, results = RandomSearch.optimize(state, key, _func, 100, 10)
    assert results[0].shape == (1000, 1)
    assert results[0].min() < 0.1
    assert results[0].max() > 0.9


def test_grid_search_init():
    def _func(x: float) -> float:
        return x**2

    space = {"x": sp.DiscreteSpace([0, 0.5, 1])}
    state = GridSearch.init(space)
    assert state.space_idx == {"x": jnp.zeros(1)}


def test_grid_search_init_with_non_discrete_space():
    space = {"x": sp.LinearSpace(0, 1)}
    with pytest.raises(AssertionError):
        GridSearch.init(space)


def test_grid_search():
    def _func(x: float) -> float:
        return x**2

    space = {"x": sp.DiscreteSpace([0, 0.5, 1])}
    key = jax.random.PRNGKey(0)
    state = GridSearch.init(space)
    state, results = GridSearch.optimize(state, key, _func, 100, 10)
    assert results[0].shape == (1000, 1)
    assert results[0].min() == 0
    assert results[0].max() == 1


# def test_update_space_idx():
#     space_idx = {"x": jnp.array([0]), "y": jnp.array([0]), "xy": {"z": jnp.array([0])}}
#     space = {
#         "x": sp.DiscreteSpace([0, 0.5, 1]),
#         "y": sp.DiscreteSpace([1.0, 1.5, 2.0]),
#         "xy": {"z": sp.DiscreteSpace([3.0, 3.5, 4.0])},
#     }
#     new_space_idx = update_state(space_idx, space, 0)
