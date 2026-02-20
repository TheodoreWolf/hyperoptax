import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.random import GridSearch, GridSearchState, RandomSearch, update_state


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


def test_update_space_idx():
    space_idx = {"x": 0, "y": 0, "xy": {"z": 0}}
    space = {
        "x": sp.DiscreteSpace([0, 0.5, 1]),
        "y": sp.DiscreteSpace([1.0, 1.5, 2.0]),
        "xy": {"z": sp.DiscreteSpace([3.0, 3.5, 4.0])},
    }
    state = GridSearchState(space, space_idx, 0)
    def scan_update_state(state, _):
        new_state = update_state(state)
        return new_state, new_state
    state, state_hist = jax.lax.scan(scan_update_state, state, length = 30)
    a = 1



def test_search_with_fns():
    space = {
        "y": sp.DiscreteSpace([1, 2]),
        "xy": {"z": sp.DiscreteSpace([3, 4, 5])},
    }
    branches = [lambda i=i: i for i in range(10)]

    def _func(y, xy):
        return jax.lax.switch(y, branches) + jax.lax.switch(xy["z"], branches)

    key = jax.random.PRNGKey(0)
    state = GridSearch.init(space)
    state, results = GridSearch.optimize(state, key, _func, 100, 1)
    assert results[0].max() == 7
