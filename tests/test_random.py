import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.base import OptimizerState
from hyperoptax.random import RandomSearch


class TestRandomSearchGetNextParams:
    def test_samples_within_bounds(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state = OptimizerState(space=space)
        key = jax.random.PRNGKey(0)
        params = RandomSearch.get_next_params(state, key)
        assert 0.0 <= float(params["x"][0]) <= 1.0

    def test_different_keys_give_different_params(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state = OptimizerState(space=space)
        params1 = RandomSearch.get_next_params(state, jax.random.PRNGKey(0))
        params2 = RandomSearch.get_next_params(state, jax.random.PRNGKey(1))
        assert not jnp.allclose(params1["x"], params2["x"])

    def test_same_key_gives_same_params(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state = OptimizerState(space=space)
        key = jax.random.PRNGKey(42)
        params1 = RandomSearch.get_next_params(state, key)
        params2 = RandomSearch.get_next_params(state, key)
        assert jnp.allclose(params1["x"], params2["x"])

    def test_nested_pytree_space(self):
        space = {"lr": sp.LinearSpace(1e-4, 1e-1), "reg": {"l1": sp.LinearSpace(0.0, 1.0)}}
        state = OptimizerState(space=space)
        params = RandomSearch.get_next_params(state, jax.random.PRNGKey(0))
        assert "lr" in params
        assert "l1" in params["reg"]

    def test_discrete_space_samples_from_values(self):
        values = [0, 1, 2, 3]
        space = {"x": sp.DiscreteSpace(values)}
        state = OptimizerState(space=space)
        params = RandomSearch.get_next_params(state, jax.random.PRNGKey(0))
        assert int(params["x"][0]) in values

    def test_log_space_samples_within_bounds(self):
        space = {"lr": sp.LogSpace(1e-4, 1e-1)}
        state = OptimizerState(space=space)
        params = RandomSearch.get_next_params(state, jax.random.PRNGKey(0))
        assert 1e-4 <= float(params["lr"][0]) <= 1e-1


class TestRandomSearchUpdateState:
    def test_update_state_is_memoryless(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state = OptimizerState(space=space)
        key = jax.random.PRNGKey(0)
        results = jnp.array(0.5)
        new_state = RandomSearch.update_state(state, key, results)
        assert new_state.space is state.space

    def test_update_state_repeated_calls_unchanged(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state = OptimizerState(space=space)
        key = jax.random.PRNGKey(0)
        for _ in range(5):
            state = RandomSearch.update_state(state, key, jnp.array(1.0))
        # State structure should remain the same regardless of results
        assert state.space == OptimizerState(space=space).space
