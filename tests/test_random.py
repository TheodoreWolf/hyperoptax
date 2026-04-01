import jax
import jax.numpy as jnp
import pytest

from hyperoptax import random as rand
from hyperoptax import spaces as sp


class TestValidateFunc:
    def test_single_arg_raises(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        with pytest.raises(TypeError, match="fn\\(key, config\\)"):
            optimizer.optimize(
                state, jax.random.PRNGKey(0), lambda x: x, n_iterations=1
            )

    def test_single_arg_raises_scan(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        with pytest.raises(TypeError, match="fn\\(key, config\\)"):
            optimizer.optimize_scan(
                state, jax.random.PRNGKey(0), lambda x: x, n_iterations=1
            )


class TestRandomSearchGetNextParams:
    def test_samples_within_bounds(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        params = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        # n_parallel=1: params["x"] has shape (1,)
        assert 0.0 <= float(params["x"][0]) <= 1.0

    def test_different_keys_give_different_params(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        params1 = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        params2 = optimizer.get_next_params(state, jax.random.PRNGKey(1))
        assert not jnp.allclose(params1["x"], params2["x"])

    def test_same_key_gives_same_params(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        key = jax.random.PRNGKey(42)
        params1 = optimizer.get_next_params(state, key)
        params2 = optimizer.get_next_params(state, key)
        assert jnp.allclose(params1["x"], params2["x"])

    def test_nested_pytree_space(self):
        space = {
            "lr": sp.LinearSpace(1e-4, 1e-1),
            "reg": {"l1": sp.LinearSpace(0.0, 1.0)},
        }
        state, optimizer = rand.RandomSearch.init(space)
        params = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        assert "lr" in params
        assert "l1" in params["reg"]

    def test_discrete_space_samples_from_values(self):
        values = [0, 1, 2, 3]
        space = {"x": sp.DiscreteSpace(values)}
        state, optimizer = rand.RandomSearch.init(space)
        params = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        assert int(params["x"][0]) in values

    def test_log_space_samples_within_bounds(self):
        space = {"lr": sp.LogSpace(1e-4, 1e-1)}
        state, optimizer = rand.RandomSearch.init(space)
        params = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        assert 1e-4 <= float(params["lr"][0]) <= 1e-1

    def test_n_parallel_batch_shape(self):
        space = {"lr": sp.LinearSpace(1e-4, 1e-1)}
        state, optimizer = rand.RandomSearch.init(space, n_parallel=3)
        params = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        assert params["lr"].shape == (3,)

    def test_n_parallel_values_within_bounds(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space, n_parallel=5)
        params = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        assert jnp.all(params["x"] >= 0.0) and jnp.all(params["x"] <= 1.0)


class TestRandomSearchUpdateState:
    def test_update_state_is_memoryless(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        key = jax.random.PRNGKey(0)
        new_state = optimizer.update_state(state, key, jnp.array([0.5]))
        assert new_state.space is state.space

    def test_update_state_repeated_calls_unchanged(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        key = jax.random.PRNGKey(0)
        for _ in range(5):
            state = optimizer.update_state(state, key, jnp.array([1.0]))
        assert state.space == rand.RandomSearch.init(space)[0].space


class TestOptimizeScan:
    def _setup(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space)
        # func receives a scalar config["x"] (from _index_batch)
        func = lambda key, config: config["x"] ** 2
        return state, optimizer, func

    def test_history_length_matches_n_iterations(self):
        state, optimizer, func = self._setup()
        _, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert params_hist["x"].shape[0] == 5
        assert results_hist.shape[0] == 5

    def test_single_iteration(self):
        state, optimizer, func = self._setup()
        _, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=1
        )
        assert params_hist["x"].shape[0] == 1
        assert results_hist.shape[0] == 1

    def test_params_within_bounds(self):
        state, optimizer, func = self._setup()
        _, (params_hist, _) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=10
        )
        assert jnp.all(params_hist["x"] >= 0.0)
        assert jnp.all(params_hist["x"] <= 1.0)

    def test_results_match_func(self):
        state, optimizer, func = self._setup()
        _, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        # params_hist["x"] shape: (5, 1); results_hist shape: (5, 1)
        expected = params_hist["x"].squeeze() ** 2
        assert jnp.allclose(results_hist.squeeze(), expected)

    def test_matches_optimize_output(self):
        """optimize_scan and optimize should produce the same sequence of params."""
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        key = jax.random.PRNGKey(7)
        func = lambda key, config: config["x"]

        state1, opt1 = rand.RandomSearch.init(space)
        state2, opt2 = rand.RandomSearch.init(space)

        _, (params_list, _) = opt1.optimize(state1, key, func, n_iterations=4)
        _, (params_stacked, _) = opt2.optimize_scan(state2, key, func, n_iterations=4)

        # params_list items have shape (1,); params_stacked["x"] has shape (4, 1)
        expected = jnp.array([p["x"][0] for p in params_list])
        assert jnp.allclose(params_stacked["x"].squeeze(), expected)

    def test_nested_space(self):
        space = {
            "lr": sp.LinearSpace(1e-4, 1e-1),
            "reg": {"l1": sp.LinearSpace(0.0, 1.0)},
        }
        state, optimizer = rand.RandomSearch.init(space)
        func = lambda key, config: config["lr"] + config["reg"]["l1"]
        _, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=6
        )
        assert params_hist["lr"].shape[0] == 6
        assert params_hist["reg"]["l1"].shape[0] == 6
        assert results_hist.shape[0] == 6

    @pytest.mark.parametrize("n", [1, 2, 3, 10])
    def test_various_n_iterations(self, n):
        state, optimizer, func = self._setup()
        _, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=n
        )
        assert params_hist["x"].shape[0] == n
        assert results_hist.shape[0] == n

    def test_n_parallel_optimize_scan(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = rand.RandomSearch.init(space, n_parallel=3)
        func = lambda key, config: config["x"] ** 2
        _, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=4
        )
        # params_hist["x"]: (4, 3); results_hist: (4, 3)
        assert params_hist["x"].shape == (4, 3)
        assert results_hist.shape == (4, 3)
