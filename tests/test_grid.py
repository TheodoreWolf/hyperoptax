import dataclasses

import jax
import jax.numpy as jnp
import pytest

from hyperoptax import grid
from hyperoptax import spaces as sp

_KEY = jax.random.PRNGKey(0)
_DUMMY_RESULTS = jnp.array([0.0])


class TestGridSearchInit:
    def test_init_discrete_1d(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        assert isinstance(state, grid.GridSearchState)
        assert state.grid_idx == 0
        assert not optimizer.shuffle

    def test_init_discrete_2d(self):
        space = {"x": sp.DiscreteSpace([0, 1, 2]), "y": sp.DiscreteSpace([0.0, 0.5])}
        state, optimizer = grid.GridSearch.init(space)
        assert isinstance(state, grid.GridSearchState)
        assert state.grid_idx == 0

    def test_init_shuffle(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space, shuffle=True)
        assert optimizer.shuffle

    def test_init_raises_for_continuous_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        with pytest.raises(ValueError):
            grid.GridSearch.init(space)

    def test_init_raises_for_mixed_space(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.LinearSpace(0.0, 1.0)}
        with pytest.raises(ValueError):
            grid.GridSearch.init(space)


class TestGridSearchUpdateState:
    def test_update_state_increments_idx(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        assert state.grid_idx == 0
        state = optimizer.update_state(state, _KEY, _DUMMY_RESULTS)
        assert state.grid_idx == 1

    def test_update_state_does_not_mutate(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        new_state = optimizer.update_state(state, _KEY, _DUMMY_RESULTS)
        assert state.grid_idx == 0
        assert new_state.grid_idx == 1

    def test_update_state_increments_repeatedly(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        for i in range(5):
            assert state.grid_idx == i
            state = optimizer.update_state(state, _KEY, _DUMMY_RESULTS)

    def test_update_state_increments_by_n_parallel(self):
        space = {"x": sp.DiscreteSpace(list(range(10)))}
        state, optimizer = grid.GridSearch.init(space, n_parallel=3)
        state = optimizer.update_state(state, _KEY, _DUMMY_RESULTS)
        assert state.grid_idx == 3


class TestGridSearchShuffle:
    def test_shuffle_reorders_grid(self):
        space = {"x": sp.DiscreteSpace(list(range(20)))}
        state_unshuffled, _ = grid.GridSearch.init(space)
        state_shuffled, _ = grid.GridSearch.init(
            space, shuffle=True, key=jax.random.PRNGKey(0)
        )
        assert not jnp.allclose(state_unshuffled.grid, state_shuffled.grid)

    def test_shuffle_preserves_size(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, _ = grid.GridSearch.init(space, shuffle=True, key=jax.random.PRNGKey(0))
        assert state.grid.shape == (6, 2)

    def test_shuffle_preserves_values(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state_unshuffled, _ = grid.GridSearch.init(space)
        state_shuffled, _ = grid.GridSearch.init(
            space, shuffle=True, key=jax.random.PRNGKey(0)
        )
        assert jnp.allclose(
            jnp.sort(state_unshuffled.grid, axis=0),
            jnp.sort(state_shuffled.grid, axis=0),
        )

    def test_different_keys_give_different_orderings(self):
        space = {"x": sp.DiscreteSpace(list(range(20)))}
        state_a, _ = grid.GridSearch.init(
            space, shuffle=True, key=jax.random.PRNGKey(0)
        )
        state_b, _ = grid.GridSearch.init(
            space, shuffle=True, key=jax.random.PRNGKey(1)
        )
        assert not jnp.allclose(state_a.grid, state_b.grid)

    def test_no_shuffle_by_default(self):
        space = {"x": sp.DiscreteSpace([0, 1, 2])}
        state, _ = grid.GridSearch.init(space)
        assert jnp.allclose(
            state.grid[:, 0], jnp.array([0, 1, 2], dtype=state.grid.dtype)
        )


class TestGridSearchGrid:
    def test_flat_space_size_is_product_of_dim_sizes(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, _ = grid.GridSearch.init(space)
        assert state.grid.shape[0] == 6  # 2 * 3 = 6
        assert state.grid.shape[1] == 2  # 2 params


class TestGridSearchGetNextParams:
    def test_get_next_params_first_index(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        params = optimizer.get_next_params(state, _KEY)
        assert params is not None

    def test_get_next_params_leaf_shape_n_parallel(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        params = optimizer.get_next_params(state, _KEY)
        # n_parallel=1: each leaf has shape (1,)
        assert params["x"].shape == (1,)

    def test_get_next_params_changes_after_update(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        params_0 = optimizer.get_next_params(state, _KEY)
        state = optimizer.update_state(state, _KEY, _DUMMY_RESULTS)
        params_1 = optimizer.get_next_params(state, _KEY)
        assert not jnp.allclose(params_0["x"], params_1["x"])

    def test_get_next_params_2d_grid(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, optimizer = grid.GridSearch.init(space)
        params = optimizer.get_next_params(state, _KEY)
        assert params is not None

    def test_get_next_params_n_parallel_batch(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space, n_parallel=2)
        params = optimizer.get_next_params(state, _KEY)
        assert params["x"].shape == (2,)

    def test_get_next_params_raises_on_overflow(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space, n_parallel=2)
        # Move to last valid position
        state = dataclasses.replace(state, grid_idx=2)
        with pytest.raises(ValueError, match="Not enough grid points"):
            optimizer.get_next_params(state, _KEY)


class TestGridSearchOptimizeScan:
    def test_optimize_scan_runs(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=3
        )
        assert params_hist["x"].shape[0] == 3
        assert results_hist.shape[0] == 3


class TestGridSearchOptimize:
    def test_optimize_runs_n_iterations(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), lambda key, config: -(config["x"] ** 2), 3
        )
        assert len(params_hist) == 3
        assert len(results_hist) == 3

    def test_optimize_increments_grid_idx(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = grid.GridSearch.init(space)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), lambda key, config: -(config["x"] ** 2), 2
        )
        assert state.grid_idx == 2

    def test_state_space_accessible_via_inheritance(self):
        # Regression: GridSearchState used to duplicate the `space` field from
        # OptimizerState, which could shadow the parent field. Verify space is
        # still accessible after removing the redundant definition.
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = grid.GridSearch.init(space)
        leaves = jax.tree.leaves(state.space, is_leaf=lambda x: isinstance(x, sp.Space))
        assert len(leaves) == 1

    def test_optimize_n_parallel(self):
        space = {"x": sp.DiscreteSpace(list(range(6)))}
        state, optimizer = grid.GridSearch.init(space, n_parallel=2)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=3
        )
        assert len(params_hist) == 3
        assert state.grid_idx == 6  # 3 * 2 = 6
