import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.grid import GridSearch, GridSearchState


class TestGridSearchInit:
    def test_init_discrete_1d(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        assert isinstance(state, GridSearchState)
        assert state.space_idx == 0
        assert not optimizer.shuffle

    def test_init_discrete_2d(self):
        space = {"x": sp.DiscreteSpace([0, 1, 2]), "y": sp.DiscreteSpace([0.0, 0.5])}
        state, optimizer = GridSearch.init(space)
        assert isinstance(state, GridSearchState)
        assert state.space_idx == 0

    def test_init_shuffle(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space, shuffle=True)
        assert optimizer.shuffle

    def test_init_raises_for_continuous_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        with pytest.raises(ValueError):
            GridSearch.init(space)

    def test_init_raises_for_mixed_space(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.LinearSpace(0.0, 1.0)}
        with pytest.raises(ValueError):
            GridSearch.init(space)


class TestGridSearchUpdateState:
    def test_update_state_increments_idx(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        assert state.space_idx == 0
        state = optimizer.update_state(state)
        assert state.space_idx == 1

    def test_update_state_does_not_mutate(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        new_state = optimizer.update_state(state)
        assert state.space_idx == 0
        assert new_state.space_idx == 1

    def test_update_state_increments_repeatedly(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        for i in range(5):
            assert state.space_idx == i
            state = optimizer.update_state(state)


class TestGridSearchShuffle:
    def test_shuffle_reorders_space_flat(self):
        space = {"x": sp.DiscreteSpace(list(range(20)))}
        state_unshuffled, _ = GridSearch.init(space)
        state_shuffled, _ = GridSearch.init(space, shuffle=True, key=jax.random.PRNGKey(0))
        assert not jnp.allclose(state_unshuffled.space_flat, state_shuffled.space_flat)

    def test_shuffle_preserves_size(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, _ = GridSearch.init(space, shuffle=True, key=jax.random.PRNGKey(0))
        assert state.space_flat.shape == (6, 2)

    def test_shuffle_preserves_values(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state_unshuffled, _ = GridSearch.init(space)
        state_shuffled, _ = GridSearch.init(space, shuffle=True, key=jax.random.PRNGKey(0))
        assert jnp.allclose(
            jnp.sort(state_unshuffled.space_flat, axis=0),
            jnp.sort(state_shuffled.space_flat, axis=0),
        )

    def test_different_keys_give_different_orderings(self):
        space = {"x": sp.DiscreteSpace(list(range(20)))}
        state_a, _ = GridSearch.init(space, shuffle=True, key=jax.random.PRNGKey(0))
        state_b, _ = GridSearch.init(space, shuffle=True, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(state_a.space_flat, state_b.space_flat)

    def test_no_shuffle_by_default(self):
        space = {"x": sp.DiscreteSpace([0, 1, 2])}
        state, _ = GridSearch.init(space)
        assert jnp.allclose(state.space_flat[:, 0], jnp.array([0, 1, 2], dtype=state.space_flat.dtype))


class TestGridSearchSpaceFlat:
    def test_flat_space_size_is_product_of_dim_sizes(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, _ = GridSearch.init(space)
        assert state.space_flat.shape[0] == 6  # 2 * 3 = 6
        assert state.space_flat.shape[1] == 2  # 2 params


class TestGridSearchGetNextParams:
    def test_get_next_params_first_index(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        params = optimizer.get_next_params(state)
        assert params is not None

    def test_get_next_params_changes_after_update(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        params_0 = optimizer.get_next_params(state)
        state = optimizer.update_state(state)
        params_1 = optimizer.get_next_params(state)
        assert not jnp.allclose(params_0["x"], params_1["x"])

    def test_get_next_params_2d_grid(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, optimizer = GridSearch.init(space)
        params = optimizer.get_next_params(state)
        assert params is not None


class TestGridSearchOptimizeScan:
    def test_optimize_scan_runs(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=3
        )
        assert params_hist["x"].shape[0] == 3
        assert results_hist.shape[0] == 3


class TestGridSearchOptimize:
    def test_optimize_runs_n_iterations(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), lambda key, config: -(config["x"]**2), 3
        )
        assert len(params_hist) == 3
        assert len(results_hist) == 3

    def test_optimize_increments_space_idx(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = GridSearch.init(space)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), lambda key, config: -(config["x"]**2), 2
        )
        assert state.space_idx == 2

    def test_state_space_accessible_via_inheritance(self):
        # Regression: GridSearchState used to duplicate the `space` field from
        # OptimizerState, which could shadow the parent field. Verify space is
        # still accessible after removing the redundant definition.
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = GridSearch.init(space)
        leaves = jax.tree.leaves(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        assert len(leaves) == 1
