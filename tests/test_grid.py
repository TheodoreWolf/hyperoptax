import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.grid import GridSearch, GridSearchState


class TestGridSearchInit:
    def test_init_discrete_1d(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state = GridSearch.init(space)
        assert isinstance(state, GridSearchState)
        assert state.space_idx == 0
        assert not state.random_shuffle

    def test_init_discrete_2d(self):
        space = {"x": sp.DiscreteSpace([0, 1, 2]), "y": sp.DiscreteSpace([0.0, 0.5])}
        state = GridSearch.init(space)
        assert isinstance(state, GridSearchState)
        assert state.space_idx == 0

    def test_init_random_shuffle(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state = GridSearch.init(space, random_shuffle=True)
        assert state.random_shuffle

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
        state = GridSearch.init(space)
        assert state.space_idx == 0
        state = GridSearch.update_state(state)
        assert state.space_idx == 1

    def test_update_state_does_not_mutate(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state = GridSearch.init(space)
        new_state = GridSearch.update_state(state)
        assert state.space_idx == 0
        assert new_state.space_idx == 1

    def test_update_state_increments_repeatedly(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state = GridSearch.init(space)
        for i in range(5):
            assert state.space_idx == i
            state = GridSearch.update_state(state)


class TestGridSearchSpaceFlat:
    def test_flat_space_size_is_product_of_dim_sizes(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state = GridSearch.init(space)
        assert state.space_flat.shape[0] == 6 # 2 * 3 = 6
        assert state.space_flat.shape[1] == 2  # 2 params


class TestGridSearchGetNextParams:
    def test_get_next_params_first_index(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state = GridSearch.init(space)
        params = GridSearch.get_next_params(state)
        assert params is not None

    def test_get_next_params_changes_after_update(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state = GridSearch.init(space)
        params_0 = GridSearch.get_next_params(state)
        state = GridSearch.update_state(state)
        params_1 = GridSearch.get_next_params(state)
        assert not jnp.allclose(params_0["x"], params_1["x"])

    def test_get_next_params_2d_grid(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state = GridSearch.init(space)
        params = GridSearch.get_next_params(state)
        assert params is not None
