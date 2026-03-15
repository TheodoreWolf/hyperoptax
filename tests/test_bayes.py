import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.bayesian import BayesianOptimizerState, BayesianSearch
from hyperoptax.kernels import RBF
from hyperoptax.acquisition import EI, PI


class TestBayesianSearchInit:
    def test_state_shapes(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = BayesianSearch.init(space, n_max=20)
        assert state.X.shape == (20, 1)
        assert state.y.shape == (20,)
        assert state.mask.shape == (20,)
        assert state.log_length_scale.shape == (1,)  # 1D space → 1 length scale

    def test_state_initial_values(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = BayesianSearch.init(space, n_max=10)
        assert not state.mask.any()
        assert jnp.all(state.X == 0.0)
        assert jnp.all(state.y == 0.0)

    def test_log_length_scale_initialized_from_kernel(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = BayesianSearch.init(space, kernel=RBF(length_scale=2.0))
        assert jnp.allclose(state.log_length_scale, jnp.log(2.0))

    def test_2d_space_n_params(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, _ = BayesianSearch.init(space, n_max=10)
        assert state.X.shape == (10, 2)
        assert state.log_length_scale.shape == (2,)  # 2D space → 2 length scales

    def test_returns_optimizer_instance(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        _, optimizer = BayesianSearch.init(space)
        assert isinstance(optimizer, BayesianSearch)

    def test_custom_kernel_and_acquisition(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        _, optimizer = BayesianSearch.init(space, kernel=RBF(), acquisition=EI())
        assert isinstance(optimizer.kernel, RBF)
        assert isinstance(optimizer.acquisition, EI)


class TestBayesianSearchUpdateState:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        self.state, self.optimizer = BayesianSearch.init(
            self.space, n_max=10, n_hparam_steps=0
        )
        self.key = jax.random.PRNGKey(0)

    def test_mask_updated(self):
        x_new = jnp.array([0.5])
        new_state = self.optimizer.update_state(self.state, self.key, 0.5, x_new)
        assert new_state.mask[0]
        assert not new_state.mask[1:].any()

    def test_X_updated(self):
        x_new = jnp.array([0.5])
        new_state = self.optimizer.update_state(self.state, self.key, 0.5, x_new)
        assert jnp.allclose(new_state.X[0], x_new)

    def test_y_updated(self):
        x_new = jnp.array([0.5])
        new_state = self.optimizer.update_state(self.state, self.key, 0.75, x_new)
        assert jnp.allclose(new_state.y[0], 0.75)

    def test_y_stores_raw_value_when_minimizing(self):
        _, optimizer = BayesianSearch.init(
            self.space, n_max=10, maximize=False, n_hparam_steps=0
        )
        x_new = jnp.array([0.5])
        new_state = optimizer.update_state(self.state, self.key, 0.75, x_new)
        # raw value stored unchanged; GP sees negated value internally
        assert jnp.allclose(new_state.y[0], 0.75)

    def test_fixed_size_maintained(self):
        x_new = jnp.array([0.5])
        new_state = self.optimizer.update_state(self.state, self.key, 0.5, x_new)
        assert new_state.X.shape == self.state.X.shape
        assert new_state.y.shape == self.state.y.shape

    def test_sequential_updates(self):
        state = self.state
        for x_val, y_val in [(0.0, 0.1), (0.5, 0.9), (1.0, 0.4)]:
            x_new = jnp.array([x_val])
            state = self.optimizer.update_state(state, self.key, y_val, x_new)
        assert int(state.mask.sum()) == 3
        assert jnp.allclose(state.y[:3], jnp.array([0.1, 0.9, 0.4]))

    def test_overflow_raises(self):
        state, optimizer = BayesianSearch.init(
            self.space, n_max=2, n_hparam_steps=0
        )
        state = optimizer.update_state(state, self.key, 0.5, jnp.array([0.5]))
        state = optimizer.update_state(state, self.key, 0.9, jnp.array([1.0]))
        with pytest.raises(ValueError, match="capacity exceeded"):
            optimizer.update_state(state, self.key, 0.1, jnp.array([0.0]))


class TestBayesianSearchGetNextParams:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        self.state, self.optimizer = BayesianSearch.init(self.space, n_max=20)
        self.key = jax.random.PRNGKey(0)

    def test_random_pick_when_no_observations(self):
        params, x_new = self.optimizer.get_next_params(self.state, self.key)
        assert "x" in params
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(x_new[0]) in valid_values

    def test_returns_valid_candidate_value(self):
        params, x_new = self.optimizer.get_next_params(self.state, self.key)
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(params["x"]) in valid_values

    def test_gp_pick_after_observations(self):
        state = self.state
        for x_val, y_val in [(0.0, 0.1), (1.0, 0.9)]:
            x_new = jnp.array([x_val])
            state = self.optimizer.update_state(state, self.key, y_val, x_new)
        params, x_new = self.optimizer.get_next_params(state, self.key)
        assert "x" in params
        assert float(x_new[0]) not in [0.0, 1.0]  # seen candidates should not be re-selected

    def test_x_new_matches_params(self):
        params, x_new = self.optimizer.get_next_params(self.state, self.key)
        assert jnp.allclose(x_new[0], params["x"])

    def test_2d_space(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        params, x_new = optimizer.get_next_params(state, self.key)
        assert "x" in params
        assert "y" in params
        assert x_new.shape == (2,)


class TestBayesianSearchOptimize:
    def test_optimize_returns_correct_shapes(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        func = lambda key, config: -(config["x"]**2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert len(params_hist) == 5
        assert len(results_hist) == 5

    def test_optimize_fills_state(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        func = lambda key, config: -(config["x"]**2)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert int(state.mask.sum()) == 5

    def test_optimize_finds_optimum(self):
        # With enough iterations to cover the space, should find x=0
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        func = lambda key, config: -(config["x"]**2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        best_idx = results_hist.index(max(results_hist, key=float))
        assert float(params_hist[best_idx]["x"]) == pytest.approx(0.0)

    def test_optimize_with_array_result(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        func = lambda key, config: jnp.array([-(config["x"]**2)])
        state, (_, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=3
        )
        assert int(state.mask.sum()) == 3

    def test_optimize_no_repeated_candidates(self):
        from hyperoptax.acquisition import UCB
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, acquisition=UCB(stochastic_multiplier=1)
        )
        func = lambda key, config: -(config["x"]**2)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        seen_X = state.X[:5]
        assert len(jnp.unique(seen_X)) == 5

    def test_optimize_continuous_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        func = lambda key, config: -(config["x"]**2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert len(params_hist) == 5
        assert int(state.mask.sum()) == 5

    def test_optimize_continuous_with_ei_uses_observed_y_max(self):
        space = {"x": sp.LinearSpace(0.0, 1.0), "y": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=20, acquisition=EI())
        x_new = jnp.array([0.5, 0.5])
        state = optimizer.update_state(state, jax.random.PRNGKey(0), 100.0, x_new)
        params, _ = optimizer.get_next_params(state, jax.random.PRNGKey(1))
        assert "x" in params and "y" in params

    def test_optimize_minimize(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10, maximize=False)
        func = lambda key, config: config["x"]**2  # minimum at x=0
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert int(state.mask.sum()) == 5
        # Raw values stored, best is the minimum
        assert float(optimizer.best_result(state)) == pytest.approx(0.0)


class TestBestParamsResult:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        self.key = jax.random.PRNGKey(0)

    def _state_with_obs(self, optimizer, observations):
        state, _ = BayesianSearch.init(
            self.space, n_max=10, n_hparam_steps=0, maximize=optimizer.maximize
        )
        for x_val, y_val in observations:
            state = optimizer.update_state(
                state, self.key, y_val, jnp.array([x_val])
            )
        return state

    def test_best_result_maximize(self):
        _, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=0)
        state = self._state_with_obs(optimizer, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.3)])
        assert float(optimizer.best_result(state)) == pytest.approx(0.9)

    def test_best_result_minimize(self):
        _, optimizer = BayesianSearch.init(
            self.space, n_max=10, n_hparam_steps=0, maximize=False
        )
        state = self._state_with_obs(optimizer, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.2)])
        assert float(optimizer.best_result(state)) == pytest.approx(0.2)

    def test_best_params_maximize(self):
        _, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=0)
        state = self._state_with_obs(optimizer, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.3)])
        params = optimizer.best_params(state)
        assert float(params["x"][0]) == pytest.approx(0.75)

    def test_best_params_minimize(self):
        _, optimizer = BayesianSearch.init(
            self.space, n_max=10, n_hparam_steps=0, maximize=False
        )
        state = self._state_with_obs(optimizer, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.2)])
        params = optimizer.best_params(state)
        assert float(params["x"][0]) == pytest.approx(0.5)

    def test_best_result_after_full_optimize(self):
        state, optimizer = BayesianSearch.init(self.space, n_max=10)
        func = lambda key, config: -(config["x"]**2)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        # best_result should match the max of what func returned
        assert float(optimizer.best_result(state)) == pytest.approx(
            float(jnp.max(state.y, where=state.mask, initial=-jnp.inf))
        )


class TestHparamTuning:
    def setup_method(self):
        self.space = {"x": sp.LinearSpace(0.0, 1.0)}
        self.key = jax.random.PRNGKey(0)

    def test_log_length_scale_changes_after_observations(self):
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=20)
        initial_log_ls = state.log_length_scale
        state = optimizer.update_state(state, self.key, 0.5, jnp.array([0.3]))
        state = optimizer.update_state(state, self.key, 0.8, jnp.array([0.7]))
        # Two observations triggers tuning; length scale should have moved
        assert not jnp.allclose(state.log_length_scale, initial_log_ls)

    def test_log_length_scale_unchanged_when_n_hparam_steps_0(self):
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=0)
        initial_log_ls = state.log_length_scale
        state = optimizer.update_state(state, self.key, 0.5, jnp.array([0.3]))
        state = optimizer.update_state(state, self.key, 0.8, jnp.array([0.7]))
        assert jnp.allclose(state.log_length_scale, initial_log_ls)

    def test_log_length_scale_unchanged_with_single_observation(self):
        # Tuning requires n_seen >= 2 (need at least 2 points for MLL)
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=20)
        initial_log_ls = state.log_length_scale
        state = optimizer.update_state(state, self.key, 0.5, jnp.array([0.3]))
        assert jnp.allclose(state.log_length_scale, initial_log_ls)

    def test_tuned_length_scale_used_in_gp(self):
        # Tuning runs without error and the optimizer still produces valid params
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=20)
        func = lambda key, config: -(config["x"]**2)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert int(state.mask.sum()) == 5

    def test_length_scale_initialized_from_kernel(self):
        state, _ = BayesianSearch.init(
            self.space, n_max=10, kernel=RBF(length_scale=3.0)
        )
        assert jnp.allclose(jnp.exp(state.log_length_scale), 3.0)


class TestNInitialRandom:
    def test_n_initial_random_default(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        _, optimizer = BayesianSearch.init(space)
        assert optimizer.n_initial_random == 1

    def test_n_initial_random_runs_correctly(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=20, n_initial_random=4)
        func = lambda key, config: -(config["x"]**2)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=8
        )
        assert int(state.mask.sum()) == 8

    def test_n_initial_random_3_discrete(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, n_initial_random=3, n_hparam_steps=0
        )
        func = lambda key, config: -(config["x"]**2)
        state, _ = optimizer.optimize(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert int(state.mask.sum()) == 5


class TestValidateFunc:
    def test_single_arg_raises(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        with pytest.raises(TypeError, match="fn\\(key, config\\)"):
            optimizer.optimize(state, jax.random.PRNGKey(0), lambda x: x, n_iterations=1)

    def test_zero_arg_raises(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        with pytest.raises(TypeError):
            optimizer.optimize(state, jax.random.PRNGKey(0), lambda: 1.0, n_iterations=1)

    def test_two_arg_passes(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        # Should not raise
        optimizer.optimize(state, jax.random.PRNGKey(0),
                           lambda key, config: config["x"][0], n_iterations=1)


class TestBayesianSearchOptimizeScan:
    def test_optimize_scan_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        func = lambda key, config: -(config["x"][0] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func, n_iterations=5
        )
        assert params_hist["x"].shape[0] == 5
        assert results_hist.shape[0] == 5
        assert int(state.mask.sum()) == 5


class TestMaximizeMinimize:
    def test_maximize_default_is_true(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        _, optimizer = BayesianSearch.init(space)
        assert optimizer.maximize is True

    def test_minimize_stores_raw_y(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = BayesianSearch.init(
            self.space if hasattr(self, "space") else space,
            n_max=10, maximize=False, n_hparam_steps=0
        )
        x_new = jnp.array([0.5])
        state = optimizer.update_state(state, jax.random.PRNGKey(0), 3.14, x_new)
        assert float(state.y[0]) == pytest.approx(3.14)

    def test_best_result_minimize_returns_minimum(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, maximize=False, n_hparam_steps=0
        )
        key = jax.random.PRNGKey(0)
        for x_val, y_val in [(0.0, 5.0), (0.5, 2.0), (1.0, 8.0)]:
            state = optimizer.update_state(
                state, key, y_val, jnp.array([x_val])
            )
        assert float(optimizer.best_result(state)) == pytest.approx(2.0)

    def test_effective_y_negated_for_minimize(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, maximize=False, n_hparam_steps=0
        )
        key = jax.random.PRNGKey(0)
        state = optimizer.update_state(state, key, 3.0, jnp.array([0.5]))
        eff = optimizer._effective_y(state)
        # Raw stored as 3.0, effective should be -3.0
        assert float(eff[0]) == pytest.approx(-3.0)
