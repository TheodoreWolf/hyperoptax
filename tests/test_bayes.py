import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax.bayesian import BayesianOptimizerState, BayesianSearch
from hyperoptax.kernels import RBF
from hyperoptax.acquisition import EI, PI, UCB, MeanLiar, SampleLiar, UCBLiar, ConstantLiar


class TestBayesianSearchInit:
    def test_state_shapes(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = BayesianSearch.init(space, n_max=20)
        assert state.X.shape == (20, 1)
        assert state.y.shape == (20,)
        assert state.mask.shape == (20,)
        assert state.log_length_scale.shape == (1,)

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
        assert state.log_length_scale.shape == (2,)

    def test_returns_optimizer_instance(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        _, optimizer = BayesianSearch.init(space)
        assert isinstance(optimizer, BayesianSearch)

    def test_custom_kernel_and_acquisition(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        _, optimizer = BayesianSearch.init(space, kernel=RBF(), acquisition=EI())
        assert isinstance(optimizer.kernel, RBF)
        assert isinstance(optimizer.acquisition, EI)

    def test_default_n_max(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, _ = BayesianSearch.init(space)
        assert state.X.shape[0] == 200


class TestBayesianSearchUpdateState:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        self.state, self.optimizer = BayesianSearch.init(
            self.space, n_max=10, n_hparam_steps=0
        )
        self.key = jax.random.PRNGKey(0)

    def test_mask_updated(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(self.state, self.key, jnp.array([0.5]), x_new)
        assert new_state.mask[0]
        assert not new_state.mask[1:].any()

    def test_X_updated(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(self.state, self.key, jnp.array([0.5]), x_new)
        assert jnp.allclose(new_state.X[0], x_new[0])

    def test_y_updated(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(self.state, self.key, jnp.array([0.75]), x_new)
        assert jnp.allclose(new_state.y[0], 0.75)

    def test_y_stores_raw_value_when_minimizing(self):
        _, optimizer = BayesianSearch.init(
            self.space, n_max=10, maximize=False, n_hparam_steps=0
        )
        x_new = jnp.array([[0.5]])
        new_state = optimizer.update_state(self.state, self.key, jnp.array([0.75]), x_new)
        assert jnp.allclose(new_state.y[0], 0.75)

    def test_fixed_size_maintained(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(self.state, self.key, jnp.array([0.5]), x_new)
        assert new_state.X.shape == self.state.X.shape
        assert new_state.y.shape == self.state.y.shape

    def test_sequential_updates(self):
        state = self.state
        for x_val, y_val in [(0.0, 0.1), (0.5, 0.9), (1.0, 0.4)]:
            x_new = jnp.array([[x_val]])
            state = self.optimizer.update_state(state, self.key, jnp.array([y_val]), x_new)
        assert int(state.mask.sum()) == 3
        assert jnp.allclose(state.y[:3], jnp.array([0.1, 0.9, 0.4]))

    def test_overflow_truncates_to_remaining_slots(self):
        """When buffer has k < n_parallel slots left, only k observations are stored."""
        state, optimizer = BayesianSearch.init(
            self.space, n_max=3, n_hparam_steps=0
        )
        # fill 2 of 3 slots
        state = optimizer.update_state(state, self.key, jnp.array([0.1]), jnp.array([[0.0]]))
        state = optimizer.update_state(state, self.key, jnp.array([0.2]), jnp.array([[0.5]]))
        assert int(state.mask.sum()) == 2

        # try to store 5 at once — only 1 slot remains
        x_batch = jnp.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        r_batch = jnp.array([0.3, 0.4, 0.5, 0.6, 0.7])
        state = optimizer.update_state(state, self.key, r_batch, x_batch)
        assert int(state.mask.sum()) == 3  # only 1 extra stored
        assert jnp.allclose(state.y[2], 0.3)  # first result from batch

    def test_overflow_at_zero_remaining_is_noop(self):
        """Calling update_state on a full buffer returns state unchanged."""
        state, optimizer = BayesianSearch.init(
            self.space, n_max=2, n_hparam_steps=0
        )
        state = optimizer.update_state(state, self.key, jnp.array([0.5]), jnp.array([[0.5]]))
        state = optimizer.update_state(state, self.key, jnp.array([0.9]), jnp.array([[1.0]]))
        assert int(state.mask.sum()) == 2

        state_after = optimizer.update_state(state, self.key, jnp.array([0.1]), jnp.array([[0.0]]))
        assert int(state_after.mask.sum()) == 2  # unchanged
        assert jnp.array_equal(state_after.y, state.y)

    def test_n_parallel_overflow_stores_only_remainder(self):
        """With n_max=7 and n_parallel=4: 1 full iter (4 stored) + overflow stores 3."""
        state, optimizer = BayesianSearch.init(
            self.space, n_max=7, n_parallel=4, n_hparam_steps=0
        )
        x4 = jnp.zeros((4, 1))
        r4 = jnp.ones(4) * 0.5
        state = optimizer.update_state(state, self.key, r4, x4)
        assert int(state.mask.sum()) == 4

        # 3 slots remain; try to store 4
        state = optimizer.update_state(state, self.key, r4, x4)
        assert int(state.mask.sum()) == 7  # capped at n_max


class TestNIterations:
    """Tests for the _n_iterations helper and the resulting loop count."""

    def setup_method(self):
        self.space = {"x": sp.LinearSpace(0.0, 1.0)}
        self.func = lambda key, config: -(config["x"] ** 2)

    def _run(self, n_max, n_parallel):
        state, opt = BayesianSearch.init(
            self.space, n_max=n_max, n_parallel=n_parallel, n_hparam_steps=0
        )
        state, (params_hist, results_hist) = opt.optimize(
            state, jax.random.PRNGKey(0), self.func
        )
        return state, params_hist, results_hist

    def test_exact_fit_iterations(self):
        """n_max divisible by n_parallel: exactly n_max//n_parallel iterations."""
        state, params_hist, _ = self._run(n_max=10, n_parallel=2)
        assert len(params_hist) == 5
        assert int(state.mask.sum()) == 10

    def test_exact_fit_parallel_1(self):
        state, params_hist, _ = self._run(n_max=7, n_parallel=1)
        assert len(params_hist) == 7
        assert int(state.mask.sum()) == 7

    def test_overflow_one_extra_iteration(self):
        """n_max=9, n_parallel=4: 2 full iters + 1 overflow = 3 iters, 9 stored."""
        state, params_hist, _ = self._run(n_max=9, n_parallel=4)
        assert len(params_hist) == 3  # ceil(9/4)
        assert int(state.mask.sum()) == 9

    def test_overflow_remainder_1(self):
        """n_max=11, n_parallel=5: 2 full + 1 overflow (stores 1) = 3 iters."""
        state, params_hist, _ = self._run(n_max=11, n_parallel=5)
        assert len(params_hist) == 3
        assert int(state.mask.sum()) == 11

    def test_overflow_n_max_less_than_n_parallel(self):
        """n_max < n_parallel: single overflow iteration, stores n_max."""
        state, params_hist, _ = self._run(n_max=3, n_parallel=10)
        assert len(params_hist) == 1
        assert int(state.mask.sum()) == 3

    def test_overflow_n_max_equals_n_parallel(self):
        """n_max == n_parallel: exactly 1 iteration, all slots filled."""
        state, params_hist, _ = self._run(n_max=5, n_parallel=5)
        assert len(params_hist) == 1
        assert int(state.mask.sum()) == 5

    def test_n_iterations_from_partial_state(self):
        """Starting from a pre-populated state uses remaining slots only."""
        state, opt = BayesianSearch.init(
            self.space, n_max=10, n_parallel=3, n_hparam_steps=0
        )
        # pre-load 4 observations
        for v in [0.1, 0.2, 0.3, 0.4]:
            state = opt.update_state(
                state, jax.random.PRNGKey(0), jnp.array([v]), jnp.array([[v]])
            )
        assert int(state.mask.sum()) == 4  # 6 slots remaining

        state, (params_hist, _) = opt.optimize(
            state, jax.random.PRNGKey(1), self.func
        )
        # 6 remaining // 3 = 2 full + 0 overflow = 2 iters, ends at 10
        assert len(params_hist) == 2
        assert int(state.mask.sum()) == 10

    def test_n_iterations_from_partial_state_with_overflow(self):
        """Remaining slots not divisible by n_parallel → overflow iteration."""
        state, opt = BayesianSearch.init(
            self.space, n_max=10, n_parallel=3, n_hparam_steps=0
        )
        # pre-load 5 observations → 5 remaining
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            state = opt.update_state(
                state, jax.random.PRNGKey(0), jnp.array([v]), jnp.array([[v]])
            )
        state, (params_hist, _) = opt.optimize(
            state, jax.random.PRNGKey(1), self.func
        )
        # 5 remaining // 3 = 1 full + 1 overflow = 2 iters, ends at 10
        assert len(params_hist) == 2
        assert int(state.mask.sum()) == 10

    def test_full_buffer_runs_zero_iterations(self):
        """Calling optimize on a full buffer runs 0 iterations."""
        state, opt = BayesianSearch.init(
            self.space, n_max=2, n_parallel=1, n_hparam_steps=0
        )
        state = opt.update_state(
            state, jax.random.PRNGKey(0), jnp.array([0.5]), jnp.array([[0.5]])
        )
        state = opt.update_state(
            state, jax.random.PRNGKey(0), jnp.array([0.9]), jnp.array([[0.9]])
        )
        state2, (params_hist, results_hist) = opt.optimize(
            state, jax.random.PRNGKey(0), self.func
        )
        assert len(params_hist) == 0
        assert int(state2.mask.sum()) == 2  # unchanged

    def test_results_hist_length_matches_params_hist(self):
        state, params_hist, results_hist = self._run(n_max=9, n_parallel=4)
        assert len(params_hist) == len(results_hist)

    def test_each_results_item_has_n_parallel_shape(self):
        """Every results item has shape (n_parallel,) including overflow iteration."""
        state, opt = BayesianSearch.init(
            self.space, n_max=9, n_parallel=4, n_hparam_steps=0
        )
        state, (_, results_hist) = opt.optimize(
            state, jax.random.PRNGKey(0), self.func
        )
        for r in results_hist:
            assert r.shape == (4,)


class TestBayesianSearchGetNextParams:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        self.state, self.optimizer = BayesianSearch.init(self.space, n_max=20)
        self.key = jax.random.PRNGKey(0)

    def test_random_pick_when_no_observations(self):
        params, x_new = self.optimizer.get_next_params(self.state, self.key)
        assert "x" in params
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(x_new[0, 0]) in valid_values

    def test_returns_valid_candidate_value(self):
        params, x_new = self.optimizer.get_next_params(self.state, self.key)
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(params["x"][0]) in valid_values

    def test_gp_pick_after_observations(self):
        state = self.state
        for x_val, y_val in [(0.0, 0.1), (1.0, 0.9)]:
            x_new = jnp.array([[x_val]])
            state = self.optimizer.update_state(state, self.key, jnp.array([y_val]), x_new)
        params, x_new = self.optimizer.get_next_params(state, self.key)
        assert "x" in params
        assert float(x_new[0, 0]) not in [0.0, 1.0]

    def test_x_new_matches_params(self):
        params, x_new = self.optimizer.get_next_params(self.state, self.key)
        assert jnp.allclose(x_new[0, 0], params["x"][0])

    def test_2d_space(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, optimizer = BayesianSearch.init(space, n_max=10)
        params, x_new = optimizer.get_next_params(state, self.key)
        assert "x" in params
        assert "y" in params
        assert x_new.shape == (1, 2)

    def test_n_parallel_discrete(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=20, n_parallel=3)
        params, x_new = optimizer.get_next_params(state, self.key)
        assert params["x"].shape == (3,)
        assert x_new.shape == (3, 1)


class TestBayesianSearchOptimize:
    def test_optimize_returns_correct_shapes(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert len(params_hist) == 5
        assert len(results_hist) == 5

    def test_optimize_fills_state(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 5

    def test_optimize_finds_optimum(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=20)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func
        )
        best_idx = results_hist.index(max(results_hist, key=lambda r: float(r[0])))
        assert float(params_hist[best_idx]["x"][0]) == pytest.approx(0.0)

    def test_optimize_with_array_result(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=3)
        func = lambda key, config: jnp.array([-(config["x"] ** 2)])
        state, (_, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert int(state.mask.sum()) == 3

    def test_optimize_converges_toward_optimum(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=20, acquisition=UCB()
        )
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        assert float(jnp.min(state.X[:20, 0])) == pytest.approx(0.0)

    def test_optimize_continuous_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert len(params_hist) == 5
        assert int(state.mask.sum()) == 5

    def test_optimize_continuous_with_ei_uses_observed_y_max(self):
        space = {"x": sp.LinearSpace(0.0, 1.0), "y": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=20, acquisition=EI())
        x_new = jnp.array([[0.5, 0.5]])
        state = optimizer.update_state(state, jax.random.PRNGKey(0), jnp.array([100.0]), x_new)
        params, _ = optimizer.get_next_params(state, jax.random.PRNGKey(1))
        assert "x" in params and "y" in params

    def test_optimize_minimize(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10, maximize=False)
        func = lambda key, config: config["x"] ** 2
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 10
        assert float(optimizer.best_result(state)) == pytest.approx(0.0)

    def test_optimize_n_parallel_fills_buffer(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(space, n_max=10, n_parallel=2)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert len(params_hist) == 5  # 10 // 2
        assert int(state.mask.sum()) == 10
        assert results_hist[0].shape == (2,)


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
                state, self.key, jnp.array([y_val]), jnp.array([[x_val]])
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
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
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
        state = optimizer.update_state(state, self.key, jnp.array([0.5]), jnp.array([[0.3]]))
        state = optimizer.update_state(state, self.key, jnp.array([0.8]), jnp.array([[0.7]]))
        assert not jnp.allclose(state.log_length_scale, initial_log_ls)

    def test_log_length_scale_unchanged_when_n_hparam_steps_0(self):
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=0)
        initial_log_ls = state.log_length_scale
        state = optimizer.update_state(state, self.key, jnp.array([0.5]), jnp.array([[0.3]]))
        state = optimizer.update_state(state, self.key, jnp.array([0.8]), jnp.array([[0.7]]))
        assert jnp.allclose(state.log_length_scale, initial_log_ls)

    def test_log_length_scale_unchanged_with_single_observation(self):
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=20)
        initial_log_ls = state.log_length_scale
        state = optimizer.update_state(state, self.key, jnp.array([0.5]), jnp.array([[0.3]]))
        assert jnp.allclose(state.log_length_scale, initial_log_ls)

    def test_tuned_length_scale_used_in_gp(self):
        state, optimizer = BayesianSearch.init(self.space, n_max=10, n_hparam_steps=20)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 10

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
        state, optimizer = BayesianSearch.init(space, n_max=8, n_initial_random=4)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 8

    def test_n_initial_random_3_discrete(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=5, n_initial_random=3, n_hparam_steps=0
        )
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 5


class TestMixedSpace:
    def test_get_next_params_mixed(self):
        space = {"lr": sp.LogSpace(1e-4, 1e-1), "layers": sp.DiscreteSpace([1, 2, 3, 4])}
        state, optimizer = BayesianSearch.init(space, n_max=20)
        params, x_new = optimizer.get_next_params(state, jax.random.PRNGKey(0))
        assert "lr" in params and "layers" in params
        assert x_new.shape == (1, 2)
        assert float(params["layers"][0]) in [1, 2, 3, 4]

    def test_optimize_mixed_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0), "n": sp.DiscreteSpace([1, 2, 4, 8])}
        state, optimizer = BayesianSearch.init(space, n_max=6)
        func = lambda key, config: -(config["x"] ** 2) + config["n"] * 0.1
        state, (params_hist, results_hist) = optimizer.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert int(state.mask.sum()) == 6
        for p in params_hist:
            assert float(p["n"][0]) in [1, 2, 4, 8]


class TestValidateFunc:
    def test_single_arg_raises(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        with pytest.raises(TypeError, match="fn\\(key, config\\)"):
            optimizer.optimize(state, jax.random.PRNGKey(0), lambda x: x)

    def test_zero_arg_raises(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        with pytest.raises(TypeError):
            optimizer.optimize(state, jax.random.PRNGKey(0), lambda: 1.0)

    def test_two_arg_passes(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        optimizer.optimize(state, jax.random.PRNGKey(0), lambda key, config: config["x"])


class TestBayesianSearchOptimizeScan:
    def test_optimize_scan_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=5)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func
        )
        assert params_hist["x"].shape[0] == 5
        assert results_hist.shape[0] == 5
        assert int(state.mask.sum()) == 5

    def test_optimize_scan_overflow(self):
        """optimize_scan also handles overflow correctly."""
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(space, n_max=7, n_parallel=3)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = optimizer.optimize_scan(
            state, jax.random.PRNGKey(0), func
        )
        # 7 // 3 = 2 full + 1 overflow = 3 iters
        assert params_hist["x"].shape[0] == 3
        assert int(state.mask.sum()) == 7


class TestMaximizeMinimize:
    def test_maximize_default_is_true(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        _, optimizer = BayesianSearch.init(space)
        assert optimizer.maximize is True

    def test_minimize_stores_raw_y(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, maximize=False, n_hparam_steps=0
        )
        x_new = jnp.array([[0.5]])
        state = optimizer.update_state(state, jax.random.PRNGKey(0), jnp.array([3.14]), x_new)
        assert float(state.y[0]) == pytest.approx(3.14)

    def test_best_result_minimize_returns_minimum(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, maximize=False, n_hparam_steps=0
        )
        key = jax.random.PRNGKey(0)
        for x_val, y_val in [(0.0, 5.0), (0.5, 2.0), (1.0, 8.0)]:
            state = optimizer.update_state(
                state, key, jnp.array([y_val]), jnp.array([[x_val]])
            )
        assert float(optimizer.best_result(state)) == pytest.approx(2.0)

    def test_effective_y_negated_for_minimize(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, optimizer = BayesianSearch.init(
            space, n_max=10, maximize=False, n_hparam_steps=0
        )
        key = jax.random.PRNGKey(0)
        state = optimizer.update_state(state, key, jnp.array([3.0]), jnp.array([[0.5]]))
        eff = optimizer._effective_y(state)
        assert float(eff[0]) == pytest.approx(-3.0)


class TestGPInternals:
    def setup_method(self):
        self.space = {"x": sp.LinearSpace(0.0, 1.0)}
        self.state, self.optimizer = BayesianSearch.init(
            self.space, n_max=10, n_hparam_steps=0
        )
        key = jax.random.PRNGKey(0)
        for x_val, y_val in [(0.1, 0.2), (0.5, 0.8), (0.9, 0.3)]:
            self.state = self.optimizer.update_state(
                self.state, key, jnp.array([y_val]), jnp.array([[x_val]])
            )

    def test_gp_posterior_matches_fit_predict(self):
        ls = jnp.exp(self.state.log_length_scale)
        eff_y = self.optimizer._effective_y(self.state)
        X_test = jnp.array([[0.3], [0.7]])
        mean_post, std_post = self.optimizer._gp_posterior(
            self.state.X, eff_y, self.state.mask, X_test, ls
        )
        L, alpha, ymean = self.optimizer._gp_fit(
            self.state.X, eff_y, self.state.mask, ls
        )
        mean_fp, std_fp = self.optimizer._gp_predict(
            X_test, L, alpha, ymean, self.state.X, ls
        )
        assert jnp.allclose(mean_post, mean_fp)
        assert jnp.allclose(std_post, std_fp)

    def test_gp_posterior_returns_correct_shapes(self):
        ls = jnp.exp(self.state.log_length_scale)
        eff_y = self.optimizer._effective_y(self.state)
        X_test = jnp.array([[0.2], [0.4], [0.6], [0.8]])
        mean, std = self.optimizer._gp_posterior(
            self.state.X, eff_y, self.state.mask, X_test, ls
        )
        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert jnp.all(std >= 0.0)


class TestLBFGSImprovement:
    def test_lbfgs_improves_over_seed(self):
        space = {"x": sp.LinearSpace(0.0, 1.0), "y": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(
            space,
            n_max=30,
            n_candidates=200,
            n_restarts=4,
            n_lbfgs_steps=50,
            n_hparam_steps=0,
        )
        key = jax.random.PRNGKey(42)
        obs = [
            ([0.3, 0.7], 1.0),
            ([0.0, 0.0], 0.0),
            ([1.0, 1.0], 0.1),
            ([0.5, 0.5], 0.3),
            ([0.3, 0.5], 0.6),
        ]
        for (x, y_), r in obs:
            state = optimizer.update_state(
                state, key, jnp.array([r]), jnp.array([[x, y_]])
            )
        params, x_new = optimizer.get_next_params(state, key)
        assert x_new.shape == (1, 2)
        assert 0.0 <= float(params["x"][0]) <= 1.0
        assert 0.0 <= float(params["y"][0]) <= 1.0


class TestKrigingBelieverLiar:
    """Tests for pluggable KB hallucination strategies."""

    def _make_state_with_obs(self, space, n_max, n_parallel, liar=None, n_obs=3):
        kwargs = dict(n_max=n_max, n_parallel=n_parallel, n_hparam_steps=0,
                      n_restarts=1, n_lbfgs_steps=5, n_candidates=50)
        if liar is not None:
            kwargs["liar"] = liar
        state, optimizer = BayesianSearch.init(space, **kwargs)
        key = jax.random.PRNGKey(0)
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        n_params = len(leaves)
        for i in range(n_obs):
            x = jnp.zeros((1, n_params)) + i * 0.1
            state = optimizer.update_state(state, key, jnp.array([float(i)]), x)
        return state, optimizer

    def test_default_liar_is_mean(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        _, optimizer = BayesianSearch.init(space)
        assert isinstance(optimizer.liar, MeanLiar)

    def test_mean_liar_optimize_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = self._make_state_with_obs(
            space, n_max=9, n_parallel=3, liar=MeanLiar()
        )
        key = jax.random.PRNGKey(1)
        func = lambda k, p: p["x"] ** 2
        state, _ = optimizer.optimize(state, key, func)
        assert int(state.mask.sum()) == 9

    def test_sample_liar_optimize_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = self._make_state_with_obs(
            space, n_max=9, n_parallel=3, liar=SampleLiar()
        )
        key = jax.random.PRNGKey(2)
        func = lambda k, p: p["x"] ** 2
        state, _ = optimizer.optimize(state, key, func)
        assert int(state.mask.sum()) == 9

    def test_ucb_liar_optimize_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = self._make_state_with_obs(
            space, n_max=9, n_parallel=3, liar=UCBLiar(kappa=1.5)
        )
        key = jax.random.PRNGKey(3)
        func = lambda k, p: p["x"] ** 2
        state, _ = optimizer.optimize(state, key, func)
        assert int(state.mask.sum()) == 9

    def test_constant_liar_optimize_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = self._make_state_with_obs(
            space, n_max=9, n_parallel=3, liar=ConstantLiar()
        )
        key = jax.random.PRNGKey(4)
        func = lambda k, p: p["x"] ** 2
        state, _ = optimizer.optimize(state, key, func)
        assert int(state.mask.sum()) == 9

    def test_different_liars_produce_different_batches(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state_mean, opt_mean = self._make_state_with_obs(
            space, n_max=20, n_parallel=3, liar=MeanLiar()
        )
        state_sample, opt_sample = self._make_state_with_obs(
            space, n_max=20, n_parallel=3, liar=SampleLiar()
        )
        key = jax.random.PRNGKey(99)
        _, x_mean = opt_mean.get_next_params(state_mean, key)
        _, x_sample = opt_sample.get_next_params(state_sample, key)
        # At least one of the parallel points should differ between strategies
        assert not jnp.allclose(x_mean, x_sample)

    def test_liar_used_only_after_n_initial_random(self):
        # With n_initial_random > n_seen, random init path is taken regardless of liar
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, optimizer = BayesianSearch.init(
            space, n_max=20, n_parallel=2, n_initial_random=5,
            n_hparam_steps=0, liar=SampleLiar()
        )
        key = jax.random.PRNGKey(7)
        # Only 2 observations — below n_initial_random=5, so random path taken
        state = optimizer.update_state(state, key, jnp.array([1.0, 0.5]),
                                       jnp.array([[0.1], [0.9]]))
        params, x_new = optimizer.get_next_params(state, key)
        assert x_new.shape == (2, 1)
        assert jnp.all((x_new >= 0.0) & (x_new <= 1.0))
