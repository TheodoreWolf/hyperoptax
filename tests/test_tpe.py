import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax import tpe


class TestTPESearchInit:
    def test_state_shapes(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = tpe.TPESearch.init(space, n_max=20)
        assert state.X.shape == (20, 1)
        assert state.y.shape == (20,)
        assert state.mask.shape == (20,)

    def test_state_initial_values(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, _ = tpe.TPESearch.init(space, n_max=10)
        assert not state.mask.any()
        assert jnp.all(state.X == 0.0)
        assert jnp.all(state.y == 0.0)

    def test_2d_space_n_params(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, _ = tpe.TPESearch.init(space, n_max=10)
        assert state.X.shape == (10, 2)

    def test_returns_optimizer_instance(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        _, optimizer = tpe.TPESearch.init(space)
        assert isinstance(optimizer, tpe.TPESearch)

    def test_default_field_values(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        _, opt = tpe.TPESearch.init(space)
        assert opt.gamma == pytest.approx(0.15)
        assert opt.n_candidates == 24
        assert opt.n_warmup == 10
        assert opt.min_bandwidth_factor == pytest.approx(0.03)
        assert opt.prior_weight == pytest.approx(1.0)
        assert opt.maximize is True
        assert opt.n_parallel == 1
        assert opt.lambda_aa == pytest.approx(0.25)

    def test_custom_gamma_and_n_candidates(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        _, opt = tpe.TPESearch.init(space, gamma=0.3, n_candidates=64)
        assert opt.gamma == pytest.approx(0.3)
        assert opt.n_candidates == 64

    def test_default_n_max(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, _ = tpe.TPESearch.init(space)
        assert state.X.shape[0] == 200


class TestTPESearchUpdateState:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        self.state, self.optimizer = tpe.TPESearch.init(self.space, n_max=10)
        self.key = jax.random.PRNGKey(0)

    def test_mask_updated(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(
            self.state, self.key, jnp.array([0.5]), x_new
        )
        assert new_state.mask[0]
        assert not new_state.mask[1:].any()

    def test_X_updated(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(
            self.state, self.key, jnp.array([0.5]), x_new
        )
        assert jnp.allclose(new_state.X[0], x_new[0])

    def test_y_updated(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(
            self.state, self.key, jnp.array([0.75]), x_new
        )
        assert jnp.allclose(new_state.y[0], 0.75)

    def test_y_stores_raw_value_when_minimizing(self):
        _, optimizer = tpe.TPESearch.init(self.space, n_max=10, maximize=False)
        x_new = jnp.array([[0.5]])
        new_state = optimizer.update_state(
            self.state, self.key, jnp.array([0.75]), x_new
        )
        assert jnp.allclose(new_state.y[0], 0.75)

    def test_fixed_size_maintained(self):
        x_new = jnp.array([[0.5]])
        new_state = self.optimizer.update_state(
            self.state, self.key, jnp.array([0.5]), x_new
        )
        assert new_state.X.shape == self.state.X.shape
        assert new_state.y.shape == self.state.y.shape

    def test_sequential_updates(self):
        state = self.state
        for x_val, y_val in [(0.0, 0.1), (0.5, 0.9), (1.0, 0.4)]:
            x_new = jnp.array([[x_val]])
            state = self.optimizer.update_state(
                state, self.key, jnp.array([y_val]), x_new
            )
        assert int(state.mask.sum()) == 3
        assert jnp.allclose(state.y[:3], jnp.array([0.1, 0.9, 0.4]))

    def test_overflow_truncates_to_remaining_slots(self):
        state, optimizer = tpe.TPESearch.init(self.space, n_max=3)
        state = optimizer.update_state(
            state, self.key, jnp.array([0.1]), jnp.array([[0.0]])
        )
        state = optimizer.update_state(
            state, self.key, jnp.array([0.2]), jnp.array([[0.5]])
        )
        x_batch = jnp.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        r_batch = jnp.array([0.3, 0.4, 0.5, 0.6, 0.7])
        state = optimizer.update_state(state, self.key, r_batch, x_batch)
        assert int(state.mask.sum()) == 3
        assert jnp.allclose(state.y[2], 0.3)

    def test_overflow_at_zero_remaining_is_noop(self):
        state, optimizer = tpe.TPESearch.init(self.space, n_max=2)
        state = optimizer.update_state(
            state, self.key, jnp.array([0.5]), jnp.array([[0.5]])
        )
        state = optimizer.update_state(
            state, self.key, jnp.array([0.9]), jnp.array([[1.0]])
        )
        assert int(state.mask.sum()) == 2
        state_after = optimizer.update_state(
            state, self.key, jnp.array([0.1]), jnp.array([[0.0]])
        )
        assert int(state_after.mask.sum()) == 2
        assert jnp.array_equal(state_after.y, state.y)


class TestNIterations:
    def setup_method(self):
        self.space = {"x": sp.LinearSpace(0.0, 1.0)}
        self.func = lambda key, config: -(config["x"] ** 2)

    def _run(self, n_max, n_parallel, n_warmup=0):
        state, opt = tpe.TPESearch.init(
            self.space, n_max=n_max, n_parallel=n_parallel, n_warmup=n_warmup
        )
        state, (params_hist, results_hist) = opt.optimize(
            state, jax.random.PRNGKey(0), self.func
        )
        return state, params_hist, results_hist

    def test_exact_fit_iterations(self):
        state, params_hist, _ = self._run(n_max=10, n_parallel=2)
        assert len(params_hist) == 5
        assert int(state.mask.sum()) == 10

    def test_exact_fit_parallel_1(self):
        state, params_hist, _ = self._run(n_max=7, n_parallel=1)
        assert len(params_hist) == 7
        assert int(state.mask.sum()) == 7

    def test_overflow_one_extra_iteration(self):
        state, params_hist, _ = self._run(n_max=9, n_parallel=4)
        assert len(params_hist) == 3
        assert int(state.mask.sum()) == 9

    def test_overflow_n_max_less_than_n_parallel(self):
        state, params_hist, _ = self._run(n_max=3, n_parallel=10)
        assert len(params_hist) == 1
        assert int(state.mask.sum()) == 3

    def test_full_buffer_runs_zero_iterations(self):
        state, opt = tpe.TPESearch.init(self.space, n_max=2, n_parallel=1)
        state = opt.update_state(
            state, jax.random.PRNGKey(0), jnp.array([0.5]), jnp.array([[0.5]])
        )
        state = opt.update_state(
            state, jax.random.PRNGKey(0), jnp.array([0.9]), jnp.array([[0.9]])
        )
        state2, (params_hist, _) = opt.optimize(state, jax.random.PRNGKey(0), self.func)
        assert len(params_hist) == 0
        assert int(state2.mask.sum()) == 2


class TestTPESearchGetNextParams:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        self.state, self.optimizer = tpe.TPESearch.init(
            self.space, n_max=20, n_warmup=2
        )
        self.key = jax.random.PRNGKey(0)

    def test_random_pick_when_no_observations(self):
        params = self.optimizer.get_next_params(self.state, self.key)
        assert "x" in params
        valid = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(params["x"][0]) in valid

    def test_returns_valid_candidate_value(self):
        params = self.optimizer.get_next_params(self.state, self.key)
        valid = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(params["x"][0]) in valid

    def test_tpe_branch_after_n_warmup(self):
        state = self.state
        for x_val, y_val in [(0.0, 0.1), (1.0, 0.9), (0.5, 0.5)]:
            state = self.optimizer.update_state(
                state, self.key, jnp.array([y_val]), jnp.array([[x_val]])
            )
        params = self.optimizer.get_next_params(state, self.key)
        valid = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert float(params["x"][0]) in valid

    def test_2d_space_shapes(self):
        space = {"x": sp.DiscreteSpace([0, 1]), "y": sp.DiscreteSpace([0, 1, 2])}
        state, opt = tpe.TPESearch.init(space, n_max=10, n_parallel=4)
        params = opt.get_next_params(state, self.key)
        assert "x" in params and "y" in params
        assert params["x"].shape == (4,)
        assert params["y"].shape == (4,)

    def test_n_parallel_continuous(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=20, n_parallel=3, n_warmup=2)
        params = opt.get_next_params(state, self.key)
        assert params["x"].shape == (3,)
        assert jnp.all((params["x"] >= 0.0) & (params["x"] <= 1.0))


class TestTPEInternals:
    def setup_method(self):
        self.space = {"x": sp.LinearSpace(0.0, 1.0)}
        self.state, self.optimizer = tpe.TPESearch.init(
            self.space, n_max=10, n_warmup=2
        )
        key = jax.random.PRNGKey(0)
        for x_val, y_val in [(0.1, 0.2), (0.5, 0.8), (0.9, 0.3), (0.7, 0.6)]:
            self.state = self.optimizer.update_state(
                self.state, key, jnp.array([y_val]), jnp.array([[x_val]])
            )

    def test_split_puts_top_gamma_in_good(self):
        eff_y = self.optimizer._effective_y(self.state)
        w_l, w_g, y_split = self.optimizer._split_and_weight(eff_y, self.state.mask)
        # All observed eff_y values where w_l > 0 must satisfy eff_y >= y_split
        observed_good = (w_l > 0) & self.state.mask
        observed_bad = (w_g > 0) & self.state.mask
        assert jnp.all(jnp.where(observed_good, eff_y >= y_split, True))
        assert jnp.all(jnp.where(observed_bad, eff_y < y_split, True))
        # Padded rows have zero weight everywhere
        assert jnp.all(w_l[~self.state.mask] == 0.0)
        assert jnp.all(w_g[~self.state.mask] == 0.0)

    def test_bandwidth_respects_magic_clip_floor(self):
        leaf_kinds = self.optimizer._leaf_kinds(self.state.space)
        lowers, uppers = self.optimizer._space_bounds(self.state.space)
        eff_y = self.optimizer._effective_y(self.state)
        w_l, _, _ = self.optimizer._split_and_weight(eff_y, self.state.mask)
        bw = self.optimizer._per_dim_bandwidth(
            self.state.X, w_l, leaf_kinds, lowers, uppers
        )
        floor = self.optimizer.min_bandwidth_factor * (uppers - lowers)
        assert jnp.all(bw >= floor - 1e-6)

    def test_sample_from_kde_within_bounds(self):
        leaf_kinds = self.optimizer._leaf_kinds(self.state.space)
        lowers, uppers = self.optimizer._space_bounds(self.state.space)
        eff_y = self.optimizer._effective_y(self.state)
        w_l, _, _ = self.optimizer._split_and_weight(eff_y, self.state.mask)
        bw_l = self.optimizer._per_dim_bandwidth(
            self.state.X, w_l, leaf_kinds, lowers, uppers
        )
        cands = self.optimizer._sample_from_kde(
            jax.random.PRNGKey(1),
            self.state.X,
            w_l,
            bw_l,
            leaf_kinds,
            lowers,
            uppers,
            50,
        )
        assert cands.shape == (50, 1)
        assert jnp.all(cands[:, 0] >= lowers[0])
        assert jnp.all(cands[:, 0] <= uppers[0])

    def test_log_kde_is_finite_on_observed(self):
        leaf_kinds = self.optimizer._leaf_kinds(self.state.space)
        lowers, uppers = self.optimizer._space_bounds(self.state.space)
        eff_y = self.optimizer._effective_y(self.state)
        w_l, w_g, _ = self.optimizer._split_and_weight(eff_y, self.state.mask)
        bw_l = self.optimizer._per_dim_bandwidth(
            self.state.X, w_l, leaf_kinds, lowers, uppers
        )
        bw_g = self.optimizer._per_dim_bandwidth(
            self.state.X, w_g, leaf_kinds, lowers, uppers
        )
        # Test on the actually observed points
        X_test = self.state.X[self.state.mask][:, None].reshape(-1, 1)
        log_l = self.optimizer._log_kde(
            X_test, self.state.X, w_l, bw_l, leaf_kinds, lowers, uppers
        )
        log_g = self.optimizer._log_kde(
            X_test, self.state.X, w_g, bw_g, leaf_kinds, lowers, uppers
        )
        assert jnp.all(jnp.isfinite(log_l))
        assert jnp.all(jnp.isfinite(log_g))

    def test_log_kde_higher_for_good_near_observed_max(self):
        """Near the observed max, l(x) should exceed g(x)."""
        leaf_kinds = self.optimizer._leaf_kinds(self.state.space)
        lowers, uppers = self.optimizer._space_bounds(self.state.space)
        eff_y = self.optimizer._effective_y(self.state)
        w_l, w_g, _ = self.optimizer._split_and_weight(eff_y, self.state.mask)
        bw_l = self.optimizer._per_dim_bandwidth(
            self.state.X, w_l, leaf_kinds, lowers, uppers
        )
        bw_g = self.optimizer._per_dim_bandwidth(
            self.state.X, w_g, leaf_kinds, lowers, uppers
        )
        # Observed max effective_y is at x=0.5 (y=0.8)
        x_near_max = jnp.array([[0.5]])
        log_l = self.optimizer._log_kde(
            x_near_max, self.state.X, w_l, bw_l, leaf_kinds, lowers, uppers
        )
        log_g = self.optimizer._log_kde(
            x_near_max, self.state.X, w_g, bw_g, leaf_kinds, lowers, uppers
        )
        assert float(log_l[0]) > float(log_g[0])


class TestTPESearchOptimize:
    def test_optimize_returns_correct_shapes(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, opt = tpe.TPESearch.init(space, n_max=5, n_parallel=1, n_warmup=2)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = opt.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert len(params_hist) == 5
        assert len(results_hist) == 5

    def test_optimize_fills_state(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, opt = tpe.TPESearch.init(space, n_max=5, n_parallel=1, n_warmup=2)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 5

    def test_optimize_finds_optimum_discrete(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, opt = tpe.TPESearch.init(space, n_max=25, n_parallel=1, n_warmup=5)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert float(opt.best_result(state)) == pytest.approx(0.0)

    def test_optimize_continuous_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=5, n_parallel=1, n_warmup=2)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, _) = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert len(params_hist) == 5
        assert int(state.mask.sum()) == 5

    def test_optimize_converges_continuous(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=40, n_parallel=1, n_warmup=10)
        # Maximum at x = 0.7
        func = lambda key, config: -((config["x"] - 0.7) ** 2)
        state, _ = opt.optimize(state, jax.random.PRNGKey(0), func)
        best_params = opt.best_params(state)
        assert float(best_params["x"]) == pytest.approx(0.7, abs=0.15)

    def test_optimize_minimize(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, opt = tpe.TPESearch.init(
            space, n_max=20, n_parallel=1, n_warmup=5, maximize=False
        )
        func = lambda key, config: config["x"] ** 2
        state, _ = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert float(opt.best_result(state)) == pytest.approx(0.0)

    def test_optimize_n_parallel_fills_buffer(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        state, opt = tpe.TPESearch.init(space, n_max=10, n_parallel=2, n_warmup=4)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = opt.optimize(
            state, jax.random.PRNGKey(0), func
        )
        assert len(params_hist) == 5
        assert int(state.mask.sum()) == 10
        assert results_hist[0].shape == (2,)


class TestTPEMixedSpace:
    def test_get_next_params_mixed(self):
        space = {
            "lr": sp.LogSpace(1e-4, 1e-1),
            "layers": sp.DiscreteSpace([1, 2, 3, 4]),
        }
        state, opt = tpe.TPESearch.init(space, n_max=20, n_parallel=1, n_warmup=2)
        params = opt.get_next_params(state, jax.random.PRNGKey(0))
        assert "lr" in params and "layers" in params
        assert params["lr"].shape == (1,)
        assert float(params["layers"][0]) in [1, 2, 3, 4]

    def test_optimize_mixed_space(self):
        space = {"x": sp.LinearSpace(0.0, 1.0), "n": sp.DiscreteSpace([1, 2, 4, 8])}
        state, opt = tpe.TPESearch.init(space, n_max=8, n_warmup=3)
        func = lambda key, config: -(config["x"] ** 2) + config["n"] * 0.1
        state, (params_hist, _) = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 8
        for p in params_hist:
            assert float(p["n"][0]) in [1, 2, 4, 8]

    def test_optimize_quantized_log(self):
        space = {"units": sp.QLogSpace(8, 512)}
        state, opt = tpe.TPESearch.init(space, n_max=8, n_warmup=3)
        func = lambda key, config: jnp.float32(config["units"])
        state, (params_hist, _) = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 8
        for p in params_hist:
            # QLogSpace returns int32 values
            assert int(p["units"][0]) >= 8
            assert int(p["units"][0]) <= 512


class TestBestParamsResult:
    def setup_method(self):
        self.space = {"x": sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])}
        self.key = jax.random.PRNGKey(0)

    def _state_with_obs(self, optimizer, observations):
        state, _ = tpe.TPESearch.init(self.space, n_max=10, maximize=optimizer.maximize)
        for x_val, y_val in observations:
            state = optimizer.update_state(
                state, self.key, jnp.array([y_val]), jnp.array([[x_val]])
            )
        return state

    def test_best_result_maximize(self):
        _, opt = tpe.TPESearch.init(self.space, n_max=10)
        state = self._state_with_obs(opt, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.3)])
        assert float(opt.best_result(state)) == pytest.approx(0.9)

    def test_best_result_minimize(self):
        _, opt = tpe.TPESearch.init(self.space, n_max=10, maximize=False)
        state = self._state_with_obs(opt, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.2)])
        assert float(opt.best_result(state)) == pytest.approx(0.2)

    def test_best_params_maximize(self):
        _, opt = tpe.TPESearch.init(self.space, n_max=10)
        state = self._state_with_obs(opt, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.3)])
        params = opt.best_params(state)
        assert float(params["x"]) == pytest.approx(0.75)

    def test_best_params_minimize(self):
        _, opt = tpe.TPESearch.init(self.space, n_max=10, maximize=False)
        state = self._state_with_obs(opt, [(0.25, 0.5), (0.75, 0.9), (0.5, 0.2)])
        params = opt.best_params(state)
        assert float(params["x"]) == pytest.approx(0.5)


class TestNWarmup:
    def test_warmup_path_runs_with_no_obs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=5, n_warmup=3)
        params = opt.get_next_params(state, jax.random.PRNGKey(0))
        assert "x" in params
        assert float(params["x"][0]) >= 0.0
        assert float(params["x"][0]) <= 1.0

    def test_warmup_completes_full_run(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=8, n_warmup=4)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 8


class TestPrior:
    def test_zero_prior_weight_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=10, n_warmup=2, prior_weight=0.0)
        func = lambda key, config: -(config["x"] ** 2)
        state, _ = opt.optimize(state, jax.random.PRNGKey(0), func)
        assert int(state.mask.sum()) == 10


class TestMaximizeMinimize:
    def test_maximize_default_is_true(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        _, opt = tpe.TPESearch.init(space)
        assert opt.maximize is True

    def test_minimize_stores_raw_y(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, opt = tpe.TPESearch.init(space, n_max=10, maximize=False)
        x_new = jnp.array([[0.5]])
        state = opt.update_state(state, jax.random.PRNGKey(0), jnp.array([3.14]), x_new)
        assert float(state.y[0]) == pytest.approx(3.14)

    def test_effective_y_negated_for_minimize(self):
        space = {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])}
        state, opt = tpe.TPESearch.init(space, n_max=10, maximize=False)
        key = jax.random.PRNGKey(0)
        state = opt.update_state(state, key, jnp.array([3.0]), jnp.array([[0.5]]))
        eff = opt._effective_y(state)
        assert float(eff[0]) == pytest.approx(-3.0)


class TestTPESearchOptimizeScan:
    def test_optimize_scan_runs(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=5, n_parallel=1, n_warmup=2)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = opt.optimize_scan(
            state, jax.random.PRNGKey(0), func
        )
        assert params_hist["x"].shape[0] == 5
        assert results_hist.shape[0] == 5
        assert int(state.mask.sum()) == 5

    def test_optimize_scan_overflow(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=7, n_parallel=3, n_warmup=3)
        func = lambda key, config: -(config["x"] ** 2)
        state, (params_hist, results_hist) = opt.optimize_scan(
            state, jax.random.PRNGKey(0), func
        )
        assert params_hist["x"].shape[0] == 3
        assert int(state.mask.sum()) == 7


class TestValidateFunc:
    def test_single_arg_raises(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=5)
        with pytest.raises(TypeError, match="fn\\(key, config\\)"):
            opt.optimize(state, jax.random.PRNGKey(0), lambda x: x)

    def test_two_arg_passes(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        state, opt = tpe.TPESearch.init(space, n_max=5)
        opt.optimize(state, jax.random.PRNGKey(0), lambda key, config: config["x"])
