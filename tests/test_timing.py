import time

import jax
import jax.numpy as jnp

from hyperoptax import bayesian
from hyperoptax import spaces as sp


class TestBayesianTiming:
    SPACE = {"x": sp.LinearSpace(0.0, 1.0), "y": sp.LinearSpace(0.0, 1.0)}

    def _make_state_with_obs(self, optimizer, n_obs=5):
        key = jax.random.PRNGKey(0)
        state, _ = bayesian.BayesianSearch.init(
            self.SPACE,
            n_max=50,
            n_candidates=500,
            n_restarts=3,
            n_lbfgs_steps=20,
            n_hparam_steps=0,
        )
        for i in range(n_obs):
            x = jnp.array([[i * 0.2, i * 0.15]])
            state = optimizer.update_state(state, key, jnp.array([float(i)]), x)
        return state

    def test_get_next_params_throughput(self, capsys):
        _, optimizer = bayesian.BayesianSearch.init(
            self.SPACE,
            n_max=50,
            n_candidates=500,
            n_restarts=3,
            n_lbfgs_steps=20,
            n_hparam_steps=0,
        )
        state = self._make_state_with_obs(optimizer)
        key = jax.random.PRNGKey(1)
        # warm-up (triggers compilation)
        params = optimizer.get_next_params(state, key)
        jax.block_until_ready(jax.tree.leaves(params)[0])

        n_calls = 10
        t0 = time.perf_counter()
        for _ in range(n_calls):
            key, k = jax.random.split(key)
            params = optimizer.get_next_params(state, k)
        jax.block_until_ready(jax.tree.leaves(params)[0])
        elapsed = time.perf_counter() - t0
        ms = elapsed / n_calls * 1000
        print(f"\nget_next_params: {ms:.1f} ms/call ({n_calls} calls)")
        assert ms < 5000  # sanity bound

    def test_optimize_throughput(self, capsys):
        n_max = 20
        _, optimizer = bayesian.BayesianSearch.init(
            self.SPACE,
            n_max=n_max,
            n_candidates=500,
            n_restarts=3,
            n_lbfgs_steps=20,
            n_hparam_steps=0,
        )
        state, _ = bayesian.BayesianSearch.init(self.SPACE, n_max=n_max)
        func = lambda key, cfg: -(cfg["x"] ** 2 + cfg["y"] ** 2)
        t0 = time.perf_counter()
        state, _ = optimizer.optimize(state, jax.random.PRNGKey(0), func)
        elapsed = time.perf_counter() - t0
        ms_per_iter = elapsed / n_max * 1000
        print(f"\noptimize: {ms_per_iter:.1f} ms/iter over {n_max} iters")
        assert ms_per_iter < 10000  # sanity bound
