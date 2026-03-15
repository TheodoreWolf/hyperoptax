import dataclasses
import functools

import jax
import jax.numpy as jnp
import optax
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax.acquisition import UCB, BaseAcquisition
from hyperoptax.base import Optimizer, OptimizerState, _validate_func
from hyperoptax.kernels import BaseKernel, Matern

MASK_VARIANCE = 1e12

#TODO
# clean up the discrete and continuous spaces
# handle good shapes for args: (1,) -> squeeze()

@struct.dataclass
class BayesianOptimizerState(OptimizerState):
    X: jax.Array  # (n_max, n_params) padded with zeros
    y: jax.Array  # (n_max,) padded with zeros — raw (un-negated) results
    mask: jax.Array  # (n_max,) bool, True for valid entries
    log_length_scale: jax.Array  # (n_params,) per-dimension ARD length scales


@dataclasses.dataclass
class BayesianSearch(Optimizer):
    jitter: float = 1e-6
    kernel: BaseKernel = dataclasses.field(
        default_factory=lambda: Matern(length_scale=1.0, nu=2.5)
    )
    acquisition: BaseAcquisition = dataclasses.field(
        default_factory=lambda: UCB(kappa=2.0)
    )
    n_candidates: int = 1000  # random candidates sampled for continuous spaces
    n_restarts: int = 2  # number of L-BFGS restarts (seeded from top candidates)
    n_lbfgs_steps: int = 10  # gradient steps per restart
    n_hparam_steps: int = 20  # Adam steps to tune log_length_scale each iteration
    n_initial_random: int = 1  # pure-random evaluations before GP kicks in
    maximize: bool = True  # set False to minimize the objective

    @functools.cached_property
    def _lbfgs_fn(self):
        """JIT-compiled single-start L-BFGS runner, built lazily on first use.

        ``kernel``, ``acquisition``, and ``n_lbfgs_steps`` are captured as
        Python constants so the same compiled XLA program is reused across
        iterations. All GP arrays are explicit JAX arguments.
        """
        kernel = self.kernel
        acquisition = self.acquisition
        solver = optax.lbfgs()
        n_steps = self.n_lbfgs_steps

        @jax.jit
        def run(x0, L, alpha, ymean, X_train, y_max, lowers, uppers, length_scale):
            def neg_acq(x):
                K_star = kernel(x[None], X_train, length_scale=length_scale)
                mean = K_star @ alpha + ymean
                v = jax.scipy.linalg.cho_solve((L, True), K_star.T)
                std = jnp.sqrt(jnp.clip(1.0 - jnp.sum(K_star * v.T, axis=1), 0.0))
                return -acquisition(mean, std, y_max=y_max)[0]

            def step(carry, _):
                x, s = carry
                val, grad = jax.value_and_grad(neg_acq)(x)
                updates, new_s = solver.update(
                    grad, s, x, value=val, grad=grad, value_fn=neg_acq
                )
                return (
                    jnp.clip(optax.apply_updates(x, updates), lowers, uppers),
                    new_s,
                ), None

            (x_final, _), _ = jax.lax.scan(
                step, (x0, solver.init(x0)), None, length=n_steps
            )
            return x_final

        return run

    @classmethod
    def init(cls, space, n_max=100, **kwargs):
        # Create the optimizer first so we can read kernel.length_scale for init.
        optimizer = cls(**kwargs)
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        state = BayesianOptimizerState(
            space=space,
            X=jnp.zeros((n_max, len(leaves))),
            y=jnp.zeros(n_max),
            mask=jnp.zeros(n_max, dtype=bool),
            log_length_scale=jnp.log(
                jnp.ones(len(leaves)) * float(optimizer.kernel.length_scale)
            ),
        )
        return state, optimizer

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def best_result(self, state: BayesianOptimizerState) -> jax.Array:
        """Return the best observed raw result (max if maximize, min if minimize)."""
        if self.maximize:
            return jnp.max(state.y, where=state.mask, initial=-jnp.inf)
        else:
            return jnp.min(state.y, where=state.mask, initial=jnp.inf)

    def best_params(self, state: BayesianOptimizerState):
        """Return the parameter pytree that achieved the best observed result."""
        if self.maximize:
            best_n = int(jnp.argmax(jnp.where(state.mask, state.y, -jnp.inf)))
        else:
            best_n = int(jnp.argmin(jnp.where(state.mask, state.y, jnp.inf)))
        x_best = state.X[best_n]
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        return treedef.unflatten([x_best[i : i + 1] for i in range(treedef.num_leaves)])

    # ------------------------------------------------------------------
    # Space helpers
    # ------------------------------------------------------------------

    def _is_discrete(self, space):
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        return all(isinstance(leaf, sp.DiscreteSpace) for leaf in leaves)

    def _candidates(self, space):
        """Build (n_total, n_params) array from the full discrete grid."""
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        values_list = [jnp.array(leaf.values) for leaf in leaves]
        grids = jnp.meshgrid(*values_list, indexing="ij")
        return jnp.stack([g.ravel() for g in grids], axis=-1)

    def _sample_candidates(self, space, key, n):
        """Sample n random candidates from a continuous space."""
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        keys_per_leaf = jax.random.split(key, len(leaves))
        cols = [
            jax.vmap(lambda k: leaf.sample(k).squeeze())(
                jax.random.split(keys_per_leaf[j], n)
            )
            for j, leaf in enumerate(leaves)
        ]
        return jnp.stack(cols, axis=-1)  # (n, n_params)

    def _space_bounds(self, space):
        """Returns (lowers, uppers) arrays of shape (n_params,)."""
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        lowers = jnp.array([leaf.lower_bound for leaf in leaves])
        uppers = jnp.array([leaf.upper_bound for leaf in leaves])
        return lowers, uppers

    # ------------------------------------------------------------------
    # GP helpers
    # ------------------------------------------------------------------

    def _effective_y(self, state: BayesianOptimizerState) -> jax.Array:
        """y in 'higher is better' orientation for GP fitting."""
        return state.y if self.maximize else -state.y

    def _gp_fit(self, X, y, mask, length_scale):
        """Fit the GP: return (L, alpha, ymean) for use in predictions."""
        ymean = jnp.mean(y, where=mask)
        y_centered = (y - ymean) * mask
        K = self.kernel(X, X, length_scale=length_scale)
        M = jnp.outer(mask.astype(float), mask.astype(float))
        K = K * M + self.jitter * jnp.eye(X.shape[0])
        K += jnp.diag((1.0 - mask.astype(float)) * MASK_VARIANCE)
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_centered)
        return L, alpha, ymean

    def _gp_predict(self, X_test, L, alpha, ymean, X_train, length_scale):
        """GP posterior mean and std at X_test given a fitted GP."""
        K_star = self.kernel(X_test, X_train, length_scale=length_scale)  # (m, n)
        mean = K_star @ alpha + ymean
        v = jax.scipy.linalg.cho_solve((L, True), K_star.T)  # (n, m)
        var = jnp.clip(1.0 - jnp.sum(K_star * v.T, axis=1), 0.0)
        return mean, jnp.sqrt(var)

    def _gp_posterior(self, X, y, mask, X_test, length_scale):
        """Convenience: fit + predict in one call."""
        L, alpha, ymean = self._gp_fit(X, y, mask, length_scale)
        return self._gp_predict(X_test, L, alpha, ymean, X, length_scale)

    @functools.cached_property
    def _tune_hparams_fn(self):
        """JIT-compiled hparam tuner, built lazily on first use.

        Accepts all varying data as explicit JAX arguments so the compiled
        XLA program is reused across iterations regardless of how many
        observations have accumulated (no recompilation per new n_seen).
        """
        kernel = self.kernel
        jitter = self.jitter
        n_steps = self.n_hparam_steps

        @jax.jit
        def tune(X, y, mask, log_length_scale):
            def neg_log_ml(log_ls):
                ls = jnp.exp(log_ls)
                ymean = jnp.sum(y * mask) / jnp.sum(mask)
                y_c = (y - ymean) * mask
                K = kernel(X, X, length_scale=ls)
                M = jnp.outer(mask.astype(float), mask.astype(float))
                K = K * M + jitter * jnp.eye(X.shape[0])
                K += jnp.diag((1.0 - mask.astype(float)) * MASK_VARIANCE)
                L = jnp.linalg.cholesky(K)
                alpha = jax.scipy.linalg.cho_solve((L, True), y_c)
                return 0.5 * y_c @ alpha + jnp.sum(jnp.log(jnp.diag(L)))

            adam = optax.adam(0.1)
            opt_state = adam.init(log_length_scale)

            def step(carry, _):
                log_ls, opt_state = carry
                grad = jax.grad(neg_log_ml)(log_ls)
                updates, new_opt_state = adam.update(grad, opt_state)
                return (optax.apply_updates(log_ls, updates), new_opt_state), None

            (log_ls, _), _ = jax.lax.scan(
                step, (log_length_scale, opt_state), None, length=n_steps
            )
            return log_ls

        return tune

    def _tune_hparams(self, state: BayesianOptimizerState) -> jax.Array:
        return self._tune_hparams_fn(
            state.X, self._effective_y(state), state.mask, state.log_length_scale
        )

    # ------------------------------------------------------------------
    # Acquisition optimisation (continuous)
    # ------------------------------------------------------------------

    def _lbfgs_maximize(
        self, L, alpha, ymean, X_train, y_max, x0, lowers, uppers, length_scale
    ):
        """
        Maximise the acquisition function from x0 using the cached JIT runner.
        Box constraints are enforced by clipping after each step.
        Returns (x_best, acq_value).
        """
        x_final = self._lbfgs_fn(
            x0, L, alpha, ymean, X_train, y_max, lowers, uppers, length_scale
        )
        # Evaluate acquisition at the final point to return its value
        mean, std = self._gp_predict(
            x_final[None], L, alpha, ymean, X_train, length_scale
        )
        return x_final, self.acquisition(mean, std, y_max=y_max)[0]

    # ------------------------------------------------------------------
    # Parameter selection
    # ------------------------------------------------------------------

    def _get_next_params_discrete(self, state, key):
        X_cands = self._candidates(state.space)
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        n_seen = int(state.mask.sum())
        length_scale = jnp.exp(state.log_length_scale)

        if n_seen < self.n_initial_random:
            idx = int(jax.random.randint(key, (), 0, X_cands.shape[0]))
        else:
            mean, std = self._gp_posterior(
                state.X, self._effective_y(state), state.mask, X_cands, length_scale
            )
            seen_mask = jnp.any(
                jnp.all(X_cands[:, None, :] == state.X[None, :, :], axis=-1)
                & state.mask[None, :],
                axis=1,
            )
            idx = int(
                self.acquisition.get_stochastic_argmax(
                    mean, std, seen_mask, n_points=1, key=key
                )[0]
            )

        x = X_cands[idx]
        params = treedef.unflatten([x[i] for i in range(treedef.num_leaves)])
        return params, x

    def _get_next_params_continuous(self, state, key):
        key_sample, key_init = jax.random.split(key)
        leaves = jax.tree.leaves(state.space, is_leaf=lambda x: isinstance(x, sp.Space))
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        lowers, uppers = self._space_bounds(state.space)
        n_seen = int(state.mask.sum())
        length_scale = jnp.exp(state.log_length_scale)

        X_cands = self._sample_candidates(state.space, key_sample, self.n_candidates)

        if n_seen < self.n_initial_random:
            x = X_cands[int(jax.random.randint(key_init, (), 0, self.n_candidates))]
        else:
            eff_y = self._effective_y(state)
            L, alpha, ymean = self._gp_fit(state.X, eff_y, state.mask, length_scale)
            mean_cands, std_cands = self._gp_predict(
                X_cands, L, alpha, ymean, state.X, length_scale
            )
            acq_vals = self.acquisition(mean_cands, std_cands)
            y_max = jnp.max(eff_y, where=state.mask, initial=-jnp.inf)

            # Seed L-BFGS from the top-n_restarts candidates
            n_seeds = min(self.n_restarts, self.n_candidates)
            seed_idxs = jnp.argsort(acq_vals)[-n_seeds:]
            seeds = X_cands[seed_idxs]

            # Fall back to the best random candidate if all L-BFGS runs diverge
            best_x = seeds[-1]
            best_val = acq_vals[seed_idxs[-1]]
            for seed in seeds:
                x_refined, val = self._lbfgs_maximize(
                    L, alpha, ymean, state.X, y_max, seed, lowers, uppers, length_scale
                )
                if float(val) > float(best_val):
                    best_x, best_val = x_refined, val
            x = best_x

        # Apply per-leaf transforms (rounds QLinearSpace/QLogSpace to integers, etc.)
        x_out = jnp.stack(
            [leaf.transform(x[i : i + 1]).squeeze() for i, leaf in enumerate(leaves)]
        )
        params = treedef.unflatten(
            [x_out[i : i + 1] for i in range(treedef.num_leaves)]
        )
        return params, x_out

    def get_next_params(self, state, key, params=None, results=None):
        """
        Returns (params_pytree, x_new) where:
          - params_pytree: the next hyperparameters to evaluate
          - x_new: flat (n_params,) array of those values (for storing in state)
        """
        if self._is_discrete(state.space):
            return self._get_next_params_discrete(state, key)
        return self._get_next_params_continuous(state, key)

    def update_state(self, state, key, results, x_new):
        n = int(state.mask.sum())
        n_max = state.X.shape[0]
        if n >= n_max:
            raise ValueError(
                f"State capacity exceeded: {n_max} observations already stored. "
                "Increase n_max when calling BayesianSearch.init()."
            )
        state = state.replace(
            X=state.X.at[n].set(x_new),
            y=state.y.at[n].set(jnp.squeeze(results)),  # store raw (un-negated)
            mask=state.mask.at[n].set(True),
        )
        if int(state.mask.sum()) >= 2 and self.n_hparam_steps > 0:
            log_ls = self._tune_hparams(state)
            state = state.replace(log_length_scale=log_ls)
        return state

    def optimize_scan(self, state, key, func, n_iterations):
        """Like optimize, but returns stacked arrays instead of lists.

        BayesianSearch cannot use jax.lax.scan because get_next_params uses
        Python-level control flow (int(state.mask.sum())). This override runs
        a Python loop and stacks the results into arrays with a leading
        n_iterations dimension, matching the base optimize_scan output format.
        """
        _validate_func(func)
        params_hist, results_hist = [], []
        params, results = None, None
        for _ in range(n_iterations):
            key, key_get, key_func, key_update = jax.random.split(key, 4)
            params, x_new = self.get_next_params(state, key_get, params, results)
            results = func(key_func, params)
            state = self.update_state(state, key_update, results, x_new)
            params_hist.append(params)
            results_hist.append(results)
        params_stacked = jax.tree.map(lambda *arrs: jnp.stack(arrs), *params_hist)
        results_stacked = jnp.stack(results_hist)
        return state, (params_stacked, results_stacked)

    def optimize(self, state, key, func, n_iterations):
        _validate_func(func)
        params_hist, results_hist = [], []
        params, results = None, None
        for _ in range(n_iterations):
            key, key_get, key_func, key_update = jax.random.split(key, 4)
            params, x_new = self.get_next_params(state, key_get, params, results)
            results = func(key_func, params)
            state = self.update_state(state, key_update, results, x_new)
            params_hist.append(params)
            results_hist.append(results)

        return state, (params_hist, results_hist)
