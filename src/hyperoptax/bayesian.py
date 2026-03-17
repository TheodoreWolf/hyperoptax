import dataclasses
import functools

import jax
import jax.numpy as jnp
import optax
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax.acquisition import EI, BaseAcquisition, BaseLiar, MeanLiar
from hyperoptax.base import Optimizer, OptimizerState, _validate_func
from hyperoptax.kernels import BaseKernel, Matern

MASK_VARIANCE = 1e12


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
        default_factory=lambda: EI(xi=0.01)
    )
    n_candidates: int = 1000  # random candidates sampled for continuous spaces
    n_restarts: int = 2  # number of L-BFGS restarts (seeded from top candidates)
    n_lbfgs_steps: int = 10  # gradient steps per restart
    n_hparam_steps: int = 20  # Adam steps to tune log_length_scale each iteration
    n_initial_random: int = 1  # pure-random evaluations before GP kicks in
    maximize: bool = True  # set False to minimize the objective
    n_parallel: int = 1
    liar: BaseLiar = dataclasses.field(default_factory=MeanLiar)

    @classmethod
    def init(cls, space, n_max=200, **kwargs):
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
    # Parameter selection
    # ------------------------------------------------------------------

    def _get_next_params_continuous(self, state, key):
        key_sample, key_rest = jax.random.split(key)
        leaves = jax.tree.leaves(state.space, is_leaf=lambda x: isinstance(x, sp.Space))
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        lowers, uppers = self._space_bounds(state.space)
        length_scale = jnp.exp(state.log_length_scale)
        n_params = len(leaves)
        n_max = state.X.shape[0]

        X_cands = self._sample_candidates(
            state.space, key_sample, self.n_candidates
        ).astype(jnp.float32)

        def _random_branch(key_rest):
            idxs = jax.random.choice(
                key_rest, self.n_candidates, (self.n_parallel,), replace=False
            )
            return X_cands[idxs]  # (n_parallel, n_params)

        def _gp_branch(key_rest):
            # Kriging Believer: sequential L-BFGS with GP mean hallucination
            eff_y = self._effective_y(state)
            X_ext = jnp.concatenate(
                [state.X, jnp.zeros((self.n_parallel, n_params))], axis=0
            )
            y_ext = jnp.concatenate([eff_y, jnp.zeros(self.n_parallel)], axis=0)
            mask_ext = jnp.concatenate(
                [state.mask, jnp.zeros(self.n_parallel, dtype=bool)], axis=0
            )

            xs_raw_list = []
            for i in range(self.n_parallel):
                key_rest, key_liar = jax.random.split(key_rest)
                L, alpha, ymean = self._gp_fit(X_ext, y_ext, mask_ext, length_scale)
                mean_cands, std_cands = self._gp_predict(
                    X_cands, L, alpha, ymean, X_ext, length_scale
                )
                acq_vals = self.acquisition(mean_cands, std_cands)
                y_max = jnp.max(y_ext, where=mask_ext, initial=-jnp.inf)

                n_seeds = min(self.n_restarts, self.n_candidates)
                seed_idxs = jnp.argsort(acq_vals)[-n_seeds:]
                seeds = X_cands[seed_idxs]  # (n_seeds, n_params)

                # L-BFGS restarts: pick best via jnp.where so this is JAX-traceable
                solver = optax.lbfgs()

                def neg_acq(x):
                    K_star = self.kernel(x[None], X_ext, length_scale=length_scale)
                    mean = K_star @ alpha + ymean
                    v = jax.scipy.linalg.cho_solve((L, True), K_star.T)
                    std = jnp.sqrt(jnp.clip(1.0 - jnp.sum(K_star * v.T, axis=1), 0.0))
                    return -self.acquisition(mean, std, y_max=y_max)[0]

                def lbfgs_step(carry, _):
                    x, s = carry
                    val, grad = jax.value_and_grad(neg_acq)(x)
                    updates, new_s = solver.update(
                        grad, s, x, value=val, grad=grad, value_fn=neg_acq
                    )
                    return (
                        jnp.clip(optax.apply_updates(x, updates), lowers, uppers),
                        new_s,
                    ), None

                def _lbfgs_restart(carry, x0):
                    best_x, best_val = carry
                    (x_refined, _), _ = jax.lax.scan(
                        lbfgs_step, (x0, solver.init(x0)), None,
                        length=self.n_lbfgs_steps
                    )
                    mean_r, std_r = self._gp_predict(
                        x_refined[None], L, alpha, ymean, X_ext, length_scale
                    )
                    val = self.acquisition(mean_r, std_r, y_max=y_max)[0]
                    best_x = jnp.where(val > best_val, x_refined, best_x)
                    best_val = jnp.where(val > best_val, val, best_val)
                    return (best_x, best_val), None

                (best_x, _), _ = jax.lax.scan(
                    _lbfgs_restart,
                    (seeds[-1], acq_vals[seed_idxs[-1]]),
                    seeds,
                )

                # Hallucinate: use liar strategy to generate pseudo-observation
                mean_i, std_i = self._gp_predict(
                    best_x[None], L, alpha, ymean, X_ext, length_scale
                )
                X_ext = X_ext.at[n_max + i].set(best_x)
                y_ext = y_ext.at[n_max + i].set(
                    self.liar(mean_i, std_i, key_liar, y_max)
                )
                mask_ext = mask_ext.at[n_max + i].set(True)
                xs_raw_list.append(best_x)

            return jnp.stack(xs_raw_list)  # (n_parallel, n_params)

        # Use lax.cond so this is JAX-traceable (required for lax.scan / vmap)
        xs_raw = jax.lax.cond(
            state.mask.sum() < self.n_initial_random,
            _random_branch,
            _gp_branch,
            key_rest,
        )

        # Apply per-leaf transforms (rounds QLinearSpace/QLogSpace to integers, etc.)
        xs_out = jnp.stack(
            [
                jnp.stack(
                    [
                        leaf.transform(xs_raw[j, i : i + 1]).squeeze()
                        for i, leaf in enumerate(leaves)
                    ]
                )
                for j in range(self.n_parallel)
            ]
        )  # (n_parallel, n_params)

        batch_params = treedef.unflatten(
            [xs_out[:, i] for i in range(treedef.num_leaves)]
        )
        return batch_params, xs_out

    def get_next_params(self, state, key, params=None, results=None):
        """
        Returns (batch_params, xs) where:
          - batch_params: batched pytree, each leaf has shape (n_parallel, ...)
          - xs: (n_parallel, n_params) flat array of those values (for update_state)
        """
        return self._get_next_params_continuous(state, key)

    def update_state(self, state, key, results, x_new):
        """
        Args:
            results: (n_parallel,) array of observed results
            x_new: (n_parallel, n_params) array of evaluated parameter vectors

        Writes each observation individually with a lax.cond guard so that
        out-of-bounds slots are silently dropped.  This is fully JAX-traceable
        (compatible with lax.scan and vmap) and correctly handles overflow.
        """
        results = jnp.atleast_1d(jnp.squeeze(results))
        n_parallel = results.shape[0]  # static Python int
        n_max = state.X.shape[0]  # static Python int
        x_new = jnp.atleast_2d(x_new)
        n = state.mask.sum()  # dynamic JAX scalar

        # Write each slot individually; skip if out of bounds.
        for i in range(n_parallel):
            slot = n + i  # dynamic JAX scalar
            state = jax.lax.cond(
                slot < n_max,
                lambda s, slot=slot, i=i: s.replace(
                    X=jax.lax.dynamic_update_slice(s.X, x_new[i : i + 1], (slot, 0)),
                    y=jax.lax.dynamic_update_slice(s.y, results[i : i + 1], (slot,)),
                    mask=jax.lax.dynamic_update_slice(
                        s.mask, jnp.ones(1, dtype=bool), (slot,)
                    ),
                ),
                lambda s: s,
                state,
            )

        if self.n_hparam_steps > 0:
            log_ls = jax.lax.cond(
                n + n_parallel >= 2,
                self._tune_hparams,
                lambda s: s.log_length_scale,
                state,
            )
            state = state.replace(log_length_scale=log_ls)
        return state

    def _n_iterations(self, state):
        """Number of optimize iterations derived from buffer capacity and n_parallel."""
        remaining = state.X.shape[0] - int(state.mask.sum())
        n_full = remaining // self.n_parallel
        has_overflow = (remaining % self.n_parallel) > 0
        return n_full + (1 if has_overflow else 0)

    def optimize_scan(self, state, key, func, n_iterations=None):
        """Like the base optimize_scan but uses jax.lax.scan.

        BayesianSearch.get_next_params returns (batch_params, xs_raw) instead
        of just batch_params, so we override to unpack xs_raw and pass it to
        update_state.  Because get_next_params is now fully JAX-traceable this
        method is JIT-able and vmap-able.
        """
        _validate_func(func)
        if n_iterations is None:
            n_iterations = self._n_iterations(state)

        # Step 0 outside scan to infer pytree structure / shapes for the carry.
        key, key_get, key_funcs, key_update = jax.random.split(key, 4)
        params0, x_new0 = self.get_next_params(state, key_get, None, None)
        func_keys0 = jax.random.split(key_funcs, self.n_parallel)
        results0 = jax.vmap(func)(func_keys0, params0)
        state = self.update_state(state, key_update, results0, x_new0)
        first_params, first_results = params0, results0

        def step(carry, _):
            state, key, params, results, x_new = carry
            key, key_get, key_funcs, key_update = jax.random.split(key, 4)
            params, x_new = self.get_next_params(state, key_get, params, results)
            func_keys = jax.random.split(key_funcs, self.n_parallel)
            results = jax.vmap(func)(func_keys, params)
            state = self.update_state(state, key_update, results, x_new)
            return (state, key, params, results, x_new), (params, results)

        (final_state, _, _, _, _), (params_hist, results_hist) = jax.lax.scan(
            step,
            (state, key, params0, results0, x_new0),
            None,
            length=n_iterations - 1,
        )

        params_hist = jax.tree.map(
            lambda first, rest: jnp.concatenate([first[None], rest]),
            first_params,
            params_hist,
        )
        results_hist = jnp.concatenate([first_results[None], results_hist])
        return final_state, (params_hist, results_hist)

    def optimize(self, state, key, func):
        """Run Bayesian optimisation until the observation buffer is full.

        The number of iterations is derived from the buffer capacity:
        ``n_max // n_parallel`` full iterations plus one overflow iteration if
        ``n_max % n_parallel != 0``.  In the overflow iteration all
        ``n_parallel`` candidates are evaluated but only the remaining slots in
        the buffer are stored.
        """
        _validate_func(func)
        n_iterations = self._n_iterations(state)
        params_hist, results_hist = [], []
        params, results = None, None
        for _ in range(n_iterations):
            key, key_get, key_funcs, key_update = jax.random.split(key, 4)
            params, x_new = self.get_next_params(state, key_get, params, results)
            func_keys = jax.random.split(key_funcs, self.n_parallel)
            results = jax.vmap(func)(func_keys, params)  # (n_parallel,)
            state = self.update_state(state, key_update, results, x_new)
            params_hist.append(params)
            results_hist.append(results)

        return state, (params_hist, results_hist)
