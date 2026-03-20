import dataclasses
import functools

import jax
import jax.numpy as jnp
import optax
from flax import struct

from hyperoptax import acquisition as acq
from hyperoptax import base, kernels
from hyperoptax import spaces as sp

MASK_VARIANCE = 1e12  # large diagonal added to masked rows to isolate them from GP fit


@struct.dataclass
class BayesianSearchState(base.OptimizerState):
    X: jax.Array  # (n_max, n_params) padded with zeros
    y: jax.Array  # (n_max,) padded with zeros — raw (un-negated) results
    mask: jax.Array  # (n_max,) bool, True for valid entries
    log_length_scale: jax.Array  # (n_params,) per-dimension ARD length scales


@dataclasses.dataclass
class BayesianSearch(base.Optimizer):
    jitter: float = 1e-6
    kernel: kernels.BaseKernel = dataclasses.field(
        default_factory=lambda: kernels.Matern(length_scale=1.0, nu=2.5)
    )
    acquisition: acq.BaseAcquisition = dataclasses.field(
        default_factory=lambda: acq.EI(xi=0.01)
    )
    n_candidates: int = 1000  # random candidates sampled for continuous spaces
    n_restarts: int = 2  # number of L-BFGS restarts (seeded from top candidates)
    n_lbfgs_steps: int = 10  # gradient steps per restart
    n_hparam_steps: int = 20  # Adam steps to tune log_length_scale each iteration
    n_warmup: int = 1  # pure-random evaluations before GP kicks in
    maximize: bool = True  # set False to minimize the objective
    n_parallel: int = 1
    hallucination: acq.BaseHallucination = dataclasses.field(
        default_factory=acq.MeanHallucination
    )

    @classmethod
    def init(cls, space, n_max=200, **kwargs):
        # Create the optimizer first so we can read kernel.length_scale for init.
        optimizer = cls(**kwargs)
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        state = BayesianSearchState(
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

    def best_result(self, state: BayesianSearchState) -> jax.Array:
        """Return the best observed raw result (max if maximize, min if minimize)."""
        if self.maximize:
            return jnp.max(state.y, where=state.mask, initial=-jnp.inf)
        else:
            return jnp.min(state.y, where=state.mask, initial=jnp.inf)

    def best_params(self, state: BayesianSearchState):
        """Return the parameter pytree that achieved the best observed result."""
        if self.maximize:
            best_n = int(jnp.argmax(jnp.where(state.mask, state.y, -jnp.inf)))
        else:
            best_n = int(jnp.argmin(jnp.where(state.mask, state.y, jnp.inf)))
        x_best = state.X[best_n]
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        # Return scalar leaves (shape ()) — one value per parameter.
        return treedef.unflatten([x_best[i] for i in range(treedef.num_leaves)])

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

    def _effective_y(self, state: BayesianSearchState) -> jax.Array:
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
        n_steps = self.n_hparam_steps

        @jax.jit
        def tune(X, y, mask, log_length_scale):
            def neg_log_ml(log_ls):
                ls = jnp.exp(log_ls)
                L, alpha, ymean = self._gp_fit(X, y, mask, ls)
                y_c = (y - ymean) * mask
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

    def _tune_hparams(self, state: BayesianSearchState) -> jax.Array:
        return self._tune_hparams_fn(
            state.X, self._effective_y(state), state.mask, state.log_length_scale
        )

    # ------------------------------------------------------------------
    # Parameter selection
    # ------------------------------------------------------------------

    def _random_select(self, state, key, X_cands):
        """Randomly pick n_parallel candidates (used during warmup)."""
        idxs = jax.random.choice(
            key, self.n_candidates, (self.n_parallel,), replace=False
        )
        return X_cands[idxs]  # (n_parallel, n_params)

    def _gp_select(self, state, key, X_cands, lowers, uppers, length_scale):
        """Kriging Believer: sequential L-BFGS with GP hallucination."""
        eff_y = self._effective_y(state)
        n_params = state.X.shape[1]
        n_max = state.X.shape[0]

        X_ext = jnp.concatenate(
            [state.X, jnp.zeros((self.n_parallel, n_params))], axis=0
        )
        y_ext = jnp.concatenate([eff_y, jnp.zeros(self.n_parallel)], axis=0)
        mask_ext = jnp.concatenate(
            [state.mask, jnp.zeros(self.n_parallel, dtype=bool)], axis=0
        )

        xs_raw_list = []
        for i in range(self.n_parallel):
            key, key_liar = jax.random.split(key)
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
                    lbfgs_step,
                    (x0, solver.init(x0)),
                    None,
                    length=self.n_lbfgs_steps,
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
                self.hallucination(mean_i, std_i, key_liar, y_max)
            )
            mask_ext = mask_ext.at[n_max + i].set(True)
            xs_raw_list.append(best_x)

        return jnp.stack(xs_raw_list)  # (n_parallel, n_params)

    def _select_next_x(self, state, key):
        key_sample, key_rest = jax.random.split(key)
        leaves = jax.tree.leaves(state.space, is_leaf=lambda x: isinstance(x, sp.Space))
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        lowers, uppers = self._space_bounds(state.space)
        length_scale = jnp.exp(state.log_length_scale)

        X_cands = self._sample_candidates(
            state.space, key_sample, self.n_candidates
        ).astype(jnp.float32)

        # Use lax.cond so this is JAX-traceable (required for lax.scan / vmap)
        xs_raw = jax.lax.cond(
            state.mask.sum() < self.n_warmup,
            lambda k: self._random_select(state, k, X_cands),
            lambda k: self._gp_select(state, k, X_cands, lowers, uppers, length_scale),
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
        return batch_params

    def get_next_params(self, state, key, params=None, results=None):
        return self._select_next_x(state, key)

    def _write_observation_batch(self, state, x_new, results, n):
        """Write n_parallel observations to the padded state buffers starting at slot n.

        Uses fori_loop to avoid unrolling into n_parallel separate cond nodes in the
        XLA graph (prevents linear compile-time growth with n_parallel).
        """
        n_max = state.X.shape[0]
        n_params = state.X.shape[1]
        n_parallel = results.shape[0]

        def body(i, s):
            slot = n + i
            x_row = jax.lax.dynamic_slice(x_new, (i, 0), (1, n_params))
            y_scalar = jax.lax.dynamic_slice(results, (i,), (1,))
            return jax.lax.cond(
                slot < n_max,
                lambda s: s.replace(
                    X=jax.lax.dynamic_update_slice(s.X, x_row, (slot, 0)),
                    y=jax.lax.dynamic_update_slice(s.y, y_scalar, (slot,)),
                    mask=jax.lax.dynamic_update_slice(
                        s.mask, jnp.ones(1, dtype=bool), (slot,)
                    ),
                ),
                lambda s: s,
                s,
            )

        return jax.lax.fori_loop(0, n_parallel, body, state)

    def update_state(self, state, key, results, params):
        """
        Args:
            results: (n_parallel,) array of observed results
            params: either the batched params pytree from get_next_params
                    (each leaf shape (n_parallel,)) or a raw (n_parallel, n_params)
                    flat array.
        """
        results = jnp.atleast_1d(jnp.squeeze(results))
        n_parallel = results.shape[0]  # static Python int
        if isinstance(params, jax.Array):
            x_new = jnp.atleast_2d(params)
        else:
            x_new = jnp.stack(jax.tree.leaves(params), axis=-1)
        n = state.mask.sum()  # dynamic JAX scalar

        state = self._write_observation_batch(state, x_new, results, n)

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
        """Number of optimize iterations derived from buffer capacity and n_parallel.

        Note: int(state.mask.sum()) forces a device sync — acceptable here since
        optimize() is a Python loop.
        """
        remaining = state.X.shape[0] - int(state.mask.sum())
        n_full = remaining // self.n_parallel
        has_overflow = (remaining % self.n_parallel) > 0
        return n_full + (1 if has_overflow else 0)

    def optimize(self, state, key, func, n_iterations=None):
        if n_iterations is None:
            n_iterations = self._n_iterations(state)
        return super().optimize(state, key, func, n_iterations)

    def optimize_scan(self, state, key, func, n_iterations=None):
        if n_iterations is None:
            n_iterations = self._n_iterations(state)
        return super().optimize_scan(state, key, func, n_iterations)
