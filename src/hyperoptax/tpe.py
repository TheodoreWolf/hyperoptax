import dataclasses

import jax
import jax.numpy as jnp
from jax.scipy import special as jss

from hyperoptax import base, bayesian, utils
from hyperoptax import spaces as sp

SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TPESearchState(base.OptimizerState):
    """Fixed-size padded observation buffers (see :class:`TPESearch`)."""

    X: jax.Array  # (n_max, n_params) padded with zeros
    y: jax.Array  # (n_max,) padded with zeros — raw (un-negated) results
    mask: jax.Array  # (n_max,) bool, True for valid entries


def _classify_leaf(leaf):
    # Order matters: QLogSpace -> QLinearSpace -> LogSpace -> LinearSpace -> Discrete
    # because the quantized / log variants subclass their continuous parents.
    if isinstance(leaf, sp.QLogSpace):
        return "qlog", {"base": float(leaf.base)}
    if isinstance(leaf, sp.QLinearSpace):
        return "qlinear", {}
    if isinstance(leaf, sp.LogSpace):
        return "log", {"base": float(leaf.base)}
    if isinstance(leaf, sp.LinearSpace):
        return "linear", {}
    if isinstance(leaf, sp.DiscreteSpace):
        return "discrete", {"values": jnp.array(leaf.values, dtype=jnp.float32)}
    raise TypeError(f"Unsupported space type: {type(leaf).__name__}")


@dataclasses.dataclass
class TPESearch(bayesian.BayesianSearch):
    """Multivariate Tree-Parzen Estimator (Watanabe 2023, arXiv:2304.11127).

    EI-weighted observations split at quantile ``gamma``; Gaussian KDE on
    continuous dims (in log-space for LogSpace/QLogSpace), Aitchison–Aitken
    kernel on DiscreteSpace dims, uniform-prior mixture, and Constant Liar
    batching for parallel selection.
    """

    gamma: float = 0.15  # top fraction forming the "good" set
    n_candidates: int = 24  # candidates drawn from l(x) per iteration
    n_warmup: int = 10  # pure-random evaluations before TPE kicks in
    min_bandwidth_factor: float = 0.03  # magic-clip floor, fraction of (hi - lo)
    prior_weight: float = 1.0  # uniform-prior mixture weight (0.0 disables)
    eps: float = 1e-12  # numerical floor for weights and logs
    maximize: bool = True  # set False to minimise the objective
    n_parallel: int = 1
    lambda_aa: float = 0.25  # Aitchison–Aitken smoothing for discrete kernels

    @classmethod
    def init(cls, space, n_max=200, **kwargs):
        optimizer = cls(**kwargs)
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        state = TPESearchState(
            space=space,
            X=jnp.zeros((n_max, len(leaves))),
            y=jnp.zeros(n_max),
            mask=jnp.zeros(n_max, dtype=bool),
        )
        return state, optimizer

    def _leaf_kinds(self, space):
        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        return [_classify_leaf(leaf) for leaf in leaves]

    def _split_and_weight(self, eff_y, mask):
        """Split observations at the gamma quantile and assign EI weights.

        Returns (w_l, w_g, y_split): each weight array shape (n_max,) with
        masked rows set to 0, and y_split is the lowest "good" effective_y.
        """
        n_valid = mask.sum()
        n_max = eff_y.shape[0]
        eff_y_safe = jnp.where(mask, eff_y, -jnp.inf)
        sorted_eff = jnp.sort(eff_y_safe)  # ascending; masked rows pinned to -inf
        n_good = jnp.maximum(1, jnp.ceil(self.gamma * n_valid).astype(jnp.int32))
        # (n_good)-th largest value sits at index (n_max - n_good) in ascending sort.
        y_split = sorted_eff[n_max - n_good]
        good = mask & (eff_y >= y_split)
        bad = mask & (eff_y < y_split)
        w_l = jnp.where(good, jax.nn.relu(eff_y - y_split) + self.eps, 0.0)
        w_g = jnp.where(bad, jax.nn.relu(y_split - eff_y) + self.eps, 0.0)
        return w_l, w_g, y_split

    def _transform(self, kind, extras, x, lo, hi):
        """Map a continuous dim into the space the Gaussian KDE lives in
        (identity for linear, log-base for log). Returns (x_t, lo_t, hi_t)."""
        if kind in ("log", "qlog"):
            lb = jnp.log(extras["base"])
            return jnp.log(jnp.maximum(x, lo)) / lb, jnp.log(lo) / lb, jnp.log(hi) / lb
        return x, lo, hi

    def _untransform(self, kind, extras, x_t):
        return jnp.power(extras["base"], x_t) if kind in ("log", "qlog") else x_t

    def _per_dim_bandwidth(self, X, weights, leaf_kinds, lowers, uppers):
        """Weighted Silverman bandwidth per dim with magic-clip floor.

        Returns shape (n_params,). Discrete dims get a placeholder (unused —
        the Aitchison–Aitken kernel needs no bandwidth). EI weights enter via
        the weighted mean/variance, not via a position transform.
        """
        w_sum = jnp.sum(weights) + self.eps
        w = weights / w_sum
        n_eff = jnp.maximum(1.0 / (jnp.sum(w * w) + self.eps), 1.0)
        bws = []
        for d, (kind, extras) in enumerate(leaf_kinds):
            if kind == "discrete":
                bws.append(jnp.array(1.0))  # placeholder, unused for discrete
                continue
            xs_t, lo_t, hi_t = self._transform(
                kind, extras, X[:, d], lowers[d], uppers[d]
            )
            mean = jnp.sum(w * xs_t)
            var = jnp.sum(w * (xs_t - mean) ** 2)
            std = jnp.sqrt(jnp.maximum(var, 0.0))
            bw = 1.06 * std * jnp.power(n_eff, -0.2)
            bw = jnp.maximum(bw, self.min_bandwidth_factor * (hi_t - lo_t))
            bws.append(bw)
        return jnp.stack(bws)

    def _sample_from_kde(
        self, key, X, weights, bandwidths, leaf_kinds, lowers, uppers, n
    ):
        """Draw n samples from the weighted KDE mixed with the uniform prior."""
        n_params = X.shape[1]
        w_sum = jnp.sum(weights)
        p_prior = self.prior_weight / (w_sum + self.prior_weight + self.eps)

        key_prior, key_parent, key_dim = jax.random.split(key, 3)
        use_prior = jax.random.bernoulli(key_prior, p_prior, shape=(n,))
        # Force categorical to ignore masked rows: log(0) -> -inf in log-probs.
        log_w = jnp.where(
            weights > 0, jnp.log(jnp.maximum(weights, self.eps)), -jnp.inf
        )
        parent_idx = jax.random.categorical(key_parent, log_w, shape=(n,))

        dim_keys = jax.random.split(key_dim, n_params)
        cols = []
        for d, (kind, extras) in enumerate(leaf_kinds):
            k_noise, k_prior_sample, k_aux = jax.random.split(dim_keys[d], 3)
            parent_vals = X[parent_idx, d]
            if kind == "discrete":
                values = extras["values"]
                K = int(values.shape[0])
                if K == 1:
                    samples = jnp.full((n,), values[0])
                else:
                    stay = jax.random.bernoulli(
                        k_noise, 1.0 - self.lambda_aa, shape=(n,)
                    )
                    parent_in_vals = jnp.argmin(
                        jnp.abs(values[None, :] - parent_vals[:, None]), axis=1
                    )
                    # offset in [1, K-1] guarantees alt_idx != parent_in_vals (mod K)
                    offset = jax.random.randint(k_aux, (n,), 1, K)
                    alt_vals = values[(parent_in_vals + offset) % K]
                    prior_vals = values[jax.random.randint(k_prior_sample, (n,), 0, K)]
                    samples = jnp.where(stay, parent_vals, alt_vals)
                    samples = jnp.where(use_prior, prior_vals, samples)
            else:
                parent_t, lo_t, hi_t = self._transform(
                    kind, extras, parent_vals, lowers[d], uppers[d]
                )
                samples_t = parent_t + bandwidths[d] * jax.random.normal(k_noise, (n,))
                prior_t = jax.random.uniform(
                    k_prior_sample, (n,), minval=lo_t, maxval=hi_t
                )
                samples_t = jnp.clip(
                    jnp.where(use_prior, prior_t, samples_t), lo_t, hi_t
                )
                samples = self._untransform(kind, extras, samples_t)
            cols.append(samples)
        return jnp.stack(cols, axis=-1)  # (n, n_params)

    def _log_kde(
        self, X_test, X_train, weights, bandwidths, leaf_kinds, lowers, uppers
    ):
        """Log-density of each test point under the weighted KDE + uniform prior.

        Continuous dims contribute Gaussian (truncation ignored — same
        constant cancels in log l - log g), log dims use Gaussian in
        log-space (Jacobian also cancels), and discrete dims use the
        Aitchison–Aitken kernel.
        """
        n_test = X_test.shape[0]
        n_train = X_train.shape[0]
        log_kernel = jnp.zeros((n_test, n_train))
        log_prior_per_dim_sum = jnp.zeros(n_test)
        for d, (kind, extras) in enumerate(leaf_kinds):
            xs = X_train[:, d]
            xt = X_test[:, d]
            if kind == "discrete":
                values = extras["values"]
                K = int(values.shape[0])
                eq = jnp.isclose(xt[:, None], xs[None, :])
                kernel = jnp.where(
                    eq, 1.0 - self.lambda_aa, self.lambda_aa / max(K - 1, 1)
                )
                log_k = jnp.log(jnp.maximum(kernel, self.eps))
                log_p = -jnp.log(float(K)) * jnp.ones(n_test)
            else:
                xs_t, lo_t, hi_t = self._transform(
                    kind, extras, xs, lowers[d], uppers[d]
                )
                xt_t, _, _ = self._transform(kind, extras, xt, lowers[d], uppers[d])
                bw = bandwidths[d]
                diff = xt_t[:, None] - xs_t[None, :]
                log_k = -0.5 * (diff / bw) ** 2 - jnp.log(bw * SQRT_2PI)
                log_p = -jnp.log(hi_t - lo_t) * jnp.ones(n_test)
            log_kernel = log_kernel + log_k
            log_prior_per_dim_sum = log_prior_per_dim_sum + log_p

        log_w = jnp.where(
            weights > 0, jnp.log(jnp.maximum(weights, self.eps)), -jnp.inf
        )
        log_data = log_kernel + log_w[None, :]  # (n_test, n_train)
        log_prior_term = (
            jnp.log(self.prior_weight + self.eps) + log_prior_per_dim_sum
        )  # (n_test,)
        combined = jnp.concatenate([log_data, log_prior_term[:, None]], axis=1)
        log_num = jss.logsumexp(combined, axis=1)
        log_denom = jnp.log(jnp.sum(weights) + self.prior_weight + self.eps)
        return log_num - log_denom

    def _tpe_select(self, state, key, lowers, uppers, leaf_kinds):
        """Constant Liar: sequential TPE acquisition with mean hallucination."""
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

        xs_list = []
        for i in range(self.n_parallel):
            key, key_cands = jax.random.split(key)
            w_l, w_g, _ = self._split_and_weight(y_ext, mask_ext)
            bw_l = self._per_dim_bandwidth(X_ext, w_l, leaf_kinds, lowers, uppers)
            bw_g = self._per_dim_bandwidth(X_ext, w_g, leaf_kinds, lowers, uppers)
            cands = self._sample_from_kde(
                key_cands,
                X_ext,
                w_l,
                bw_l,
                leaf_kinds,
                lowers,
                uppers,
                self.n_candidates,
            )
            log_l = self._log_kde(cands, X_ext, w_l, bw_l, leaf_kinds, lowers, uppers)
            log_g = self._log_kde(cands, X_ext, w_g, bw_g, leaf_kinds, lowers, uppers)
            best_idx = jnp.argmax(log_l - log_g)
            best_x = cands[best_idx]
            # Constant Liar: hallucinate mean(effective_y) at the chosen x
            y_fake = jnp.sum(jnp.where(mask_ext, y_ext, 0.0)) / jnp.maximum(
                mask_ext.sum(), 1
            )
            X_ext = X_ext.at[n_max + i].set(best_x)
            y_ext = y_ext.at[n_max + i].set(y_fake)
            mask_ext = mask_ext.at[n_max + i].set(True)
            xs_list.append(best_x)
        return jnp.stack(xs_list)  # (n_parallel, n_params)

    def get_next_params(self, state, key, params=None, results=None):
        """Select the next batch of ``n_parallel`` candidates.

        During the first ``n_warmup`` iterations, candidates are chosen
        uniformly at random. Afterwards, TPE samples candidates from the
        good-set KDE and selects those maximising ``log l(x) - log g(x)``,
        with Constant Liar hallucination for the parallel slots.
        """
        key_sample, key_rest = jax.random.split(key)
        leaves = jax.tree.leaves(state.space, is_leaf=lambda x: isinstance(x, sp.Space))
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        lowers, uppers = self._space_bounds(state.space)
        leaf_kinds = self._leaf_kinds(state.space)

        X_cands = utils.sample_space_array(
            state.space, key_sample, self.n_candidates
        ).astype(jnp.float32)

        xs_raw = jax.lax.cond(
            state.mask.sum() < self.n_warmup,
            lambda k: self._random_select(state, k, X_cands),
            lambda k: self._tpe_select(state, k, lowers, uppers, leaf_kinds),
            key_rest,
        )

        # Transforms are element-wise, so vectorise the n_parallel axis.
        xs_out = jnp.stack(
            [leaf.transform(xs_raw[:, i]) for i, leaf in enumerate(leaves)],
            axis=-1,
        )  # (n_parallel, n_params)

        batch_params = treedef.unflatten(
            [xs_out[:, i] for i in range(treedef.num_leaves)]
        )
        return batch_params

    def update_state(self, state, key, results, params):
        """Record new observations in the padded buffers."""
        results = jnp.atleast_1d(jnp.squeeze(results)).astype(state.y.dtype)
        if isinstance(params, jax.Array):
            x_new = jnp.atleast_2d(params)
        else:
            x_new = jnp.stack(jax.tree.leaves(params), axis=-1)
        x_new = x_new.astype(state.X.dtype)
        n = state.mask.sum()
        return self._write_observation_batch(state, x_new, results, n)
