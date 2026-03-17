import jax
import jax.numpy as jnp
import pytest

from hyperoptax.acquisition import (
    EI, PI, UCB, BaseAcquisition,
    BaseHallucination, MeanHallucination, SampleHallucination, UCBHallucination, ConstantHallucination,
)


class TestGetArgmax:
    def test_ucb_get_argmax_selects_correct_unseen_index(self):
        ucb = UCB(kappa=2.0)
        # UCB = mean + 2*std → [1.2(seen), 0.2, 0.4]; best unseen is index 2
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        seen_mask = jnp.array([True, False, False])
        idx = ucb.get_argmax(mean, std, seen_mask, n_points=1)
        assert int(idx[0]) == 2

    def test_ucb_values_match_formula(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.5])
        std = jnp.array([0.2, 0.3])
        vals = ucb(mean, std)
        assert jnp.allclose(vals, mean + 2.0 * std)

    def test_ei_values_match_formula(self):
        from jax.scipy.stats import norm
        ei = EI(xi=0.0)
        mean = jnp.array([1.5, 0.5])
        std = jnp.array([0.5, 0.5])
        y_max = jnp.array(1.0)
        vals = ei(mean, std, y_max=y_max)
        a = mean - y_max
        z = a / std
        expected = a * norm.cdf(z) + std * norm.pdf(z)
        assert jnp.allclose(vals, expected)


class TestUCB:
    def test_get_argmax_when_none_seen(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_mask = jnp.array([False, False])

        max_val = X[ucb.get_argmax(mean, std, seen_mask)]
        assert jnp.allclose(max_val, jnp.array([[2.0, 2.0]]))

    def test_get_argmax_excludes_seen(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])

        max_val = X[ucb.get_argmax(mean, std, seen_mask)]
        assert jnp.allclose(max_val, jnp.array([[0.0, 0.0]]))

    def test_get_argmax_when_jitted(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])

        argmax = jax.jit(ucb.get_argmax)(mean, std, seen_mask)
        assert jnp.allclose(X[argmax], jnp.array([[0.0, 0.0]]))



class TestEI:
    def test_get_argmax_when_none_seen(self):
        ei = EI(xi=0.01)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_mask = jnp.array([False, False])

        max_val = X[ei.get_argmax(mean, std, seen_mask)]
        assert jnp.allclose(max_val, jnp.array([[2.0, 2.0]]))

    def test_get_argmax_when_jitted(self):
        ei = EI(xi=0.01)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])

        argmax = jax.jit(ei.get_argmax)(mean, std, seen_mask)
        assert jnp.allclose(X[argmax], jnp.array([[0.0, 0.0]]))


class TestPI:
    def test_pi_output_is_probability(self):
        pi = PI(xi=0.01)
        mean = jnp.array([1.0, 0.5, 0.0])
        std = jnp.array([0.5, 0.5, 0.5])
        vals = pi(mean, std)
        assert jnp.all(vals >= 0.0) and jnp.all(vals <= 1.0)

    def test_pi_higher_for_larger_improvement(self):
        pi = PI(xi=0.01)
        # y_max = max(mean) = 2.0; higher mean -> higher PI
        mean = jnp.array([2.0, 1.0, 0.0])
        std = jnp.array([0.5, 0.5, 0.5])
        vals = pi(mean, std)
        assert vals[0] > vals[1] > vals[2]

    def test_pi_with_explicit_y_max(self):
        pi = PI(xi=0.01)
        mean = jnp.array([1.5, 0.5])
        std = jnp.array([0.5, 0.5])
        # With y_max=2.0, both z < 0, so PI < 0.5
        vals = pi(mean, std, y_max=jnp.array(2.0))
        assert vals[0] > vals[1]
        assert float(vals[0]) < 0.5

    def test_pi_excludes_seen(self):
        pi = PI(xi=0.01)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])
        result = X[pi.get_argmax(mean, std, seen_mask)]
        # Index 0 is seen (best PI), so should return index 2 (highest unseen PI)
        assert not jnp.allclose(result, jnp.array([[2.0, 2.0]]))

    def test_pi_jitted(self):
        pi = PI(xi=0.01)
        mean = jnp.array([1.0, 0.5])
        std = jnp.array([0.3, 0.3])
        vals = jax.jit(pi)(mean, std)
        assert vals.shape == (2,)


class TestBaseAcquisition:
    def test_call_raises_not_implemented(self):
        acq = BaseAcquisition()
        with pytest.raises(NotImplementedError):
            acq(jnp.array([1.0]), jnp.array([0.1]))

    def test_ei_uses_y_max_from_mean_when_none(self):
        # EI with y_max=None uses max(mean) as reference
        ei = EI(xi=0.0)
        mean = jnp.array([2.0, 1.0])
        std = jnp.array([0.5, 0.5])
        vals_auto = ei(mean, std, y_max=None)
        vals_explicit = ei(mean, std, y_max=jnp.max(mean))
        assert jnp.allclose(vals_auto, vals_explicit)

    def test_pi_uses_y_max_from_mean_when_none(self):
        pi = PI(xi=0.0)
        mean = jnp.array([2.0, 1.0])
        std = jnp.array([0.5, 0.5])
        vals_auto = pi(mean, std, y_max=None)
        vals_explicit = pi(mean, std, y_max=jnp.max(mean))
        assert jnp.allclose(vals_auto, vals_explicit)


class TestHallucinationStrategies:
    def setup_method(self):
        self.mean = jnp.array([1.5])
        self.std = jnp.array([0.3])
        self.key = jax.random.PRNGKey(0)
        self.y_max = jnp.array(1.0)

    def test_mean_hallucination_returns_mean(self):
        h = MeanHallucination()
        out = h(self.mean, self.std, self.key, self.y_max)
        assert jnp.allclose(out, self.mean[0])

    def test_sample_hallucination_is_stochastic(self):
        h = SampleHallucination()
        key1, key2 = jax.random.split(self.key)
        out1 = h(self.mean, self.std, key1, self.y_max)
        out2 = h(self.mean, self.std, key2, self.y_max)
        assert not jnp.allclose(out1, out2)

    def test_sample_hallucination_mean_is_posterior_mean(self):
        h = SampleHallucination()
        keys = jax.random.split(self.key, 5000)
        samples = jnp.array([h(self.mean, self.std, k, self.y_max) for k in keys])
        assert jnp.abs(jnp.mean(samples) - self.mean[0]) < 0.05

    def test_ucb_hallucination_formula(self):
        kappa = 3.0
        h = UCBHallucination(kappa=kappa)
        out = h(self.mean, self.std, self.key, self.y_max)
        assert jnp.allclose(out, self.mean[0] + kappa * self.std[0])

    def test_constant_hallucination_uses_y_max(self):
        h = ConstantHallucination()
        out = h(self.mean, self.std, self.key, self.y_max)
        assert jnp.allclose(out, self.y_max)

    def test_constant_hallucination_fixed_value(self):
        h = ConstantHallucination(value=42.0)
        out = h(self.mean, self.std, self.key, self.y_max)
        assert jnp.allclose(out, jnp.array(42.0))

    def test_constant_hallucination_fixed_value_ignores_y_max(self):
        h = ConstantHallucination(value=42.0)
        out1 = h(self.mean, self.std, self.key, jnp.array(0.0))
        out2 = h(self.mean, self.std, self.key, jnp.array(999.0))
        assert jnp.allclose(out1, out2)

    def test_all_hallucinations_return_scalar(self):
        strategies = [
            MeanHallucination(), SampleHallucination(), UCBHallucination(),
            ConstantHallucination(), ConstantHallucination(value=1.0),
        ]
        for h in strategies:
            out = h(self.mean, self.std, self.key, self.y_max)
            assert out.ndim == 0, f"{type(h).__name__} did not return scalar"
