import jax
import jax.numpy as jnp

from hyperoptax.acquisition import EI, PI, UCB


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
    def test_get_max_when_none_seen(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_mask = jnp.array([False, False])

        max_val = ucb.get_max(mean, std, X, seen_mask)
        assert jnp.allclose(max_val, jnp.array([2.0, 2.0]))

    def test_get_max_excludes_seen(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])

        max_val = ucb.get_max(mean, std, X, seen_mask)
        assert jnp.allclose(max_val, jnp.array([0.0, 0.0]))

    def test_get_max_when_jitted(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])

        max_val = jax.jit(ucb.get_max)(mean, std, X, seen_mask)
        assert jnp.allclose(max_val, jnp.array([0.0, 0.0]))

    def test_get_stochastic_argmax_when_stochastic_multiplier_is_1(self):
        ucb = UCB(kappa=2.0, stochastic_multiplier=1)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        seen_mask = jnp.array([True, False, False])
        key = jax.random.PRNGKey(0)

        # when stochastic_multiplier is 1, the two methods are equivalent
        argmax_val = ucb.get_argmax(mean, std, seen_mask, 1)
        stochastic_argmax_val = ucb.get_stochastic_argmax(mean, std, seen_mask, 1, key)
        assert jnp.allclose(argmax_val, stochastic_argmax_val)


class TestEI:
    def test_get_max_when_none_seen(self):
        ei = EI(xi=0.01)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_mask = jnp.array([False, False])

        max_val = ei.get_max(mean, std, X, seen_mask)
        assert jnp.allclose(max_val, jnp.array([2.0, 2.0]))

    def test_get_max_when_jitted(self):
        ei = EI(xi=0.01)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])

        max_val = jax.jit(ei.get_max)(mean, std, X, seen_mask)
        assert jnp.allclose(max_val, jnp.array([0.0, 0.0]))


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
        pi = PI(xi=0.01, stochastic_multiplier=1)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_mask = jnp.array([True, False, False])
        result = pi.get_max(mean, std, X, seen_mask)
        # Index 0 is seen (best PI), so should return index 2 (highest unseen PI)
        assert not jnp.allclose(result, jnp.array([2.0, 2.0]))

    def test_pi_jitted(self):
        pi = PI(xi=0.01)
        mean = jnp.array([1.0, 0.5])
        std = jnp.array([0.3, 0.3])
        vals = jax.jit(pi)(mean, std)
        assert vals.shape == (2,)
