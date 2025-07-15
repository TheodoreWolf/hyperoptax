import jax
import jax.numpy as jnp

from hyperoptax.acquisition import EI, UCB


class TestUCB:
    def test_get_max_when_no_seen_idx(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_idx = jnp.array([])

        max_val = ucb.get_max(mean, std, X, seen_idx)
        assert jnp.allclose(max_val, jnp.array([2.0, 2.0]))

    def test_get_max_when_seen_idx(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_idx = jnp.array([0])

        max_val = ucb.get_max(mean, std, X, seen_idx)
        assert jnp.allclose(max_val, jnp.array([0.0, 0.0]))

    def test_get_max_when_jitted(self):
        ucb = UCB(kappa=2.0)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_idx = jnp.array([0])

        max_val = jax.jit(ucb.get_max)(mean, std, X, seen_idx)
        assert jnp.allclose(max_val, jnp.array([0.0, 0.0]))

    def test_get_stochastic_argmax_when_stochastic_multiplier_is_1(self):
        ucb = UCB(kappa=2.0, stochastic_multiplier=1)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        seen_idx = jnp.array([0])
        key = jax.random.PRNGKey(0)

        # when stochastic_multiplier is 1, the two methods are equivalent
        argmax_val = ucb.get_argmax(mean, std, seen_idx, 1)
        stochastic_argmax_val = ucb.get_stochastic_argmax(mean, std, seen_idx, 1, key)
        assert jnp.allclose(argmax_val, stochastic_argmax_val)


class TestEI:
    def test_get_max_when_no_seen_idx(self):
        ei = EI(xi=0.01)
        mean = jnp.array([1.0, 0.0])
        std = jnp.array([0.1, 0.1])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0]])
        seen_idx = jnp.array([])

        max_val = ei.get_max(mean, std, X, seen_idx)
        assert jnp.allclose(max_val, jnp.array([2.0, 2.0]))

    def test_get_max_when_jitted(self):
        ei = EI(xi=0.01)
        mean = jnp.array([1.0, 0.0, 0.0])
        std = jnp.array([0.1, 0.1, 0.2])
        X = jnp.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
        seen_idx = jnp.array([0])

        max_val = jax.jit(ei.get_max)(mean, std, X, seen_idx)
        assert jnp.allclose(max_val, jnp.array([0.0, 0.0]))
