import warnings
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from hyperoptax.base import Optimizer, OptimizerState, _validate_func


class TestValidateFunc:
    def test_valid_two_arg_function_passes(self):
        _validate_func(lambda key, config: config)

    def test_one_arg_raises(self):
        with pytest.raises(TypeError, match="fn\\(key, config\\)"):
            _validate_func(lambda x: x)

    def test_zero_arg_raises(self):
        with pytest.raises(TypeError):
            _validate_func(lambda: 1)

    def test_uninspectable_function_warns(self):
        # Simulate inspect.signature raising a non-TypeError ValueError
        with patch("inspect.signature", side_effect=ValueError("no signature")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _validate_func(lambda key, config: config)
            assert len(w) == 1
            assert "introspect" in str(w[0].message).lower()

    def test_uninspectable_function_returns_none(self):
        with patch("inspect.signature", side_effect=ValueError("no signature")):
            result = _validate_func(lambda key, config: config)
        assert result is None


class TestOptimizerBase:
    def test_init_returns_state_and_optimizer(self):
        space = {"x": jnp.array(0.0)}
        state, optimizer = Optimizer.init(space)
        assert isinstance(state, OptimizerState)
        assert isinstance(optimizer, Optimizer)

    def test_init_state_stores_space(self):
        space = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        state, _ = Optimizer.init(space)
        assert state.space == space

    def test_update_state_raises_not_implemented(self):
        state, optimizer = Optimizer.init({})
        with pytest.raises(NotImplementedError):
            optimizer.update_state(state, jax.random.PRNGKey(0), jnp.array(1.0))

    def test_get_next_params_raises_not_implemented(self):
        state, optimizer = Optimizer.init({})
        with pytest.raises(NotImplementedError):
            optimizer.get_next_params(state, jax.random.PRNGKey(0))
