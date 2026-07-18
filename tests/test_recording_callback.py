import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperoptax import BatchCallback, BatchCompleted, BayesianSearch, RandomSearch, base
from hyperoptax import spaces as sp


def _objective(key, params):
    return params["x"] ** 2 + 0.01 * jax.random.normal(key)


def _assert_trees_equal(left, right):
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


class _UpdateError(RuntimeError):
    pass


class _FailingUpdateOptimizer(base.Optimizer):
    def get_next_params(self, state, key, params=None, results=None):
        del state, key, params, results
        return {"x": jnp.ones((self.n_parallel,))}

    def update_state(self, state, key, results, params=None):
        del state, key, results, params
        raise _UpdateError("update failed")


def test_callback_is_keyword_only_and_exported():
    callback_parameter = inspect.signature(base.Optimizer.optimize).parameters[
        "callback"
    ]
    assert callback_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert BatchCompleted.__module__ == "hyperoptax.recording"
    assert BatchCallback is not None


def test_callback_receives_one_host_event_per_completed_batch(monkeypatch):
    space = {"x": sp.LinearSpace(0.0, 1.0)}
    state, optimizer = RandomSearch.init(space, n_parallel=3)
    events = []
    device_get_calls = 0
    original_device_get = jax.device_get

    def counted_device_get(value):
        nonlocal device_get_calls
        device_get_calls += 1
        return original_device_get(value)

    monkeypatch.setattr(base.jax, "device_get", counted_device_get)
    _, (params_history, results_history) = optimizer.optimize(
        state,
        jax.random.PRNGKey(0),
        _objective,
        n_iterations=4,
        callback=events.append,
    )

    assert device_get_calls == 4
    assert [event.batch_index for event in events] == [0, 1, 2, 3]
    assert all(event.duration_seconds is not None for event in events)
    assert all(event.duration_seconds >= 0 for event in events)
    for event, params, results in zip(
        events, params_history, results_history, strict=True
    ):
        assert isinstance(event, BatchCompleted)
        assert isinstance(event.results, np.ndarray)
        assert all(
            isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(event.params)
        )
        _assert_trees_equal(event.params, params)
        np.testing.assert_array_equal(event.results, np.asarray(results))


def test_batch_duration_must_not_be_negative():
    with pytest.raises(ValueError, match="duration_seconds"):
        BatchCompleted(
            batch_index=0,
            params={"x": np.asarray([1.0])},
            results=np.asarray([1.0]),
            duration_seconds=-1.0,
        )


def test_omitting_callback_preserves_seeded_state_and_history(monkeypatch):
    space = {"x": sp.LinearSpace(0.0, 1.0)}
    initial_state, optimizer = RandomSearch.init(space, n_parallel=3)
    key = jax.random.PRNGKey(7)

    device_get_calls = 0
    original_device_get = jax.device_get

    def counted_device_get(value):
        nonlocal device_get_calls
        device_get_calls += 1
        return original_device_get(value)

    monkeypatch.setattr(base.jax, "device_get", counted_device_get)
    state_without, history_without = optimizer.optimize(
        initial_state, key, _objective, n_iterations=3
    )
    assert device_get_calls == 0

    events = []
    state_with, history_with = optimizer.optimize(
        initial_state,
        key,
        _objective,
        n_iterations=3,
        callback=events.append,
    )

    assert state_with == state_without
    _assert_trees_equal(history_with, history_without)
    assert len(events) == 3


def test_zero_iterations_emit_no_events():
    state, optimizer = RandomSearch.init({"x": sp.LinearSpace(0.0, 1.0)})
    events = []

    _, history = optimizer.optimize(
        state,
        jax.random.PRNGKey(0),
        _objective,
        n_iterations=0,
        callback=events.append,
    )

    assert events == []
    assert history == ([], [])


def test_objective_failure_emits_no_event():
    class ObjectiveError(RuntimeError):
        pass

    def failing_objective(key, params):
        del key, params
        raise ObjectiveError("objective failed")

    state, optimizer = RandomSearch.init({"x": sp.LinearSpace(0.0, 1.0)})
    events = []

    with pytest.raises(ObjectiveError, match="objective failed"):
        optimizer.optimize(
            state,
            jax.random.PRNGKey(0),
            failing_objective,
            n_iterations=1,
            callback=events.append,
        )

    assert events == []


def test_update_failure_emits_no_event():
    state, optimizer = _FailingUpdateOptimizer.init({"x": sp.LinearSpace(0.0, 1.0)})
    events = []

    with pytest.raises(_UpdateError, match="update failed"):
        optimizer.optimize(
            state,
            jax.random.PRNGKey(0),
            _objective,
            n_iterations=1,
            callback=events.append,
        )

    assert events == []


def test_callback_exception_propagates_unchanged():
    class CallbackError(RuntimeError):
        pass

    error = CallbackError("callback failed")

    def failing_callback(event):
        del event
        raise error

    state, optimizer = RandomSearch.init({"x": sp.LinearSpace(0.0, 1.0)})

    with pytest.raises(CallbackError) as exc_info:
        optimizer.optimize(
            state,
            jax.random.PRNGKey(0),
            _objective,
            n_iterations=1,
            callback=failing_callback,
        )

    assert exc_info.value is error


@pytest.mark.parametrize("infer_iterations", [False, True])
def test_bayesian_search_forwards_callback(infer_iterations):
    state, optimizer = BayesianSearch.init(
        {"x": sp.DiscreteSpace([0.0, 0.5, 1.0])},
        n_max=4,
        n_parallel=2,
        n_hparam_steps=0,
    )
    events = []
    kwargs = {} if infer_iterations else {"n_iterations": 2}

    optimizer.optimize(
        state,
        jax.random.PRNGKey(0),
        _objective,
        callback=events.append,
        **kwargs,
    )

    assert [event.batch_index for event in events] == [0, 1]


def test_optimize_scan_does_not_accept_callback():
    assert "callback" not in inspect.signature(base.Optimizer.optimize_scan).parameters
    assert "callback" not in inspect.signature(BayesianSearch.optimize_scan).parameters
