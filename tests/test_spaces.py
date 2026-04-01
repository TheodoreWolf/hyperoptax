import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp


def test_space():
    space = sp.LinearSpace(0, 1)
    assert space.sample(jax.random.PRNGKey(0)).shape == (1,)


def test_space_sample_in_pytree():
    space = {"a": sp.LinearSpace(0, 1), "b": sp.LinearSpace(2, 3)}
    key = jax.random.PRNGKey(0)
    sample = jax.tree.map(
        lambda x: x.sample(key), space, is_leaf=lambda x: isinstance(x, sp.Space)
    )
    assert sample["a"].shape == (1,)
    assert sample["b"].shape == (1,)


def test_discrete_space():
    space = sp.DiscreteSpace([0, 1, 2, 3])
    assert space.sample(jax.random.PRNGKey(0)).shape == (1,)
    assert space.sample(jax.random.PRNGKey(0)) in [0, 1, 2, 3]


def test_discrete_space_sample_in_pytree():
    space = {"a": sp.DiscreteSpace([0, 1, 2, 3]), "b": sp.DiscreteSpace([4, 5, 6, 7])}
    key = jax.random.PRNGKey(0)
    sample = jax.tree.map(
        lambda x: x.sample(key),
        space,
        is_leaf=lambda x: isinstance(x, sp.DiscreteSpace),
    )
    assert sample["a"].shape == (1,)
    assert sample["b"].shape == (1,)
    assert sample["a"] in [0, 1, 2, 3]
    assert sample["b"] in [4, 5, 6, 7]


def test_log_space():
    space = sp.LogSpace(1e-4, 1e-1)
    assert space.sample(jax.random.PRNGKey(0)).shape == (1,)
    assert space.sample(jax.random.PRNGKey(0)) > 1e-4
    assert space.sample(jax.random.PRNGKey(0)) < 1e-1


def test_log_space_sample_in_pytree():
    space = {"a": sp.LogSpace(1e-4, 1e-1), "b": sp.LogSpace(1e-3, 1e-2)}
    key = jax.random.PRNGKey(0)
    sample = jax.tree.map(
        lambda x: x.sample(key), space, is_leaf=lambda x: isinstance(x, sp.LogSpace)
    )
    assert sample["a"].shape == (1,)
    assert sample["b"].shape == (1,)
    assert sample["a"] > 1e-4
    assert sample["a"] < 1e-1
    assert sample["b"] < 1e-2
    assert sample["b"] > 1e-3


def test_logspace_with_different_bases():
    space = sp.LogSpace(2, 64, base=2)
    sample = space.sample(jax.random.PRNGKey(0))
    assert sample.shape == (1,)
    assert sample > 2
    assert sample < 64


def test_qspace():
    space = sp.QLinearSpace(0, 100)
    sample = space.sample(jax.random.PRNGKey(0))
    assert sample.shape == (1,)
    assert sample.dtype == jnp.int32
    assert sample >= 0
    assert sample <= 100


def test_qlogspace():
    space = sp.QLogSpace(2, 128, base=2)
    sample = space.sample(jax.random.PRNGKey(0))
    assert sample.shape == (1,)
    assert sample.dtype == jnp.int32
    assert sample >= 2
    assert sample <= 128


def test_space_post_init():
    sp.LogSpace(0, 10, base=2)
    with pytest.raises(AssertionError):
        sp.LogSpace(10, 0)
    with pytest.raises(AssertionError):
        sp.LogSpace(0, 10, base=1)


def test_space_base_sample_raises():
    # Space.sample is abstract — subclasses must override it
    space = sp.Space()
    with pytest.raises(NotImplementedError):
        space.sample(jax.random.PRNGKey(0))


def test_discrete_space_lower_upper_bound():
    space = sp.DiscreteSpace([3, 1, 4, 1, 5, 9])
    assert space.lower_bound == 1.0
    assert space.upper_bound == 9.0


def test_discrete_space_transform_snaps_to_nearest():
    space = sp.DiscreteSpace([0.0, 0.25, 0.5, 0.75, 1.0])
    assert float(space.transform(jnp.array([0.1]))[0]) == pytest.approx(0.0)
    assert float(space.transform(jnp.array([0.4]))[0]) == pytest.approx(0.5)
    assert float(space.transform(jnp.array([0.9]))[0]) == pytest.approx(1.0)


def test_discrete_space_transform_preserves_shape():
    space = sp.DiscreteSpace([0, 1, 2])
    assert space.transform(jnp.array([1.4])).shape == (1,)
    assert space.transform(jnp.array(1.4)).shape == ()


def test_linear_space_lower_upper_bound():
    space = sp.LinearSpace(2.0, 8.0)
    assert space.lower_bound == 2.0
    assert space.upper_bound == 8.0
