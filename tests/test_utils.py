import jax
import jax.numpy as jnp
import pytest

from hyperoptax import spaces as sp
from hyperoptax import utils


class TestSampleSpacePytree:
    def test_single_linear_shape_and_bounds(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        samples = utils.sample_space_pytree(space, jax.random.PRNGKey(0), n=8)
        assert samples["x"].shape == (8,)
        assert jnp.all(samples["x"] >= 0.0) and jnp.all(samples["x"] <= 1.0)

    def test_heterogeneous_flat_dict(self):
        space = {
            "lr": sp.LogSpace(1e-4, 1e-1),
            "momentum": sp.LinearSpace(0.0, 1.0),
            "optimizer": sp.DiscreteSpace((0, 1, 2)),
            "layers": sp.QLinearSpace(1, 8),
            "units": sp.QLogSpace(16, 1024),
        }
        n = 16
        samples = utils.sample_space_pytree(space, jax.random.PRNGKey(0), n=n)

        # Structure preserved
        assert set(samples.keys()) == set(space.keys())
        # All leaves shaped (n,)
        for v in jax.tree.leaves(samples):
            assert v.shape == (n,)

        # Per-Space bounds
        assert jnp.all(samples["lr"] >= 1e-4) and jnp.all(samples["lr"] <= 1e-1)
        assert jnp.all(samples["momentum"] >= 0.0) and jnp.all(samples["momentum"] <= 1.0)
        # DiscreteSpace: every sample must be one of the values
        for v in samples["optimizer"]:
            assert int(v) in (0, 1, 2)
        # Quantised spaces: integer dtype + within bounds
        assert jnp.issubdtype(samples["layers"].dtype, jnp.integer)
        assert jnp.issubdtype(samples["units"].dtype, jnp.integer)
        assert jnp.all(samples["layers"] >= 1) and jnp.all(samples["layers"] <= 8)
        assert jnp.all(samples["units"] >= 16) and jnp.all(samples["units"] <= 1024)

    def test_nested_heterogeneous_pytree(self):
        space = {
            "opt": {
                "lr": sp.LogSpace(1e-5, 1e-1),
                "sched": {"warmup": sp.QLinearSpace(0, 1000)},
            },
            "model": {
                "depth": sp.QLinearSpace(2, 12),
                "activation": sp.DiscreteSpace((0, 1, 2)),
            },
        }
        samples = utils.sample_space_pytree(space, jax.random.PRNGKey(1), n=5)
        assert samples["opt"]["lr"].shape == (5,)
        assert samples["opt"]["sched"]["warmup"].shape == (5,)
        assert samples["model"]["depth"].shape == (5,)
        assert samples["model"]["activation"].shape == (5,)
        # Bounds on continuous / quantised leaves
        assert jnp.all(samples["opt"]["lr"] >= 1e-5)
        assert jnp.all(samples["opt"]["lr"] <= 1e-1)
        assert jnp.all(samples["opt"]["sched"]["warmup"] >= 0)
        assert jnp.all(samples["opt"]["sched"]["warmup"] <= 1000)

    def test_determinism_same_key(self):
        space = {
            "a": sp.LinearSpace(0.0, 1.0),
            "b": sp.LogSpace(1e-3, 1.0),
            "c": sp.DiscreteSpace((0, 1, 2, 3)),
        }
        key = jax.random.PRNGKey(42)
        s1 = utils.sample_space_pytree(space, key, n=4)
        s2 = utils.sample_space_pytree(space, key, n=4)
        for leaf1, leaf2 in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
            assert jnp.allclose(leaf1, leaf2)

    def test_different_keys_differ(self):
        space = {"a": sp.LinearSpace(0.0, 1.0), "b": sp.LinearSpace(0.0, 1.0)}
        s1 = utils.sample_space_pytree(space, jax.random.PRNGKey(0), n=8)
        s2 = utils.sample_space_pytree(space, jax.random.PRNGKey(1), n=8)
        assert not jnp.allclose(s1["a"], s2["a"])
        assert not jnp.allclose(s1["b"], s2["b"])

    def test_n_one(self):
        space = {"x": sp.LinearSpace(0.0, 1.0)}
        samples = utils.sample_space_pytree(space, jax.random.PRNGKey(0), n=1)
        assert samples["x"].shape == (1,)

    @pytest.mark.parametrize("n", [1, 2, 7, 100])
    def test_various_n(self, n):
        space = {
            "lr": sp.LogSpace(1e-4, 1.0),
            "k": sp.DiscreteSpace((0.1, 0.5, 0.9)),
        }
        samples = utils.sample_space_pytree(space, jax.random.PRNGKey(0), n=n)
        assert samples["lr"].shape == (n,)
        assert samples["k"].shape == (n,)

    def test_jit_compatible(self):
        space = {
            "lr": sp.LogSpace(1e-4, 1.0),
            "n": sp.QLinearSpace(1, 16),
        }
        fn = jax.jit(lambda k: utils.sample_space_pytree(space, k, n=8))
        samples = fn(jax.random.PRNGKey(0))
        assert samples["lr"].shape == (8,)
        assert samples["n"].shape == (8,)


class TestSampleSpaceArray:
    def test_flat_shape(self):
        space = {
            "a": sp.LinearSpace(0.0, 1.0),
            "b": sp.LogSpace(1e-3, 1.0),
            "c": sp.DiscreteSpace((0, 1, 2)),
        }
        out = utils.sample_space_array(space, jax.random.PRNGKey(0), n=10)
        assert out.shape == (10, 3)

    def test_heterogeneous_column_bounds(self):
        space = {
            "lr": sp.LogSpace(1e-4, 1e-1),
            "p": sp.LinearSpace(0.0, 1.0),
            "k": sp.QLinearSpace(1, 5),
        }
        out = utils.sample_space_array(space, jax.random.PRNGKey(0), n=32)
        # leaf order matches jax.tree.leaves
        leaf_order = list(
            jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        )
        # By Python dict-insertion order of the pytree: k, lr, p?
        # We don't assume order — instead just confirm each *Space's* bounds
        # are satisfied by *some* column. That's enough to catch a swap.
        cols = [out[:, i] for i in range(out.shape[1])]
        # Check via leaf order (jax.tree.leaves is deterministic given the same pytree)
        for leaf, col in zip(leaf_order, cols):
            assert jnp.all(col >= leaf.lower_bound - 1e-6)
            assert jnp.all(col <= leaf.upper_bound + 1e-6)

    def test_matches_pytree_version(self):
        space = {
            "a": sp.LinearSpace(0.0, 1.0),
            "b": sp.LogSpace(1e-3, 1.0),
        }
        key = jax.random.PRNGKey(7)
        tree_samples = utils.sample_space_pytree(space, key, n=6)
        arr = utils.sample_space_array(space, key, n=6)
        expected = jnp.stack(jax.tree.leaves(tree_samples), axis=-1)
        assert jnp.allclose(arr, expected)

    def test_nested_pytree(self):
        space = {
            "opt": {"lr": sp.LogSpace(1e-4, 1.0), "wd": sp.LinearSpace(0.0, 0.1)},
            "model": {"depth": sp.QLinearSpace(1, 8)},
        }
        out = utils.sample_space_array(space, jax.random.PRNGKey(0), n=4)
        assert out.shape == (4, 3)
