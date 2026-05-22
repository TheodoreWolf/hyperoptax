import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from hyperoptax import spaces as sp


def make_key_tree(
    pytree: PyTree,
    subkey: jax.random.PRNGKey,
) -> PyTree:
    """Split ``subkey`` into a pytree of PRNGKeys matching the structure of ``pytree``.

    :class:`~hyperoptax.spaces.Space` objects are treated as leaves, so the
    returned tree has one key per space in the search-space pytree.

    Args:
        pytree: A pytree whose structure determines how many keys are produced.
        subkey: PRNGKey to split.

    Returns:
        A pytree with the same structure as ``pytree`` where each leaf is a
        fresh PRNGKey.
    """
    tree = jax.tree_util.tree_structure(
        pytree, is_leaf=lambda x: isinstance(x, sp.Space)
    )
    keys = jax.random.split(subkey, tree.num_leaves)
    return jax.tree_util.tree_unflatten(tree, keys)


def sample_space_pytree(
    space: PyTree,
    key: jax.random.PRNGKey,
    n: int,
) -> PyTree:
    """Sample ``n`` configurations from a pytree of Space leaves.

    Per-leaf dispatch to the correct ``Space.sample`` method is handled by a
    static :func:`jax.tree.map`; the ``n`` samples axis is parallelised with
    :func:`jax.vmap` over PRNG keys.

    Args:
        space: A pytree whose leaves are :class:`~hyperoptax.spaces.Space`
            instances. Heterogeneous Space types may be freely mixed.
        key: PRNGKey used to seed sampling.
        n: Number of independent configurations to draw.

    Returns:
        A pytree with the same structure as ``space`` where each Space leaf is
        replaced by a JAX array of shape ``(n,)`` containing that parameter's
        ``n`` samples.
    """

    def sample_once(k):
        subkeys = make_key_tree(space, k)
        return jax.tree.map(
            lambda x, sk: x.sample(sk).squeeze(),
            space,
            subkeys,
            is_leaf=lambda x: isinstance(x, sp.Space),
        )

    keys = jax.random.split(key, n)
    return jax.vmap(sample_once)(keys)


def sample_space_array(
    space: PyTree,
    key: jax.random.PRNGKey,
    n: int,
) -> jax.Array:
    """Sample ``n`` configurations stacked as a flat ``(n, n_params)`` array.

    Convenience wrapper around :func:`sample_space_pytree` that flattens the
    per-leaf result tree into a single 2-D array. Leaf order matches
    ``jax.tree.leaves(space, is_leaf=...)``.
    """
    samples = sample_space_pytree(space, key, n)
    leaves = jax.tree.leaves(samples)
    return jnp.stack(leaves, axis=-1)
