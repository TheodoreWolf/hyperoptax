import jax
from flax import struct

from hyperoptax import spaces as sp


def make_key_tree(
    pytree: struct.PyTreeNode,
    subkey: jax.random.PRNGKey,
) -> struct.PyTreeNode:
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
