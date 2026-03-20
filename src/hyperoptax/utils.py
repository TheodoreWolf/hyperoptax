import jax
from flax import struct

from hyperoptax import spaces as sp


def make_key_tree(
    pytree: struct.PyTreeNode,
    subkey: jax.random.PRNGKey,
) -> struct.PyTreeNode:
    tree = jax.tree_util.tree_structure(
        pytree, is_leaf=lambda x: isinstance(x, sp.Space)
    )
    keys = jax.random.split(subkey, tree.num_leaves)
    return jax.tree_util.tree_unflatten(tree, keys)
