from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax import utils
from hyperoptax.base import Optimizer, OptimizerState


@struct.dataclass
class GridSearchState(OptimizerState):
    space: struct.PyTreeNode
    space_flat: jax.Array
    space_idx: int
    random_shuffle: bool


class GridSearch(Optimizer):
    @classmethod
    def init(cls, space, random_shuffle=False):
        is_discrete = jax.tree.map(
            lambda x: isinstance(x, sp.DiscreteSpace),
            space,
            is_leaf=lambda x: isinstance(x, sp.Space),
        )
        if not all(jax.tree.leaves(is_discrete)):
            raise ValueError("GridSearch requires all spaces to be DiscreteSpace.")

        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        values_list = [jnp.array(leaf.values) for leaf in leaves]
        grids = jnp.meshgrid(*values_list, indexing="ij")
        # Flatten into (n_total, n_leaves) so space_flat[i] is the i-th param combination
        space_flat = jnp.stack([g.ravel() for g in grids], axis=-1)

        return GridSearchState(
            space=space,
            space_flat=space_flat,
            space_idx=0,
            random_shuffle=random_shuffle,
        )

    @classmethod
    def get_next_params(cls, state: GridSearchState, key=None) -> struct.PyTreeNode:
        flat_params = state.space_flat[state.space_idx]
        _, treedef = jax.tree.flatten(state.space, is_leaf=lambda x: isinstance(x, sp.Space))
        return treedef.unflatten([flat_params[i] for i in range(treedef.num_leaves)])

    @classmethod
    def update_state(cls, state: GridSearchState, key=None, results=None) -> GridSearchState:
        return state.replace(space_idx=state.space_idx + 1)

