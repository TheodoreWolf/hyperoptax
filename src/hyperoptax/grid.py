import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax.base import Optimizer, OptimizerState


@struct.dataclass
class GridSearchState(OptimizerState):
    space_flat: jax.Array
    space_idx: int


@dataclasses.dataclass
class GridSearch(Optimizer):
    shuffle: bool = False

    @classmethod
    def init(cls, space, key=None, **kwargs):
        is_discrete = jax.tree.map(
            lambda x: isinstance(x, sp.DiscreteSpace),
            space,
            is_leaf=lambda x: isinstance(x, sp.Space),
        )
        if not all(jax.tree.leaves(is_discrete)):
            raise ValueError("GridSearch requires all spaces to be DiscreteSpace.")

        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        values_list = [jnp.array(leaf.values) for leaf in leaves]
        # TODO: use indexes so that we don't generate the full grid.
        grids = jnp.meshgrid(*values_list, indexing="ij")
        # Flatten into (n_total, n_leaves) so space_flat[i] is the i-th param combination
        space_flat = jnp.stack([g.ravel() for g in grids], axis=-1)
        if kwargs.get("shuffle", False):
            key = key if key is not None else jax.random.PRNGKey(0)
            space_flat = jax.random.permutation(key, space_flat)
        state = GridSearchState(
            space=space,
            space_flat=space_flat,
            space_idx=0,
        )
        return state, cls(**kwargs)

    def get_next_params(
        self, state: GridSearchState, key=None, params=None, results=None
    ) -> struct.PyTreeNode:
        flat_params = state.space_flat[state.space_idx]
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        return treedef.unflatten([flat_params[i] for i in range(treedef.num_leaves)])

    def update_state(
        self, state: GridSearchState, key=None, results=None, params=None
    ) -> GridSearchState:
        return state.replace(space_idx=state.space_idx + 1)
