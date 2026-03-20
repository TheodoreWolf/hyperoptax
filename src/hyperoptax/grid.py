import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax import base
from hyperoptax import spaces as sp


@struct.dataclass
class GridSearchState(base.OptimizerState):
    grid: jax.Array
    grid_idx: int


@dataclasses.dataclass
class GridSearch(base.Optimizer):
    shuffle: bool = False
    n_parallel: int = 1

    @classmethod
    def init(cls, space, key=None, **kwargs):
        is_discrete = jax.tree.map(
            lambda x: isinstance(x, sp.DiscreteSpace),
            space,
            is_leaf=lambda x: isinstance(x, sp.Space),
        )
        if not all(jax.tree.leaves(is_discrete)):
            raise ValueError("GridSearch requires all spaces to be DiscreteSpace.")

        optimizer = cls(**kwargs)

        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        values_list = [jnp.array(leaf.values) for leaf in leaves]
        # TODO: use indexes so that we don't generate the full grid.
        grids = jnp.meshgrid(*values_list, indexing="ij")
        # Flatten into (n_total, n_leaves) so grid[i] is the i-th param combination
        grid = jnp.stack([g.ravel() for g in grids], axis=-1)
        if optimizer.shuffle:
            # key=None falls back to PRNGKey(0); pass key explicitly for reproducibility
            key = key if key is not None else jax.random.PRNGKey(0)
            grid = jax.random.permutation(key, grid)
        n_usable = (len(grid) // optimizer.n_parallel) * optimizer.n_parallel
        grid = grid[:n_usable]
        state = GridSearchState(
            space=space,
            grid=grid,
            grid_idx=0,
        )
        return state, optimizer

    def get_next_params(
        self, state: GridSearchState, key, params=None, results=None
    ) -> struct.PyTreeNode:
        # Only check eagerly; inside lax.scan grid_idx is an abstract tracer.
        if not isinstance(state.grid_idx, jax.core.Tracer):
            if int(state.grid_idx) + self.n_parallel > state.grid.shape[0]:
                raise ValueError(
                    f"Not enough grid points remaining "
                    f"(grid_idx={int(state.grid_idx)}, n_parallel={self.n_parallel}, "
                    f"grid_size={state.grid.shape[0]})."
                )
        # Extract n_parallel rows; use dynamic slice for scan compatibility
        rows = jax.lax.dynamic_slice_in_dim(
            state.grid, state.grid_idx, self.n_parallel, axis=0
        )  # (n_parallel, n_leaves)
        _, treedef = jax.tree.flatten(
            state.space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        # Each leaf gets shape (n_parallel,)
        return treedef.unflatten([rows[:, i] for i in range(treedef.num_leaves)])

    def update_state(
        self, state: GridSearchState, key, results, params=None
    ) -> GridSearchState:
        return state.replace(grid_idx=state.grid_idx + self.n_parallel)
