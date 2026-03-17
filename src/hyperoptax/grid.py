import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax.base import Optimizer, OptimizerState


@struct.dataclass
class GridSearchState(OptimizerState):
    grid: jax.Array
    grid_idx: int


@dataclasses.dataclass
class GridSearch(Optimizer):
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

        leaves = jax.tree.leaves(space, is_leaf=lambda x: isinstance(x, sp.Space))
        values_list = [jnp.array(leaf.values) for leaf in leaves]
        # TODO: use indexes so that we don't generate the full grid.
        grids = jnp.meshgrid(*values_list, indexing="ij")
        # Flatten into (n_total, n_leaves) so grid[i] is the i-th param combination
        grid = jnp.stack([g.ravel() for g in grids], axis=-1)
        if kwargs.get("shuffle", False):
            key = key if key is not None else jax.random.PRNGKey(0)
            grid = jax.random.permutation(key, grid)
        state = GridSearchState(
            space=space,
            grid=grid,
            grid_idx=0,
        )
        return state, cls(**kwargs)

    def get_next_params(
        self, state: GridSearchState, key, params=None, results=None
    ) -> struct.PyTreeNode:
        n_total = state.grid.shape[0]
        # Bounds check only works outside JIT/scan where grid_idx is concrete.
        try:
            idx_val = int(state.grid_idx)
            if idx_val + self.n_parallel > n_total:
                raise ValueError(
                    f"Not enough grid points remaining: need {self.n_parallel} but only "
                    f"{n_total - idx_val} left (n_total={n_total}, "
                    f"grid_idx={idx_val})."
                )
        except Exception as e:
            if "concrete" in str(e).lower() or "tracer" in str(e).lower():
                pass  # Inside JIT/scan - grid_idx is traced, skip bounds check
            else:
                raise
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
