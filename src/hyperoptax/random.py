import logging
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax import utils

logger = logging.getLogger(__name__)


@struct.dataclass
class OptimizerState:
    space: struct.PyTreeNode


class Optimizer(struct.PyTreeNode):
    @classmethod
    def init(cls, space, n_seeds: int = 1) -> OptimizerState:
        return cls._init(space, n_seeds)

    @classmethod
    def _init(cls, space, n_seeds: int = 1):
        pass

    @classmethod
    def optimize(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
        n_parallel: int,
    ) -> tuple[OptimizerState, jax.Array]:
        state, results = cls._optimize(state, key, func, n_iterations, n_parallel)
        return state, results

    @classmethod
    def _optimize(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
        n_parallel: int,
    ) -> tuple[OptimizerState, jax.Array]:
        raise NotImplementedError


class RandomSearch(Optimizer):
    @classmethod
    def _init(cls, space, n_seeds):
        return OptimizerState(space=space)

    @classmethod
    def _optimize(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
        n_parallel: int = 1,
    ) -> tuple[OptimizerState, jax.Array]:
        keys = jax.random.split(key, n_parallel)
        state, results = jax.vmap(cls._loop, in_axes=(None, 0, None, None, None))(
            state, keys, func, n_iterations, n_parallel
        )
        results = jax.tree.map(
            lambda x: x.reshape((n_iterations * n_parallel, -1)), results
        )
        return state, results

    @classmethod
    def _loop(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
        n_parallel: int = 1,
    ) -> tuple[OptimizerState, tuple[jax.Array, struct.PyTreeNode]]:
        def _step(
            carry: tuple[OptimizerState, jax.random.PRNGKey], _
        ) -> tuple[OptimizerState, tuple[jax.Array, struct.PyTreeNode]]:
            state, key = carry
            key, subkey = jax.random.split(key)
            keys = utils.make_key_tree(state.space, subkey)
            next_params = jax.tree.map(
                lambda x, k: x.sample(k),
                state.space,
                keys,
                is_leaf=lambda x: isinstance(x, sp.Space),
            )
            results = func(**next_params)

            return (state, key), (results, next_params)

        state, (results, params) = jax.lax.scan(
            _step, (state, key), length=n_iterations
        )
        return state, (results, params)


@struct.dataclass
class GridSearchState:
    space: struct.PyTreeNode
    space_idx: struct.PyTreeNode
    idx_pointer: int
    random_shuffle = False


class GridSearch(Optimizer):
    @classmethod
    def _init(cls, space, n_seed):
        # assert the space is discrete for grid search
        assert any(
            jax.tree.flatten(
                jax.tree.map(
                    lambda x: isinstance(x, sp.DiscreteSpace),
                    space,
                    is_leaf=lambda x: isinstance(x, sp.Space),
                ),
            )[0]
        ), "GridSearch with non-Discrete spaces is not possible."
        space_idx = jax.tree.map(
            lambda x: 0, space, is_leaf=lambda x: isinstance(x, sp.Space)
        )
        return GridSearchState(space, space_idx, 0)

    @classmethod
    def _optimize(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
        n_parallel: int = 1,
    ) -> tuple[OptimizerState, jax.Array]:
        keys = jax.random.split(key, n_parallel)
        state, results = jax.vmap(cls._loop, in_axes=(None, 0, None, None, None))(
            state, keys, func, n_iterations, n_parallel
        )
        results = jax.tree.map(
            lambda x: x.reshape((n_iterations * n_parallel, -1)), results
        )
        return state, results

    @classmethod
    def _loop(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
        n_parallel: int = 1,
    ) -> tuple[OptimizerState, jax.Array]:
        def _step(carry, _):
            key, state = carry
            key, subkey = jax.random.split(key)
            keys = utils.make_key_tree(state.space, subkey)
            next_params = jax.tree.map(
                lambda x, y: jnp.array(x.values).at[y].get(),
                state.space,
                state.space_idx,
                is_leaf=lambda x: isinstance(x, sp.Space),
            )
            results = func(**next_params)
            new_state = update_state(state)

            return (key, new_state), (results, next_params)

        (key, state), (results, params) = jax.lax.scan(
            _step, (key, state), length=n_iterations
        )
        return state, (results, params)


def update_state(state: GridSearchState) -> GridSearchState:
    idx_leaves, idx_treedef = jax.tree_util.tree_flatten(state.space_idx)
    space_leaves, _ = jax.tree_util.tree_flatten(
        state.space, is_leaf=lambda x: isinstance(x, sp.Space)
    )
    N = len(idx_leaves)
    idx_arr = jnp.stack(idx_leaves).reshape((N,))
    lengths = [len(x.values) for x in space_leaves]
    lengths = jnp.array(lengths, dtype=idx_arr.dtype)
    pointer = jnp.asarray(state.idx_pointer)
    out_of_bounds = pointer >= N
    cur_idx = jax.lax.dynamic_slice(idx_arr, (pointer,), (1,))[0]
    max_idx_for_pointer = lengths[pointer] - 1

    def return_unchanged():
        return state

    def handle_in_bounds():
        def inc_pointer():
            new_pointer = pointer + 1
            new_idx_arr = idx_arr.at[pointer].set(0)
            new_space_idx = jax.tree_util.tree_unflatten(
                idx_treedef, new_idx_arr.reshape((N,))
            )
            return state.replace(
                idx_pointer=jnp.asarray(new_pointer), space_idx=new_space_idx
            )

        def inc_idx():
            new_idx_arr = idx_arr.at[pointer].add(1)
            new_space_idx = jax.tree_util.tree_unflatten(
                idx_treedef, new_idx_arr.reshape((N,))
            )
            return state.replace(space_idx=new_space_idx)

        is_full = cur_idx == max_idx_for_pointer
        return jax.lax.cond(is_full, inc_pointer, inc_idx)

    new_state = jax.lax.cond(out_of_bounds, return_unchanged, handle_in_bounds)
    return new_state
