from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax.spaces import Space


@struct.dataclass
class OptimizerState:
    space: struct.PyTreeNode


class Optimizer(struct.PyTreeNode):
    @classmethod
    def init(cls, space, n_seeds: int = 1) -> OptimizerState:
        cls._init(space, n_seeds)
        return OptimizerState(space=space)

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
        results = results.reshape((n_iterations * n_parallel, -1))
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
            state, key = carry
            key, subkey = jax.random.split(key)
            tree = jax.tree_util.tree_structure(
                state.space, is_leaf=lambda x: isinstance(x, Space)
            )
            keys = jax.random.split(subkey, tree.num_leaves)
            keys = jax.tree_util.tree_unflatten(tree, keys)
            next_params = jax.tree.map(
                lambda x, k: x.sample(k),
                state.space,
                keys,
                is_leaf=lambda x: isinstance(x, Space),
            )
            results = func(**next_params)

            return (state, key), results

        state, results = jax.lax.scan(_step, (state, key), length=n_iterations)
        return state, results
