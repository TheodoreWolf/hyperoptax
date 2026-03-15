import jax
from flax import struct

from hyperoptax import spaces as sp
from hyperoptax import utils
from hyperoptax.base import Optimizer, OptimizerState


class RandomSearch(Optimizer):
    def get_next_params(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        params=None,
        results=None,
    ) -> struct.PyTreeNode:
        key, subkey = jax.random.split(key)
        keys = utils.make_key_tree(state.space, subkey)
        next_params = jax.tree.map(
            lambda x, k: x.sample(k),
            state.space,
            keys,
            is_leaf=lambda x: isinstance(x, sp.Space),
        )
        return next_params

    def update_state(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        results: jax.Array,
        params=None,
    ) -> OptimizerState:
        """
        RandomSearch is memoryless, no state to update.
        """
        return state
