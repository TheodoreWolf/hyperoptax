import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from hyperoptax import base, utils
from hyperoptax import spaces as sp


@dataclasses.dataclass
class RandomSearch(base.Optimizer):
    """Stateless random search — samples each space independently each iteration.

    No model is fitted and no history is maintained, so this is the cheapest
    optimizer and useful as a strong baseline.

    Attributes:
        n_parallel: Number of random configurations evaluated per iteration.
    """

    n_parallel: int = 1

    @classmethod
    def init(cls, space, **kwargs):
        return base.OptimizerState(space=space), cls(**kwargs)

    def get_next_params(
        self,
        state: base.OptimizerState,
        key: jax.random.PRNGKey,
        params=None,
        results=None,
    ) -> struct.PyTreeNode:
        """Sample ``n_parallel`` independent configurations from the search space."""
        def sample_once(k):
            subkeys = utils.make_key_tree(state.space, k)
            sample = jax.tree.map(
                lambda x, sk: x.sample(sk),
                state.space,
                subkeys,
                is_leaf=lambda x: isinstance(x, sp.Space),
            )
            # Squeeze (1,) per-leaf values to scalars for stacking
            return jax.tree.map(lambda leaf: leaf.squeeze(), sample)

        keys = jax.random.split(key, self.n_parallel)
        samples = [sample_once(k) for k in keys]
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *samples)

    def update_state(
        self,
        state: base.OptimizerState,
        key: jax.random.PRNGKey,
        results: jax.Array,
        params=None,
    ) -> base.OptimizerState:
        """
        RandomSearch is memoryless, no state to update.
        """
        return state
