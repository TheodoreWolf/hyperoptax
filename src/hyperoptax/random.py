import dataclasses

import jax
from jaxtyping import PyTree

from hyperoptax import base, utils


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
    ) -> PyTree:
        """Sample ``n_parallel`` independent configurations from the search space."""
        return utils.sample_space_pytree(state.space, key, self.n_parallel)

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
