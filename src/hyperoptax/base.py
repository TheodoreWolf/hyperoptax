from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class OptimizerState:
    space: struct.PyTreeNode


class Optimizer(struct.PyTreeNode):
    @classmethod
    def init(cls, space, **kwargs) -> OptimizerState:
        return OptimizerState(space=space)


    @classmethod
    def optimize(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
    ) -> tuple[OptimizerState, jax.Array]:
        """
        High Level API for optimizing a function over a space.
        Not recommended if you want to do fancy things
        with parallel computation.
        """

        def _step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            next_params = cls.get_next_params(state, subkey)
            results = func(**next_params)
            state = cls.update_state(state, subkey, results)
            return (state, key), results

        (state, _), results = jax.lax.scan(_step, (state, key), None, length=n_iterations)
        return state, results

    @classmethod
    def update_state(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        results: jax.Array,
    ) -> OptimizerState:
        """
        Updates the optimizer state based on the results of the function.
        """
        raise NotImplementedError

    @classmethod
    def get_next_params(
        cls,
        state: OptimizerState,
        key: jax.random.PRNGKey,
    ) -> struct.PyTreeNode:
        """
        Gets the next parameters to sample from the space.
        """
        raise NotImplementedError
