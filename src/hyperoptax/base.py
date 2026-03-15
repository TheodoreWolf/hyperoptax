import inspect
import warnings
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct


def _validate_func(func):
    try:
        sig = inspect.signature(func)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) < 2:
            raise TypeError(
                f"func must have signature fn(key, config) — "
                f"received a function with {len(positional)} positional parameter(s). "
                "Did you forget the key argument?"
            )
    except (ValueError, TypeError) as e:
        if "func must have" in str(e):
            raise
        warnings.warn(
            "Can't introspect function signature - ensure that the "
            "function has a signature fn(key, config)."
        )
        return


@struct.dataclass
class OptimizerState:
    space: struct.PyTreeNode


class Optimizer:
    @classmethod
    def init(cls, space, **kwargs) -> OptimizerState:
        return OptimizerState(space=space), cls()

    def optimize(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
    ) -> tuple[OptimizerState, tuple[struct.PyTreeNode, jax.Array]]:
        """
        High Level API for optimizing a function over a space.
        Not recommended if you want to do fancy things
        with parallel computation.
        """
        _validate_func(func)
        params_hist, results_hist = [], []
        params, results = None, None
        for _ in range(n_iterations):
            key, key_get, key_func, key_update = jax.random.split(key, 4)
            params = self.get_next_params(state, key_get, params, results)
            results = func(key_func, params)
            state = self.update_state(state, key_update, results, params)
            params_hist.append(params)
            results_hist.append(results)
        return state, (params_hist, results_hist)

    def optimize_scan(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        func: Callable,
        n_iterations: int,
    ) -> tuple[OptimizerState, tuple[struct.PyTreeNode, jax.Array]]:
        """
        Like optimize, but uses jax.lax.scan for the inner loop.

        Requires func to be JAX-traceable (jit-compilable). Returns stacked
        arrays instead of lists: params_hist is a pytree where each leaf has
        a leading n_iterations dimension, and results_hist is a stacked array
        of shape (n_iterations, ...).
        """
        _validate_func(func)
        # Run one step outside scan to determine pytree structure and result shape.
        key, key_get, key_func, key_update = jax.random.split(key, 4)
        params0 = self.get_next_params(state, key_get, None, None)
        results0 = func(key_func, params0)
        state = self.update_state(state, key_update, results0, params0)
        # Save step-0 outputs before scan overwrites these names via carry.
        first_params, first_results = params0, results0

        def step(carry, _):
            state, key, params, results = carry
            key, key_get, key_func, key_update = jax.random.split(key, 4)
            params = self.get_next_params(state, key_get, params, results)
            results = func(key_func, params)
            state = self.update_state(state, key_update, results, params)
            return (state, key, params, results), (
                params,
                results,
            )

        (final_state, _, _, _), (params_hist, results_hist) = jax.lax.scan(
            step,
            (state, key, params0, results0),
            None,
            length=n_iterations - 1,
        )

        # Prepend step 0 so the output has n_iterations total entries.
        params_hist = jax.tree.map(
            lambda first, rest: jnp.concatenate([first[None], rest]),
            first_params,
            params_hist,
        )
        results_hist = jnp.concatenate([first_results[None], results_hist])
        return final_state, (params_hist, results_hist)

    def update_state(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        results: jax.Array,
        params: struct.PyTreeNode = None,
    ) -> OptimizerState:
        """
        Updates the optimizer state based on the results of the function.
        """
        raise NotImplementedError

    def get_next_params(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
        params: struct.PyTreeNode = None,
        results: jax.Array = None,
    ) -> struct.PyTreeNode:
        """
        Gets the next parameters to sample from the space.
        params and results are the previous iteration's values (None on first call).
        """
        raise NotImplementedError
