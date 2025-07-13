import jax.numpy as jnp
import pytest

from hyperoptax.base import BaseOptimizer
from hyperoptax.spaces import LinearSpace


class TestBaseOptimizer:
    def setup_method(self):
        self.optimizer = BaseOptimizer(
            domain={"x": LinearSpace(0, 1, 10)}, f=lambda x: x
        )

    def test_when_no_results_are_found(self):
        with pytest.raises(AssertionError):
            self.optimizer.max
        with pytest.raises(AssertionError):
            self.optimizer.min

    def test_max(self):
        # manually set the results
        self.optimizer.results = (
            self.optimizer.domain,
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )
        assert self.optimizer.max["target"] == 10

    def test_min(self):
        # manually set the results
        self.optimizer.results = (
            self.optimizer.domain,
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )
        assert self.optimizer.min["target"] == 1


# class TestDomain:
#     def test_domain(self):
#         domain = {
#             "x": LinearSpace(0, 1, 10),
#             "y": LinearSpace(0, 1, 10),
#             "z": LinearSpace(0, 1, 10),
#             "agent_kwargs": {
#                 "lr": LinearSpace(0, 1, 10),
#                 "batch_size": LinearSpace(0, 1, 10),
#             },
#         }

#         def f(x, y, z, agent_kwargs):
#             return x + y + z + agent_kwargs["lr"] + agent_kwargs["batch_size"]

#         f(**domain)
