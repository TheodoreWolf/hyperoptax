"""Dependency-light recording types for Python-loop optimization."""

from dataclasses import dataclass
from typing import Callable, TypeAlias

from jaxtyping import ArrayLike, PyTree

__all__ = ["BatchCallback", "BatchCompleted"]


@dataclass(frozen=True)
class BatchCompleted:
    """Host-valued parameters and results from one completed optimizer batch."""

    batch_index: int
    params: PyTree[ArrayLike]
    results: ArrayLike
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")


BatchCallback: TypeAlias = Callable[[BatchCompleted], None]
