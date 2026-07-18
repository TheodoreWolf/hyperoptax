"""Storage protocol and domain errors for dashboard records."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from hyperoptax.dashboard.models import (
    BatchWriteResult,
    Study,
    StudyStatus,
    Trial,
    TrialInput,
)


class DashboardStorageError(RuntimeError):
    """Base error for durable dashboard state."""


class SchemaVersionError(DashboardStorageError):
    """Raised when a database schema cannot be opened safely."""


class StudyNotFoundError(DashboardStorageError, KeyError):
    """Raised when a requested study does not exist."""


class TrialNotFoundError(DashboardStorageError, KeyError):
    """Raised when a requested trial does not exist in a study."""


class StudyAlreadyExistsError(DashboardStorageError):
    """Raised when creating a study would replace an existing one."""


class InvalidStatusTransitionError(DashboardStorageError):
    """Raised when a study lifecycle transition is invalid."""


class BatchConflictError(DashboardStorageError):
    """Raised when an idempotency key is reused for different batch data."""


@runtime_checkable
class StudyStore(Protocol):
    """Storage-independent contract consumed by :class:`StudyQueryService`."""

    @property
    def path(self) -> Path:
        """Location of the store when one exists."""

    def create_study(self, study: Study) -> Study:
        """Insert a new study without replacing an existing identity."""

    def get_study(self, study_id: str) -> Study:
        """Return one study or raise :class:`StudyNotFoundError`."""

    def list_studies(self) -> tuple[Study, ...]:
        """Return studies ordered newest first."""

    def set_study_status(
        self,
        study_id: str,
        status: StudyStatus,
        *,
        finished_at: datetime | None = None,
    ) -> Study:
        """Apply one validated lifecycle transition."""

    def append_batch(
        self,
        study_id: str,
        batch_index: int,
        trials: Sequence[TrialInput],
    ) -> BatchWriteResult:
        """Atomically append one idempotent batch and advance its revision."""

    def list_trials(self, study_id: str) -> tuple[Trial, ...]:
        """Return all trials in stable evaluation order."""

    def get_study_with_trials(
        self,
        study_id: str,
        *,
        updated_since: int | None = None,
    ) -> tuple[Study, tuple[Trial, ...]]:
        """Read study metadata and trials from one consistent snapshot."""

    def get_trial(self, study_id: str, trial_id: str) -> Trial:
        """Return a trial, raising :class:`StudyNotFoundError` if absent."""

    def get_trials_updated_since(
        self, study_id: str, since_revision: int
    ) -> tuple[Trial, ...]:
        """Return every trial with ``updated_revision > since_revision``."""

    def close(self) -> None:
        """Release persistent resources, if the implementation has any."""


__all__ = [
    "BatchConflictError",
    "DashboardStorageError",
    "InvalidStatusTransitionError",
    "SchemaVersionError",
    "StudyAlreadyExistsError",
    "StudyNotFoundError",
    "StudyStore",
    "TrialNotFoundError",
]
