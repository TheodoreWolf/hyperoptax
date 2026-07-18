"""SQLite recording paths for live batches and completed histories."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, TypeAlias
from uuid import UUID, uuid5

import jax
import numpy as np

from hyperoptax.dashboard.models import (
    Study,
    StudyConfig,
    StudyStatus,
    TrialInput,
    TrialState,
    to_json_value,
    utc_now,
)
from hyperoptax.dashboard.sqlite_store import SQLiteStore
from hyperoptax.recording import BatchCompleted

_HostBatch: TypeAlias = tuple[Any, np.ndarray]


def _trial_id(study_id: str, batch_index: int, batch_slot: int) -> str:
    namespace = UUID(study_id)
    return str(uuid5(namespace, f"batch:{batch_index}:slot:{batch_slot}"))


def _validate_param_batch(params: Any, n_parallel: int) -> None:
    for leaf in jax.tree_util.tree_leaves(params):
        array = np.asarray(leaf)
        if array.ndim == 0 or array.shape[0] != n_parallel:
            raise ValueError(
                "every parameter leaf must have a leading n_parallel axis; "
                f"received shape {array.shape}, expected {n_parallel}"
            )


def _completed_trials(
    study: Study,
    batch_index: int,
    params: Any,
    results: np.ndarray,
    *,
    finished_at: datetime | None,
    duration_seconds: float | None = None,
) -> tuple[TrialInput, ...]:
    started_at = (
        finished_at - timedelta(seconds=duration_seconds)
        if finished_at is not None and duration_seconds is not None
        else None
    )
    return tuple(
        TrialInput(
            trial_id=_trial_id(study.study_id, batch_index, batch_slot),
            batch_index=batch_index,
            batch_slot=batch_slot,
            evaluation_index=batch_index * study.n_parallel + batch_slot,
            state=TrialState.COMPLETED,
            params=to_json_value(
                jax.tree_util.tree_map(
                    lambda leaf: np.asarray(leaf)[batch_slot], params
                ),
                path="params",
            ),
            objective_value=float(results[batch_slot]),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            metadata=(
                {"duration_scope": "parallel_batch"}
                if duration_seconds is not None
                else {}
            ),
        )
        for batch_slot in range(study.n_parallel)
    )


def _normalize_list_history(
    params_history: Any,
    results_history: list[Any] | tuple[Any, ...],
    *,
    n_parallel: int,
) -> tuple[_HostBatch, ...]:
    if not isinstance(params_history, (list, tuple)):
        raise ValueError("list results_history requires list or tuple params_history")
    if len(params_history) != len(results_history):
        raise ValueError(
            "params_history and results_history must contain the same number "
            f"of batches; got {len(params_history)} and {len(results_history)}"
        )

    batches: list[_HostBatch] = []
    for batch_index, (params, results) in enumerate(
        zip(params_history, results_history, strict=True)
    ):
        result_array = np.asarray(results)
        if result_array.ndim != 1 or result_array.shape[0] != n_parallel:
            raise ValueError(
                "each results_history batch must have shape (n_parallel,); "
                f"batch {batch_index} has shape {result_array.shape}, expected "
                f"({n_parallel},)"
            )
        _validate_param_batch(params, n_parallel)
        batches.append((params, result_array))
    return tuple(batches)


def _normalize_stacked_history(
    params_history: Any,
    results_history: Any,
    *,
    n_parallel: int,
) -> tuple[_HostBatch, ...]:
    result_array = np.asarray(results_history)
    if result_array.ndim != 2 or result_array.shape[1] != n_parallel:
        raise ValueError(
            "stacked results_history must have shape (n_batches, n_parallel); "
            f"got {result_array.shape}, expected (*, {n_parallel})"
        )

    n_batches = result_array.shape[0]
    for leaf in jax.tree_util.tree_leaves(params_history):
        array = np.asarray(leaf)
        if array.ndim < 2 or array.shape[:2] != (n_batches, n_parallel):
            raise ValueError(
                "every stacked parameter leaf must have leading "
                "(n_batches, n_parallel) axes; "
                f"received shape {array.shape}, expected "
                f"({n_batches}, {n_parallel}, ...)"
            )

    return tuple(
        (
            jax.tree_util.tree_map(
                lambda leaf: np.asarray(leaf)[batch_index], params_history
            ),
            result_array[batch_index],
        )
        for batch_index in range(n_batches)
    )


def _normalize_history(
    params_history: Any,
    results_history: Any,
    *,
    n_parallel: int,
) -> tuple[_HostBatch, ...]:
    """Transfer one complete history to the host and validate its batch axes."""

    host_params, host_results = jax.device_get((params_history, results_history))
    if isinstance(host_results, (list, tuple)):
        return _normalize_list_history(host_params, host_results, n_parallel=n_parallel)
    return _normalize_stacked_history(host_params, host_results, n_parallel=n_parallel)


class SQLiteRecorder:
    """Record each completed optimizer batch in one SQLite transaction.

    The recorder is deliberately synchronous: after ``__call__`` returns, all
    trials in the batch and the corresponding study revision are committed.
    Use it as a context manager so study completion or failure is durable too.
    """

    def __init__(
        self,
        database: str | Path,
        *,
        study: StudyConfig | Study | Mapping[str, Any],
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        self.store = SQLiteStore(database, busy_timeout_seconds=busy_timeout_seconds)
        if isinstance(study, Mapping):
            study = StudyConfig(**study)
        if isinstance(study, StudyConfig):
            study_record = study.create_study(status=StudyStatus.RUNNING)
        elif isinstance(study, Study):
            if study.status == StudyStatus.CREATED:
                study_record = dataclasses.replace(
                    study, status=StudyStatus.RUNNING, finished_at=None
                )
            elif study.status == StudyStatus.RUNNING:
                study_record = study
            else:
                raise ValueError(
                    "SQLiteRecorder study must have created or running status"
                )
        else:  # pragma: no cover - guarded by the public annotation
            raise TypeError("study must be a StudyConfig, Study, or mapping")

        self._study = study_record
        self._active = False
        self._closed = False
        self._last_batch_index: int | None = None

    @property
    def study_id(self) -> str:
        return self._study.study_id

    @property
    def study(self) -> Study:
        """Return the latest durable study metadata."""

        if self._active:
            return self.store.get_study(self.study_id)
        return self._study

    def __enter__(self) -> "SQLiteRecorder":
        if self._closed:
            raise RuntimeError("SQLiteRecorder cannot be reused after exit")
        if self._active:
            raise RuntimeError("SQLiteRecorder is already active")
        self._study = self.store.create_study(self._study)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_value, traceback
        if not self._active:
            return False
        target = StudyStatus.COMPLETED if exc_type is None else StudyStatus.FAILED
        try:
            self._study = self.store.set_study_status(
                self.study_id, target, finished_at=utc_now()
            )
        finally:
            self._active = False
            self._closed = True
            self.store.close()
        return False

    def __call__(self, event: BatchCompleted) -> None:
        if not self._active:
            raise RuntimeError("SQLiteRecorder must be entered before recording")
        if not isinstance(event, BatchCompleted):
            raise TypeError("SQLiteRecorder expects a BatchCompleted event")
        self._validate_batch_order(event.batch_index)

        results = np.asarray(event.results)
        if results.ndim != 1:
            raise ValueError(
                f"BatchCompleted.results must be one-dimensional; got {results.shape}"
            )
        if results.shape[0] != self._study.n_parallel:
            raise ValueError(
                "BatchCompleted result count does not match study n_parallel: "
                f"{results.shape[0]} != {self._study.n_parallel}"
            )
        _validate_param_batch(event.params, self._study.n_parallel)

        finished_at = utc_now()
        trials = _completed_trials(
            self._study,
            event.batch_index,
            event.params,
            results,
            finished_at=finished_at,
            duration_seconds=event.duration_seconds,
        )
        self.store.append_batch(self.study_id, event.batch_index, trials)
        self._last_batch_index = event.batch_index

    def _validate_batch_order(self, batch_index: int) -> None:
        if batch_index < 0:
            raise ValueError("batch_index must not be negative")
        if self._last_batch_index is None:
            if batch_index != 0:
                raise ValueError(
                    f"first recorded batch_index must be 0; received {batch_index}"
                )
            return
        if batch_index not in {
            self._last_batch_index,
            self._last_batch_index + 1,
        }:
            raise ValueError(
                "batch_index must repeat the current batch for an idempotent retry "
                "or advance by one"
            )


def import_history(
    database: str | Path,
    *,
    study: StudyConfig | Mapping[str, Any],
    params_history: Any,
    results_history: Any,
    busy_timeout_seconds: float = 5.0,
) -> Study:
    """Import a completed ``optimize`` or ``optimize_scan`` history.

    The complete history is transferred to the host with one ``device_get``.
    Trial timing is left unset because neither history representation contains
    that information; the study timestamps describe this import operation.
    """

    if isinstance(study, Mapping):
        study = StudyConfig(**study)
    if not isinstance(study, StudyConfig):  # pragma: no cover - public annotation
        raise TypeError("study must be a StudyConfig or mapping")

    metadata = {
        **study.metadata,
        "recording_mode": "bulk_history",
        "trial_timing_available": False,
    }
    study_config = dataclasses.replace(study, metadata=metadata)
    batches = _normalize_history(
        params_history,
        results_history,
        n_parallel=study_config.n_parallel,
    )
    study_record = study_config.create_study(status=StudyStatus.RUNNING)
    trial_batches = tuple(
        _completed_trials(
            study_record,
            batch_index,
            params,
            results,
            finished_at=None,
        )
        for batch_index, (params, results) in enumerate(batches)
    )

    store = SQLiteStore(database, busy_timeout_seconds=busy_timeout_seconds)
    try:
        store.create_study(study_record)
        for batch_index, trials in enumerate(trial_batches):
            store.append_batch(study_record.study_id, batch_index, trials)
        return store.set_study_status(
            study_record.study_id,
            StudyStatus.COMPLETED,
        )
    finally:
        store.close()


__all__ = ["SQLiteRecorder", "import_history"]
