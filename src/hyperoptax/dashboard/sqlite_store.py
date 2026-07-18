"""Host-local SQLite implementation of the dashboard study store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from hyperoptax.dashboard.models import (
    SCHEMA_VERSION,
    BatchWriteResult,
    Direction,
    JsonValue,
    Study,
    StudyStatus,
    Trial,
    TrialInput,
    TrialState,
    normalize_datetime,
    to_json_value,
    utc_now,
)
from hyperoptax.dashboard.store import (
    BatchConflictError,
    InvalidStatusTransitionError,
    SchemaVersionError,
    StudyAlreadyExistsError,
    StudyNotFoundError,
    TrialNotFoundError,
)

_MIGRATION_1 = (
    """
    CREATE TABLE studies (
        study_id TEXT PRIMARY KEY,
        sweep_id TEXT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        direction TEXT NOT NULL,
        optimizer_name TEXT NOT NULL,
        optimizer_config_json TEXT NOT NULL,
        search_space_json TEXT NOT NULL,
        seed_json TEXT NOT NULL,
        n_parallel INTEGER NOT NULL CHECK (n_parallel >= 1),
        code_version TEXT NOT NULL,
        source_revision TEXT,
        variant TEXT,
        replicate TEXT,
        tags_json TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        schema_version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE trials (
        trial_id TEXT PRIMARY KEY,
        study_id TEXT NOT NULL REFERENCES studies(study_id),
        batch_index INTEGER NOT NULL CHECK (batch_index >= 0),
        batch_slot INTEGER NOT NULL CHECK (batch_slot >= 0),
        evaluation_index INTEGER,
        state TEXT NOT NULL,
        params_json TEXT NOT NULL,
        objective_value REAL,
        started_at TEXT,
        finished_at TEXT,
        duration_seconds REAL,
        seed_json TEXT NOT NULL,
        worker TEXT,
        error_json TEXT,
        metadata_json TEXT NOT NULL,
        created_revision INTEGER NOT NULL CHECK (created_revision >= 1),
        updated_revision INTEGER NOT NULL CHECK (
            updated_revision >= created_revision
        ),
        UNIQUE (study_id, batch_index, batch_slot)
    )
    """,
    "CREATE INDEX idx_trials_study ON trials(study_id)",
    """
    CREATE INDEX idx_trials_study_evaluation
    ON trials(study_id, evaluation_index)
    """,
    "CREATE INDEX idx_trials_study_state ON trials(study_id, state)",
    """
    CREATE INDEX idx_trials_study_objective
    ON trials(study_id, objective_value)
    """,
    """
    CREATE INDEX idx_trials_study_updated_revision
    ON trials(study_id, updated_revision)
    """,
)

_VALID_TRANSITIONS: dict[StudyStatus, frozenset[StudyStatus]] = {
    StudyStatus.CREATED: frozenset(
        {StudyStatus.RUNNING, StudyStatus.FAILED, StudyStatus.CANCELLED}
    ),
    StudyStatus.RUNNING: frozenset(
        {StudyStatus.COMPLETED, StudyStatus.FAILED, StudyStatus.CANCELLED}
    ),
    StudyStatus.COMPLETED: frozenset(),
    StudyStatus.FAILED: frozenset(),
    StudyStatus.CANCELLED: frozenset(),
}


def _json_dumps(value: Any) -> str:
    return json.dumps(
        to_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_loads(value: str) -> JsonValue:
    return json.loads(value)


def _datetime_to_db(value: datetime | None) -> str | None:
    if value is None:
        return None
    return normalize_datetime(value, field_name="timestamp").isoformat()


def _datetime_from_db(value: str | None) -> datetime | None:
    if value is None:
        return None
    return normalize_datetime(datetime.fromisoformat(value), field_name="timestamp")


class SQLiteStore:
    """A short-transaction SQLite store for one local recorder and readers.

    Connections are opened per operation so the query layer never holds a read
    transaction while serializing results. Journal mode is intentionally left
    at SQLite's default; WAL is not implied by this implementation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        if busy_timeout_seconds < 0:
            raise ValueError("busy_timeout_seconds must not be negative")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and self._path.is_dir():
            raise IsADirectoryError(self._path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = round(busy_timeout_seconds * 1_000)
        self._migrate()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms:d}")
        return connection

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._write_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
            current_version = int(row["version"])
            if current_version > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database schema {current_version} is newer than supported "
                    f"schema {SCHEMA_VERSION}"
                )
            if current_version < 1:
                for statement in _MIGRATION_1:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, _datetime_to_db(utc_now())),
                )

    def create_study(self, study: Study) -> Study:
        if study.schema_version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"study schema {study.schema_version} does not match store schema "
                f"{SCHEMA_VERSION}"
            )
        values = (
            study.study_id,
            study.sweep_id,
            study.name,
            _datetime_to_db(study.created_at),
            _datetime_to_db(study.finished_at),
            study.status.value,
            study.direction.value,
            study.optimizer_name,
            _json_dumps(study.optimizer_config),
            _json_dumps(study.search_space),
            _json_dumps(study.seed),
            study.n_parallel,
            study.code_version,
            study.source_revision,
            study.variant,
            study.replicate,
            _json_dumps(study.tags),
            _json_dumps(study.metadata),
            study.revision,
            study.schema_version,
        )
        try:
            with self._write_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO studies (
                        study_id, sweep_id, name, created_at, finished_at,
                        status, direction, optimizer_name,
                        optimizer_config_json, search_space_json, seed_json,
                        n_parallel, code_version, source_revision, variant,
                        replicate, tags_json, metadata_json, revision,
                        schema_version
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values,
                )
        except sqlite3.IntegrityError as error:
            raise StudyAlreadyExistsError(
                f"study already exists: {study.study_id}"
            ) from error
        return self.get_study(study.study_id)

    def get_study(self, study_id: str) -> Study:
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
        if row is None:
            raise StudyNotFoundError(f"study not found: {study_id}")
        return self._study_from_row(row)

    def list_studies(self) -> tuple[Study, ...]:
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM studies ORDER BY created_at DESC, study_id"
            ).fetchall()
        return tuple(self._study_from_row(row) for row in rows)

    def set_study_status(
        self,
        study_id: str,
        status: StudyStatus,
        *,
        finished_at: datetime | None = None,
    ) -> Study:
        target = StudyStatus(status)
        with self._write_connection() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
            if row is None:
                raise StudyNotFoundError(f"study not found: {study_id}")
            current = StudyStatus(row["status"])
            if current == target:
                return self._study_from_row(row)
            if target not in _VALID_TRANSITIONS[current]:
                raise InvalidStatusTransitionError(
                    f"cannot transition study {study_id} from "
                    f"{current.value} to {target.value}"
                )
            terminal = target in {
                StudyStatus.COMPLETED,
                StudyStatus.FAILED,
                StudyStatus.CANCELLED,
            }
            completed_at = finished_at or utc_now() if terminal else None
            connection.execute(
                """
                UPDATE studies
                SET status = ?, finished_at = ?
                WHERE study_id = ?
                """,
                (target.value, _datetime_to_db(completed_at), study_id),
            )
            updated = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
            return self._study_from_row(updated)

    def append_batch(
        self,
        study_id: str,
        batch_index: int,
        trials: Sequence[TrialInput],
    ) -> BatchWriteResult:
        trial_items = tuple(trials)
        self._validate_batch(batch_index, trial_items)
        try:
            with self._write_connection() as connection:
                study_row = connection.execute(
                    "SELECT * FROM studies WHERE study_id = ?", (study_id,)
                ).fetchone()
                if study_row is None:
                    raise StudyNotFoundError(f"study not found: {study_id}")
                study = self._study_from_row(study_row)

                existing_rows = connection.execute(
                    """
                    SELECT * FROM trials
                    WHERE study_id = ? AND batch_index = ?
                    ORDER BY batch_slot
                    """,
                    (study_id, batch_index),
                ).fetchall()
                if existing_rows:
                    existing = tuple(self._trial_from_row(row) for row in existing_rows)
                    if not self._batch_matches(existing, trial_items):
                        raise BatchConflictError(
                            f"batch ({study_id}, {batch_index}) already contains "
                            "different trial data"
                        )
                    return BatchWriteResult(
                        revision=study.revision,
                        inserted=False,
                        trial_count=len(existing),
                    )

                if study.status not in {StudyStatus.CREATED, StudyStatus.RUNNING}:
                    raise InvalidStatusTransitionError(
                        f"cannot append a new batch to {study.status.value} study "
                        f"{study_id}"
                    )
                expected_slots = set(range(study.n_parallel))
                actual_slots = {trial.batch_slot for trial in trial_items}
                if actual_slots != expected_slots:
                    raise ValueError(
                        f"batch slots must be exactly 0..{study.n_parallel - 1}; "
                        f"received {sorted(actual_slots)}"
                    )

                revision = study.revision + 1
                connection.execute(
                    "UPDATE studies SET revision = ? WHERE study_id = ?",
                    (revision, study_id),
                )
                for trial in trial_items:
                    self._insert_trial(connection, study_id, revision, trial)
                return BatchWriteResult(
                    revision=revision,
                    inserted=True,
                    trial_count=len(trial_items),
                )
        except sqlite3.IntegrityError as error:
            raise BatchConflictError(
                f"batch ({study_id}, {batch_index}) violates trial identity constraints"
            ) from error

    @staticmethod
    def _validate_batch(batch_index: int, trials: tuple[TrialInput, ...]) -> None:
        if batch_index < 0:
            raise ValueError("batch_index must not be negative")
        if not trials:
            raise ValueError("a completed batch must contain at least one trial")
        if any(trial.batch_index != batch_index for trial in trials):
            raise ValueError("every trial must use the append batch_index")
        slots = [trial.batch_slot for trial in trials]
        if len(slots) != len(set(slots)):
            raise ValueError("batch slots must be unique")
        trial_ids = [trial.trial_id for trial in trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("trial IDs must be unique within a batch")

    @staticmethod
    def _batch_matches(
        existing: tuple[Trial, ...], incoming: tuple[TrialInput, ...]
    ) -> bool:
        if len(existing) != len(incoming):
            return False
        incoming_by_slot = {trial.batch_slot: trial for trial in incoming}
        for stored in existing:
            candidate = incoming_by_slot.get(stored.batch_slot)
            if candidate is None:
                return False
            # Completion timestamps are assigned at observation time and may
            # differ on an explicit retry. Durable identity and result content
            # must otherwise be identical.
            if (
                stored.trial_id != candidate.trial_id
                or stored.batch_index != candidate.batch_index
                or stored.evaluation_index != candidate.evaluation_index
                or stored.state != candidate.state
                or stored.params != candidate.params
                or stored.objective_value != candidate.objective_value
                or stored.duration_seconds != candidate.duration_seconds
                or stored.seed != candidate.seed
                or stored.worker != candidate.worker
                or stored.error != candidate.error
                or stored.metadata != candidate.metadata
            ):
                return False
        return True

    @staticmethod
    def _insert_trial(
        connection: sqlite3.Connection,
        study_id: str,
        revision: int,
        trial: TrialInput,
    ) -> None:
        connection.execute(
            """
            INSERT INTO trials (
                trial_id, study_id, batch_index, batch_slot, evaluation_index,
                state, params_json, objective_value, started_at, finished_at,
                duration_seconds, seed_json, worker, error_json, metadata_json,
                created_revision, updated_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.trial_id,
                study_id,
                trial.batch_index,
                trial.batch_slot,
                trial.evaluation_index,
                trial.state.value,
                _json_dumps(trial.params),
                trial.objective_value,
                _datetime_to_db(trial.started_at),
                _datetime_to_db(trial.finished_at),
                trial.duration_seconds,
                _json_dumps(trial.seed),
                trial.worker,
                _json_dumps(trial.error) if trial.error is not None else None,
                _json_dumps(trial.metadata),
                revision,
                revision,
            ),
        )

    def list_trials(self, study_id: str) -> tuple[Trial, ...]:
        _, trials = self.get_study_with_trials(study_id)
        return trials

    def get_study_with_trials(
        self,
        study_id: str,
        *,
        updated_since: int | None = None,
    ) -> tuple[Study, tuple[Trial, ...]]:
        if updated_since is not None and updated_since < 0:
            raise ValueError("updated_since must not be negative")
        with self._read_connection() as connection:
            connection.execute("BEGIN")
            study_row = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
            if study_row is None:
                raise StudyNotFoundError(f"study not found: {study_id}")
            where_revision = "" if updated_since is None else "AND updated_revision > ?"
            parameters: tuple[Any, ...] = (
                (study_id,) if updated_since is None else (study_id, updated_since)
            )
            rows = connection.execute(
                f"""
                SELECT * FROM trials
                WHERE study_id = ? {where_revision}
                ORDER BY evaluation_index IS NULL, evaluation_index,
                         batch_index, batch_slot
                """,
                parameters,
            ).fetchall()
            connection.commit()
        return self._study_from_row(study_row), tuple(
            self._trial_from_row(row) for row in rows
        )

    def get_trial(self, study_id: str, trial_id: str) -> Trial:
        self.get_study(study_id)
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM trials
                WHERE study_id = ? AND trial_id = ?
                """,
                (study_id, trial_id),
            ).fetchone()
        if row is None:
            raise TrialNotFoundError(
                f"trial {trial_id!r} not found in study {study_id}"
            )
        return self._trial_from_row(row)

    def get_trials_updated_since(
        self, study_id: str, since_revision: int
    ) -> tuple[Trial, ...]:
        if since_revision < 0:
            raise ValueError("since_revision must not be negative")
        _, trials = self.get_study_with_trials(study_id, updated_since=since_revision)
        return trials

    def close(self) -> None:
        """No-op because operations use short-lived connections."""

    @staticmethod
    def _study_from_row(row: sqlite3.Row) -> Study:
        optimizer_config = _json_loads(row["optimizer_config_json"])
        search_space = _json_loads(row["search_space_json"])
        metadata = _json_loads(row["metadata_json"])
        tags = _json_loads(row["tags_json"])
        if not isinstance(optimizer_config, dict):
            raise SchemaVersionError("optimizer_config_json is not an object")
        if not isinstance(search_space, dict):
            raise SchemaVersionError("search_space_json is not an object")
        if not isinstance(metadata, dict):
            raise SchemaVersionError("metadata_json is not an object")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise SchemaVersionError("tags_json is not a string list")
        return Study(
            study_id=row["study_id"],
            sweep_id=row["sweep_id"],
            name=row["name"],
            created_at=_datetime_from_db(row["created_at"]),
            finished_at=_datetime_from_db(row["finished_at"]),
            status=StudyStatus(row["status"]),
            direction=Direction(row["direction"]),
            optimizer_name=row["optimizer_name"],
            optimizer_config=optimizer_config,
            search_space=search_space,
            seed=_json_loads(row["seed_json"]),
            n_parallel=int(row["n_parallel"]),
            code_version=row["code_version"],
            source_revision=row["source_revision"],
            variant=row["variant"],
            replicate=row["replicate"],
            tags=tuple(tags),
            metadata=metadata,
            revision=int(row["revision"]),
            schema_version=int(row["schema_version"]),
        )

    @staticmethod
    def _trial_from_row(row: sqlite3.Row) -> Trial:
        error = _json_loads(row["error_json"]) if row["error_json"] else None
        metadata = _json_loads(row["metadata_json"])
        if error is not None and not isinstance(error, dict):
            raise SchemaVersionError("error_json is not an object")
        if not isinstance(metadata, dict):
            raise SchemaVersionError("metadata_json is not an object")
        return Trial(
            trial_id=row["trial_id"],
            study_id=row["study_id"],
            batch_index=int(row["batch_index"]),
            batch_slot=int(row["batch_slot"]),
            evaluation_index=(
                int(row["evaluation_index"])
                if row["evaluation_index"] is not None
                else None
            ),
            state=TrialState(row["state"]),
            params=_json_loads(row["params_json"]),
            objective_value=(
                float(row["objective_value"])
                if row["objective_value"] is not None
                else None
            ),
            started_at=_datetime_from_db(row["started_at"]),
            finished_at=_datetime_from_db(row["finished_at"]),
            duration_seconds=(
                float(row["duration_seconds"])
                if row["duration_seconds"] is not None
                else None
            ),
            seed=_json_loads(row["seed_json"]),
            worker=row["worker"],
            error=error,
            metadata=metadata,
            created_revision=int(row["created_revision"]),
            updated_revision=int(row["updated_revision"]),
        )


__all__ = ["SQLiteStore"]
