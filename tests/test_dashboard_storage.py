import sqlite3
from pathlib import Path

import numpy as np
import pytest

from hyperoptax.dashboard.models import (
    Direction,
    FilterOperator,
    SortDirection,
    StudyConfig,
    StudyStatus,
    TrialFilter,
    TrialInput,
    TrialQuery,
    TrialSort,
    TrialState,
    utc_now,
)
from hyperoptax.dashboard.queries import (
    InvalidQueryError,
    StudyQueryService,
    flatten_parameter_fields,
    split_parameter_path,
)
from hyperoptax.dashboard.recorder import SQLiteRecorder
from hyperoptax.dashboard.sqlite_store import SQLiteStore
from hyperoptax.dashboard.store import (
    BatchConflictError,
    InvalidStatusTransitionError,
    SchemaVersionError,
)
from hyperoptax.recording import BatchCompleted


def _study_config(
    *,
    n_parallel: int = 2,
    direction: Direction = Direction.MAXIMIZE,
) -> StudyConfig:
    return StudyConfig(
        name="dashboard test",
        direction=direction,
        optimizer_name="RandomSearch",
        optimizer_config={"n_parallel": n_parallel},
        search_space={"group": {"learning.rate": {"scale": "log"}}},
        n_parallel=n_parallel,
        tags=("test",),
        metadata={"suite": "storage"},
    )


def _event(
    batch_index: int,
    results: list[float],
    *,
    params: dict | None = None,
) -> BatchCompleted:
    if params is None:
        params = {"x": np.arange(len(results)) + batch_index * len(results)}
    return BatchCompleted(
        batch_index=batch_index,
        params=params,
        results=np.asarray(results),
    )


def _trial(
    trial_id: str,
    *,
    batch_index: int,
    batch_slot: int,
    objective: float,
) -> TrialInput:
    return TrialInput(
        trial_id=trial_id,
        batch_index=batch_index,
        batch_slot=batch_slot,
        evaluation_index=batch_index * 2 + batch_slot,
        state=TrialState.COMPLETED,
        params={"x": batch_index * 2 + batch_slot},
        objective_value=objective,
        finished_at=utc_now(),
    )


def test_recorder_commits_nested_batch_and_advances_revision(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite3"
    recorder = SQLiteRecorder(database, study=_study_config())

    with recorder:
        recorder(
            _event(
                0,
                [0.25, 0.75],
                params={
                    "model": {
                        "learning.rate": np.asarray([0.01, 0.02]),
                        "schedule": np.asarray([[1, 2], [3, 4]]),
                    },
                    "width": np.asarray([16, 32]),
                },
            )
        )

        service = StudyQueryService(database)
        changes = service.get_changes(recorder.study_id, 0)
        assert changes.study_revision == 1
        assert changes.study_status == StudyStatus.RUNNING
        assert len(changes.trials) == 2
        assert {trial.created_revision for trial in changes.trials} == {1}
        assert changes.trials[0].params == {
            "model": {"learning.rate": 0.01, "schedule": [1, 2]},
            "width": 16,
        }
        assert changes.trials[1].evaluation_index == 1

        recorder(_event(1, [1.0, 1.5]))
        incremental = service.get_changes(recorder.study_id, 1)
        assert incremental.study_revision == 2
        assert [trial.batch_index for trial in incremental.trials] == [1, 1]

    assert recorder.study.status == StudyStatus.COMPLETED
    assert recorder.study.finished_at is not None
    assert recorder.study.revision == 2


def test_recorder_persists_callback_batch_wall_time(tmp_path: Path) -> None:
    recorder = SQLiteRecorder(tmp_path / "timing.sqlite3", study=_study_config())
    event = BatchCompleted(
        batch_index=0,
        params={"x": np.asarray([0.25, 0.75])},
        results=np.asarray([1.0, 2.0]),
        duration_seconds=1.25,
    )

    with recorder:
        recorder(event)
        trials = recorder.store.list_trials(recorder.study_id)

    assert {trial.duration_seconds for trial in trials} == {1.25}
    assert all(trial.started_at is not None for trial in trials)
    assert all(trial.finished_at is not None for trial in trials)
    assert all(
        (trial.finished_at - trial.started_at).total_seconds() == pytest.approx(1.25)
        for trial in trials
    )


def test_recorder_retry_is_idempotent_and_conflicts_are_explicit(
    tmp_path: Path,
) -> None:
    recorder = SQLiteRecorder(tmp_path / "retry.sqlite3", study=_study_config())
    event = _event(0, [1.0, 2.0])

    with recorder:
        recorder(event)
        recorder(event)
        assert recorder.study.revision == 1
        assert len(recorder.store.list_trials(recorder.study_id)) == 2

        with pytest.raises(BatchConflictError, match="different trial data"):
            recorder(_event(0, [1.0, 9.0]))

        assert recorder.study.revision == 1
        assert [
            trial.objective_value
            for trial in recorder.store.list_trials(recorder.study_id)
        ] == [1.0, 2.0]


def test_pareto_front_is_direction_aware_and_excludes_dominated_trials(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "pareto.sqlite3")
    study = _study_config(n_parallel=4).create_study(status=StudyStatus.RUNNING)
    store.create_study(study)
    candidates = (
        ("fast", 1.0, 1.0),
        ("balanced", 2.0, 3.0),
        ("dominated", 3.0, 2.0),
        ("accurate", 4.0, 4.0),
    )
    store.append_batch(
        study.study_id,
        0,
        tuple(
            TrialInput(
                trial_id=trial_id,
                batch_index=0,
                batch_slot=index,
                evaluation_index=index,
                state=TrialState.COMPLETED,
                params={"width": index + 1},
                objective_value=objective,
                duration_seconds=duration,
                finished_at=utc_now(),
            )
            for index, (trial_id, duration, objective) in enumerate(candidates)
        ),
    )

    service = StudyQueryService(store)
    front = service.pareto_front(study.study_id)

    assert [point.trial_id for point in front.points] == [
        "fast",
        "balanced",
        "accurate",
    ]
    assert front.trials_considered == 4
    assert front.x_direction is Direction.MINIMIZE
    assert front.y_direction is Direction.MAXIMIZE

    minimise_front = service.pareto_front(
        study.study_id,
        y_direction=Direction.MINIMIZE,
    )
    assert [point.trial_id for point in minimise_front.points] == ["fast"]

    with pytest.raises(InvalidQueryError, match="two different fields"):
        service.pareto_front(
            study.study_id,
            x_field="objective_value",
            y_field="objective_value",
        )


def test_batch_insert_rolls_back_all_rows_and_revision_on_identity_error(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "atomic.sqlite3")
    study = _study_config().create_study(status=StudyStatus.RUNNING)
    store.create_study(study)
    store.append_batch(
        study.study_id,
        0,
        (
            _trial("shared-id", batch_index=0, batch_slot=0, objective=1.0),
            _trial("batch-0-slot-1", batch_index=0, batch_slot=1, objective=2.0),
        ),
    )

    with pytest.raises(BatchConflictError, match="identity constraints"):
        store.append_batch(
            study.study_id,
            1,
            (
                _trial("new-id", batch_index=1, batch_slot=0, objective=3.0),
                _trial("shared-id", batch_index=1, batch_slot=1, objective=4.0),
            ),
        )

    assert store.get_study(study.study_id).revision == 1
    trials = store.list_trials(study.study_id)
    assert len(trials) == 2
    assert all(trial.batch_index == 0 for trial in trials)


def test_context_failure_marks_study_failed_and_cached_study_survives_close(
    tmp_path: Path,
) -> None:
    recorder = SQLiteRecorder(tmp_path / "failed.sqlite3", study=_study_config())

    with pytest.raises(RuntimeError, match="objective failed"):
        with recorder:
            recorder(_event(0, [1.0, 2.0]))
            raise RuntimeError("objective failed")

    assert recorder.study.status == StudyStatus.FAILED
    assert recorder.study.finished_at is not None
    assert recorder.study.revision == 1
    with pytest.raises(RuntimeError, match="cannot be reused"):
        recorder.__enter__()


def test_recorder_validates_order_and_host_batch_shapes(tmp_path: Path) -> None:
    recorder = SQLiteRecorder(tmp_path / "validation.sqlite3", study=_study_config())

    with recorder:
        with pytest.raises(ValueError, match="first recorded batch_index must be 0"):
            recorder(_event(1, [1.0, 2.0]))
        with pytest.raises(ValueError, match="result count"):
            recorder(_event(0, [1.0]))
        with pytest.raises(ValueError, match="leading n_parallel axis"):
            recorder(
                _event(
                    0,
                    [1.0, 2.0],
                    params={"x": np.asarray(3.0)},
                )
            )

        recorder(_event(0, [1.0, 2.0]))
        with pytest.raises(ValueError, match="advance by one"):
            recorder(_event(2, [3.0, 4.0]))


def test_schema_migration_is_repeatable_and_rejects_newer_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "migration.sqlite3"
    SQLiteStore(database)
    SQLiteStore(database)

    with sqlite3.connect(database) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert versions == [(1,)]
        connection.execute("UPDATE schema_migrations SET version = 99")

    with pytest.raises(SchemaVersionError, match="newer than supported"):
        SQLiteStore(database)


def test_study_status_transitions_are_validated(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "status.sqlite3")
    study = _study_config().create_study()
    store.create_study(study)

    running = store.set_study_status(study.study_id, StudyStatus.RUNNING)
    assert running.status == StudyStatus.RUNNING
    completed = store.set_study_status(study.study_id, StudyStatus.COMPLETED)
    assert completed.status == StudyStatus.COMPLETED
    assert completed.finished_at is not None
    with pytest.raises(InvalidStatusTransitionError, match="cannot transition"):
        store.set_study_status(study.study_id, StudyStatus.RUNNING)


def test_query_filters_sorts_missing_values_and_escaped_paths(tmp_path: Path) -> None:
    database = tmp_path / "query.sqlite3"
    recorder = SQLiteRecorder(database, study=_study_config())
    with recorder:
        recorder(
            _event(
                0,
                [1.0, 3.0],
                params={
                    "learning.rate": np.asarray([0.1, 0.2]),
                    "nested": {"width": np.asarray([16, 32])},
                },
            )
        )
        recorder(
            _event(
                1,
                [2.0, 4.0],
                params={"learning.rate": np.asarray([0.3, 0.4])},
            )
        )

    service = StudyQueryService(database)
    learning_rate = r"params.learning\.rate"
    result = service.query_trials(
        recorder.study_id,
        TrialQuery(
            columns=("trial_id", "objective_value", learning_rate),
            filters=(
                TrialFilter(
                    field=learning_rate,
                    op=FilterOperator.BETWEEN,
                    value=[0.15, 0.35],
                ),
            ),
            sort=(
                TrialSort(
                    field="objective_value",
                    direction=SortDirection.DESC,
                ),
            ),
        ),
    )
    assert result.study_revision == 2
    assert result.rows_total == 2
    assert result.rows_returned == 2
    assert [row["objective_value"] for row in result.rows] == [3.0, 2.0]
    assert [row[learning_rate] for row in result.rows] == [0.2, 0.3]
    assert not result.sampled

    missing = service.query_trials(
        recorder.study_id,
        TrialQuery(
            filters=(
                TrialFilter(
                    field="params.nested.width",
                    op=FilterOperator.IS_MISSING,
                ),
            )
        ),
    )
    assert missing.rows_total == 2
    assert {row["batch_index"] for row in missing.rows} == {1}

    limited = service.query_trials(recorder.study_id, TrialQuery(limit=1))
    assert limited.rows_total == 4
    assert limited.rows_returned == 1
    assert limited.point_limit == 1

    with pytest.raises(InvalidQueryError, match="unknown trial fields"):
        service.query_trials(
            recorder.study_id,
            TrialQuery(columns=("params.does_not_exist",)),
        )


def test_parameter_path_escaping_round_trips() -> None:
    params = {"a.b": {r"c\d": 3}, "items": [4, 5]}
    flattened = flatten_parameter_fields(params)

    dotted = r"params.a\.b.c\\d"
    assert flattened[dotted] == 3
    assert split_parameter_path(dotted) == ("a.b", r"c\d")
    assert flattened["params.items.1"] == 5


@pytest.mark.parametrize(
    ("direction", "expected_curve", "expected_best"),
    [
        (Direction.MAXIMIZE, [3.0, 3.0, 3.0], 3.0),
        (Direction.MINIMIZE, [3.0, 1.0, 1.0], 1.0),
    ],
)
def test_best_so_far_respects_study_direction(
    tmp_path: Path,
    direction: Direction,
    expected_curve: list[float],
    expected_best: float,
) -> None:
    database = tmp_path / f"{direction.value}.sqlite3"
    recorder = SQLiteRecorder(
        database,
        study=_study_config(n_parallel=1, direction=direction),
    )
    with recorder:
        for batch_index, objective in enumerate([3.0, 1.0, 2.0]):
            recorder(_event(batch_index, [objective]))

    service = StudyQueryService(database)
    assert [
        point.best_objective_value for point in service.best_so_far(recorder.study_id)
    ] == expected_curve
    assert service.best_trial(recorder.study_id).objective_value == expected_best
    description = service.describe_study(recorder.study_id)
    assert description.best_trial.objective_value == expected_best
    assert description.trial_counts[TrialState.COMPLETED.value] == 3


def test_change_query_returns_committed_tail_and_terminal_status(
    tmp_path: Path,
) -> None:
    database = tmp_path / "changes.sqlite3"
    recorder = SQLiteRecorder(database, study=_study_config())
    service = StudyQueryService(database)

    with recorder:
        recorder(_event(0, [1.0, 2.0]))
        first = service.get_changes(recorder.study_id, 0)
        assert first.study_revision == 1
        assert first.study_status == StudyStatus.RUNNING
        assert len(first.trials) == 2

        recorder(_event(1, [3.0, 4.0]))
        second = service.get_changes(recorder.study_id, 1)
        assert second.study_revision == 2
        assert {trial.batch_index for trial in second.trials} == {1}

    terminal = service.get_changes(recorder.study_id, 2)
    assert terminal.study_revision == 2
    assert terminal.study_status == StudyStatus.COMPLETED
    assert terminal.trials == ()
    with pytest.raises(InvalidQueryError, match="newer than current revision"):
        service.get_changes(recorder.study_id, 3)
