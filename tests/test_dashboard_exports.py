import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

import hyperoptax.dashboard.export as export_module
import hyperoptax.dashboard.snapshot as snapshot_module
from hyperoptax.dashboard.export import (
    ExportFormat,
    export_study_data,
    trial_query_result_payload,
)
from hyperoptax.dashboard.models import (
    Direction,
    FilterOperator,
    SortDirection,
    StudyConfig,
    StudyStatus,
    TrialFilter,
    TrialQuery,
    TrialSort,
)
from hyperoptax.dashboard.queries import (
    DEFAULT_TRIAL_COLUMNS,
    StudyQueryService,
)
from hyperoptax.dashboard.recorder import SQLiteRecorder
from hyperoptax.dashboard.snapshot import snapshot_database
from hyperoptax.recording import BatchCompleted


def _config() -> StudyConfig:
    return StudyConfig(
        name="export test",
        direction=Direction.MAXIMIZE,
        optimizer_name="RandomSearch",
        n_parallel=2,
    )


def _event(batch_index: int, results: list[float]) -> BatchCompleted:
    first = batch_index * 2
    return BatchCompleted(
        batch_index=batch_index,
        params={
            "x": np.asarray([first, first + 1]),
            "group": {"width": np.asarray([16 + first, 17 + first])},
        },
        results=np.asarray(results),
    )


def _completed_database(tmp_path: Path) -> tuple[Path, str]:
    database = tmp_path / "results.sqlite3"
    recorder = SQLiteRecorder(database, study=_config())
    with recorder:
        recorder(_event(0, [0.1, 0.9]))
        recorder(_event(1, [0.5, 1.2]))
    return database, recorder.study_id


def test_snapshot_is_a_consistent_point_in_time_database(tmp_path: Path) -> None:
    database = tmp_path / "live.sqlite3"
    snapshot = tmp_path / "snapshot.sqlite3"
    recorder = SQLiteRecorder(database, study=_config())

    with recorder:
        recorder(_event(0, [0.1, 0.9]))
        result = snapshot_database(database, snapshot)
        recorder(_event(1, [0.5, 1.2]))

    assert result.output == snapshot.resolve()
    assert result.size_bytes == snapshot.stat().st_size
    snapshot_service = StudyQueryService(snapshot)
    snapshot_study = snapshot_service.describe_study(recorder.study_id).study
    assert snapshot_study.status == StudyStatus.RUNNING
    assert snapshot_study.revision == 1
    assert len(snapshot_service.query_trials(recorder.study_id).rows) == 2

    source_service = StudyQueryService(database)
    source_study = source_service.describe_study(recorder.study_id).study
    assert source_study.status == StudyStatus.COMPLETED
    assert source_study.revision == 2


def test_snapshot_rejects_same_path_and_existing_hard_link(tmp_path: Path) -> None:
    database, _ = _completed_database(tmp_path)

    with pytest.raises(ValueError, match="must be different"):
        snapshot_database(database, database, overwrite=True)

    hard_link = tmp_path / "same-inode.sqlite3"
    os.link(database, hard_link)
    with pytest.raises(ValueError, match="must be different"):
        snapshot_database(database, hard_link, overwrite=True)


def test_snapshot_requires_explicit_overwrite(tmp_path: Path) -> None:
    database, study_id = _completed_database(tmp_path)
    output = tmp_path / "existing.sqlite3"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        snapshot_database(database, output)
    assert output.read_bytes() == b"keep me"

    snapshot_database(database, output, overwrite=True)
    service = StudyQueryService(output)
    assert service.describe_study(study_id).study.revision == 2


def test_snapshot_failure_never_publishes_partial_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _ = _completed_database(tmp_path)
    output = tmp_path / "protected.sqlite3"
    output.write_bytes(b"original")

    def fail_backup(source: Path, temporary: Path) -> None:
        del source
        temporary.write_bytes(b"partial")
        raise RuntimeError("injected backup failure")

    monkeypatch.setattr(snapshot_module, "_backup_database", fail_backup)
    with pytest.raises(RuntimeError, match="injected backup failure"):
        snapshot_database(database, output, overwrite=True)

    assert output.read_bytes() == b"original"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_json_export_is_exact_filtered_query_with_fidelity_metadata(
    tmp_path: Path,
) -> None:
    database, study_id = _completed_database(tmp_path)
    output = tmp_path / "filtered.json"
    query = TrialQuery(
        columns=("trial_id", "objective_value", "params.x"),
        filters=(
            TrialFilter(
                field="objective_value",
                op=FilterOperator.GTE,
                value=0.5,
            ),
        ),
        sort=(
            TrialSort(
                field="objective_value",
                direction=SortDirection.DESC,
            ),
        ),
        limit=2,
    )
    expected = StudyQueryService(database).query_trials(study_id, query)

    exported = export_study_data(database, study_id, output, query=query)
    payload = json.loads(output.read_text())

    assert payload == trial_query_result_payload(expected)
    assert payload["rows_total"] == 3
    assert payload["rows_returned"] == 2
    assert payload["sampled"] is False
    assert payload["sampling_method"] is None
    assert payload["aggregation"] is None
    assert payload["point_limit"] == 2
    assert [row["objective_value"] for row in payload["rows"]] == [1.2, 0.9]
    assert exported.format == ExportFormat.JSON
    assert exported.rows_total == 3
    assert exported.rows_exported == 2
    assert exported.columns == query.columns


def test_csv_export_preserves_requested_column_order_and_nested_json(
    tmp_path: Path,
) -> None:
    database, study_id = _completed_database(tmp_path)
    output = tmp_path / "filtered.csv"
    columns = ("objective_value", "params", "trial_id")
    query = TrialQuery(
        columns=columns,
        sort=(TrialSort(field="evaluation_index"),),
    )

    exported = export_study_data(database, study_id, output, query=query)
    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    assert tuple(reader.fieldnames) == columns
    assert [row["objective_value"] for row in rows] == ["0.1", "0.9", "0.5", "1.2"]
    assert rows[0]["params"] == '{"group":{"width":16},"x":0}'
    assert exported.columns == columns
    assert exported.rows_exported == 4


def test_empty_csv_still_has_stable_default_columns(tmp_path: Path) -> None:
    database, study_id = _completed_database(tmp_path)
    output = tmp_path / "empty.csv"
    query = TrialQuery(
        filters=(
            TrialFilter(
                field="objective_value",
                op=FilterOperator.GT,
                value=100,
            ),
        )
    )

    exported = export_study_data(database, study_id, output, query=query)
    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == DEFAULT_TRIAL_COLUMNS
        assert list(reader) == []
    assert exported.columns == DEFAULT_TRIAL_COLUMNS
    assert exported.rows_exported == 0


def test_export_rejects_database_as_output_and_requires_overwrite(
    tmp_path: Path,
) -> None:
    database, study_id = _completed_database(tmp_path)
    with pytest.raises(ValueError, match="must be different"):
        export_study_data(
            database,
            study_id,
            database,
            format=ExportFormat.CSV,
            overwrite=True,
        )

    output = tmp_path / "existing.json"
    output.write_text("original")
    with pytest.raises(FileExistsError, match="already exists"):
        export_study_data(database, study_id, output)
    assert output.read_text() == "original"

    export_study_data(database, study_id, output, overwrite=True)
    assert json.loads(output.read_text())["rows_returned"] == 4


def test_export_failure_keeps_existing_output_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, study_id = _completed_database(tmp_path)
    output = tmp_path / "protected.json"
    output.write_text("original")

    def fail_write(path: Path, result: object) -> None:
        del result
        path.write_text("partial")
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(export_module, "_write_json", fail_write)
    with pytest.raises(RuntimeError, match="injected export failure"):
        export_study_data(database, study_id, output, overwrite=True)

    assert output.read_text() == "original"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_export_format_can_be_explicit_when_suffix_is_not(tmp_path: Path) -> None:
    database, study_id = _completed_database(tmp_path)
    output = tmp_path / "trials.data"
    result = export_study_data(
        database,
        study_id,
        output,
        format=ExportFormat.JSON,
    )
    assert result.format == ExportFormat.JSON
    assert json.loads(output.read_text())["study_revision"] == 2

    with pytest.raises(ValueError, match="must be supplied"):
        export_study_data(database, study_id, tmp_path / "trials.unknown")
