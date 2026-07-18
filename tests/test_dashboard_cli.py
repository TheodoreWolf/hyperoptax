import json
from pathlib import Path

import numpy as np
import pytest

from hyperoptax.dashboard.cli import main
from hyperoptax.dashboard.models import Direction, StudyConfig
from hyperoptax.dashboard.queries import StudyQueryService
from hyperoptax.dashboard.recorder import SQLiteRecorder
from hyperoptax.recording import BatchCompleted


def _database(tmp_path: Path) -> tuple[Path, str]:
    database = tmp_path / "results.sqlite3"
    config = StudyConfig(
        name="CLI test",
        direction=Direction.MAXIMIZE,
        optimizer_name="RandomSearch",
        n_parallel=2,
    )
    with SQLiteRecorder(database, study=config) as recorder:
        recorder(
            BatchCompleted(
                batch_index=0,
                params={"x": np.asarray([0.25, 0.75])},
                results=np.asarray([1.0, 2.0]),
            )
        )
    return database, recorder.study_id


def test_snapshot_subcommand_creates_verified_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, study_id = _database(tmp_path)
    output = tmp_path / "snapshot.sqlite3"

    assert (
        main(
            [
                "snapshot",
                str(database),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert "Snapshot created:" in capsys.readouterr().out
    assert StudyQueryService(output).describe_study(study_id).study.revision == 1


def test_export_subcommand_accepts_inline_query_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, study_id = _database(tmp_path)
    output = tmp_path / "best.json"
    query = {
        "columns": ["evaluation_index", "objective_value", "params.x"],
        "filters": [{"field": "objective_value", "op": "gte", "value": 1.5}],
        "sort": [{"field": "objective_value", "direction": "desc"}],
    }

    assert (
        main(
            [
                "export",
                str(database),
                "--study",
                study_id,
                "--output",
                str(output),
                "--query",
                json.dumps(query),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows_total"] == 1
    assert payload["rows"] == [
        {"evaluation_index": 1, "objective_value": 2.0, "params.x": 0.75}
    ]
    assert "Exported 1/1 rows" in capsys.readouterr().out


def test_export_subcommand_reads_query_file_and_reports_invalid_json(
    tmp_path: Path,
) -> None:
    database, study_id = _database(tmp_path)
    query_file = tmp_path / "query.json"
    query_file.write_text('{"columns": ["trial_id"]}', encoding="utf-8")

    assert (
        main(
            [
                "export",
                str(database),
                "--study",
                study_id,
                "--output",
                str(tmp_path / "trials.csv"),
                "--query",
                f"@{query_file}",
            ]
        )
        == 0
    )

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "export",
                str(database),
                "--study",
                study_id,
                "--output",
                str(tmp_path / "invalid.json"),
                "--query",
                "[]",
            ]
        )
