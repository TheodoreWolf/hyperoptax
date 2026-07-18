from pathlib import Path
from uuid import uuid4

import jax
import numpy as np
import pytest

import hyperoptax.dashboard.recorder as recorder_module
from hyperoptax import RandomSearch
from hyperoptax import spaces as sp
from hyperoptax.dashboard import import_history
from hyperoptax.dashboard.models import StudyConfig, StudyStatus, TrialState
from hyperoptax.dashboard.sqlite_store import SQLiteStore


def _config(*, study_id: str | None = None, n_parallel: int = 2) -> StudyConfig:
    return StudyConfig(
        name="history import test",
        direction="maximize",
        optimizer_name="RandomSearch",
        optimizer_config={"n_parallel": n_parallel},
        search_space={
            "learning_rate": {"lower": 1e-4, "upper": 1e-1},
            "regularization": {"l1": {"lower": 0.0, "upper": 1.0}},
        },
        study_id=study_id,
        n_parallel=n_parallel,
        metadata={"source": "test"},
    )


def _matching_histories() -> tuple[tuple, tuple]:
    space = {
        "learning_rate": sp.LogSpace(1e-4, 1e-1),
        "regularization": {"l1": sp.LinearSpace(0.0, 1.0)},
    }
    list_state, list_optimizer = RandomSearch.init(space, n_parallel=2)
    scan_state, scan_optimizer = RandomSearch.init(space, n_parallel=2)
    key = jax.random.PRNGKey(7)

    def objective(_key, params):
        return params["learning_rate"] + params["regularization"]["l1"]

    _, list_history = list_optimizer.optimize(
        list_state,
        key,
        objective,
        n_iterations=3,
    )
    _, scan_history = scan_optimizer.optimize_scan(
        scan_state,
        key,
        objective,
        n_iterations=3,
    )
    return list_history, scan_history


def test_list_and_scan_histories_import_as_identical_timeless_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_history, scan_history = _matching_histories()
    study_id = str(uuid4())
    config = _config(study_id=study_id)

    original_device_get = recorder_module.jax.device_get
    device_get_calls = 0

    def count_device_get(value):
        nonlocal device_get_calls
        device_get_calls += 1
        return original_device_get(value)

    monkeypatch.setattr(recorder_module.jax, "device_get", count_device_get)

    list_database = tmp_path / "list.sqlite3"
    list_study = import_history(
        list_database,
        study=config,
        params_history=list_history[0],
        results_history=list_history[1],
    )
    assert device_get_calls == 1

    scan_database = tmp_path / "scan.sqlite3"
    scan_study = import_history(
        scan_database,
        study=config,
        params_history=scan_history[0],
        results_history=scan_history[1],
    )
    assert device_get_calls == 2

    list_trials = SQLiteStore(list_database).list_trials(study_id)
    scan_trials = SQLiteStore(scan_database).list_trials(study_id)
    assert list_trials == scan_trials

    for study in (list_study, scan_study):
        assert study.status == StudyStatus.COMPLETED
        assert study.finished_at is not None
        assert study.revision == 3
        assert study.n_parallel == 2
        assert study.metadata == {
            "source": "test",
            "recording_mode": "bulk_history",
            "trial_timing_available": False,
        }

    assert len(list_trials) == 6
    assert [trial.batch_index for trial in list_trials] == [0, 0, 1, 1, 2, 2]
    assert [trial.batch_slot for trial in list_trials] == [0, 1, 0, 1, 0, 1]
    assert [trial.evaluation_index for trial in list_trials] == list(range(6))
    assert all(trial.state == TrialState.COMPLETED for trial in list_trials)
    assert all(trial.started_at is None for trial in list_trials)
    assert all(trial.finished_at is None for trial in list_trials)
    assert all(trial.duration_seconds is None for trial in list_trials)
    assert all(trial.error is None for trial in list_trials)
    assert all("regularization" in trial.params for trial in list_trials)
    assert all("l1" in trial.params["regularization"] for trial in list_trials)


def test_empty_list_history_creates_completed_study_without_trials(
    tmp_path: Path,
) -> None:
    database = tmp_path / "empty.sqlite3"
    study = import_history(
        database,
        study=_config(n_parallel=3),
        params_history=[],
        results_history=[],
    )

    assert study.status == StudyStatus.COMPLETED
    assert study.finished_at is not None
    assert study.revision == 0
    assert study.n_parallel == 3
    assert SQLiteStore(database).list_trials(study.study_id) == ()


@pytest.mark.parametrize(
    ("params_history", "results_history", "message"),
    [
        (
            [{"x": np.asarray([1.0, 2.0])}],
            [],
            "same number of batches",
        ),
        (
            [{"x": np.asarray([1.0])}],
            [np.asarray([1.0, 2.0])],
            "leading n_parallel axis",
        ),
        (
            {"x": np.ones((2, 2))},
            np.ones((3, 2)),
            "leading \\(n_batches, n_parallel\\) axes",
        ),
    ],
)
def test_invalid_history_is_rejected_before_database_creation(
    tmp_path: Path,
    params_history,
    results_history,
    message: str,
) -> None:
    database = tmp_path / "invalid.sqlite3"

    with pytest.raises(ValueError, match=message):
        import_history(
            database,
            study=_config(),
            params_history=params_history,
            results_history=results_history,
        )

    assert not database.exists()
