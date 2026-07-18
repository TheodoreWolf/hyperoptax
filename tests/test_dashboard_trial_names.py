import re
from pathlib import Path

from hyperoptax.dashboard.models import (
    Direction,
    StudyConfig,
    StudyStatus,
    TrialInput,
    TrialQuery,
    TrialState,
)
from hyperoptax.dashboard.queries import StudyQueryService
from hyperoptax.dashboard.sqlite_store import SQLiteStore
from hyperoptax.dashboard.trial_names import make_trial_name


def test_trial_name_format_and_hash_mapping_are_stable() -> None:
    assert (
        make_trial_name(
            "00000000-0000-0000-0000-000000000000",
            evaluation_index=0,
            batch_index=0,
        )
        == "misty-falcon-1"
    )
    assert (
        make_trial_name("trial-a", evaluation_index=None, batch_index=2)
        == "wild-dolphin-3"
    )


def test_existing_database_trials_receive_stable_queryable_names(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trial-names.sqlite3"
    store = SQLiteStore(database)
    study = StudyConfig(
        name="trial names",
        direction=Direction.MAXIMIZE,
        optimizer_name="RandomSearch",
        n_parallel=2,
    ).create_study(status=StudyStatus.RUNNING)
    store.create_study(study)
    store.append_batch(
        study.study_id,
        0,
        tuple(
            TrialInput(
                trial_id=f"trial-{index}",
                batch_index=0,
                batch_slot=index,
                evaluation_index=index,
                state=TrialState.COMPLETED,
                params={"x": index},
                objective_value=float(index),
            )
            for index in range(2)
        ),
    )

    first_read = [trial.trial_name for trial in store.list_trials(study.study_id)]
    reopened = StudyQueryService(database)
    second_read = [
        trial.trial_name for trial in reopened.store.list_trials(study.study_id)
    ]
    rows = reopened.query_trials(
        study.study_id,
        TrialQuery(columns=("trial_id", "trial_name")),
    ).rows

    assert first_read == second_read
    assert len(set(first_read)) == 2
    assert all(re.fullmatch(r"[a-z]+-[a-z]+-[1-9][0-9]*", name) for name in first_read)
    assert [row["trial_name"] for row in rows] == first_read
    assert "trial_name" in reopened.available_fields(study.study_id)
