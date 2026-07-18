from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from hyperoptax.dashboard.importance import hyperparameter_importance
from hyperoptax.dashboard.models import Trial, TrialState

pytest.importorskip("sklearn")


def _trials() -> list[Trial]:
    study_id = str(uuid4())
    values = np.linspace(-1.0, 1.0, 32)
    return [
        Trial(
            trial_id=f"trial-{index}",
            study_id=study_id,
            batch_index=index,
            batch_slot=0,
            evaluation_index=index,
            state=TrialState.COMPLETED,
            params={
                "signal": float(value),
                "noise": float(np.sin(index * 2.7)),
                "constant": 1.0,
                "label": "ignored",
            },
            objective_value=float(4.0 * value + 0.03 * np.cos(index)),
            created_revision=1,
            updated_revision=1,
        )
        for index, value in enumerate(values)
    ]


def test_random_forest_importance_and_pearson_correlation_are_deterministic() -> None:
    result = hyperparameter_importance(_trials(), study_revision=7)
    repeated = hyperparameter_importance(_trials(), study_revision=7)

    by_field = {item.field: item for item in result.parameters}
    assert result.n_trials == 32
    assert result.study_revision == 7
    assert result.reason is None
    assert sum(item.importance for item in result.parameters) == pytest.approx(1.0)
    assert by_field["params.signal"].importance > by_field["params.noise"].importance
    assert by_field["params.signal"].correlation == pytest.approx(1.0, abs=0.01)
    assert by_field["params.constant"].correlation is None
    assert result == repeated


def test_importance_explains_when_there_are_too_few_trials() -> None:
    result = hyperparameter_importance(_trials()[:1], study_revision=1)

    assert result.parameters == ()
    assert (
        result.reason
        == "At least two completed trials with finite objectives are required."
    )
