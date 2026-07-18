"""Study-level hyperparameter importance estimates for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from hyperoptax.dashboard.models import Trial, TrialState


@dataclass(frozen=True, slots=True)
class HyperparameterImportance:
    """One numeric hyperparameter's model and linear association estimates."""

    field: str
    importance: float
    correlation: float | None


@dataclass(frozen=True, slots=True)
class HyperparameterImportanceResult:
    """A deterministic random-forest importance analysis for one study."""

    parameters: tuple[HyperparameterImportance, ...]
    n_trials: int
    n_estimators: int
    random_state: int
    study_revision: int
    reason: str | None = None


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def hyperparameter_importance(
    trials: Sequence[Trial],
    *,
    study_revision: int,
    n_estimators: int = 200,
    random_state: int = 0,
) -> HyperparameterImportanceResult:
    """Fit a random forest and calculate Pearson correlations on numeric params.

    Only completed trials with a finite objective participate. A parameter must
    be present and finite for every participating trial, so the model never
    silently imputes values or changes the effective study population by field.
    """

    from hyperoptax.dashboard.queries import flatten_parameter_fields

    rows: list[tuple[dict[str, object], float]] = []
    for trial in trials:
        if trial.state is not TrialState.COMPLETED:
            continue
        objective = _numeric(trial.objective_value)
        if objective is None:
            continue
        flattened = flatten_parameter_fields(trial.params)
        rows.append((flattened, objective))

    if len(rows) < 2:
        return HyperparameterImportanceResult(
            parameters=(),
            n_trials=len(rows),
            n_estimators=n_estimators,
            random_state=random_state,
            study_revision=study_revision,
            reason="At least two completed trials with finite objectives are required.",
        )

    common_fields = set(rows[0][0])
    for params, _ in rows[1:]:
        common_fields.intersection_update(params)
    fields = tuple(
        field
        for field in sorted(common_fields)
        if all(_numeric(params[field]) is not None for params, _ in rows)
    )
    if not fields:
        return HyperparameterImportanceResult(
            parameters=(),
            n_trials=len(rows),
            n_estimators=n_estimators,
            random_state=random_state,
            study_revision=study_revision,
            reason="No numeric hyperparameters are available in every completed trial.",
        )

    features = np.asarray(
        [[_numeric(params[field]) for field in fields] for params, _ in rows],
        dtype=float,
    )
    objectives = np.asarray([objective for _, objective in rows], dtype=float)

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as error:  # pragma: no cover - dashboard extra installs sklearn
        raise ImportError(
            "Hyperparameter importance requires scikit-learn. "
            "Install hyperoptax[dashboard]."
        ) from error

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )
    forest.fit(features, objectives)
    objective_std = float(np.std(objectives))
    rows_out = []
    for index, field in enumerate(fields):
        feature = features[:, index]
        correlation = None
        if objective_std > 0 and float(np.std(feature)) > 0:
            correlation = float(np.corrcoef(feature, objectives)[0, 1])
        rows_out.append(
            HyperparameterImportance(
                field=field,
                importance=float(forest.feature_importances_[index]),
                correlation=correlation,
            )
        )

    return HyperparameterImportanceResult(
        parameters=tuple(
            sorted(rows_out, key=lambda item: (-item.importance, item.field))
        ),
        n_trials=len(rows),
        n_estimators=n_estimators,
        random_state=random_state,
        study_revision=study_revision,
    )


__all__ = [
    "HyperparameterImportance",
    "HyperparameterImportanceResult",
    "hyperparameter_importance",
]
