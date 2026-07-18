"""Typed, storage-independent queries over dashboard study records."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hyperoptax.dashboard.models import (
    BestSoFarPoint,
    Direction,
    FilterOperator,
    JsonObject,
    JsonValue,
    ParetoFront,
    ParetoPoint,
    SortDirection,
    Study,
    StudyChanges,
    StudyDescription,
    Trial,
    TrialFilter,
    TrialQuery,
    TrialQueryResult,
    TrialSort,
    TrialState,
    to_json_value,
)
from hyperoptax.dashboard.sqlite_store import SQLiteStore
from hyperoptax.dashboard.store import StudyStore

_MISSING = object()
DEFAULT_TRIAL_COLUMNS = (
    "trial_id",
    "trial_name",
    "study_id",
    "batch_index",
    "batch_slot",
    "evaluation_index",
    "state",
    "params",
    "objective_value",
    "started_at",
    "finished_at",
    "duration_seconds",
    "seed",
    "worker",
    "error",
    "metadata",
    "created_revision",
    "updated_revision",
)
_BASE_FIELDS = DEFAULT_TRIAL_COLUMNS


class InvalidQueryError(ValueError):
    """Raised when a serialized trial query is not safe or meaningful."""


def escape_path_segment(segment: str) -> str:
    """Escape a parameter path segment for reversible dotted display paths."""

    return segment.replace("\\", "\\\\").replace(".", "\\.")


def split_parameter_path(field: str) -> tuple[str, ...]:
    """Parse ``params.a.b`` while respecting escaped dots and backslashes."""

    if not field.startswith("params."):
        raise InvalidQueryError(f"not a nested parameter field: {field!r}")
    raw = field[len("params.") :]
    if not raw:
        raise InvalidQueryError("parameter field must include at least one segment")
    segments: list[str] = []
    current: list[str] = []
    escaped = False
    for character in raw:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ".":
            if not current:
                raise InvalidQueryError(f"empty parameter path segment in {field!r}")
            segments.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        raise InvalidQueryError(f"trailing path escape in {field!r}")
    if not current:
        raise InvalidQueryError(f"empty parameter path segment in {field!r}")
    segments.append("".join(current))
    return tuple(segments)


def flatten_parameter_fields(
    params: JsonValue,
    *,
    prefix: str = "params",
) -> dict[str, JsonValue]:
    """Flatten nested JSON params into deterministic, reversible display paths."""

    flattened: dict[str, JsonValue] = {}

    def visit(value: JsonValue, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{escape_path_segment(key)}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        else:
            flattened[path] = value

    visit(params, prefix)
    return flattened


def trial_to_row(trial: Trial) -> JsonObject:
    """Serialize a full trial without exposing storage-specific JSON columns."""

    return {
        "trial_id": trial.trial_id,
        "trial_name": trial.trial_name,
        "study_id": trial.study_id,
        "batch_index": trial.batch_index,
        "batch_slot": trial.batch_slot,
        "evaluation_index": trial.evaluation_index,
        "state": trial.state.value,
        "params": trial.params,
        "objective_value": trial.objective_value,
        "started_at": _serialize_value(trial.started_at),
        "finished_at": _serialize_value(trial.finished_at),
        "duration_seconds": trial.duration_seconds,
        "seed": trial.seed,
        "worker": trial.worker,
        "error": trial.error,
        "metadata": trial.metadata,
        "created_revision": trial.created_revision,
        "updated_revision": trial.updated_revision,
    }


class StudyQueryService:
    """Semantic query surface shared by Python, HTTP, and future agents."""

    def __init__(self, store: StudyStore | str | Path) -> None:
        self.store: StudyStore
        if isinstance(store, (str, Path)):
            self.store = SQLiteStore(store)
        else:
            self.store = store

    def list_studies(self) -> tuple[Study, ...]:
        return self.store.list_studies()

    def describe_study(self, study_id: str) -> StudyDescription:
        study, trials = self.store.get_study_with_trials(study_id)
        counts = {state.value: 0 for state in TrialState}
        for trial in trials:
            counts[trial.state.value] += 1
        return StudyDescription(
            study=study,
            trial_counts=counts,
            best_trial=_best_trial(study, trials),
        )

    def available_fields(self, study_id: str) -> tuple[str, ...]:
        _, trials = self.store.get_study_with_trials(study_id)
        fields = set(_BASE_FIELDS)
        for trial in trials:
            fields.update(flatten_parameter_fields(trial.params))
        return tuple(sorted(fields))

    def hyperparameter_importance(self, study_id: str) -> Any:
        """Return deterministic random-forest and Pearson estimates per parameter."""

        from hyperoptax.dashboard.importance import hyperparameter_importance

        study, trials = self.store.get_study_with_trials(study_id)
        return hyperparameter_importance(trials, study_revision=study.revision)

    def pareto_front(
        self,
        study_id: str,
        *,
        x_field: str = "duration_seconds",
        y_field: str = "objective_value",
        x_direction: Direction | str = Direction.MINIMIZE,
        y_direction: Direction | str | None = None,
    ) -> ParetoFront:
        """Return the two-dimensional front over finite completed trials."""

        study, trials = self.store.get_study_with_trials(study_id)
        available = _available_fields(trials)
        unknown = sorted({x_field, y_field} - available)
        if unknown:
            raise InvalidQueryError(f"unknown trial fields: {', '.join(unknown)}")
        if x_field == y_field:
            raise InvalidQueryError("Pareto axes must use two different fields")

        x_direction = Direction(x_direction)
        y_direction = Direction(y_direction or study.direction)
        candidates: list[tuple[Trial, float, float]] = []
        for trial in trials:
            if trial.state != TrialState.COMPLETED:
                continue
            x_value = _finite_number(_field_value(trial, x_field))
            y_value = _finite_number(_field_value(trial, y_field))
            if x_value is None or y_value is None:
                continue
            candidates.append((trial, x_value, y_value))

        def rank(value: float, direction: Direction) -> float:
            return value if direction == Direction.MINIMIZE else -value

        ordered = sorted(
            candidates,
            key=lambda item: (
                rank(item[1], x_direction),
                rank(item[2], y_direction),
                item[0].trial_id,
            ),
        )
        points: list[ParetoPoint] = []
        best_y_rank = math.inf
        index = 0
        while index < len(ordered):
            x_rank = rank(ordered[index][1], x_direction)
            group: list[tuple[Trial, float, float]] = []
            while (
                index < len(ordered) and rank(ordered[index][1], x_direction) == x_rank
            ):
                group.append(ordered[index])
                index += 1
            group_best = min(rank(item[2], y_direction) for item in group)
            if group_best >= best_y_rank:
                continue
            points.extend(
                ParetoPoint(trial_id=trial.trial_id, x=x_value, y=y_value)
                for trial, x_value, y_value in group
                if rank(y_value, y_direction) == group_best
            )
            best_y_rank = group_best

        return ParetoFront(
            x_field=x_field,
            y_field=y_field,
            x_direction=x_direction,
            y_direction=y_direction,
            points=tuple(points),
            trials_considered=len(candidates),
            study_revision=study.revision,
        )

    def query_trials(
        self,
        study_id: str,
        query: TrialQuery | Mapping[str, Any] | None = None,
    ) -> TrialQueryResult:
        normalized = _normalize_query(query)
        study, trials = self.store.get_study_with_trials(study_id)
        available = _available_fields(trials)
        _validate_fields(normalized, available)

        matching = _apply_filters(trials, normalized.filters)
        ordered = _apply_sort(matching, normalized.sort)
        limited = ordered[: normalized.limit]
        if normalized.columns is None:
            rows = tuple(trial_to_row(trial) for trial in limited)
        else:
            rows = tuple(
                {
                    field: _serialize_value(_field_value(trial, field))
                    for field in normalized.columns
                }
                for trial in limited
            )
        return TrialQueryResult(
            rows=rows,
            rows_total=len(matching),
            rows_returned=len(rows),
            sampled=False,
            sampling_method=None,
            aggregation=None,
            point_limit=normalized.limit,
            study_revision=study.revision,
        )

    def get_trial(self, study_id: str, trial_id: str) -> Trial:
        return self.store.get_trial(study_id, trial_id)

    def get_changes(self, study_id: str, since_revision: int) -> StudyChanges:
        if since_revision < 0:
            raise InvalidQueryError("since_revision must not be negative")
        study, trials = self.store.get_study_with_trials(
            study_id, updated_since=since_revision
        )
        if since_revision > study.revision:
            raise InvalidQueryError(
                f"since_revision {since_revision} is newer than current revision "
                f"{study.revision}"
            )
        return StudyChanges(
            study_revision=study.revision,
            study_status=study.status,
            trials=trials,
        )

    def changes(self, study_id: str, since_revision: int) -> StudyChanges:
        """Backward-compatible spelling for Python callers."""

        return self.get_changes(study_id, since_revision)

    def best_trial(
        self,
        study_id: str,
        query: TrialQuery | Mapping[str, Any] | None = None,
    ) -> Trial | None:
        normalized = _normalize_query(query)
        study, trials = self.store.get_study_with_trials(study_id)
        available = _available_fields(trials)
        _validate_fields(normalized, available)
        return _best_trial(study, _apply_filters(trials, normalized.filters))

    def best_so_far(
        self,
        study_id: str,
        query: TrialQuery | Mapping[str, Any] | None = None,
    ) -> tuple[BestSoFarPoint, ...]:
        normalized = _normalize_query(query)
        study, trials = self.store.get_study_with_trials(study_id)
        available = _available_fields(trials)
        _validate_fields(normalized, available)
        candidates = sorted(
            (
                trial
                for trial in _apply_filters(trials, normalized.filters)
                if trial.state == TrialState.COMPLETED
                and trial.objective_value is not None
                and trial.evaluation_index is not None
            ),
            key=lambda trial: (trial.evaluation_index, trial.trial_id),
        )
        points: list[BestSoFarPoint] = []
        current_best: float | None = None
        for trial in candidates:
            objective = trial.objective_value
            if current_best is None or _is_better(
                objective, current_best, study.direction
            ):
                current_best = objective
            points.append(
                BestSoFarPoint(
                    trial_id=trial.trial_id,
                    evaluation_index=trial.evaluation_index,
                    objective_value=objective,
                    best_objective_value=current_best,
                )
            )
        return tuple(points)

    def close(self) -> None:
        self.store.close()


def _normalize_query(
    query: TrialQuery | Mapping[str, Any] | None,
) -> TrialQuery:
    if query is None:
        return TrialQuery()
    if isinstance(query, TrialQuery):
        return query
    if isinstance(query, Mapping):
        try:
            return TrialQuery(**query)
        except (TypeError, ValueError) as error:
            raise InvalidQueryError(str(error)) from error
    raise TypeError("query must be a TrialQuery, mapping, or None")


def _available_fields(trials: Sequence[Trial]) -> set[str]:
    fields = set(_BASE_FIELDS)
    for trial in trials:
        fields.update(flatten_parameter_fields(trial.params))
    return fields


def _validate_fields(query: TrialQuery, available: set[str]) -> None:
    fields: list[str] = []
    if query.columns is not None:
        if len(query.columns) != len(set(query.columns)):
            raise InvalidQueryError("query columns must not contain duplicates")
        fields.extend(query.columns)
    fields.extend(item.field for item in query.filters)
    fields.extend(item.field for item in query.sort)
    unknown = sorted(set(fields) - available)
    if unknown:
        raise InvalidQueryError(f"unknown trial fields: {', '.join(unknown)}")
    for item in query.filters:
        _validate_filter(item)


def _validate_filter(item: TrialFilter) -> None:
    if item.op == FilterOperator.BETWEEN:
        if not isinstance(item.value, list) or len(item.value) != 2:
            raise InvalidQueryError("between filter value must contain two bounds")
    elif item.op in {FilterOperator.IN, FilterOperator.NOT_IN}:
        if not isinstance(item.value, list):
            raise InvalidQueryError(f"{item.op.value} filter value must be a list")
    elif item.op == FilterOperator.CONTAINS and not isinstance(item.value, str):
        raise InvalidQueryError("contains filter value must be a string")
    elif item.op in {FilterOperator.IS_MISSING, FilterOperator.IS_NOT_MISSING}:
        if item.value is not None:
            raise InvalidQueryError(f"{item.op.value} filter does not accept a value")


def _apply_filters(
    trials: Iterable[Trial], filters: Sequence[TrialFilter]
) -> tuple[Trial, ...]:
    return tuple(
        trial
        for trial in trials
        if all(_matches_filter(trial, item) for item in filters)
    )


def _matches_filter(trial: Trial, item: TrialFilter) -> bool:
    actual = _field_value(trial, item.field)
    missing = actual is _MISSING or actual is None
    if item.op == FilterOperator.IS_MISSING:
        return missing
    if item.op == FilterOperator.IS_NOT_MISSING:
        return not missing
    if missing:
        return False

    expected = item.value
    try:
        if item.op == FilterOperator.EQ:
            return actual == expected
        if item.op == FilterOperator.NE:
            return actual != expected
        if item.op == FilterOperator.LT:
            return actual < expected
        if item.op == FilterOperator.LTE:
            return actual <= expected
        if item.op == FilterOperator.GT:
            return actual > expected
        if item.op == FilterOperator.GTE:
            return actual >= expected
        if item.op == FilterOperator.BETWEEN:
            return expected[0] <= actual <= expected[1]
        if item.op == FilterOperator.IN:
            return actual in expected
        if item.op == FilterOperator.NOT_IN:
            return actual not in expected
        if item.op == FilterOperator.CONTAINS:
            return isinstance(actual, str) and expected in actual
    except TypeError as error:
        raise InvalidQueryError(
            f"filter {item.op.value!r} is incompatible with field {item.field!r}"
        ) from error
    raise InvalidQueryError(f"unsupported filter operator: {item.op.value}")


def _apply_sort(
    trials: Sequence[Trial], sort_items: Sequence[TrialSort]
) -> tuple[Trial, ...]:
    if not sort_items:
        return tuple(trials)

    def compare(left: Trial, right: Trial) -> int:
        for item in sort_items:
            left_value = _field_value(left, item.field)
            right_value = _field_value(right, item.field)
            left_missing = left_value is _MISSING or left_value is None
            right_missing = right_value is _MISSING or right_value is None
            if left_missing != right_missing:
                return 1 if left_missing else -1
            if left_missing:
                continue
            try:
                comparison = (left_value > right_value) - (left_value < right_value)
            except TypeError as error:
                raise InvalidQueryError(
                    f"field {item.field!r} contains values that cannot be sorted"
                ) from error
            if comparison:
                if item.direction == SortDirection.DESC:
                    comparison = -comparison
                return comparison
        return (left.trial_id > right.trial_id) - (left.trial_id < right.trial_id)

    return tuple(sorted(trials, key=cmp_to_key(compare)))


def _field_value(trial: Trial, field: str) -> Any:
    if field == "params":
        return trial.params
    if field.startswith("params."):
        value: Any = trial.params
        for segment in split_parameter_path(field):
            if isinstance(value, dict):
                if segment not in value:
                    return _MISSING
                value = value[segment]
            elif isinstance(value, list):
                try:
                    index = int(segment)
                except ValueError:
                    return _MISSING
                if index < 0 or index >= len(value):
                    return _MISSING
                value = value[index]
            else:
                return _MISSING
        return value
    if field not in _BASE_FIELDS:
        return _MISSING
    return _serialize_value(getattr(trial, field))


def _serialize_value(value: Any) -> JsonValue:
    if value is _MISSING:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    return to_json_value(value)


def _finite_number(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _best_trial(study: Study, trials: Iterable[Trial]) -> Trial | None:
    best: Trial | None = None
    for trial in trials:
        if trial.state != TrialState.COMPLETED or trial.objective_value is None:
            continue
        if best is None or _is_better(
            trial.objective_value, best.objective_value, study.direction
        ):
            best = trial
    return best


def _is_better(candidate: float, incumbent: float, direction: Direction) -> bool:
    if direction == Direction.MAXIMIZE:
        return candidate > incumbent
    return candidate < incumbent


__all__ = [
    "DEFAULT_TRIAL_COLUMNS",
    "InvalidQueryError",
    "StudyQueryService",
    "escape_path_segment",
    "flatten_parameter_fields",
    "split_parameter_path",
    "trial_to_row",
]
