"""Dependency-light records shared by dashboard storage and query clients."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeAlias
from uuid import UUID, uuid4

from hyperoptax.dashboard.trial_names import make_trial_name

SCHEMA_VERSION = 1
MAX_QUERY_LIMIT = 10_000

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class Direction(str, Enum):
    """Whether a study's scalar objective should be maximized or minimized."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class StudyStatus(str, Enum):
    """Lifecycle state of one optimizer execution."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrialState(str, Enum):
    """Lifecycle state of one candidate evaluation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FilterOperator(str, Enum):
    """Supported structured-query comparisons."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    IS_MISSING = "is_missing"
    IS_NOT_MISSING = "is_not_missing"
    CONTAINS = "contains"


class SortDirection(str, Enum):
    """Sort direction for a query field."""

    ASC = "asc"
    DESC = "desc"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def normalize_datetime(value: datetime, *, field_name: str) -> datetime:
    """Normalize a timestamp to UTC and reject naive values."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def to_json_value(value: Any, *, path: str = "value") -> JsonValue:
    """Convert common host/JAX values into strict JSON-compatible values.

    This intentionally rejects non-finite numbers and arbitrary objects. It is
    used before durable writes so invalid metadata cannot create records that
    the API is unable to serialize later.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value, path=path)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, datetime):
        return normalize_datetime(value, field_name=path).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_json_value(
                getattr(value, item.name), path=f"{path}.{item.name}"
            )
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        converted: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}")
            converted[key] = to_json_value(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            to_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            pass
        else:
            if scalar is not value:
                return to_json_value(scalar, path=path)

    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        return to_json_value(list_method(), path=path)

    raise TypeError(f"{path} contains unsupported value {value!r}")


def _json_object(value: Mapping[str, Any], *, path: str) -> JsonObject:
    converted = to_json_value(value, path=path)
    if not isinstance(converted, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError(f"{path} must be a JSON object")
    return converted


def _uuid(value: str | UUID | None, *, field_name: str) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(str(value)))
    except ValueError as error:
        raise ValueError(f"{field_name} must be a UUID") from error


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """User-supplied configuration from which a durable study is created."""

    name: str
    direction: Direction | str
    optimizer_name: str
    n_parallel: int = 1
    optimizer_config: Mapping[str, Any] = field(default_factory=dict)
    search_space: Mapping[str, Any] = field(default_factory=dict)
    study_id: str | UUID | None = None
    sweep_id: str | UUID | None = None
    seed: Any = None
    code_version: str = "unknown"
    source_revision: str | None = None
    variant: str | None = None
    replicate: str | None = None
    tags: tuple[str, ...] | list[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.optimizer_name.strip():
            raise ValueError("optimizer_name must not be empty")
        if self.n_parallel < 1:
            raise ValueError("n_parallel must be at least 1")

        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "study_id", _uuid(self.study_id, field_name="study_id")
        )
        if self.sweep_id is not None:
            object.__setattr__(
                self, "sweep_id", _uuid(self.sweep_id, field_name="sweep_id")
            )
        object.__setattr__(
            self,
            "optimizer_config",
            _json_object(self.optimizer_config, path="optimizer_config"),
        )
        search_space = to_json_value(self.search_space, path="search_space")
        if not isinstance(search_space, dict):  # pragma: no cover - annotated Mapping
            raise TypeError("search_space must be a JSON object")
        object.__setattr__(self, "search_space", search_space)
        object.__setattr__(self, "seed", to_json_value(self.seed, path="seed"))
        object.__setattr__(self, "tags", tuple(self.tags))
        if any(not isinstance(tag, str) or not tag for tag in self.tags):
            raise ValueError("tags must contain non-empty strings")
        object.__setattr__(
            self, "metadata", _json_object(self.metadata, path="metadata")
        )

    def create_study(
        self,
        *,
        status: StudyStatus = StudyStatus.CREATED,
        created_at: datetime | None = None,
    ) -> "Study":
        """Create the initial durable representation for this configuration."""

        return Study(
            study_id=str(self.study_id),
            sweep_id=str(self.sweep_id) if self.sweep_id is not None else None,
            name=self.name,
            created_at=created_at or utc_now(),
            finished_at=None,
            status=status,
            direction=Direction(self.direction),
            optimizer_name=self.optimizer_name,
            optimizer_config=dict(self.optimizer_config),
            search_space=dict(self.search_space),
            seed=self.seed,
            n_parallel=self.n_parallel,
            code_version=self.code_version,
            source_revision=self.source_revision,
            variant=self.variant,
            replicate=self.replicate,
            tags=tuple(self.tags),
            metadata=dict(self.metadata),
            revision=0,
            schema_version=SCHEMA_VERSION,
        )


@dataclass(frozen=True, slots=True)
class Study:
    """Durable metadata for one optimizer execution."""

    study_id: str
    sweep_id: str | None
    name: str
    created_at: datetime
    finished_at: datetime | None
    status: StudyStatus
    direction: Direction
    optimizer_name: str
    optimizer_config: JsonObject
    search_space: JsonObject
    seed: JsonValue
    n_parallel: int
    code_version: str
    source_revision: str | None
    variant: str | None
    replicate: str | None
    tags: tuple[str, ...]
    metadata: JsonObject
    revision: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "study_id", _uuid(self.study_id, field_name="study_id")
        )
        if self.sweep_id is not None:
            object.__setattr__(
                self, "sweep_id", _uuid(self.sweep_id, field_name="sweep_id")
            )
        object.__setattr__(self, "status", StudyStatus(self.status))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self,
            "created_at",
            normalize_datetime(self.created_at, field_name="created_at"),
        )
        if self.finished_at is not None:
            object.__setattr__(
                self,
                "finished_at",
                normalize_datetime(self.finished_at, field_name="finished_at"),
            )
        if self.n_parallel < 1:
            raise ValueError("n_parallel must be at least 1")
        if self.revision < 0:
            raise ValueError("revision must not be negative")
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        object.__setattr__(
            self,
            "optimizer_config",
            _json_object(self.optimizer_config, path="optimizer_config"),
        )
        object.__setattr__(
            self, "search_space", _json_object(self.search_space, path="search_space")
        )
        object.__setattr__(self, "seed", to_json_value(self.seed, path="seed"))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(
            self, "metadata", _json_object(self.metadata, path="metadata")
        )


@dataclass(frozen=True, slots=True)
class TrialInput:
    """A trial before the store assigns batch revision metadata."""

    trial_id: str
    batch_index: int
    batch_slot: int
    evaluation_index: int | None
    state: TrialState
    params: JsonValue
    objective_value: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    seed: JsonValue = None
    worker: str | None = None
    error: JsonObject | None = None
    metadata: JsonObject = field(default_factory=dict)
    trial_name: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must not be empty")
        if self.batch_index < 0 or self.batch_slot < 0:
            raise ValueError("batch_index and batch_slot must not be negative")
        if self.evaluation_index is not None and self.evaluation_index < 0:
            raise ValueError("evaluation_index must not be negative")
        object.__setattr__(
            self,
            "trial_name",
            make_trial_name(
                self.trial_id,
                evaluation_index=self.evaluation_index,
                batch_index=self.batch_index,
            ),
        )
        object.__setattr__(self, "state", TrialState(self.state))
        object.__setattr__(self, "params", to_json_value(self.params, path="params"))
        if self.objective_value is not None:
            objective = float(self.objective_value)
            if not math.isfinite(objective):
                raise ValueError("objective_value must be finite")
            object.__setattr__(self, "objective_value", objective)
        if self.started_at is not None:
            object.__setattr__(
                self,
                "started_at",
                normalize_datetime(self.started_at, field_name="started_at"),
            )
        if self.finished_at is not None:
            object.__setattr__(
                self,
                "finished_at",
                normalize_datetime(self.finished_at, field_name="finished_at"),
            )
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")
        object.__setattr__(self, "seed", to_json_value(self.seed, path="seed"))
        if self.error is not None:
            object.__setattr__(self, "error", _json_object(self.error, path="error"))
        object.__setattr__(
            self, "metadata", _json_object(self.metadata, path="metadata")
        )


@dataclass(frozen=True, slots=True)
class Trial(TrialInput):
    """A stored trial with study and revision identity."""

    study_id: str = ""
    created_revision: int = 0
    updated_revision: int = 0

    def __post_init__(self) -> None:
        TrialInput.__post_init__(self)
        object.__setattr__(
            self, "study_id", _uuid(self.study_id, field_name="study_id")
        )
        if self.created_revision < 1 or self.updated_revision < self.created_revision:
            raise ValueError(
                "trial revisions must be positive and monotonically ordered"
            )


@dataclass(frozen=True, slots=True)
class BatchWriteResult:
    """Outcome of an atomic batch append."""

    revision: int
    inserted: bool
    trial_count: int


@dataclass(frozen=True, slots=True)
class TrialFilter:
    """One validated field comparison in a trial query."""

    field: str
    op: FilterOperator | str
    value: Any = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("filter field must not be empty")
        object.__setattr__(self, "op", FilterOperator(self.op))
        object.__setattr__(
            self, "value", to_json_value(self.value, path="filter.value")
        )


@dataclass(frozen=True, slots=True)
class TrialSort:
    """One field ordering in a trial query."""

    field: str
    direction: SortDirection | str = SortDirection.ASC

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("sort field must not be empty")
        object.__setattr__(self, "direction", SortDirection(self.direction))


@dataclass(frozen=True, slots=True)
class TrialQuery:
    """Versioned, serializable query shared by UI and agent clients."""

    columns: tuple[str, ...] | list[str] | None = None
    filters: tuple[TrialFilter, ...] | list[TrialFilter | Mapping[str, Any]] = ()
    sort: tuple[TrialSort, ...] | list[TrialSort | Mapping[str, Any]] = ()
    limit: int = 1_000
    version: int = 1

    def __post_init__(self) -> None:
        if self.columns is not None:
            columns = tuple(self.columns)
            if any(not isinstance(column, str) or not column for column in columns):
                raise ValueError("columns must contain non-empty strings")
            object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self,
            "filters",
            tuple(
                item if isinstance(item, TrialFilter) else TrialFilter(**item)
                for item in self.filters
            ),
        )
        object.__setattr__(
            self,
            "sort",
            tuple(
                item if isinstance(item, TrialSort) else TrialSort(**item)
                for item in self.sort
            ),
        )
        if not 1 <= self.limit <= MAX_QUERY_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
        if self.version != 1:
            raise ValueError(f"unsupported TrialQuery version: {self.version}")


@dataclass(frozen=True, slots=True)
class TrialQueryResult:
    """Bounded query rows with explicit fidelity metadata."""

    rows: tuple[JsonObject, ...]
    rows_total: int
    rows_returned: int
    sampled: bool
    sampling_method: str | None
    aggregation: str | None
    point_limit: int
    study_revision: int


@dataclass(frozen=True, slots=True)
class StudyChanges:
    """Trials inserted or updated after a known study revision."""

    study_revision: int
    study_status: StudyStatus
    trials: tuple[Trial, ...]


@dataclass(frozen=True, slots=True)
class StudyDescription:
    """Study metadata plus the small overview needed by catalogue clients."""

    study: Study
    trial_counts: dict[str, int]
    best_trial: Trial | None


@dataclass(frozen=True, slots=True)
class BestSoFarPoint:
    """Direction-aware running optimum after one completed trial."""

    trial_id: str
    evaluation_index: int
    objective_value: float
    best_objective_value: float


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One non-dominated completed trial projected onto two numeric fields."""

    trial_id: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ParetoFront:
    """A direction-aware Pareto front suitable for UI and agent clients."""

    x_field: str
    y_field: str
    x_direction: Direction
    y_direction: Direction
    points: tuple[ParetoPoint, ...]
    trials_considered: int
    study_revision: int


__all__ = [
    "MAX_QUERY_LIMIT",
    "SCHEMA_VERSION",
    "BatchWriteResult",
    "BestSoFarPoint",
    "Direction",
    "FilterOperator",
    "JsonObject",
    "JsonValue",
    "ParetoFront",
    "ParetoPoint",
    "SortDirection",
    "Study",
    "StudyChanges",
    "StudyConfig",
    "StudyDescription",
    "StudyStatus",
    "Trial",
    "TrialFilter",
    "TrialInput",
    "TrialQuery",
    "TrialQueryResult",
    "TrialSort",
    "TrialState",
    "normalize_datetime",
    "to_json_value",
    "utc_now",
]
