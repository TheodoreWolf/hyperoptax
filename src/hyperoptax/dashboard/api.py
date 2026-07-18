"""Read-only HTTP API and static dashboard application.

The web layer depends only on the public ``StudyQueryService`` behavior.  It is
kept separate from SQLite so tests, future stores, and agent transports can use
the same query semantics.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover - exercised without dashboard extra
    raise ImportError(
        "The dashboard requires optional dependencies. Install hyperoptax[dashboard]."
    ) from exc


_T = TypeVar("_T")


class TrialFilterPayload(BaseModel):
    """One validated filter in an HTTP trial query."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    op: str = Field(min_length=1)
    value: Any = None


class TrialSortPayload(BaseModel):
    """One validated sort key in an HTTP trial query."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    direction: str = "asc"


class TrialQueryPayload(BaseModel):
    """Transport model for the versioned trial-query endpoint."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str] | None = None
    filters: list[TrialFilterPayload] = Field(default_factory=list)
    sort: list[TrialSortPayload] = Field(default_factory=list)
    limit: int = Field(default=10_000, ge=1, le=10_000)
    version: Literal[1] = 1


def _normalise_base_path(base_path: str) -> str:
    if not base_path or base_path == "/":
        return ""
    normalised = "/" + base_path.strip("/")
    if any(part in {".", ".."} for part in normalised.split("/")):
        raise ValueError("base_path must not contain '.' or '..' segments")
    return normalised


def _serialisable(value: Any) -> Any:
    """Convert service results without teaching the API their storage shape."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return jsonable_encoder(value)


async def _resolve(value: _T | Awaitable[_T]) -> _T:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_service(
    operation: Callable[..., Any],
    *args: Any,
    missing_message: str,
    **kwargs: Any,
) -> Any:
    try:
        return await _resolve(operation(*args, **kwargs))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=missing_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _enum_value(enum_type: type[Any] | None, value: str) -> Any:
    if enum_type is None:
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        try:
            return enum_type[value.upper()]
        except (KeyError, TypeError):
            return value


def _service_query(payload: TrialQueryPayload) -> Any:
    """Translate the HTTP model into dashboard-model dataclasses when present."""

    raw = payload.model_dump()
    try:
        from hyperoptax.dashboard.models import (  # type: ignore[attr-defined]
            FilterOperator,
            SortDirection,
            TrialFilter,
            TrialQuery,
            TrialSort,
        )
    except (ImportError, AttributeError):
        # This fallback keeps create_app useful for lightweight external query
        # services and makes the web layer independent in isolation tests.
        return raw

    filters = tuple(
        TrialFilter(
            field=item.field,
            op=_enum_value(FilterOperator, item.op),
            value=item.value,
        )
        for item in payload.filters
    )
    sort = tuple(
        TrialSort(
            field=item.field,
            direction=_enum_value(SortDirection, item.direction),
        )
        for item in payload.sort
    )
    return TrialQuery(
        columns=None if payload.columns is None else tuple(payload.columns),
        filters=filters,
        sort=sort,
        limit=payload.limit,
        version=payload.version,
    )


def _changes_operation(service: Any) -> Callable[..., Any]:
    for name in ("get_changes", "changes", "changes_since"):
        operation = getattr(service, name, None)
        if operation is not None:
            return operation
    raise RuntimeError(
        "StudyQueryService must provide get_changes(), changes(), or changes_since()."
    )


def _build_dashboard_app(service: Any, static_dir: Path) -> FastAPI:
    app = FastAPI(
        title="Hyperoptax dashboard API",
        version="1",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )
    app.state.query_service = service

    app.mount(
        "/static",
        StaticFiles(directory=static_dir, check_dir=True),
        name="static",
    )

    @app.get("/api/v1/studies")
    async def list_studies() -> Any:
        result = await _call_service(
            service.list_studies,
            missing_message="No studies were found.",
        )
        return _serialisable(result)

    @app.get("/api/v1/studies/{study_id}")
    async def describe_study(study_id: str) -> Any:
        result = await _call_service(
            service.describe_study,
            study_id,
            missing_message=f"Study {study_id!r} was not found.",
        )
        return _serialisable(result)

    @app.get("/api/v1/studies/{study_id}/fields")
    async def available_fields(study_id: str) -> Any:
        result = await _call_service(
            service.available_fields,
            study_id,
            missing_message=f"Study {study_id!r} was not found.",
        )
        return {"fields": _serialisable(result), "query_version": 1}

    @app.get("/api/v1/studies/{study_id}/importance")
    async def hyperparameter_importance(study_id: str) -> Any:
        result = await _call_service(
            service.hyperparameter_importance,
            study_id,
            missing_message=f"Study {study_id!r} was not found.",
        )
        return _serialisable(result)

    @app.get("/api/v1/studies/{study_id}/pareto")
    async def pareto_front(
        study_id: str,
        x_field: str = Query(default="duration_seconds", min_length=1),
        y_field: str = Query(default="objective_value", min_length=1),
        x_direction: Literal["minimize", "maximize"] = "minimize",
        y_direction: Literal["minimize", "maximize"] | None = None,
    ) -> Any:
        result = await _call_service(
            service.pareto_front,
            study_id,
            x_field=x_field,
            y_field=y_field,
            x_direction=x_direction,
            y_direction=y_direction,
            missing_message=f"Study {study_id!r} was not found.",
        )
        return _serialisable(result)

    @app.post("/api/v1/studies/{study_id}/trials/query")
    async def query_trials(study_id: str, query: TrialQueryPayload) -> Any:
        try:
            service_query = _service_query(query)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = await _call_service(
            service.query_trials,
            study_id,
            service_query,
            missing_message=f"Study {study_id!r} was not found.",
        )
        return _serialisable(result)

    @app.get("/api/v1/studies/{study_id}/trials/{trial_id}")
    async def get_trial(study_id: str, trial_id: str) -> Any:
        result = await _call_service(
            service.get_trial,
            study_id,
            trial_id,
            missing_message=(
                f"Trial {trial_id!r} was not found in study {study_id!r}."
            ),
        )
        return _serialisable(result)

    @app.get("/api/v1/studies/{study_id}/changes")
    async def study_changes(
        study_id: str,
        since_revision: int = Query(default=0, ge=0),
    ) -> Any:
        result = await _call_service(
            _changes_operation(service),
            study_id,
            since_revision,
            missing_message=f"Study {study_id!r} was not found.",
        )
        return _serialisable(result)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


def create_app(
    service: Any,
    *,
    base_path: str = "",
    static_dir: Path | None = None,
) -> FastAPI:
    """Create the loopback-ready dashboard app for a query service.

    ``base_path`` mounts the complete dashboard beneath a reverse-proxy or
    JupyterHub prefix. The static application derives API URLs from its own
    module URL, so the same files work at the root and under a prefix.
    """

    assets = static_dir or Path(__file__).with_name("static")
    dashboard = _build_dashboard_app(service, assets)
    prefix = _normalise_base_path(base_path)
    if not prefix:
        return dashboard

    root = FastAPI(
        title="Hyperoptax dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    root.mount(prefix, dashboard)

    @root.get("/", include_in_schema=False)
    async def redirect_to_dashboard() -> RedirectResponse:
        return RedirectResponse(f"{prefix}/")

    root.state.query_service = service
    root.state.dashboard_app = dashboard
    return root


__all__ = ["TrialFilterPayload", "TrialQueryPayload", "TrialSortPayload", "create_app"]
