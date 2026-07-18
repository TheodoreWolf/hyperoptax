"""Filtered JSON and CSV export through the dashboard query contract."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from hyperoptax.dashboard._file_utils import (
    discard_file,
    paths_refer_to_same_file,
    prepare_destination,
    publish_file,
    temporary_path_for,
)
from hyperoptax.dashboard.models import (
    JsonObject,
    JsonValue,
    TrialQuery,
    TrialQueryResult,
)
from hyperoptax.dashboard.queries import (
    DEFAULT_TRIAL_COLUMNS,
    StudyQueryService,
)
from hyperoptax.dashboard.store import StudyStore

ExportSource: TypeAlias = StudyQueryService | StudyStore | str | Path


class ExportFormat(str, Enum):
    """Dependency-light tabular export formats."""

    JSON = "json"
    CSV = "csv"


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Metadata for a completely published filtered export."""

    output: Path
    format: ExportFormat
    columns: tuple[str, ...]
    rows_total: int
    rows_exported: int
    study_revision: int


def export_study_data(
    source: ExportSource,
    study_id: str,
    output: str | Path,
    *,
    query: TrialQuery | Mapping[str, Any] | None = None,
    format: ExportFormat | str | None = None,
    overwrite: bool = False,
) -> ExportResult:
    """Run one bounded ``TrialQuery`` and atomically export exactly its rows.

    JSON preserves the full ``TrialQueryResult``, including fidelity metadata.
    CSV contains the same rows with a deterministic header: explicit query
    columns retain their requested order, otherwise the standard trial column
    order is used.
    """

    file_format = _resolve_format(output, format)
    normalized_query = _coerce_query(query)
    if isinstance(source, (str, Path)) and paths_refer_to_same_file(source, output):
        raise ValueError("export source database and output must be different files")

    service, close_service = _query_service(source)
    try:
        store_path = getattr(service.store, "path", None)
        if store_path is not None and paths_refer_to_same_file(store_path, output):
            raise ValueError(
                "export source database and output must be different files"
            )
        destination = prepare_destination(output, overwrite=overwrite)
        result = service.query_trials(study_id, normalized_query)
        columns = (
            tuple(normalized_query.columns)
            if normalized_query.columns is not None
            else DEFAULT_TRIAL_COLUMNS
        )
        temporary = temporary_path_for(destination)
        try:
            if file_format == ExportFormat.JSON:
                _write_json(temporary, result)
            else:
                _write_csv(temporary, result, columns)
            publish_file(temporary, destination, overwrite=overwrite)
        finally:
            discard_file(temporary)
    finally:
        if close_service:
            service.close()

    return ExportResult(
        output=destination,
        format=file_format,
        columns=columns,
        rows_total=result.rows_total,
        rows_exported=result.rows_returned,
        study_revision=result.study_revision,
    )


def trial_query_result_payload(result: TrialQueryResult) -> JsonObject:
    """Serialize a query result without adding export-only wrapper fields."""

    return {
        "rows": [dict(row) for row in result.rows],
        "rows_total": result.rows_total,
        "rows_returned": result.rows_returned,
        "sampled": result.sampled,
        "sampling_method": result.sampling_method,
        "aggregation": result.aggregation,
        "point_limit": result.point_limit,
        "study_revision": result.study_revision,
    }


def _query_service(source: ExportSource) -> tuple[StudyQueryService, bool]:
    if isinstance(source, StudyQueryService):
        return source, False
    if isinstance(source, (str, Path)):
        return StudyQueryService(source), True
    return StudyQueryService(source), False


def _coerce_query(
    query: TrialQuery | Mapping[str, Any] | None,
) -> TrialQuery:
    if query is None:
        return TrialQuery()
    if isinstance(query, TrialQuery):
        return query
    if isinstance(query, Mapping):
        return TrialQuery(**query)
    raise TypeError("query must be a TrialQuery, mapping, or None")


def _resolve_format(
    output: str | Path,
    file_format: ExportFormat | str | None,
) -> ExportFormat:
    if isinstance(file_format, ExportFormat):
        return file_format
    if file_format is not None:
        try:
            return ExportFormat(file_format.lower().removeprefix("."))
        except ValueError as error:
            raise ValueError(f"unsupported export format: {file_format!r}") from error
    suffix = Path(output).suffix.lower().removeprefix(".")
    try:
        return ExportFormat(suffix)
    except ValueError as error:
        raise ValueError(
            "export format must be supplied or inferred from a .json/.csv suffix"
        ) from error


def _write_json(path: Path, result: TrialQueryResult) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            trial_query_result_payload(result),
            stream,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(
    path: Path,
    result: TrialQueryResult,
    columns: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in result.rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})
        stream.flush()
        os.fsync(stream.fileno())


def _csv_value(value: JsonValue) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


__all__ = [
    "ExportFormat",
    "ExportResult",
    "export_study_data",
    "trial_query_result_payload",
]
