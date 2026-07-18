"""Consistent snapshots of live local SQLite study databases."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from hyperoptax.dashboard._file_utils import (
    discard_file,
    paths_refer_to_same_file,
    prepare_destination,
    publish_file,
    temporary_path_for,
)


class SnapshotError(RuntimeError):
    """Raised when SQLite cannot produce a verified backup."""


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Metadata for one completely published database snapshot."""

    source: Path
    output: Path
    size_bytes: int


def snapshot_database(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> SnapshotResult:
    """Create and atomically publish a consistent SQLite online backup.

    The backup is written and checked under a private temporary name in the
    output directory. The requested output path becomes visible only after the
    online backup has completed successfully.
    """

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source database does not exist: {source_path}")
    if paths_refer_to_same_file(source_path, output):
        raise ValueError("snapshot source and output must be different files")
    destination = prepare_destination(output, overwrite=overwrite)
    temporary = temporary_path_for(destination)
    try:
        _backup_database(source_path, temporary)
        publish_file(temporary, destination, overwrite=overwrite)
    finally:
        discard_file(temporary)
    return SnapshotResult(
        source=source_path,
        output=destination,
        size_bytes=destination.stat().st_size,
    )


def _backup_database(source: Path, destination: Path) -> None:
    source_uri = f"{source.as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(source_uri, uri=True)) as source_connection:
            with closing(sqlite3.connect(destination)) as destination_connection:
                source_connection.backup(destination_connection)
                check = destination_connection.execute("PRAGMA quick_check").fetchone()
                if check is None or check[0] != "ok":
                    detail = "no result" if check is None else str(check[0])
                    raise SnapshotError(f"snapshot integrity check failed: {detail}")
                destination_connection.commit()
    except sqlite3.Error as error:
        raise SnapshotError(f"SQLite backup failed: {error}") from error


__all__ = ["SnapshotError", "SnapshotResult", "snapshot_database"]
