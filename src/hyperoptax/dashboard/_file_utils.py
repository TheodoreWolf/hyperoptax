"""Atomic local-file publication helpers for dashboard artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def paths_refer_to_same_file(left: str | Path, right: str | Path) -> bool:
    """Return whether two paths resolve to the same name or existing inode."""

    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    if left_path.resolve() == right_path.resolve():
        return True
    if os.path.lexists(left_path) and os.path.lexists(right_path):
        try:
            return os.path.samefile(left_path, right_path)
        except OSError:
            return False
    return False


def prepare_destination(path: str | Path, *, overwrite: bool) -> Path:
    """Resolve a destination and reject an existing target by default."""

    requested = Path(path).expanduser()
    # Resolve the parent for a stable absolute result, but do not follow the
    # final component: overwriting a symlink must replace the link itself, not
    # mutate an unrelated target.
    destination = requested.parent.resolve() / requested.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    if os.path.lexists(destination) and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    return destination


def temporary_path_for(destination: Path) -> Path:
    """Create a private temporary file beside its eventual destination."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def publish_file(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    """Atomically publish a complete file without accidental replacement."""

    if overwrite:
        os.replace(temporary, destination)
        return

    # A hard link is an atomic no-replace publication because the temporary
    # file lives in the same directory/filesystem. ``os.replace`` would have a
    # race between the earlier existence check and publication.
    os.link(temporary, destination)
    temporary.unlink()


def discard_file(path: Path) -> None:
    """Remove an unpublished temporary file if it still exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


__all__ = [
    "discard_file",
    "paths_refer_to_same_file",
    "prepare_destination",
    "publish_file",
    "temporary_path_for",
]
