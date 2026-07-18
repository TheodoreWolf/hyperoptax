"""Command-line entry point for the local Hyperoptax dashboard."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperoptax",
        description="Inspect Hyperoptax study results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dashboard = commands.add_parser(
        "dashboard",
        help="Serve a read-only dashboard for a local study database.",
    )
    dashboard.add_argument("database", type=Path, help="Existing SQLite database")
    dashboard.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    dashboard.add_argument(
        "--port",
        default=8080,
        type=int,
        choices=range(1, 65_536),
        metavar="PORT",
        help="Preferred port (default: 8080; falls back when unavailable)",
    )
    dashboard.add_argument(
        "--base-path",
        default="",
        help="Optional URL prefix for JupyterHub or a reverse proxy",
    )

    snapshot = commands.add_parser(
        "snapshot",
        help="Create a consistent copy of a live dashboard database.",
    )
    snapshot.add_argument("database", type=Path, help="Source SQLite database")
    snapshot.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination SQLite snapshot",
    )
    snapshot.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )

    export = commands.add_parser(
        "export",
        help="Export a bounded, filtered study query to JSON or CSV.",
    )
    export.add_argument("database", type=Path, help="Source SQLite database")
    export.add_argument("--study", required=True, help="Study UUID to export")
    export.add_argument("--output", type=Path, required=True, help="Output file")
    export.add_argument(
        "--format",
        choices=("json", "csv"),
        help="Output format (otherwise inferred from the output suffix)",
    )
    export.add_argument(
        "--query",
        help=("TrialQuery JSON object, or @PATH to read one from a UTF-8 JSON file"),
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser


def _available_port(host: str, preferred: int) -> tuple[int, bool]:
    """Return the preferred port, or an ephemeral fallback when occupied."""

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, preferred))
        except OSError:
            probe.bind((host, 0))
            return int(probe.getsockname()[1]), True
    return preferred, False


def _display_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return f"[{host}]" if ":" in host else host


def _run_dashboard(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    database = args.database.expanduser().resolve()
    if not database.is_file():
        parser.error(f"database does not exist or is not a file: {database}")

    try:
        import uvicorn

        from hyperoptax.dashboard.api import create_app
        from hyperoptax.dashboard.queries import StudyQueryService
    except ImportError as exc:
        parser.error(
            "dashboard dependencies are unavailable; install hyperoptax[dashboard]"
        )
        raise AssertionError("argparse exits") from exc

    if args.host in {"0.0.0.0", "::"}:
        print(
            "WARNING: the dashboard has no built-in authentication. "
            "Use loopback plus SSH forwarding, or put it behind an "
            "authenticated TLS proxy.",
            file=sys.stderr,
        )

    try:
        service = StudyQueryService(database)
        app = create_app(service, base_path=args.base_path)
        port, used_fallback = _available_port(args.host, args.port)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    prefix = "/" + args.base_path.strip("/") if args.base_path.strip("/") else ""
    if used_fallback:
        print(
            f"Port {args.port} is unavailable; using {port} instead.",
            file=sys.stderr,
        )
    print(f"Hyperoptax dashboard: http://{_display_host(args.host)}:{port}{prefix}/")
    uvicorn.run(app, host=args.host, port=port, log_level="info")
    return 0


def _query_payload(value: str | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"query file does not exist: {path}")
        value = path.read_text(encoding="utf-8")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("query must be a JSON object")
    return payload


def _run_snapshot(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from hyperoptax.dashboard.snapshot import snapshot_database

    try:
        result = snapshot_database(
            args.database,
            args.output,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Snapshot created: {result.output} ({result.size_bytes} bytes)")
    return 0


def _run_export(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from hyperoptax.dashboard.export import export_study_data

    try:
        query = _query_payload(args.query)
        result = export_study_data(
            args.database,
            args.study,
            args.output,
            query=query,
            format=args.format,
            overwrite=args.overwrite,
        )
    except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Exported {result.rows_exported}/{result.rows_total} rows to "
        f"{result.output} ({result.format.value}, revision {result.study_revision})"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Hyperoptax command-line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "dashboard":
        return _run_dashboard(args, parser)
    if args.command == "snapshot":
        return _run_snapshot(args, parser)
    if args.command == "export":
        return _run_export(args, parser)
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["main"]
