"""Command line entry point: ``sunwai serve | dub | worker lipsync | config``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from sunwai.core.config import Settings
from sunwai.core.legacy_env import LEGACY_VARS, export_to_environ


def _redacted(settings: Settings) -> dict[str, object]:
    """Return the effective settings with secrets masked and unset fields omitted."""
    dumped = json.loads(settings.model_dump_json())
    return {g: {k: v for k, v in fields.items() if v is not None} for g, fields in dumped.items() if fields}


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = Settings()
    export_to_environ(settings)
    port = args.port or settings.server.port or 18941
    uvicorn.run("sunwai.server.app:create_app", factory=True, host=args.host, port=port)
    return 0


def _dub(args: argparse.Namespace) -> int:
    from sunwai.pipeline.context import ChunkContext, ChunkRequest
    from sunwai.pipeline.workflow import DubWorkflow

    settings = Settings()
    export_to_environ(settings)
    request = ChunkRequest(url=args.url, language=args.lang, start_s=args.start, duration_s=args.dur)
    out = DubWorkflow().run(ChunkContext(request=request, work_dir=Path(args.out), settings=settings))
    print(out)
    return 0


def _worker(args: argparse.Namespace) -> int:
    settings = Settings()
    export_to_environ(settings)
    from sunwai.workers import lipsync

    sys.argv = ["lipsync", "--root", args.root, "--id", args.id] + (["--once"] if args.once else [])
    lipsync.main()  # type: ignore[no-untyped-call]
    return 0


def _config(args: argparse.Namespace) -> int:
    if args.action == "show":
        print(json.dumps(_redacted(Settings()), indent=2))
    else:
        for var in LEGACY_VARS:
            print(f"{var.canonical:60s} <- {var.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (separate for testing)."""
    parser = argparse.ArgumentParser(prog="sunwai", description="SunwAI Indic AI dubbing")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.set_defaults(func=_serve)

    dub = sub.add_parser("dub", help="dub one chunk from a URL")
    dub.add_argument("--url", required=True)
    dub.add_argument("--lang", required=True)
    dub.add_argument("--out", required=True)
    dub.add_argument("--start", type=float, default=0.0)
    dub.add_argument("--dur", type=float, default=90.0)
    dub.set_defaults(func=_dub)

    worker = sub.add_parser("worker", help="run the persistent lip-sync worker")
    worker.add_argument("kind", choices=["lipsync"])
    worker.add_argument("--root", required=True)
    worker.add_argument("--id", default="w1")
    worker.add_argument("--once", action="store_true")
    worker.set_defaults(func=_worker)

    config = sub.add_parser("config", help="show effective settings or the variable mapping")
    config.add_argument("action", choices=["show", "map"])
    config.set_defaults(func=_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
