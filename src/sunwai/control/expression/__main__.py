"""Command line for the server's ``run_script``: the last stdout line is a JSON result."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sunwai.control.expression import ExprError, apply_candidate, do_expression_resynth, head_revision, undo_revision


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="python -m sunwai.control.expression")
    parser.add_argument("cmd", choices=["preview", "apply", "undo", "head"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--seg", required=True)
    parser.add_argument("--intent")
    parser.add_argument("--expected-revision", type=int)
    parser.add_argument("--idem")
    parser.add_argument("--cand")
    parser.add_argument("--job-id")
    return parser


def run(args: argparse.Namespace) -> Any:
    """Dispatch the sub-command and return its result."""
    if args.cmd == "preview":
        return do_expression_resynth(
            args.out, args.lang, args.seg, args.intent, args.expected_revision, args.idem, job_id=args.job_id
        )
    if args.cmd == "apply":
        return apply_candidate(args.out, args.seg, args.cand, args.expected_revision, lang=args.lang)
    if args.cmd == "undo":
        return undo_revision(args.out, args.seg, args.expected_revision, lang=args.lang)
    return head_revision(args.out, args.lang, args.seg)


def main() -> None:
    """Run the CLI, printing ``{"ok": ...}`` JSON; expression errors are reported in-band with exit code 0."""
    try:
        print(json.dumps({"ok": True, "result": run(build_parser().parse_args())}, ensure_ascii=False))
    except ExprError as exc:
        print(json.dumps({"ok": False, "status": exc.status, "error": exc.msg}))
        sys.exit(0)


if __name__ == "__main__":
    main()
