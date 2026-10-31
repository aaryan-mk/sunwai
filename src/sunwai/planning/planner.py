#!/usr/bin/env python3
"""Prepare a video for chunked progressive dubbing.

The planner runs seven cached stages; each is skipped when its validated output already exists:

1. fetch    -> ``raw.mp4`` + ``meta.json`` + ``source.mp4``      (:mod:`sunwai.planning.fetch`)
2. audio    -> ``source16k.wav``                                  (:mod:`sunwai.planning.audio`)
3. vad      -> ``vad.json``                                       (:mod:`sunwai.planning.vad`)
4. chunks   -> ``plan.json`` (timing plan)                        (:mod:`sunwai.planning.chunks`)
5. diarize  -> speaker turns, whole file, once                    (:mod:`sunwai.planning.diarization`)
6. gender   -> per-speaker gender                                 (:mod:`sunwai.planning.gender`)
7. write    -> ``speakers_global.json`` (+ voice plan)            (:mod:`sunwai.planning.speakers`)

CLI::

    python -m sunwai.planning.planner --url URL_OR_LOCAL_FILE --out JOB_DIR [--start 0] [--dur 0] [--chunk 60]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from sunwai.planning.audio import stage_audio
from sunwai.planning.chunks import stage_chunks
from sunwai.planning.diarization import stage_diarize
from sunwai.planning.errors import PlannerError
from sunwai.planning.fetch import stage_fetch
from sunwai.planning.fileio import atomic_json, json_or_none, load_json, norm_path, remove_file
from sunwai.planning.gender import stage_gender
from sunwai.planning.rollout import decide_voice_rollout, make_plan_fn
from sunwai.planning.speakers import stage_write
from sunwai.planning.vad import stage_vad
from sunwai.planning.wavio import SAMPLE_RATE, read_mono16k

log = logging.getLogger("planner")

T = TypeVar("T")


@dataclass(frozen=True)
class PlannerArgs:
    """Validated planner inputs.

    Attributes:
        url: HTTP(S) URL or local file path.
        out_dir: Job directory.
        start: Cut start in seconds.
        dur: Cut length in seconds (0 = to the end).
        chunk: Target chunk length in seconds.
    """

    url: str
    out_dir: str
    start: float = 0.0
    dur: float = 0.0
    chunk: float = 60.0


class Planner:
    """Run the planning stages for one job directory and record their timings in ``plan.json``."""

    def __init__(self, args: PlannerArgs, voice_plan: bool) -> None:
        """Create a planner.

        Args:
            args: Validated inputs.
            voice_plan: Whether this job gets a voice plan (see :func:`sunwai.planning.rollout.decide_voice_rollout`).
        """
        self.args = args
        self.voice_plan = voice_plan
        self.out_dir = args.out_dir
        self.plan_path = os.path.join(self.out_dir, "plan.json")
        self.timings: dict[str, float] = {}

    def _path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def _timed(self, name: str, stage: Callable[[], T]) -> T:
        started = time.time()
        result = stage()
        elapsed = time.time() - started
        self.timings[name] = round(elapsed, 3)
        log.info("[%s] done in %.3fs", name, elapsed)
        return result

    def run(self) -> None:
        """Run every stage and log the summary."""
        a = self.args
        started = time.time()
        self._timed("fetch", lambda: stage_fetch(a.url, self.out_dir, a.start, a.dur))
        self._timed("audio", lambda: stage_audio(self.out_dir))
        self._timed("vad", lambda: stage_vad(self.out_dir))
        self._store_chunk_plan(self._timed("chunks", lambda: stage_chunks(self.out_dir, a.chunk)))
        diarized = self._timed("diarize", lambda: stage_diarize(self.out_dir))
        gender = self._timed("gender", lambda: stage_gender(self.out_dir, diarized.turns, diarized.vetted))
        backend_info = json_or_none(self._path("spkbackend.json")) or {}
        self._timed(
            "write",
            lambda: stage_write(
                self.out_dir,
                diarized.backend,
                diarized.turns,
                gender,
                degraded=backend_info.get("degraded"),
                degraded_reason=backend_info.get("degraded_reason"),
                plan_fn=make_plan_fn(self.out_dir) if self.voice_plan else None,
            ),
        )
        self._finalise_plan(diarized.backend)
        self._log_summary(time.time() - started)

    def _store_chunk_plan(self, plan: dict[str, Any]) -> None:
        """Merge the chunk plan into any existing ``plan.json``, keeping fields written by earlier runs."""
        try:
            existing = load_json(self.plan_path)
        except Exception:
            existing = {}
        for key in ("duration", "chunk_target", "n_chunks", "chunks"):
            existing[key] = plan[key]
        existing.setdefault("timings", {})
        atomic_json(self.plan_path, existing, indent=2)

    def _finalise_plan(self, backend: str) -> None:
        """Write the stage timings and the diarization summary into ``plan.json``."""
        try:
            plan = load_json(self.plan_path)
        except Exception:
            duration = len(read_mono16k(self._path("source16k.wav"))) / float(SAMPLE_RATE)
            plan = {"duration": duration, "chunk_target": self.args.chunk, "chunks": []}
        plan["timings"] = self.timings
        info = json_or_none(self._path("spkbackend.json")) or {}
        degraded = bool(info.get("degraded", str(backend).startswith("ecapa")))
        plan["diarization"] = {"backend": backend, "degraded": degraded, "reason": info.get("degraded_reason")}
        atomic_json(self.plan_path, plan, indent=2, ensure_ascii=False)

    def _log_summary(self, total_s: float) -> None:
        plan = load_json(self.plan_path)
        speakers = load_json(self._path("speakers_global.json"))
        log.info("=" * 60)
        log.info(
            "SUMMARY  duration=%.3fs  n_chunks=%d  backend=%s%s  total=%.1fs",
            plan.get("duration", 0.0),
            len(plan.get("chunks", [])),
            speakers["backend"],
            " (DEGRADED)" if speakers.get("degraded") else "",
            total_s,
        )
        for s in speakers["speakers"]:
            log.info(
                "   speaker %-3s %-8s p_female=%.3f total=%.2fs n=%d",
                s["id"], s["gender"], s["p_female"], s["total_s"], s["n_turns"],
            )  # fmt: skip
        log.info("outputs: %s", self.out_dir)
        print("ALL DONE")


def build_parser() -> argparse.ArgumentParser:
    """The planner's command-line parser."""
    parser = argparse.ArgumentParser(description="Prepare a video for chunked progressive dubbing.")
    parser.add_argument("--url", required=True, help="URL or local file")
    parser.add_argument("--out", required=True, help="JOB_DIR")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--dur", type=float, default=0.0, help="0 = to end")
    parser.add_argument("--chunk", type=float, default=60.0)
    return parser


def _report_error(argv: Sequence[str] | None, code: str, message: str) -> int:
    """Write ``planner_error.json`` (best effort), print the final ``PLANNER_ERROR`` line and return rc 3."""
    try:
        args = list(argv if argv is not None else sys.argv[1:])
        out = norm_path(args[args.index("--out") + 1])
        atomic_json(
            os.path.join(out, "planner_error.json"), {"error": code, "message": message, "ts": time.time()}, indent=2
        )
    except Exception:
        pass
    print(f"PLANNER_ERROR {code}: {message}", file=sys.stderr, flush=True)
    return 3


def _run_cli(argv: Sequence[str] | None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    ns = build_parser().parse_args(argv)
    out_dir = norm_path(ns.out)
    os.makedirs(out_dir, exist_ok=True)
    remove_file(os.path.join(out_dir, "planner_error.json"))
    voice_plan = decide_voice_rollout(out_dir)
    if not (ns.chunk > 1.0 and math.isfinite(ns.chunk)):
        raise PlannerError(f"--chunk must be a finite number > 1 s, got {ns.chunk}")
    Planner(PlannerArgs(ns.url, out_dir, ns.start, ns.dur, ns.chunk), voice_plan).run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Clean failures (no audio, too long, bad media, timeouts) write ``<out>/planner_error.json``, print a final
    ``PLANNER_ERROR <code>: <message>`` line on stderr (the server shows the log tail) and return 3.
    """
    try:
        return _run_cli(argv)
    except PlannerError as exc:
        return _report_error(argv, exc.code, str(exc))
    except SystemExit:
        raise
    except Exception as exc:
        log.exception("planner crashed")
        _report_error(argv, "internal", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
