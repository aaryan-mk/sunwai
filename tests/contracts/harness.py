"""Test harness for the HTTP route contracts: a real app on a temporary job root, with subprocesses faked."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ProcessFake = Callable[..., Awaitable[tuple[int, str]]]

JOB_ID = "0123456789ab"
CHUNKS = [{"idx": 0, "start": 0.0, "dur": 10.0}, {"idx": 1, "start": 10.0, "dur": 5.0}]


@dataclass
class Harness:
    """A running app plus the knobs tests need.

    Attributes:
        client: Test client bound to the app (startup hooks are not run).
        root: The temporary job root the app serves.
        app: The FastAPI application under test.
    """

    client: TestClient
    root: Path
    app: Any
    _set_process: Callable[[ProcessFake], None]
    _disable_runs: Callable[[], None]
    _occupy: Callable[[str], None]

    def fake_process(self, fake: ProcessFake) -> None:
        """Replace the subprocess runner (planner, pipeline, resynth) with ``fake``."""
        self._set_process(fake)

    def occupy_admission(self, job_id: str) -> None:
        """Mark ``job_id`` as running in the admission controller (fills one uncached-job slot)."""
        self._occupy(job_id)

    def disable_job_runs(self) -> None:
        """Make background job runs no-ops so creating a job never spawns work."""
        self._disable_runs()


def write_services_file(tmp: Path) -> Path:
    """Point every dependency at a closed local port so /health never reaches a real service."""
    path = tmp / "services.json"
    names = ("asr", "mt", "it2", "tts", "sep")
    path.write_text(json.dumps({**{n: ["127.0.0.1:1"] for n in names}, "lip_mode": "none"}), encoding="utf-8")
    return path


def build_harness(tmp: Path, env: dict[str, str]) -> Harness:
    """Create the application under test with ``env`` applied. Implemented per code generation."""
    from sunwai.server.app import build_app
    from sunwai.server.config import ServerConfig

    config = ServerConfig.from_environ({**os.environ, **env})
    app = build_app(config)
    container = app.state.container

    def set_process(fake: ProcessFake) -> None:
        container.process.run = fake

    def disable_runs() -> None:
        async def _noop(job_id: str) -> None:
            return None

        container.job_runner.run_job = _noop

    return Harness(
        TestClient(app), Path(env["DUB_ROOT"]), app, set_process, disable_runs, container.admission.running.append
    )


def seed_job(root: Path, langs: tuple[str, ...] = ("hi",), *, state: str = "done", lipsync: bool = True) -> Path:
    """Write a finished (or ``state``) two-chunk job under ``root`` and return its directory."""
    job = root / JOB_ID
    job.mkdir(parents=True)
    meta = {
        "job_id": JOB_ID,
        "url": "https://youtu.be/x",
        "langs": list(langs),
        "start": 0,
        "dur": 0,
        "lipsync": lipsync,
    }
    (job / "job.json").write_text(json.dumps(meta), encoding="utf-8")
    (job / "plan.json").write_text(json.dumps({"duration": 15.0, "chunks": CHUNKS, "speakers": [{"id": "S1"}]}))
    (job / "source.mp4").write_bytes(b"src")
    (job / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    for lang in langs:
        for chunk in CHUNKS:
            out = job / lang / f"c{chunk['idx']:04d}"
            out.mkdir(parents=True)
            (out / "status.json").write_text(json.dumps({"state": "done", "stage": "mix", "version": 3}))
            (out / f"dub_{lang}.mp4").write_bytes(b"dub-bytes")
            segs = [
                {"id": "s1", "start": 1.0, "end": 2.0, "speaker": "S1", "src_text": "a", "tgt_text": "b", "tempo": 1.1},
                {
                    "id": "s2",
                    "start": 3.0,
                    "end": 4.5,
                    "tgt_text": "d",
                    "history": ["old"],
                    "edited": True,
                    "overflow": True,
                    "overflow_s": 0.5,
                },
            ]
            (out / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
            src = out / "src.mp4"
            src.write_bytes(b"orig-bytes")
            os.utime(src, (1_000_000, 1_000_000))
    return job
