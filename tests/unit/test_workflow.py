"""Stage interface, runner policy and workflow ordering (no model calls)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from sunwai.core.config import Settings
from sunwai.core.deadlines import CancellationToken, Deadline
from sunwai.core.errors import Cancelled, DeadlineExceeded
from sunwai.pipeline.context import ChunkContext, ChunkRequest
from sunwai.pipeline.runner import Pipeline
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages import MtStage
from sunwai.pipeline.workflow import STAGE_ORDER


class Recorder:
    """Timing sink that remembers what it was given."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float]] = []

    def record(self, work_dir: Path, key: str, seconds: float) -> None:
        self.rows.append((key, seconds))


class Upper:
    name = "upper"
    policy = StagePolicy(timing_key="upper")

    def run(self, value: str, ctx: ChunkContext) -> str:
        return value.upper()

    def validate(self, value: str, ctx: ChunkContext) -> None:
        return None


def make_ctx(tmp_path: Path, **kw: Any) -> ChunkContext:
    return ChunkContext(request=ChunkRequest("u", "hi"), work_dir=tmp_path, settings=Settings(), **kw)


def test_runner_runs_validates_and_times(tmp_path: Path) -> None:
    sink = Recorder()
    assert Pipeline(sink).execute(Upper(), "ab", make_ctx(tmp_path)) == "AB"
    assert [k for k, _ in sink.rows] == ["upper"]


def test_runner_honours_cancellation_and_deadline(tmp_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        Pipeline(Recorder()).execute(Upper(), "x", make_ctx(tmp_path, cancellation=token))
    with pytest.raises(DeadlineExceeded):
        Pipeline(Recorder()).execute(Upper(), "x", make_ctx(tmp_path, deadline=Deadline.after(-1)))


def test_engine_adapter_delegates_with_legacy_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeEngine:
        def stage_mt(self, args: Any, out_dir: Path, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen.update(lang=args.lang, out=args.out, n=len(segments))
            return segments

    monkeypatch.setattr("sunwai.pipeline.stages.base.engine", lambda: FakeEngine())
    monkeypatch.setattr("sunwai.pipeline.stages.mt.engine", lambda: FakeEngine())
    out = MtStage().run([{"id": 0}], make_ctx(tmp_path))
    assert out == [{"id": 0}] and seen == {"lang": "hi", "out": str(tmp_path), "n": 1}


def test_workflow_order_matches_engine_main() -> None:
    """The workflow must call the stages in the same order as ``engine.main`` (the production path)."""
    src = (Path(__file__).parents[2] / "src/sunwai/pipeline/engine.py").read_text(encoding="utf-8")
    main = src[src.index("def main():") :]
    calls = re.findall(r"segments = (?:prep_shared|stage_(\w+))\(|(stage_separate)\(|(out_mp4) = stage_mix", main)
    order = ["prepare" if not a and not b and not c else (a or ("separate" if b else "mix")) for a, b, c in calls]
    assert tuple(order) == STAGE_ORDER
