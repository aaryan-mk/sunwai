"""Dubbing the missing (language, chunk) pairs of a job through the pipeline subprocess."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from sunwai.server.clock import utcnow
from sunwai.server.config import ServerConfig
from sunwai.server.paths import JobPaths
from sunwai.server.repository import JobRepository
from sunwai.server.scheduling import FocusRegistry, PriorityGate
from sunwai.server.services.artifacts import ArtifactChecks, Pair
from sunwai.server.services.process import ProcessRunner

log = logging.getLogger(__name__)

PIPELINE_MODULE = "sunwai.pipeline.engine"
STAGE_RE = re.compile(r"Stage\s+([A-Za-z0-9_\-]+)")


def stage_from_log(text: str) -> str:
    """Return the last pipeline stage named in ``text``, ``starting`` if none."""
    last = "starting"
    for line in text.splitlines():
        match = STAGE_RE.search(line)
        if match:
            last = match.group(1)
    return last


class ChunkDispatcher:
    """Runs chunk work with bounded fan-out; the viewer's focus decides what runs first."""

    def __init__(
        self,
        config: ServerConfig,
        repo: JobRepository,
        paths: JobPaths,
        artifacts: ArtifactChecks,
        process: ProcessRunner,
        gate: PriorityGate,
        focus: FocusRegistry,
    ) -> None:
        """Create the dispatcher.

        Args:
            config: Server configuration (fan-out, timeouts, job root).
            repo: Job storage.
            paths: Directory resolver.
            artifacts: Completion checks.
            process: Subprocess runner.
            gate: Gate that limits concurrently running pipeline processes.
            focus: Viewer focus used to rank tasks.
        """
        self._config = config
        self._repo = repo
        self._paths = paths
        self._artifacts = artifacts
        self._process = process
        self._gate = gate
        self._focus = focus

    async def enqueue(self, job_id: str, job_dir: Path, pairs: list[Pair]) -> None:
        """Run every missing pair with at most ``fanout`` tasks alive at once (bounded memory and queue).

        Whenever a slot frees, the next task is the one with the best CURRENT priority; the shared gate still decides
        which alive task gets a pipeline slot.

        Args:
            job_id: Job identifier (focus lookups).
            job_dir: Job directory holding ``source.mp4``.
            pairs: ``(lang, chunk)`` pairs still to dub.
        """
        source_file = job_dir / "source.mp4"
        if not source_file.exists():
            return
        pending = sorted(pairs, key=lambda p: (p[1]["idx"], p[0]))  # chunk-major: ties resolve in playback order
        alive: set[asyncio.Future[None]] = set()
        try:
            while pending or alive:
                while pending and len(alive) < self._config.fanout:
                    best = min(pending, key=lambda p: self._focus.priority(job_id, p[0], p[1]["idx"]))
                    pending.remove(best)
                    alive.add(asyncio.ensure_future(self.run_chunk(job_dir, str(source_file), best[0], best[1])))
                done, _ = await asyncio.wait(alive, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    alive.discard(task)
                    if not task.cancelled() and task.exception() is not None:
                        log.error("[enqueue_chunks] %s: task error %r", job_id, task.exception())
        except BaseException:
            for task in alive:
                task.cancel()
            await asyncio.gather(*alive, return_exceptions=True)
            raise

    async def run_chunk(self, job_dir: Path, url: str, lang: str, chunk: dict[str, Any]) -> None:
        """Dub one chunk in one language and publish its final state.

        Args:
            job_dir: Job directory.
            url: Source video path or URL handed to the pipeline.
            lang: Target language.
            chunk: Planned chunk (``idx``, ``start``, ``dur``).
        """
        out_dir = self._paths.resolve_chunk_dir(job_dir, lang, chunk["idx"])
        if not out_dir:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._artifacts.chunk_done(out_dir, lang):
            return
        attempt = uuid.uuid4().hex[:8]  # only the owning attempt may publish a final state
        previous = self._repo.read_status(out_dir)
        self._repo.update_status(
            out_dir,
            state="queued",
            stage="queued",
            started=None,
            finished=None,
            error_tail=None,
            version=int(previous.get("version", 0) or 0),
            attempt=attempt,
            pid=None,
            editing=False,
        )
        rc, tail = await self._run_pipeline(job_dir, out_dir, url, lang, chunk)
        if self._repo.read_status(out_dir).get("attempt") != attempt:
            return  # superseded by a newer attempt: do not clobber it
        self._publish_result(out_dir, lang, rc, tail)

    async def _run_pipeline(
        self, job_dir: Path, out_dir: Path, url: str, lang: str, chunk: dict[str, Any]
    ) -> tuple[int, str]:
        args = [
            "--url",
            url,
            "--lang",
            lang,
            "--out",
            str(out_dir),
            "--start",
            str(chunk["start"]),
            "--dur",
            str(chunk["dur"]),
        ]
        env = {"SPEAKER_MAP": str(job_dir / "speakers_global.json"), "CHUNK_START": str(chunk["start"])}
        await self._gate.acquire(job_dir.name, lang, chunk["idx"])
        try:
            self._repo.update_status(out_dir, state="running", stage="running", started=utcnow())
            try:
                return await self._process.run(
                    PIPELINE_MODULE,
                    args,
                    self._config.root,
                    out_dir / "pipeline.log",
                    extra_env=env,
                    timeout=self._config.pipeline_timeout_s,
                    kind="chunk",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return 1, f"spawn error: {exc!r}"
        finally:
            self._gate.release()

    def _publish_result(self, out_dir: Path, lang: str, rc: int, tail: str) -> None:
        stage = stage_from_log(tail)
        mp4 = out_dir / f"dub_{lang}.mp4"
        if rc == 0 and mp4.is_file() and mp4.stat().st_size > 0:
            version = int(self._repo.read_status(out_dir).get("version", 0) or 0) + 1
            self._repo.update_status(
                out_dir, state="done", stage=stage or "done", finished=utcnow(), error_tail=None, version=version
            )
        elif rc == 0:
            self._repo.update_status(
                out_dir,
                state="error",
                stage=stage or "done",
                finished=utcnow(),
                error_tail="exited 0 but dub mp4 was not produced or empty\n" + (tail[-2000:] or ""),
            )
        else:
            self._repo.update_status(
                out_dir,
                state="error",
                stage=stage or "error",
                finished=utcnow(),
                error_tail=tail[-2000:] or f"exit {rc}",
            )
