"""One background task per job: admission, planning, dubbing and the final state derived from artefacts."""

from __future__ import annotations

import asyncio
import logging
import math
from enum import Enum
from pathlib import Path

from sunwai.server.atomic import JsonObject
from sunwai.server.clock import utcnow
from sunwai.server.config import ServerConfig
from sunwai.server.paths import JobPaths
from sunwai.server.repository import JobRepository
from sunwai.server.scheduling import JobAdmission
from sunwai.server.services.artifacts import ArtifactChecks
from sunwai.server.services.chunks import ChunkDispatcher
from sunwai.server.services.health import HealthService
from sunwai.server.services.planning import PlannerService

log = logging.getLogger(__name__)

MAX_PASSES = 12


class PassOutcome(Enum):
    """How one dubbing pass ended."""

    COMPLETE = "complete"
    LANGS_CHANGED = "langs_changed"
    ABORTED = "aborted"


class JobRunner:
    """Owns the per-job asyncio tasks. Idempotent: running a job again after any crash is safe."""

    def __init__(
        self,
        config: ServerConfig,
        repo: JobRepository,
        paths: JobPaths,
        artifacts: ArtifactChecks,
        admission: JobAdmission,
        planner: PlannerService,
        dispatcher: ChunkDispatcher,
        health: HealthService,
    ) -> None:
        """Create the runner.

        Args:
            config: Server configuration (admission limits).
            repo: Job storage.
            paths: Directory resolver.
            artifacts: Completion checks.
            admission: Admission controller.
            planner: Planning service.
            dispatcher: Chunk dispatcher.
            health: Source of the "services down" hint appended to error messages.
        """
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._config = config
        self._repo = repo
        self._paths = paths
        self._artifacts = artifacts
        self._admission = admission
        self._planner = planner
        self._dispatcher = dispatcher
        self._health = health

    def ensure_running(self, job_id: str) -> str:
        """Start (or keep) the single run task of ``job_id``.

        Returns:
            ``running`` when a task is active or was started, ``cached`` when there was nothing left to do.
        """
        task = self.tasks.get(job_id)
        if task is not None and not task.done():
            return "running"
        job_dir = self._paths.job_dir(job_id)
        if not self._artifacts.job_needs_work(job_dir):
            if self._repo.read_status(job_dir).get("state") != "done":
                self._repo.update_status(job_dir, state="done", stage="done", finished=utcnow(), error_tail=None)
            return "cached"
        self._repo.update_status(
            job_dir, state="queued", stage="queued", queued_at=utcnow(), finished=None, error_tail=None
        )
        self.tasks[job_id] = asyncio.create_task(self.run_job(job_id))
        return "running"

    async def shutdown(self) -> None:
        """Cancel every job task; the cancel path kills the child process groups."""
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_job(self, job_id: str) -> None:
        """Wait for admission, (re)plan if needed, dub every missing chunk of every language, set the final state.

        The languages are re-read after each pass, so languages added meanwhile are picked up.

        Args:
            job_id: Job to run.
        """
        job_dir = self._paths.job_dir(job_id)
        try:
            await self._admission.admit(job_id)
            if await self._dub_until_stable(job_id, job_dir):
                await self._finalize(job_dir)
        except asyncio.CancelledError:
            raise  # shutdown: state stays resumable, reconcile picks it up
        except Exception as exc:
            log.exception("job %s failed", job_id)
            try:
                self._fail(job_dir, "error", f"server error: {exc!r}"[:1500])
            except Exception:
                pass
        finally:
            self.tasks.pop(job_id, None)
            await self._admission.release(job_id)

    def _fail(self, job_dir: Path, stage: str, message: str) -> None:
        self._repo.update_status(job_dir, state="error", stage=stage, finished=utcnow(), error_tail=message)

    async def _dub_until_stable(self, job_id: str, job_dir: Path) -> bool:
        """Run passes until the language set stops changing; False when a pass aborted with an error status."""
        for _ in range(MAX_PASSES):
            outcome = await self._run_pass(job_id, job_dir)
            if outcome is not PassOutcome.LANGS_CHANGED:
                return outcome is PassOutcome.COMPLETE
        return True

    async def _run_pass(self, job_id: str, job_dir: Path) -> PassOutcome:
        meta = self._repo.read_active_meta(job_dir)
        if not meta:
            self._fail(job_dir, "error", "job.json missing")
            return PassOutcome.ABORTED
        langs = list(meta["langs"])
        if not await self._planner.ensure_plan(job_dir, meta):
            return PassOutcome.ABORTED
        plan = self._repo.read_plan(job_dir)
        if not self._within_limits(job_dir, plan, langs):
            return PassOutcome.ABORTED
        pairs = self._artifacts.missing_pairs(job_dir, langs, plan)
        if pairs:
            self._repo.update_status(job_dir, state="running", stage="dubbing", finished=None, error_tail=None)
            await self._dispatcher.enqueue(job_id, job_dir, pairs)
        newer = self._repo.read_active_meta(job_dir)
        if newer and set(newer["langs"]) != set(langs):
            return PassOutcome.LANGS_CHANGED  # languages were added while we worked
        return PassOutcome.COMPLETE

    @staticmethod
    def _plan_length(plan: JsonObject) -> float:
        """End of the last planned chunk in seconds; infinity when the plan is unusable."""
        try:
            ends = [float(c["start"]) + float(c["dur"]) for c in plan["chunks"]]
            return max(ends) if all(math.isfinite(e) and e >= 0 for e in ends) else float("inf")
        except Exception:
            return float("inf")

    def _within_limits(self, job_dir: Path, plan: JsonObject, langs: list[str]) -> bool:
        """Apply the demo's video-length and total-work limits; record an error status when exceeded."""
        cfg = self._config
        length = self._plan_length(plan)
        if (
            cfg.admission
            and math.isfinite(length)
            and length <= cfg.max_video_s
            and length * len(langs) > cfg.max_work_s
        ):
            self._fail(
                job_dir,
                "planning",
                f"this demo dubs at most {cfg.max_work_s:.0f} s of video x languages per job; "
                f"{length:.0f} s x {len(langs)} languages is too much. Pick a shorter range or fewer languages.",
            )
            return False
        if not math.isfinite(length) or (cfg.admission and length > cfg.max_video_s):
            self._fail(
                job_dir,
                "planning",
                f"video is {length:.0f}s; the limit is {cfg.max_video_s:.0f}s. Pick a start/end range.",
            )
            return False
        return True

    async def _finalize(self, job_dir: Path) -> None:
        """Derive the final job state from the artefacts that exist."""
        meta = self._repo.read_active_meta(job_dir)
        plan = self._repo.read_plan(job_dir)
        left = self._artifacts.missing_pairs(job_dir, meta.get("langs", []), plan)
        if not left:
            self._repo.update_status(job_dir, state="done", stage="done", finished=utcnow(), error_tail=None)
            return
        failed = []
        for lang, chunk in left:
            out_dir = self._paths.resolve_chunk_dir(job_dir, lang, chunk["idx"])
            if out_dir and self._repo.read_status(out_dir).get("state") == "error":
                failed.append(f"{lang}/c{chunk['idx']:04d}")
        hint = await self._health.down_hint()
        summary = ("chunks failed: " + ", ".join(failed[:12])) if failed else "Incomplete state"
        self._fail(job_dir, "dubbing", summary + hint)
