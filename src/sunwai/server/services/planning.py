"""Running the planner subprocess and keeping chunk directories consistent with the plan it wrote."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from pathlib import Path

from sunwai.server.atomic import JsonObject
from sunwai.server.clock import utcnow
from sunwai.server.config import ServerConfig
from sunwai.server.repository import JobRepository
from sunwai.server.services.artifacts import ArtifactChecks, plan_signature
from sunwai.server.services.process import ProcessRunner

log = logging.getLogger(__name__)

PLANNER_MODULE = "sunwai.planning.planner"
# Planner products that a killed planner may leave PARTIAL (the planner reuses any existing file, so a crash would
# otherwise mean corrupt reuse).
PLANNER_DERIVED = (
    "source.mp4",
    "source16k.wav",
    "vad.json",
    "plan.json",
    "speakers_global.json",
    "spkturns.json",
    "spkgender.json",
    "spkbackend.json",
    "lid.json",
)
SIGNATURE_ERROR = "cannot persist the plan signature (plan.sig); please retry"


class PlannerService:
    """Plans a job and makes its chunk directories match the plan layout.

    ``plan.sig`` records the layout the chunk directories were built for. It is committed only after a purge
    succeeded and must be persisted before any chunk starts.
    """

    def __init__(
        self, config: ServerConfig, repo: JobRepository, artifacts: ArtifactChecks, process: ProcessRunner
    ) -> None:
        """Create the service.

        Args:
            config: Server configuration (planner timeout, job root).
            repo: Job storage.
            artifacts: Plan validity checks.
            process: Subprocess runner.
        """
        self._config = config
        self._repo = repo
        self._artifacts = artifacts
        self._process = process

    def _fail(self, job_dir: Path, message: str) -> bool:
        self._repo.update_status(job_dir, state="error", stage="planning", finished=utcnow(), error_tail=message)
        return False

    async def ensure_plan(self, job_dir: Path, meta: JsonObject) -> bool:
        """Make sure a valid, applied plan exists, planning (and purging stale chunks) when needed.

        Args:
            job_dir: Job directory.
            meta: Contents of ``job.json``.

        Returns:
            False after recording an error status; True when chunks can be scheduled.
        """
        if not self._artifacts.plan_valid(job_dir) and not await self._run_planner(job_dir, meta):
            return False
        applied = self._repo.read_plan_signature(job_dir)
        current = plan_signature(self._repo.read_plan(job_dir))
        if applied is None:  # legacy job: adopt the current layout
            if not self._repo.write_plan_signature(job_dir, current):
                return self._fail(job_dir, SIGNATURE_ERROR)
            return True
        if applied != current:  # crash between plan.json and purge: finish the purge now
            return await self._apply_plan(job_dir, meta, applied)
        return True

    async def _purge_in_thread(self, job_dir: Path, langs: list[str]) -> int:
        """Purge chunk dirs in a worker thread; a cancelled job WAITS for it so it cannot outlive the attempt."""
        future = asyncio.get_running_loop().run_in_executor(None, self._repo.purge_chunk_dirs, job_dir, langs)
        cancelled = False
        while not future.done():
            try:
                await asyncio.wait({future})
            except asyncio.CancelledError:
                cancelled = True  # keep waiting for the purge, re-raise afterwards
        if cancelled:
            raise asyncio.CancelledError()
        return future.result()

    async def _apply_plan(self, job_dir: Path, meta: JsonObject, old_sig: str | None) -> bool:
        new_sig = plan_signature(self._repo.read_plan(job_dir))
        if new_sig != old_sig:
            failed = await self._purge_in_thread(job_dir, list(meta["langs"]))
            if failed:
                with contextlib.suppress(OSError):
                    (job_dir / "plan.json").unlink()  # force a re-plan + re-purge; do not commit the new signature
                return self._fail(job_dir, f"cannot remove {failed} chunk dirs of the previous plan; please retry")
            log.info("[replan] %s: plan layout changed, stale chunk dirs purged", job_dir.name)
        if not self._repo.write_plan_signature(job_dir, new_sig):
            return self._fail(job_dir, SIGNATURE_ERROR)
        return True

    def _purge_names(self, job_dir: Path) -> list[str]:
        """Planner outputs to delete before a re-plan.

        A published ``speakers_global.json`` that carries an immutable voice plan (or any rollout=1 job that already
        has a map) is kept: deleting it would let the rerun publish a DIFFERENT plan (old- and new-voice chunks).
        """
        keep = False
        try:
            speakers_path = job_dir / "speakers_global.json"
            if speakers_path.exists():
                rollout = self._repo.read_json(job_dir / "voice_rollout.json")
                speakers = self._repo.read_json(speakers_path)
                keep = isinstance(speakers.get("voice_plan"), dict) or rollout.get("voice_plan") == 1
        except Exception:
            keep = (job_dir / "voice_rollout.json").exists() and (job_dir / "speakers_global.json").exists()
        return [name for name in PLANNER_DERIVED if not (keep and name == "speakers_global.json")]

    def _delete_stale_outputs(self, job_dir: Path) -> bool:
        for name in self._purge_names(job_dir):
            try:
                (job_dir / name).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                return self._fail(job_dir, f"cannot remove stale {name}: {exc!r}")
        return True

    async def _run_planner(self, job_dir: Path, meta: JsonObject) -> bool:
        attempt = uuid.uuid4().hex[:8]
        # plan.sig = last APPLIED layout, wins over a plan.json that may be newer
        old_sig = self._repo.read_plan_signature(job_dir) or plan_signature(self._repo.read_plan(job_dir))
        if not self._delete_stale_outputs(job_dir):
            return False
        self._repo.update_status(
            job_dir,
            state="planning",
            stage="planning",
            started=utcnow(),
            finished=None,
            error_tail=None,
            attempt=attempt,
        )
        args = ["--url", meta["url"], "--out", str(job_dir), "--start", str(meta["start"]), "--dur", str(meta["dur"])]
        rc, tail = await self._process.run(
            PLANNER_MODULE,
            args,
            self._config.root,
            job_dir / "planner.log",
            timeout=self._config.planner_timeout_s,
            kind="planner",
        )
        if rc == 0 and self._artifacts.plan_valid(job_dir):
            if not await self._apply_plan(job_dir, meta, old_sig):
                return False
            self._repo.update_status(job_dir, state="planning", stage="planned", error_tail=None)
            return True
        if rc == 0:
            return self._fail(job_dir, "Planner exited 0 but plan.json/source.mp4 missing or invalid")
        return self._fail(job_dir, tail[-2000:] or f"exit {rc}")
