"""Job submission (validation, language merging, admission) and restart recovery."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from pathlib import Path

from sunwai.server.atomic import JsonObject
from sunwai.server.clock import utcnow
from sunwai.server.config import ServerConfig
from sunwai.server.errors import ServiceError
from sunwai.server.paths import LANG_RE, JobPaths
from sunwai.server.repository import JobRepository
from sunwai.server.scheduling import JobAdmission, KeyedLocks
from sunwai.server.schemas import CreateJobResponse, JobRequest
from sunwai.server.services.artifacts import ArtifactChecks
from sunwai.server.services.job_runner import JobRunner
from sunwai.server.services.languages import LanguageCatalog
from sunwai.server.services.process import ProcessRunner
from sunwai.server.services.sources import SourceValidator

log = logging.getLogger(__name__)

QUEUE_SETTLE_S = 0.05


def job_id_of(url: str, start: float, dur: float) -> str:
    """Return the deterministic 12-hex-digit identifier of a (source, window) pair."""
    key = f"{url}|{float(start)}|{float(dur)}".encode()
    return hashlib.sha1(key).hexdigest()[:12]


class JobService:
    """Creates jobs, merges added languages and resumes in-flight work after a restart."""

    def __init__(
        self,
        config: ServerConfig,
        repo: JobRepository,
        paths: JobPaths,
        artifacts: ArtifactChecks,
        locks: KeyedLocks,
        admission: JobAdmission,
        runner: JobRunner,
        sources: SourceValidator,
        catalog: LanguageCatalog,
        process: ProcessRunner,
    ) -> None:
        """Create the service.

        Args:
            config: Server configuration (limits).
            repo: Job storage.
            paths: Directory resolver.
            artifacts: Completion checks.
            locks: Keyed locks (``job:<id>``).
            admission: Admission controller (queue state).
            runner: Starts the per-job task.
            sources: Source validator.
            catalog: Language catalogue.
            process: Subprocess runner (orphan sweep at startup).
        """
        self._config = config
        self._repo = repo
        self._paths = paths
        self._artifacts = artifacts
        self._locks = locks
        self._admission = admission
        self._runner = runner
        self._sources = sources
        self._catalog = catalog
        self._process = process
        self._reconcile_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ request validation
    def _normalise_langs(self, requested: list[str]) -> list[str]:
        langs = list(dict.fromkeys(lang.strip().lower() for lang in requested))
        langs = [lang for lang in langs if LANG_RE.match(lang)]
        if not langs:
            raise ServiceError(422, "no valid languages")
        allowed = self._catalog.target_codes()
        invalid = [lang for lang in langs if allowed and lang not in allowed]
        if invalid:
            raise ServiceError(422, f"Unsupported languages: {invalid}")
        if self._config.admission and len(langs) > self._config.max_targets:
            raise ServiceError(422, f"at most {self._config.max_targets} target languages per request")
        return langs

    def _validate_window(self, body: JobRequest) -> tuple[float, float]:
        start, dur = float(body.start), float(body.dur)
        if not (math.isfinite(start) and math.isfinite(dur)):
            raise ServiceError(422, "start/dur must be finite numbers")
        if start < 0:
            raise ServiceError(422, "start must be >= 0")
        if dur < 0:
            raise ServiceError(422, "dur must be >= 0")
        if start > self._config.max_start_s:
            raise ServiceError(422, f"start must be at most {self._config.max_start_s:.0f} s")
        if dur > self._config.max_dur_s:
            raise ServiceError(422, f"dur must be at most {self._config.max_dur_s:.0f} s")
        return start, dur

    def _merge_languages(self, old: JsonObject, langs: list[str], dur: float) -> list[str]:
        """Merge ``langs`` into the job's languages, enforcing the per-job language and work limits."""
        cfg = self._config
        known = list(old.get("langs", []))
        merged = list(dict.fromkeys(known + langs))
        if cfg.admission and len(merged) > max(cfg.max_targets, len(known)):
            raise ServiceError(
                422, f"a job may have at most {cfg.max_targets} target languages (it already has {old.get('langs')})"
            )
        if cfg.admission and dur > 0 and dur * len(merged) > cfg.max_work_s and dur * len(merged) > dur * len(known):
            raise ServiceError(
                422,
                f"too much work: {dur:.0f} s x {len(merged)} languages; this demo allows at most "
                f"{cfg.max_work_s:.0f} s of video x languages per job. Pick a shorter range or fewer languages.",
            )
        return merged

    def _check_capacity(self, job_id: str, job_dir: Path, old: JsonObject, langs: list[str]) -> None:
        """Answer 429 when the demo already has its quota of running and waiting jobs."""
        cfg = self._config
        busy = self._admission.busy
        limit = cfg.max_uncached_jobs + cfg.max_queued_jobs
        new_work = (not old) or bool(set(langs) - set(old.get("langs", []))) or self._artifacts.job_needs_work(job_dir)
        if cfg.admission and new_work and job_id not in self._runner.tasks and busy >= limit:
            raise ServiceError(
                429,
                f"The demo is busy: {busy} dubbing jobs are already running or waiting (limit {limit}). "
                "Please try again in a few minutes.",
                headers={"Retry-After": "60"},
            )

    # ------------------------------------------------------------------ creation
    def _decide_voice_rollout(self, job_dir: Path, old: JsonObject) -> None:
        """New jobs are rolled out on the voice plan HERE, before any language dir exists; the planner trusts it."""
        fresh = old == {} and (not job_dir.exists() or not any(job_dir.iterdir()))
        job_dir.mkdir(parents=True, exist_ok=True)
        marker = job_dir / "voice_rollout.json"
        if fresh and self._config.voice_plan and not marker.exists():
            self._repo.write_json(marker, {"voice_plan": 1, "decided": utcnow(), "by": "create_job"})

    def _write_job_meta(
        self,
        job_dir: Path,
        job_id: str,
        url: str,
        merged: list[str],
        start: float,
        dur: float,
        body: JobRequest,
        old: JsonObject,
    ) -> None:
        lipsync_given = "lipsync" in body.model_fields_set
        lipsync = bool(body.lipsync) if lipsync_given else (old.get("lipsync", True) is not False)
        meta = {
            "job_id": job_id,
            "url": url,
            "langs": merged,
            "start": start,
            "dur": dur,
            "lipsync": lipsync,
            "created": old.get("created") or utcnow(),
            "updated": utcnow(),
        }
        self._repo.write_json(job_dir / "job.json", meta)
        for lang in merged:
            (job_dir / lang).mkdir(parents=True, exist_ok=True)

    async def create_job(self, body: JobRequest) -> CreateJobResponse:
        """Create a job, or merge new languages into the existing one, and make sure it is running.

        Args:
            body: Validated request body.

        Returns:
            Job identifier, merged languages, scheduling mode and queue position.

        Raises:
            ServiceError: 422 for invalid input or limits, 429 when the demo is busy.
        """
        url = self._sources.validate(body.url)
        langs = self._normalise_langs(body.langs)
        start, dur = self._validate_window(body)
        job_id = job_id_of(url, start, dur)
        job_dir = self._paths.job_dir(job_id)
        async with self._locks.get(f"job:{job_id}"):
            old = self._repo.read_active_meta(job_dir)
            merged = self._merge_languages(old, langs, dur)
            self._check_capacity(job_id, job_dir, old, langs)
            self._decide_voice_rollout(job_dir, old)
            self._write_job_meta(job_dir, job_id, url, merged, start, dur, body, old)
            mode = self._runner.ensure_running(job_id)
        await asyncio.sleep(QUEUE_SETTLE_S)  # let run_job enter the admission queue so the position is real
        return CreateJobResponse(
            job_id=job_id,
            langs=merged,
            mode=mode,
            state=self._repo.read_status(job_dir).get("state"),
            queue_position=self._admission.position(job_id),
            queue_len=len(self._admission.queue),
        )

    # ------------------------------------------------------------------ startup
    def _is_in_flight(self, job_dir: Path, status: JsonObject) -> bool:
        if not status or status.get("state") in ("queued", "planning", "running"):
            return True  # no status.json = died right after job.json
        return any(
            self._repo.read_json(path).get("state") in ("queued", "running")
            for path in self._repo.chunk_status_files(job_dir)
        )

    def _inspect_for_resume(self, job_dir: Path) -> tuple[str, str] | None:
        """Return ``(queued-at, job id)`` when the job was in flight, else ``None``."""
        if not (job_dir / "job.json").is_file():
            return None
        restored = self._repo.restore_interrupted_edits(job_dir)
        status = self._repo.read_status(job_dir)
        inflight = self._is_in_flight(job_dir, status)
        log.info(
            "[reconcile] %s state=%s inflight=%s restored_edits=%s",
            job_dir.name,
            status.get("state"),
            inflight,
            restored,
        )
        if not inflight:
            return None
        return (status.get("queued_at") or status.get("started") or "", job_dir.name)

    async def reconcile(self) -> None:
        """Finish what a dead predecessor left half-done.

        Only jobs that were IN FLIGHT (queued/planning/running, or with a queued/running chunk) are resumed; merely
        errored old jobs wait for an explicit re-POST.
        """
        if not self._config.reconcile:
            return
        try:
            job_dirs = self._repo.list_job_dirs()
        except OSError:
            return
        candidates = []
        for job_dir in job_dirs:
            try:
                found = self._inspect_for_resume(job_dir)
            except Exception as exc:
                log.warning("[reconcile] %s error %r", job_dir.name, exc)
                continue
            if found:
                candidates.append(found)
        for _, job_id in sorted(candidates):
            try:
                log.info("[reconcile] resuming %s: %s", job_id, self._runner.ensure_running(job_id))
            except Exception as exc:
                log.warning("[reconcile] resume %s failed %r", job_id, exc)

    async def startup(self) -> None:
        """Create the job root, sweep orphaned children and start reconciliation in the background."""
        self._config.root.mkdir(parents=True, exist_ok=True)
        killed = await asyncio.get_running_loop().run_in_executor(None, self._process.sweep_orphans)
        log.info(
            "[startup] pid=%s root=%s admission=%s max_jobs=%s orphans_killed=%s",
            os.getpid(),
            self._config.root,
            self._config.admission,
            self._config.max_uncached_jobs,
            killed,
        )
        self._reconcile_task = asyncio.create_task(self.reconcile())
