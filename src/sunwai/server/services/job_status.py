"""Assembling the job status view the player polls, and the sync-quality metrics derived from it."""

from __future__ import annotations

from pathlib import Path

from sunwai.server.atomic import JsonObject
from sunwai.server.clock import utcnow
from sunwai.server.errors import ServiceError
from sunwai.server.paths import JobPaths
from sunwai.server.repository import JobRepository
from sunwai.server.scheduling import JobAdmission
from sunwai.server.schemas import ChunkState, ChunkSync, JobMetrics, JobStatus, LangMetrics, LangProgress, PlanChunk
from sunwai.server.services.artifacts import ArtifactChecks
from sunwai.server.services.health import HealthService
from sunwai.server.services.sync_metrics import chunk_sync, lang_sync

ACTIVE_STATES = ("running", "queued", "planning", "error")
NULL_NOTE = "cut_s=0 and overflowed=0 means nothing was truncated; tempo 1.0 = no speed-up"


class JobStatusService:
    """Builds ``GET /api/jobs/{id}`` and its metrics view from the files on disk."""

    def __init__(
        self,
        repo: JobRepository,
        paths: JobPaths,
        artifacts: ArtifactChecks,
        admission: JobAdmission,
        health: HealthService,
    ) -> None:
        """Create the service.

        Args:
            repo: Job storage.
            paths: Directory resolver.
            artifacts: Source-readiness check shared with the media route.
            admission: Queue state reported to the player.
            health: Source of the "services down" hint.
        """
        self._repo = repo
        self._paths = paths
        self._artifacts = artifacts
        self._admission = admission
        self._health = health

    def _chunk_state(self, job_dir: Path, lang: str, idx: int) -> tuple[ChunkState, ChunkSync | None, bool]:
        """Return the chunk's state, its sync metrics (if done) and whether it counts as done."""
        out_dir = self._paths.resolve_chunk_dir(job_dir, lang, idx)
        if not (out_dir and out_dir.exists()):
            return ChunkState(idx=idx, state="queued", stage="", version=0, has_media=False), None, False
        status = self._repo.read_status(out_dir)
        state = status.get("state", "unknown")
        sync: ChunkSync | None = None
        if state == "done":
            try:
                sync = chunk_sync(self._repo.load_segments(out_dir), out_dir, lang)
            except Exception:
                sync = None
        view = ChunkState(
            idx=idx,
            state=state,
            stage=status.get("stage", ""),
            version=status.get("version", 0),
            has_media=(out_dir / f"dub_{lang}.mp4").exists(),
            sync=sync,
            started=status.get("started"),
            finished=status.get("finished"),
            editing=bool(status.get("editing")),
        )
        return view, sync, state == "done"

    def _language_progress(self, job_dir: Path, lang: str, chunks: list[JsonObject]) -> LangProgress:
        states: list[ChunkState] = []
        syncs: list[ChunkSync | None] = []
        done = 0
        for chunk in chunks:
            view, sync, finished = self._chunk_state(job_dir, lang, chunk["idx"])
            states.append(view)
            syncs.append(sync)
            done += finished
        return LangProgress(done=done, total=len(chunks), chunks=states, sync=lang_sync(syncs, done))

    def _speakers(self, job_dir: Path, plan: JsonObject) -> object:
        speakers = plan.get("speakers", [])
        global_file = job_dir / "speakers_global.json"
        if global_file.exists():
            data = self._repo.read_json(global_file)
            if "speakers" in data:
                speakers = data["speakers"]
        return speakers

    def _load(self, job_id: str) -> tuple[Path, JsonObject]:
        job_dir = self._paths.resolve_job(job_id)
        meta = self._repo.read_job_meta(job_dir) if job_dir else {}
        if job_dir is None or not meta:
            raise ServiceError(404, "unknown job")
        return job_dir, meta

    async def get_job(self, job_id: str) -> JobStatus:
        """Return the status of a job: state, queue place, per-language chunk progress and sync metrics.

        Args:
            job_id: Job identifier from the URL.

        Returns:
            The status view.

        Raises:
            ServiceError: 404 when the job does not exist.
        """
        job_dir, meta = self._load(job_id)
        status = self._repo.read_status(job_dir)
        state = status.get("state", "unknown")
        position = self._admission.position(job_id)
        if position is not None and position > 0:
            state = "queued"  # waiting for an admission slot
        plan = self._repo.read_plan(job_dir)
        chunks = plan.get("chunks", [])
        langs = list(meta.get("langs", []))
        hint = (await self._health.down_hint()).strip(" ()") if state in ACTIVE_STATES else ""
        return JobStatus(
            job_id=job_id,
            url=meta.get("url"),
            state=state,
            queue_position=position,
            queue_len=len(self._admission.queue),
            error_tail=status.get("error_tail") if state == "error" else None,
            services_down=hint,
            duration=plan.get("duration", 0),
            lipsync=meta.get("lipsync", True) is not False,
            chunks=[
                PlanChunk.model_validate(
                    {**c, "has_source": self._artifacts.source_file(job_dir, langs, c["idx"]) is not None}
                )
                for c in chunks
            ],
            speakers=self._speakers(job_dir, plan),
            langs={lang: self._language_progress(job_dir, lang, chunks) for lang in langs},
            now=utcnow(),
        )

    async def get_metrics(self, job_id: str) -> JobMetrics:
        """Return the sync-quality numbers of ``get_job`` per language and per measured chunk."""
        data = await self.get_job(job_id)
        metrics = {
            lang: LangMetrics(total=info.sync, chunks={c.idx: c.sync for c in info.chunks if c.sync})
            for lang, info in data.langs.items()
        }
        return JobMetrics(job_id=job_id, metrics=metrics, null=NULL_NOTE)
