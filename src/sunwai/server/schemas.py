"""Public request and response models of the HTTP API.

Response models describe the JSON the web UI and the end-to-end tests already depend on. Values that originate in
pipeline-written files (segments, plan chunks) keep ``Any`` so they pass through unchanged.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------------------- requests
class JobRequest(BaseModel):
    """Create a dubbing job.

    Attributes:
        url: Source video URL.
        langs: Target language codes (at least one).
        start: Window start on the source timeline, seconds.
        dur: Window length, seconds; 0 means up to the admitted maximum.
        lipsync: False delivers aligned dubbed audio on the original picture with no lip render.
    """

    url: str
    langs: list[str] = Field(min_length=1)
    start: float = 0.0
    dur: float = 0.0
    lipsync: bool = True


class TextRequest(BaseModel):
    """Replace the text of one segment."""

    text: str = ""


class FocusRequest(BaseModel):
    """Viewer focus: the language and chunk currently being watched (drives scheduling priority)."""

    lang: str
    chunk: int


# ----------------------------------------------------------------------------- jobs
class CreateJobResponse(BaseModel):
    """Result of ``POST /api/jobs``.

    Attributes:
        job_id: Deterministic identifier of (url, start, dur).
        langs: All languages of the job after merging.
        mode: ``running`` when work was scheduled, ``cached`` when everything already existed.
        state: Job state at response time.
        queue_position: 0 running, n waiting, ``None`` not in the admission controller.
        queue_len: Jobs waiting for admission.
    """

    job_id: str
    langs: list[str]
    mode: str
    state: str | None
    queue_position: int | None
    queue_len: int


class ChunkSync(BaseModel):
    """Sync-quality metrics of one finished chunk."""

    segments: int
    overflowed: int
    cut_s: float
    refit: int
    tempo_mean: float | None
    tempo_max: float | None
    tempo_n: int
    lipsync: bool


class LangSync(BaseModel):
    """Sync-quality metrics aggregated over the finished chunks of one language."""

    chunks_measured: int
    segments: int
    overflowed: int
    cut_s: float
    refit: int
    tempo_mean: float | None
    tempo_max: float | None
    lipsync_chunks: int
    chunks_done: int
    lipsync_coverage: float


class ChunkState(BaseModel):
    """State of one chunk of one language; a chunk without a directory carries only the first five fields."""

    idx: int
    state: str
    stage: str
    version: Any
    has_media: bool
    sync: ChunkSync | None = None
    started: Any = None
    finished: Any = None
    editing: bool = False


class LangProgress(BaseModel):
    """Progress of one target language."""

    done: int
    total: int
    chunks: list[ChunkState]
    sync: LangSync


class PlanChunk(BaseModel):
    """A planned chunk as written by the planner, plus whether its original-video cut is ready."""

    model_config = ConfigDict(extra="allow")

    has_source: bool


class JobStatus(BaseModel):
    """Result of ``GET /api/jobs/{id}``."""

    job_id: str
    url: Any
    state: str
    queue_position: int | None
    queue_len: int
    error_tail: Any
    services_down: str
    duration: Any
    lipsync: bool
    chunks: list[PlanChunk]
    speakers: Any
    langs: dict[str, LangProgress]
    now: str


class LangMetrics(BaseModel):
    """Metrics of one language: the total and one entry per measured chunk index."""

    total: LangSync
    chunks: dict[int, ChunkSync]


class JobMetrics(BaseModel):
    """Result of ``GET /api/jobs/{id}/metrics``."""

    job_id: str
    metrics: dict[str, LangMetrics]
    null: str


class FocusResponse(BaseModel):
    """Result of ``POST /api/jobs/{id}/focus``."""

    ok: bool
    gen: int


# ----------------------------------------------------------------------------- segments
class SegmentView(BaseModel):
    """One transcript line with global (source-timeline) times."""

    chunk: int
    id: Any
    start: float | int
    end: float | int
    speaker: Any
    gender: Any
    src_text: Any
    mt_text: Any
    tgt_text: Any
    edited: Any
    fallback: bool
    fallback_reason: Any
    history_len: int


class SegmentList(BaseModel):
    """Result of ``GET /api/jobs/{id}/{lang}/segments``."""

    segments: list[SegmentView]


class EditResponse(BaseModel):
    """Result of a segment edit or undo.

    Attributes:
        ok: Always true on success.
        version: New chunk version; the player re-fetches media when it changes.
        segment: The segment as stored after the edit.
    """

    ok: bool
    version: int
    segment: dict[str, Any] | None


# ----------------------------------------------------------------------------- health
class DependencyStatus(BaseModel):
    """Reachability of one dependency service group."""

    up: int
    total: int
    down: list[str]


class AdmissionInfo(BaseModel):
    """Admission-control limits and current occupancy."""

    on: bool
    max_jobs: int
    running: list[str]
    queued: list[str]
    targets_max: int
    max_queued: int
    max_dur_s: float
    max_work_s: float
    fanout: int
    token_required: bool


class HealthResponse(BaseModel):
    """Result of ``GET /health``."""

    ok: bool
    ready: bool
    degraded: list[str]
    root: str
    pipeline: bool
    max_concurrent: int
    admission: AdmissionInfo
    deps: dict[str, DependencyStatus]
    lip: dict[str, int | None]
    pid: int
    time: str
