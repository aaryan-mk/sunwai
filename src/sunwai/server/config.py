"""Runtime configuration of the HTTP server, read once from the flat ``DUB_*`` environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sunwai.paths import DATA_DIR, PACKAGE_SRC, WEB_DIR

_OFF_VALUES = ("0", "false", "no", "off", "n", "f")


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    return float(environ.get(name, str(default)))


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    return int(environ.get(name, str(default)))


@dataclass(frozen=True)
class ServerConfig:
    """Every tunable of the API server.

    Attributes:
        root: Absolute job root.
        max_concurrent: Chunk subprocesses allowed to run at once.
        pipeline_timeout_s: Wall-time limit of one chunk subprocess.
        planner_timeout_s: Wall-time limit of the planner subprocess.
        resynth_timeout_s: Wall-time limit of one segment re-synthesis.
        edit_gate_timeout_s: How long an edit waits for a pipeline slot before answering 503.
        admission: Whether bounded admission control is on.
        max_uncached_jobs: Uncached jobs that may run at once; the rest queue.
        max_targets: Target languages allowed per job.
        max_start_s: Latest accepted window start, seconds.
        max_dur_s: Longest accepted window, seconds.
        max_video_s: Longest whole video accepted right after planning, seconds.
        max_edit_chars: Longest accepted segment edit.
        max_queued_jobs: New uncached jobs allowed to wait behind the running ones.
        max_work_s: Dubbed seconds times languages allowed per job.
        fanout: Chunk-by-language tasks alive at once per job.
        cors_origins: Allowed CORS origins.
        token: Shared secret for mutating ``/api/`` calls; empty disables the check.
        reconcile: Whether in-flight jobs are resumed at startup.
        voice_plan: Whether new jobs are rolled out on the voice plan.
        languages_file: Language catalogue served by ``/api/languages``.
        web_dir: Static web UI directory.
        package_src: Directory put on ``PYTHONPATH`` of child processes.
    """

    root: Path
    max_concurrent: int
    pipeline_timeout_s: float
    planner_timeout_s: float
    resynth_timeout_s: float
    edit_gate_timeout_s: float
    admission: bool
    max_uncached_jobs: int
    max_targets: int
    max_start_s: float
    max_dur_s: float
    max_video_s: float
    max_edit_chars: int
    max_queued_jobs: int
    max_work_s: float
    fanout: int
    cors_origins: tuple[str, ...]
    token: str
    reconcile: bool
    voice_plan: bool
    languages_file: Path
    web_dir: Path
    package_src: Path

    @property
    def admission_slots(self) -> int:
        """Jobs admitted at once: the configured limit, or effectively unlimited when admission is off."""
        return self.max_uncached_jobs if self.admission else 10**6

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> ServerConfig:
        """Build the configuration from environment-style variables.

        Args:
            environ: Mapping of variable names to values (normally ``os.environ``).

        Returns:
            The immutable configuration with every default applied.
        """
        max_concurrent = _env_int(environ, "MAX_CONCURRENT", 3)
        max_targets = _env_int(environ, "DUB_MAX_TARGETS", 3)
        max_video_s = _env_float(environ, "DUB_MAX_VIDEO_S", 1800)
        origins = tuple(o.strip() for o in environ.get("DUB_CORS_ORIGINS", "*").split(",") if o.strip())
        return cls(
            root=Path(environ.get("DUB_ROOT", "./jobs")).resolve(),
            max_concurrent=max_concurrent,
            pipeline_timeout_s=3600,
            planner_timeout_s=3600,
            resynth_timeout_s=_env_float(environ, "DUB_RESYNTH_TIMEOUT", 900),
            edit_gate_timeout_s=_env_float(environ, "DUB_EDIT_GATE_TIMEOUT", 240),
            admission=environ.get("DUB_ADMISSION", "1") != "0",
            max_uncached_jobs=_env_int(environ, "DUB_MAX_UNCACHED_JOBS", 1),
            max_targets=max_targets,
            max_start_s=_env_float(environ, "DUB_MAX_START_S", 1800),
            max_dur_s=_env_float(environ, "DUB_MAX_DUR_S", 1800),
            max_video_s=max_video_s,
            max_edit_chars=_env_int(environ, "DUB_MAX_EDIT_CHARS", 600),
            max_queued_jobs=_env_int(environ, "DUB_MAX_QUEUED_JOBS", 4),
            max_work_s=_env_float(environ, "DUB_MAX_WORK_S", max_targets * max_video_s),
            fanout=max(1, _env_int(environ, "DUB_FANOUT", 4 * max_concurrent)),
            cors_origins=origins or ("*",),
            token=environ.get("DUB_TOKEN", ""),
            reconcile=environ.get("DUB_RECONCILE", "1") == "1",
            voice_plan=all(
                environ.get(k, "").strip().lower() not in _OFF_VALUES for k in ("VOICE_PLAN", "VOICE_XLING")
            ),
            languages_file=DATA_DIR / "languages.json",
            web_dir=WEB_DIR,
            package_src=PACKAGE_SRC,
        )
