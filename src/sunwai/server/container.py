"""The composition root: constructs every service once and wires their dependencies explicitly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sunwai.server.config import ServerConfig
from sunwai.server.paths import JobPaths
from sunwai.server.repository import FileJobRepository, JobRepository
from sunwai.server.scheduling import FocusRegistry, JobAdmission, KeyedLocks, PriorityGate
from sunwai.server.services.artifacts import ArtifactChecks
from sunwai.server.services.chunks import PIPELINE_MODULE, ChunkDispatcher
from sunwai.server.services.focus import FocusService
from sunwai.server.services.health import HealthService, load_services
from sunwai.server.services.job_runner import JobRunner
from sunwai.server.services.job_status import JobStatusService
from sunwai.server.services.jobs import JobService
from sunwai.server.services.languages import LanguageCatalog
from sunwai.server.services.media import MediaService
from sunwai.server.services.planning import PLANNER_MODULE, PlannerService
from sunwai.server.services.process import ProcessRunner
from sunwai.server.services.segments import SegmentService
from sunwai.server.services.sources import SourceValidator


@dataclass
class Container:
    """All long-lived collaborators of the API, built by :func:`build_container`.

    Routers reach the services through FastAPI dependencies that read this object from ``app.state.container``.
    """

    config: ServerConfig
    paths: JobPaths
    repo: JobRepository
    locks: KeyedLocks
    admission: JobAdmission
    focus_registry: FocusRegistry
    gate: PriorityGate
    process: ProcessRunner
    catalog: LanguageCatalog
    health: HealthService
    artifacts: ArtifactChecks
    job_runner: JobRunner
    jobs: JobService
    job_status: JobStatusService
    segments: SegmentService
    media: MediaService
    focus: FocusService


def build_container(config: ServerConfig, environ: Mapping[str, str]) -> Container:
    """Construct the services of one application instance.

    Args:
        config: Server configuration.
        environ: Environment used to locate dependency services (``DUB_SERVICES``, ``DEP_HOST``, ``LIP_*``).

    Returns:
        The wired container; nothing has been started yet.
    """
    paths = JobPaths(config.root)
    repo = FileJobRepository(paths)
    locks, admission, focus_registry = KeyedLocks(), JobAdmission(config.admission_slots), FocusRegistry()
    gate = PriorityGate(config.max_concurrent, focus_registry.priority)
    process = ProcessRunner(config.root, paths.procs_dir, config.package_src, (PIPELINE_MODULE, PLANNER_MODULE))
    catalog = LanguageCatalog(config.languages_file, repo)
    health = HealthService(config, load_services(environ), admission)
    artifacts = ArtifactChecks(repo, paths)
    planner = PlannerService(config, repo, artifacts, process)
    dispatcher = ChunkDispatcher(config, repo, paths, artifacts, process, gate, focus_registry)
    runner = JobRunner(config, repo, paths, artifacts, admission, planner, dispatcher, health)
    sources = SourceValidator(paths.uploads_dir)
    return Container(
        config=config,
        paths=paths,
        repo=repo,
        locks=locks,
        admission=admission,
        focus_registry=focus_registry,
        gate=gate,
        process=process,
        catalog=catalog,
        health=health,
        artifacts=artifacts,
        job_runner=runner,
        jobs=JobService(config, repo, paths, artifacts, locks, admission, runner, sources, catalog, process),
        job_status=JobStatusService(repo, paths, artifacts, admission, health),
        segments=SegmentService(config, repo, paths, locks, gate, process),
        media=MediaService(repo, paths, artifacts),
        focus=FocusService(focus_registry, repo, paths),
    )
