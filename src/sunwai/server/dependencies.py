"""FastAPI dependencies: routers receive services from the container the application factory attached."""

from __future__ import annotations

from fastapi import Request

from sunwai.server.container import Container
from sunwai.server.services.focus import FocusService
from sunwai.server.services.health import HealthService
from sunwai.server.services.job_status import JobStatusService
from sunwai.server.services.jobs import JobService
from sunwai.server.services.languages import LanguageCatalog
from sunwai.server.services.media import MediaService
from sunwai.server.services.segments import SegmentService


def get_container(request: Request) -> Container:
    """Return the container of the running application."""
    container: Container = request.app.state.container
    return container


def get_job_service(request: Request) -> JobService:
    """Return the job submission service."""
    return get_container(request).jobs


def get_job_status_service(request: Request) -> JobStatusService:
    """Return the job status service."""
    return get_container(request).job_status


def get_segment_service(request: Request) -> SegmentService:
    """Return the segment service."""
    return get_container(request).segments


def get_media_service(request: Request) -> MediaService:
    """Return the media service."""
    return get_container(request).media


def get_focus_service(request: Request) -> FocusService:
    """Return the viewer-focus service."""
    return get_container(request).focus


def get_health_service(request: Request) -> HealthService:
    """Return the health service."""
    return get_container(request).health


def get_language_catalog(request: Request) -> LanguageCatalog:
    """Return the language catalogue."""
    return get_container(request).catalog
