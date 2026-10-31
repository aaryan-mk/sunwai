"""Job endpoints: create, inspect, metrics and viewer focus."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from sunwai.server.dependencies import get_focus_service, get_job_service, get_job_status_service
from sunwai.server.schemas import CreateJobResponse, FocusRequest, FocusResponse, JobMetrics, JobRequest, JobStatus
from sunwai.server.services.focus import FocusService
from sunwai.server.services.job_status import JobStatusService
from sunwai.server.services.jobs import JobService

router = APIRouter()

Jobs = Annotated[JobService, Depends(get_job_service)]
Status = Annotated[JobStatusService, Depends(get_job_status_service)]
Focus = Annotated[FocusService, Depends(get_focus_service)]


@router.post("/api/jobs", response_model=CreateJobResponse)
async def create_job(body: JobRequest, jobs: Jobs) -> CreateJobResponse:
    """Create a dubbing job, or merge new languages into the existing one, and start it."""
    return await jobs.create_job(body)


@router.get("/api/jobs/{job_id}", response_model=JobStatus, response_model_exclude_unset=True)
async def get_job(job_id: str, status: Status) -> JobStatus:
    """Return the state of a job, its queue place and per-language chunk progress."""
    return await status.get_job(job_id)


@router.get("/api/jobs/{job_id}/metrics", response_model=JobMetrics)
async def get_job_metrics(job_id: str, status: Status) -> JobMetrics:
    """Return sync-quality metrics per language and per chunk (same numbers as in the job status)."""
    return await status.get_metrics(job_id)


@router.post("/api/jobs/{job_id}/focus", response_model=FocusResponse)
async def set_focus(job_id: str, body: FocusRequest, focus: Focus) -> FocusResponse:
    """Record the chunk and language the viewer is watching; it drives scheduling priority."""
    return focus.set_focus(job_id, body)
