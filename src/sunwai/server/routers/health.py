"""Liveness and dependency readiness."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from sunwai.server.dependencies import get_health_service
from sunwai.server.schemas import HealthResponse
from sunwai.server.services.health import HealthService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(service: Annotated[HealthService, Depends(get_health_service)]) -> HealthResponse:
    """Report this process, the admission state and every dependency the pipeline needs."""
    return await service.report()
