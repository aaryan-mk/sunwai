"""Media endpoints: dubbed chunks and original-video cuts, served with HTTP range support."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from sunwai.server.dependencies import get_media_service
from sunwai.server.services.media import MediaService

router = APIRouter()

Media = Annotated[MediaService, Depends(get_media_service)]


@router.get("/media/{job_id}/original/{c}.mp4")
async def media_original_chunk(job_id: str, c: int, media: Media) -> FileResponse:
    """Serve the original-video cut of chunk ``c``."""
    return FileResponse(str(media.original_chunk(job_id, c)), media_type="video/mp4")


@router.get("/media/{job_id}/{lang}/chunks/{c}.mp4")
async def media_chunk(job_id: str, lang: str, c: int, media: Media, lip: int = 1) -> FileResponse:
    """Serve a finished dubbed chunk; ``?lip=0`` forces the aligned-audio video over the lip-sync render."""
    return FileResponse(str(media.dubbed_chunk(job_id, lang, c, lip)), media_type="video/mp4")
