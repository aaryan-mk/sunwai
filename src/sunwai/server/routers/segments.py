"""Transcript endpoints: list, edit and undo segments."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from sunwai.server.dependencies import get_segment_service
from sunwai.server.schemas import EditResponse, SegmentList, TextRequest
from sunwai.server.services.segments import SegmentService

router = APIRouter()

Segments = Annotated[SegmentService, Depends(get_segment_service)]


@router.get("/api/jobs/{job_id}/{lang}/segments", response_model=SegmentList)
async def get_segments(job_id: str, lang: str, segments: Segments) -> SegmentList:
    """List the transcript lines of every finished chunk, with global times."""
    return segments.list_segments(job_id, lang)


@router.post("/api/jobs/{job_id}/{lang}/chunks/{c}/segments/{seg_id}", response_model=EditResponse)
async def edit_segment(
    job_id: str, lang: str, c: int, seg_id: str, body: TextRequest, segments: Segments
) -> EditResponse:
    """Replace the text of one segment and re-synthesise it."""
    return await segments.edit(job_id, lang, c, seg_id, body.text)


@router.post("/api/jobs/{job_id}/{lang}/chunks/{c}/segments/{seg_id}/undo", response_model=EditResponse)
async def undo_segment(job_id: str, lang: str, c: int, seg_id: str, segments: Segments) -> EditResponse:
    """Undo the last edit of one segment and re-synthesise it."""
    return await segments.undo(job_id, lang, c, seg_id)
