"""Public request models of the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
