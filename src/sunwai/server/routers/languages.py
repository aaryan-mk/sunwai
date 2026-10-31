"""Language capability endpoint."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from sunwai.server.dependencies import get_language_catalog
from sunwai.server.services.languages import LanguageCatalog

router = APIRouter()


@router.get("/api/languages")
async def get_languages(catalog: Annotated[LanguageCatalog, Depends(get_language_catalog)]) -> dict[str, Any]:
    """Return the supported target and source languages."""
    return catalog.config()
