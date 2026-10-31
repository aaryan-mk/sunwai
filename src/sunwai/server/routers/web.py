"""The static web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> FileResponse:
    """Serve the player page."""
    web_dir: Path = request.app.state.container.config.web_dir
    return FileResponse(str(web_dir / "index.html"), media_type="text/html")


def mount_static(app: FastAPI, web_dir: Path) -> None:
    """Serve ``styles.css`` and ``js/*`` under ``/static``."""
    app.mount("/static", StaticFiles(directory=str(web_dir), check_dir=False), name="static")
