"""Filesystem locations shared across SunwAI (package data, static web UI, source root)."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_SRC = PACKAGE_DIR.parent  # directory that contains the ``sunwai`` package
REPO_ROOT = PACKAGE_SRC.parent  # repository root in a source checkout
DATA_DIR = PACKAGE_DIR / "data"  # languages.json, voice_bank.json, voice_rates.json


def _find_web_dir() -> Path:
    """Locate the static UI: ``<repo>/web`` in a checkout, ``SUNWAI_WEB_DIR`` if set."""
    import os

    override = os.environ.get("SUNWAI_WEB_DIR", "").strip()
    return Path(override) if override else REPO_ROOT / "web"


WEB_DIR = _find_web_dir()
