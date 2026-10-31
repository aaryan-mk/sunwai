"""The voice catalogue (languages + voice bank) and the revision stamp that ties a plan to it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sunwai.paths import DATA_DIR
from sunwai.planning.voices.types import Bank

PLAN_VERSION = 2
ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
_PACKAGE_DIR = Path(__file__).resolve().parent


def read_bytes(path: str | Path) -> bytes:
    """Read a file, returning ``b""`` when it is missing or unreadable."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return b""


def catalogue_rev(app: Path = DATA_DIR) -> str:
    """Hash the catalogue (languages.json + voice bank), the ECAPA model id and this package's source.

    The package source is included because the assignment, gate and optimiser implementation shape the plan.
    """
    digest = hashlib.sha256()
    for name in ("languages.json", "voice_bank.json"):
        digest.update(read_bytes(Path(app) / name))
    digest.update(ECAPA_ID.encode())
    for source in sorted(_PACKAGE_DIR.glob("*.py")):
        digest.update(read_bytes(source))
    return digest.hexdigest()[:16]


def smap_hash(speakers_global: dict[str, Any]) -> str:
    """Hash the speaker map (speakers and turns) a plan was computed for."""
    core = {"speakers": speakers_global.get("speakers", []), "turns": speakers_global.get("turns", [])}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_bank(app: Path = DATA_DIR) -> Bank:
    """Load ``voice_bank.json`` (empty dict when missing)."""
    bank: Bank = json.loads(read_bytes(Path(app) / "voice_bank.json").decode("utf-8") or "{}")
    return bank
