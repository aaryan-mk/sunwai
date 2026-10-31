"""Snapshot and restore of the chunk-level mix files, so a failed remix can be rolled back exactly."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sunwai.control.expression.revisions import expr_dir
from sunwai.control.expression.fsutil import unlink_quietly

MixSnapshot = dict[str, Any]
SCRATCH_PATTERNS = (".dubtmp_*", ".stemtmp_*")


def mix_files(out_dir: str | Path) -> list[str]:
    """List every file the mix stage publishes or overwrites: ``dub.wav``, ``dub_<lang>.mp4`` (not lip-synced) and ``tts_placed.*``."""
    names = {"dub.wav", "tts_placed.json", "tts_placed.wav"}
    names |= {p.name for p in Path(out_dir).glob("dub_*.mp4") if "lipsync" not in p.name}
    return sorted(names)


def snapshot_mix(out_dir: str | Path) -> MixSnapshot:
    """Copy the current mix files into a backup directory and return ``{"dir", "files"}`` (missing files map to ``None``)."""
    backup = expr_dir(out_dir) / "mixbak" / uuid.uuid4().hex[:8]
    backup.mkdir(parents=True)
    files: dict[str, str | None] = {}
    for name in mix_files(out_dir):
        source = Path(out_dir) / name
        if source.exists():
            shutil.copy2(source, backup / name)  # copy2 keeps mtime: lip-sync freshness (lipsync newer than dub) must not flip
            files[name] = str(backup / name)
        else:
            files[name] = None
    return {"dir": str(backup), "files": files}


def restore_mix(out_dir: str | Path, snap: MixSnapshot) -> None:
    """Restore the mix files from ``snap``, remove files created by the failed remix and drop the backup."""
    chunk = Path(out_dir)
    for name, backup in snap["files"].items():
        target = chunk / name
        if backup is None:
            unlink_quietly(target)
        else:
            tmp = chunk / f".restore_{uuid.uuid4().hex[:6]}_{name}"
            shutil.copy2(backup, tmp)
            os.replace(tmp, target)
    for created in list(chunk.glob("dub_*.mp4")) + list(chunk.glob("tts_placed.*")):
        if "lipsync" not in created.name and created.name not in snap["files"]:  # created by the failed remix
            unlink_quietly(created)
    for pattern in SCRATCH_PATTERNS:
        for scratch in chunk.glob(pattern):
            unlink_quietly(scratch)
    shutil.rmtree(snap["dir"], ignore_errors=True)


def discard_mix_snapshot(snap: MixSnapshot) -> None:
    """Drop a snapshot after its change committed."""
    shutil.rmtree(snap["dir"], ignore_errors=True)
