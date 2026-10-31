"""Atomic JSON writes, file copies and per-chunk timing records."""

from __future__ import annotations

import json
import logging
import os
import threading as _threading
from pathlib import Path
from typing import Dict


log = logging.getLogger(__name__)




def atomic_write_json(path, obj, **kw) -> None:
    """Write JSON to a temp file in the same dir, re-parse it, then os.replace (readers never see a partial file)."""
    path = str(path)
    tmp = f"{path}.tmp{os.getpid()}_{_threading.get_ident()}"
    kw.setdefault("ensure_ascii", False)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
    with open(tmp, encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp, path)


def _safe_cleanup(cid, tts_dir, made) -> None:
    """remove this attempt's piece files; never raises (logs only)."""
    try:
        files = [x for r_ in list(made or ()) for x in (r_, r_.with_name(r_.stem + ".fp"))]
        files += list(tts_dir.glob(f"{cid}_p*_trim.wav")) + list(tts_dir.glob(f"{cid}_p*_fit.wav"))
        for _p in files:
            try:
                os.remove(_p)
            except OSError:
                pass
    except Exception as e:
        try:
            log.warning("split cleanup skipped (seg %s): %s", cid, e)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def load_timings(out_dir: Path) -> Dict[str, float]:
    p = out_dir / "timings.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_timing(out_dir: Path, stage: str, secs: float):
    t = load_timings(out_dir)
    t[stage] = round(t.get(stage, 0.0) + secs, 3)
    atomic_write_json(out_dir / "timings.json", t, indent=2)


def _atomic_copy(src: Path, dst: Path):
    """copy file/dir to dst via a temp name + rename so a crash never leaves a partial file at the final path"""
    import shutil
    tmp = dst.with_name(dst.name + ".tmpcopy")
    if tmp.exists():
        shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
    if src.is_dir():
        shutil.copytree(src, tmp)
        if dst.exists():
            shutil.rmtree(dst)
    else:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _sha_file(p) -> str:
    import hashlib
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:20]
    except OSError:
        return "missing"
