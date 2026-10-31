"""Bounded ffmpeg subprocess helper."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List


log = logging.getLogger(__name__)

from sunwai.pipeline.support.env import _env_num


# ---------------------------------------------------------------------------
# Audio IO helpers
# ---------------------------------------------------------------------------
def run_ffmpeg(args: List[str], *, capture: bool = True, check: bool = True):
    import subprocess
    inputs = {str(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "-i"}
    out = str(args[-1])
    if out in inputs:
        # ffmpeg cannot write over its own input: render to a temp file, then replace
        tmp = out + ".tmp" + Path(out).suffix
        args = list(args[:-1]) + [tmp]
    cmd = ["ffmpeg", "-hide_banner", "-y", "-nostdin", "-loglevel", "error"] + args
    try:
        proc = subprocess.run(cmd, capture_output=capture, check=check,
                              timeout=_env_num("PIPE_FFMPEG_TIMEOUT", 900.0, float))
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {_env_num('PIPE_FFMPEG_TIMEOUT', 900.0, float):.0f}s: {' '.join(cmd[-4:])}")
    if out in inputs and proc.returncode == 0:
        os.replace(tmp, out)
    return proc
