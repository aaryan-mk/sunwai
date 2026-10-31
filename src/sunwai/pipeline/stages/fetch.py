"""Fetch stage: obtain the exact source window."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict


log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import StageBase

from sunwai.media.ffmpeg import run_ffmpeg
from sunwai.media.frames import _owned_frames_by_ts


def stage_fetch(args, out_dir: Path) -> Dict[str, Any]:
    meta_path = out_dir / "meta.json"
    src_mp4 = out_dir / "src.mp4"

    if meta_path.exists() and src_mp4.exists():
        log.info("Stage fetch: cached")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    url = args.url
    is_url = url.startswith("http://") or url.startswith("https://")

    if is_url:
        import yt_dlp
        ydl_opts = {
            "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "outtmpl": str(out_dir / "raw_%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            raw_path = Path(ydl.prepare_filename(info))
            title = info.get("title", "")
            uploader = info.get("uploader", "")
            license_ = info.get("license", "") or info.get("license_info", "")
    else:
        raw_path = Path(url)
        if not raw_path.exists():
            raise FileNotFoundError(f"Local input not found: {raw_path}")
        shutil.copy(raw_path, out_dir / raw_path.name)
        raw_path = out_dir / raw_path.name
        title = raw_path.stem
        uploader = ""
        license_ = ""

    start = args.start
    dur = args.dur

    # ffmpeg cut + re-encode
    # (D): the old cfr cut gave ONE FRAME TOO MANY per chunk (input -t is anchored at the first selected frame, so it reaches
    # into the next chunk) and rounded every picture to a frame slot (up to one frame of timing error). New: the owned frames are
    # chosen from the SOURCE's decoded timestamps (frame owned iff start <= pts - start_time < start+dur), passthrough video sync,
    # exactly that many frames, each at its true time. Only provably-CFR sources take this path; the result is verified (frame count)
    # and anything else falls back to the legacy command. CHUNK_EXACT_OWN=0 forces the legacy cut. Audio is cut exactly as before.
    _cut = [
        "-ss", f"{start}",
        "-t", f"{dur}",
        "-i", str(raw_path),
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac",
    ]
    _why = None
    if os.environ.get("CHUNK_EXACT_OWN", "1").strip().lower() in ("0", "false", "off", "no"):
        _why = "disabled by CHUNK_EXACT_OWN=0"
    else:
        try:
            _n, _delta = _owned_frames_by_ts(raw_path, float(start), float(dur))
            _vf = "trim=end_frame=%d,setpts=PTS-STARTPTS+%.6f/TB" % (_n, _delta)
            run_ffmpeg(_cut + ["-vsync", "0", "-vf", _vf, str(src_mp4)])
            import subprocess as _sp
            _c = _sp.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", str(src_mp4)], capture_output=True, text=True, timeout=120).stdout.strip()
            if int(_c) != _n:
                raise RuntimeError(f"cut has {_c} frames, expected {_n}")
        except Exception as _e:
            _why = f"{type(_e).__name__}: {_e}"
    if _why is not None:
        if "disabled" not in _why:
            log.warning("fetch: exact frame cut rejected (%s); legacy cfr cut for this chunk", _why)
        try:
            os.remove(src_mp4)
        except OSError:
            pass
        run_ffmpeg(_cut + [str(src_mp4)])

    meta = {
        "title": title,
        "uploader": uploader,
        "license": license_,
        "url": url,
        "start": start,
        "dur": dur,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Stage fetch: done")
    return meta


class FetchStage(StageBase[None, dict[str, Any]]):
    """Obtain the exact source window as ``src.mp4`` and write ``meta.json``."""

    name = "fetch"
    policy = StagePolicy(timing_key="fetch")

    def run(self, value: None, ctx: ChunkContext) -> dict[str, Any]:
        """Obtain the exact source window as ``src.mp4`` and write ``meta.json``."""
        return stage_fetch(ctx.legacy_args(), ctx.work_dir)  # type: ignore[no-any-return]
