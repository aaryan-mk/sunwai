"""Stage 1 -- fetch: download or copy the input into ``raw.mp4``, then cut/re-encode it into ``source.mp4``.

Outputs (each validated before reuse): ``raw.mp4``, ``meta.json`` and ``source.mp4``. Every file is published
atomically, so a crashed run can only ever leave a missing file, never a partial one that looks complete.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass

from sunwai.planning.artifacts import DERIVED_FROM_WAV, TMP_PREFIX, remove_derived
from sunwai.planning.envvars import env_flag
from sunwai.planning.errors import MediaError
from sunwai.planning.fileio import atomic_json, is_url, json_or_none, norm_path, remove_file
from sunwai.planning.subprocesses import ffprobe_duration, run_ffmpeg
from sunwai.planning.video_check import check_video
from sunwai.planning.ytdlp import Meta, copy_local, download_url, prefetched_youtube_copy

log = logging.getLogger("planner")

_STALE_TMP_S = 1800
_ENCODE_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k"]


@dataclass(frozen=True)
class CutRange:
    """The part of the raw video that becomes ``source.mp4``.

    Attributes:
        needed: False when the whole video is used.
        start: Cut start in seconds (0.0 when not needed).
        dur: Cut length in seconds, or ``None`` when not needed.
        expect: Expected length of the resulting video, or ``None`` when unknown.
    """

    needed: bool
    start: float
    dur: float | None
    expect: float | None


def plan_cut(start: float, dur: float, raw_dur: float, meta_dur: float) -> CutRange:
    """Work out the cut for the requested ``start``/``dur``.

    Args:
        start: Requested start in seconds.
        dur: Requested duration in seconds (0 means "to the end").
        raw_dur: Probed duration of ``raw.mp4`` (0 when unknown).
        meta_dur: Duration recorded in ``meta.json`` (0 when unknown).

    Raises:
        MediaError: When ``start`` lies beyond the end of the video.
    """
    needed = start > 0.0 or dur > 0.0
    src_dur = meta_dur or raw_dur
    cut_start = max(0.0, float(start)) if needed else 0.0
    known_end = raw_dur or src_dur
    if needed and known_end > 0 and cut_start >= known_end - 0.05:
        raise MediaError(f"start={cut_start:.1f}s is beyond the end of the video ({known_end:.1f}s)")
    cut_dur = (float(dur) if dur and dur > 0 else max(0.0, src_dur - cut_start)) if needed else None
    expect = None
    if raw_dur > 0:
        expect = min(cut_dur, raw_dur - cut_start) if (needed and cut_dur is not None) else raw_dur
    return CutRange(needed, cut_start, cut_dur, expect)


def cut_range_mismatch(meta: Meta, cut: CutRange) -> str | None:
    """Why an existing ``source.mp4`` was cut for a different range than requested, or ``None`` if it matches."""
    if cut.needed:
        assert cut.dur is not None
        if abs(float(meta.get("start", -1)) - cut.start) > 1e-3 or abs(float(meta.get("dur", -1)) - cut.dur) > 1e-3:
            return "cut range differs from the requested start/dur"
    elif "start" in meta or "dur" in meta:
        return "was cut for a different range"
    return None


def encode_args(raw_path: str, out_path: str, cut: CutRange, input_seek: bool) -> list[str]:
    """Ffmpeg arguments that turn ``raw_path`` into the (optionally cut) ``out_path``.

    Input-side ``-ss`` is an accurate seek with re-encode; output-side ``-ss`` decoded from t=0 (kill switch
    ``PLANNER_INPUT_SEEK=0``).
    """
    tail = ["-map", "0", *_ENCODE_ARGS] if cut.needed else _ENCODE_ARGS
    tail = [*tail, "-avoid_negative_ts", "make_zero", "-y", out_path]
    if not cut.needed:
        return ["-i", raw_path, *tail]
    ss, t = f"{cut.start:.3f}", f"{cut.dur:.3f}"
    head = ["-ss", ss, "-i", raw_path, "-t", t] if input_seek else ["-i", raw_path, "-ss", ss, "-t", t]
    return head + tail


def sweep_tmp(out_dir: str) -> None:
    """Remove leftovers of a crashed earlier attempt (temp downloads/encodes are only ever named ``.g066tmp_*``)."""
    try:
        now = time.time()
        for name in os.listdir(out_dir):
            if not name.startswith(TMP_PREFIX):
                continue
            path = os.path.join(out_dir, name)
            try:
                if now - os.path.getmtime(path) < _STALE_TMP_S:  # may belong to a live planner on this dir
                    continue
            except OSError:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                remove_file(path)
    except OSError:
        pass


def input_signature(path: str) -> list[int] | None:
    """``[size, mtime_ns]`` of a local input, or ``None`` when it cannot be stat'ed."""
    try:
        st = os.stat(norm_path(path))
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return None


class SourceFetcher:
    """Produce ``raw.mp4``, ``meta.json`` and ``source.mp4`` in a job directory, reusing validated cached files."""

    def __init__(self, url: str, out_dir: str, start: float, dur: float) -> None:
        """Prepare a fetch.

        Args:
            url: HTTP(S) URL or local file path.
            out_dir: Job directory.
            start: Cut start in seconds (0 = from the beginning).
            dur: Cut length in seconds (0 = to the end).
        """
        self.requested = url
        self.prefetched = prefetched_youtube_copy(url, out_dir) if is_url(url) else None
        self.url = self.prefetched or url
        self.out_dir = out_dir
        self.start, self.dur = start, dur
        self.raw_path = os.path.join(out_dir, "raw.mp4")
        self.meta_path = os.path.join(out_dir, "meta.json")
        self.source_path = os.path.join(out_dir, "source.mp4")
        loaded = json_or_none(self.meta_path)
        self.meta: Meta = loaded if isinstance(loaded, dict) else {}
        self.local_sig = None if is_url(self.url) else input_signature(self.url)

    def run(self) -> None:
        """Run stage 1."""
        if self.prefetched:
            log.info("[1] using pre-fetched local copy %s instead of downloading", self.prefetched)
        sweep_tmp(self.out_dir)
        if not self._raw_is_reusable():
            self._fetch_raw()
        self._ensure_source()
        log.info("[1] ready: %s", norm_path(self.source_path))

    # ---- raw.mp4 -------------------------------------------------------------------------------------------

    def _wanted_meta_url(self) -> str:
        return self.requested if is_url(self.requested) else norm_path(self.requested)

    def _raw_rejection(self) -> str | None:
        """Why the existing ``raw.mp4`` cannot be reused, or ``None`` if it is a complete, matching video."""
        meta = self.meta
        if not meta.get("url"):
            return "meta.json missing/corrupt: cannot prove which input raw.mp4 belongs to"
        if meta["url"] != self._wanted_meta_url():
            return f"belongs to another input ({meta['url']})"
        if not is_url(self.url) and (not self.local_sig or meta.get("src_sig") != self.local_sig):
            return "local input: no matching size/mtime signature in meta.json (legacy or changed file)"
        ok, why = check_video(self.raw_path, expect_dur=float(meta.get("duration") or 0.0) or None)
        return None if ok else why

    def _raw_is_reusable(self) -> bool:
        if not os.path.exists(self.raw_path):
            return False
        why = self._raw_rejection()
        if why is None:
            return True
        log.warning("[1] raw.mp4 failed validation (%s): discarding it and fetching again", why)
        remove_file(self.raw_path)
        return False

    def _fetch_raw(self) -> None:
        """Download/copy the input into a temp dir, validate it, publish ``raw.mp4`` and write ``meta.json``."""
        tmp_dir = os.path.join(self.out_dir, f"{TMP_PREFIX}dl{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        try:
            tmp_raw = os.path.join(tmp_dir, "raw.mp4")
            self.meta = download_url(self.url, tmp_dir, tmp_raw) if is_url(self.url) else copy_local(self.url, tmp_raw)
            ok, why = check_video(tmp_raw, expect_dur=float(self.meta.get("duration") or 0.0) or None)
            if not ok:
                raise MediaError(f"fetched video is not usable: {why}")
            os.replace(tmp_raw, self.raw_path)  # atomic publish: raw.mp4 only ever exists complete
            remove_derived(self.out_dir, ("source.mp4", "source16k.wav") + DERIVED_FROM_WAV)  # stale w.r.t. the new raw
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if self.prefetched:
            self._apply_prefetched_sidecar()
        if self.local_sig:
            self.meta["src_sig"] = self.local_sig
        atomic_json(self.meta_path, self.meta, indent=2)
        log.info("[1] wrote meta.json")

    def _apply_prefetched_sidecar(self) -> None:
        """Keep the source URL; take title/uploader/license from the sidecar written by ``fetch_yt_local.sh``."""
        assert self.prefetched
        self.meta["url"] = self.requested
        try:
            with open(self.prefetched[:-4] + ".json", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            for key in ("title", "uploader", "license", "duration"):
                if sidecar.get(key) is not None:
                    self.meta[key] = sidecar[key]
        except Exception:
            pass

    # ---- source.mp4 ----------------------------------------------------------------------------------------

    def _source_rejection(self, cut: CutRange) -> str | None:
        """Why the existing ``source.mp4`` cannot be reused, or ``None`` if it matches the request."""
        why = cut_range_mismatch(self.meta, cut)
        if why:
            return why
        ok, why = check_video(self.source_path, expect_dur=cut.expect, tail=True)
        if not ok:
            return why
        actual = ffprobe_duration(self.source_path)
        if cut.needed and cut.expect and actual > cut.expect + max(2.0, 0.02 * cut.expect):
            return f"duration {actual:.1f}s is longer than the requested cut {cut.expect:.1f}s"
        return None

    def _ensure_source(self) -> None:
        raw_dur = ffprobe_duration(norm_path(self.raw_path))
        cut = plan_cut(self.start, self.dur, raw_dur, float(self.meta.get("duration") or 0.0))
        if os.path.exists(self.source_path):
            why = self._source_rejection(cut)
            if why:
                log.warning("[1] source.mp4 failed validation (%s): discarding it and re-cutting", why)
                remove_file(self.source_path)
        if not os.path.exists(self.source_path):
            self._encode_source(cut)

    def _encode_source(self, cut: CutRange) -> None:
        log.info("[1] producing source.mp4 ...")
        tmp_src = os.path.join(self.out_dir, f"{TMP_PREFIX}source{os.getpid()}.mp4")
        try:
            if cut.needed:
                assert cut.dur is not None
                log.info("[1] cutting [%.3f, %.3f]s from raw", cut.start, cut.start + cut.dur)
                self.meta["start"], self.meta["dur"] = cut.start, cut.dur
            else:
                log.info("[1] remux/re-encode full -> source.mp4")
            run_ffmpeg(encode_args(norm_path(self.raw_path), tmp_src, cut, env_flag("PLANNER_INPUT_SEEK")), step=1)
            ok, why = check_video(tmp_src, expect_dur=cut.expect)
            if not ok:
                raise MediaError(f"source.mp4 encode is not usable: {why}")
            remove_derived(self.out_dir, ("source16k.wav",) + DERIVED_FROM_WAV)  # built from the previous source
            os.replace(tmp_src, self.source_path)
            if cut.needed:  # meta AFTER the replace: a crash in between only costs a re-cut, never a mislabelled source
                atomic_json(self.meta_path, self.meta, indent=2)
        finally:
            remove_file(tmp_src)


def stage_fetch(url: str, out_dir: str, start: float, dur: float) -> None:
    """Run stage 1 (see :class:`SourceFetcher`)."""
    SourceFetcher(url, out_dir, start, dur).run()
