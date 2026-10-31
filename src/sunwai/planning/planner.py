#!/usr/bin/env python3
"""planner.py — prepare a video for chunked progressive dubbing.

Stages (each cached; skipped if its output already exists):
 1 fetch        -> raw.mp4 + meta.json + source.mp4 (+ meta.json start/dur)
 2 audio        -> source16k.wav
 3 vad          -> vad.json
 4 chunks       -> plan.json  (timing plan)
 5 diarize      -> speaker turns (whole file, once)
 6 gender       -> per-speaker gender
 7 write        -> speakers_global.json (+ timings into plan.json)

CLI:
  python planner.py --url URL_OR_LOCAL_FILE --out JOB_DIR
                    [--start 0] [--dur 0] [--chunk 60]
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import logging
import subprocess
import threading
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sunwai.paths import DATA_DIR


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

log = logging.getLogger("planner")


def _norm_path(p: str) -> str:
    """Return an absolute, symlink-resolved path (so cached-output checks match
    the actual files written), regardless of how the CLI argument was spelled."""
    try:
        return os.path.realpath(os.path.abspath(os.path.expanduser(p)))
    except Exception:
        return os.path.abspath(os.path.expanduser(p))


def _is_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _fmt_clock(t: float) -> str:
    if t >= 3600:
        return f"{int(t // 3600):d}:{int((t % 3600) // 60):02d}:{t % 60:06.3f}"
    if t >= 60:
        return f"{int(t // 60):02d}:{t % 60:06.3f}"
    return f"{t:06.3f}s"


# --------------------------------------------------------------------------- #
# robustness helpers: typed errors, env parsing, timeouts, media validation, atomic writes
# --------------------------------------------------------------------------- #

class PlannerError(RuntimeError):
    """A failure the planner reports cleanly (planner_error.json + a final PLANNER_ERROR line on stderr + rc 3)."""
    code = "planner_error"


class NoAudioError(PlannerError):
    code = "no_audio"


class ResourceLimitError(PlannerError):
    code = "resource_limit"


class MediaError(PlannerError):
    code = "bad_media"


class SubprocessTimeout(PlannerError):
    code = "timeout"


class DownloadBlockedError(PlannerError):
    code = "download_blocked"


YT_BLOCKED_MSG = ("YouTube blocked this download (sign-in required). Try another link, "
                  "or ask the operator to refresh the YouTube cookies.")
_BOT_RE = None


def _env_float(name: str, default: float, lo: float = None, hi: float = None) -> float:
    """Parse a float env var; blank, non-numeric, non-finite or out-of-range values fall back to the default (warned)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(raw)
        if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError("out of range")
        return v
    except ValueError:
        log.warning("env %s=%r is invalid (need a finite number in [%s, %s]); using default %s", name, raw, lo, hi, default)
        return default


def _ffmpeg_timeout() -> float:
    return _env_float("PLANNER_FFMPEG_TIMEOUT", 3600.0, lo=5.0)


def _ffprobe_timeout() -> float:
    return _env_float("PLANNER_FFPROBE_TIMEOUT", 60.0, lo=1.0)


def _run_cmd(cmd, timeout):
    """subprocess with a hard timeout that kills the whole process group (a plain subprocess.run(timeout=) leaves
    grandchildren holding the pipes open and blocks forever). Returns (rc, stdout, stderr); raises
    subprocess.TimeoutExpired."""
    import signal
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        raise
    return p.returncode, out, err


def _atomic_json(path: str, obj, **kw) -> None:
    """Temp file in the same dir, re-parse, then os.replace: a crash never leaves a half-written JSON under the name."""
    tmp = f"{path}.tmp{os.getpid()}"
    kw.setdefault("ensure_ascii", False)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
    with open(tmp, encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp, path)


def _json_or_none(path: str):
    """Parsed JSON, or None when the file is missing or corrupt (a corrupt cache is treated as absent)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _probe(path: str):
    """One ffprobe call (with timeout) -> {"duration": float, "n_video": int, "n_audio": int}, or None when the
    file is missing/unreadable/probe timed out. Attached-picture 'video' streams are not counted."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 64:
            return None
        rc, so, _se = _run_cmd(["ffprobe", "-v", "error", "-show_entries",
                                "stream=codec_type,duration:stream_disposition=attached_pic:format=duration",
                                "-of", "json", path], _ffprobe_timeout())
        if rc != 0:
            return None
        js = json.loads(so or "{}")
        nv = na = 0
        vd = ad = a0 = 0.0

        def _f(x):
            try:
                v = float(x)
                return v if math.isfinite(v) and v > 0 else 0.0
            except (TypeError, ValueError):
                return 0.0
        for st in js.get("streams", []):
            if st.get("codec_type") == "video" and not (st.get("disposition") or {}).get("attached_pic"):
                nv += 1
                vd = max(vd, _f(st.get("duration")))
            elif st.get("codec_type") == "audio":
                na += 1
                ad = max(ad, _f(st.get("duration")))
                if na == 1:
                    a0 = _f(st.get("duration"))      # the track stage_audio extracts (0:a:0)
        dur = _f((js.get("format") or {}).get("duration")) or max(vd, ad)   # TS/fragmented: no container duration
        return {"duration": dur, "n_video": nv, "n_audio": na, "v_dur": vd, "a_dur": ad, "a0_dur": a0}
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _tail_decodes(path: str, secs: float = 3.0, v_dur: float = 0.0, f_dur: float = 0.0) -> str:
    """Decode the last `secs` seconds of the first video stream. A download cut short after the header (moov at the
    front) still probes with the full duration but cannot be decoded at its end. Returns "ok", "bad" (decode errors /
    timeout / nonzero rc) or "empty" (clean decode but no frame: indistinguishable from an audio-only tail when the
    video stream duration is unknown; the caller accepts that with a warning rather than reject a valid file)."""
    try:
        # -sseof is relative to the container end; when the video stream ends clearly earlier (audio-only tail) seek
        # relative to the VIDEO end instead, or the tail would legitimately hold no frames
        seek = ["-sseof", f"-{secs:g}"]
        if v_dur > 0 and f_dur > v_dur + 1.0:
            seek = ["-ss", f"{max(0.0, v_dur - secs):.3f}"]
        rc, _so, se = _run_cmd(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-stats"] + seek + [
                                "-i", path, "-map", "0:V:0", "-an", "-f", "null", "-"], _ffmpeg_timeout())
    except (subprocess.TimeoutExpired, OSError):
        return "bad"
    import re
    frames = re.findall(r"frame=\s*(\d+)", se or "")
    errs = [l for l in (se or "").replace("\r", "\n").splitlines() if l.strip() and "frame=" not in l]
    if rc != 0 or errs:
        return "bad"
    return "ok" if frames and int(frames[-1]) > 0 else "empty"


def _head_decodes(path: str, secs: float = 5.0) -> bool:
    """Bounded decode from the START (no seeking): proves the first seconds of video decode cleanly. Used when the
    duration is unknown and the EOF-relative tail decode is not possible."""
    try:
        rc, _so, se = _run_cmd(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-stats", "-t", f"{secs:g}",
                                "-i", path, "-map", "0:V:0", "-an", "-f", "null", "-"], _ffmpeg_timeout())
    except (subprocess.TimeoutExpired, OSError):
        return False
    import re
    frames = re.findall(r"frame=\s*(\d+)", se or "")
    errs = [l for l in (se or "").replace("\r", "\n").splitlines() if l.strip() and "frame=" not in l]
    return rc == 0 and not errs and bool(frames) and int(frames[-1]) > 0


def _check_video(path: str, expect_dur: float = None, rel_tol: float = 0.02, abs_tol: float = 2.0, tail: bool = True):
    """(ok, reason): readable, >=1 video stream, duration > 0, not shorter than expect_dur (minus tolerance), and the
    last seconds decode. Used for raw.mp4 / source.mp4 so a partial file from a crashed run is never reused."""
    info = _probe(path)
    if info is None:
        return False, "unreadable (ffprobe failed or timed out)"
    if info["n_video"] < 1:
        return False, "no video stream"
    known = info["duration"] > 0.1
    if not known:
        # duration unavailable (fragmented MP4 / TS without a header duration): do not reject a decodable file
        log.warning("%s: container reports no duration; accepting on decodability alone", os.path.basename(path))
    elif expect_dur and expect_dur > 0 and info["duration"] < expect_dur - max(abs_tol, rel_tol * expect_dur):
        return False, f"duration {info['duration']:.1f}s < expected {expect_dur:.1f}s (truncated?)"
    if tail and not known:
        log.warning("%s: duration unknown, skipping the EOF-relative tail decode; decoding the first seconds instead",
                    os.path.basename(path))
        if not _head_decodes(path):
            return False, "duration unknown and the first seconds do not decode"
    elif tail:
        st = _tail_decodes(path, v_dur=info.get("v_dur", 0.0), f_dur=info["duration"])
        if st == "bad":
            return False, "tail does not decode (truncated?)"
        if st == "empty":
            if info.get("v_dur", 0.0) > 0 and known:
                return False, "no frame in the last seconds of the video stream (truncated?)"
            log.warning("%s: no video frame in the last seconds and the video duration is unknown "
                        "(audio-only tail?); accepting", os.path.basename(path))
    return True, "ok"


def _mem_available_bytes():
    """min(MemAvailable, cgroup headroom); None when unknown."""
    cands = []
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                cands.append(int(line.split()[1]) * 1024)
                break
    except Exception:
        pass
    for base in ("/sys/fs/cgroup",):
        try:
            mx = open(os.path.join(base, "memory.max")).read().strip()
            if mx != "max":
                cur = int(open(os.path.join(base, "memory.current")).read().strip())
                cands.append(max(0, int(mx) - cur))
        except Exception:
            pass
    return min(cands) if cands else None


def _check_resources(duration: float, what: str = "audio") -> None:
    """Resource ceiling: refuse inputs longer than PLANNER_MAX_DUR_S (default 4 h), or whose in-RAM working set
    (duration x 16 kHz x 4 B x PLANNER_MEM_FACTOR, default 6: pyannote waveform + torch copy + activations) exceeds
    80% of available RAM, with a clear error instead of an OOM kill."""
    max_dur = _env_float("PLANNER_MAX_DUR_S", 4 * 3600.0, lo=1.0)
    if duration > max_dur:
        raise ResourceLimitError(f"{what} is {duration / 60:.1f} min long; the planner ceiling is {max_dur / 60:.1f} min "
                                 f"(PLANNER_MAX_DUR_S). Dub a shorter range (start/dur).")
    need = duration * 16000 * 4 * _env_float("PLANNER_MEM_FACTOR", 6.0, lo=1.0)
    avail = _mem_available_bytes()
    if avail is not None and need > 0.8 * avail:
        raise ResourceLimitError(f"{what} of {duration / 60:.0f} min needs ~{need / 2**30:.1f} GiB of RAM to plan but only "
                                 f"{avail / 2**30:.1f} GiB is available. Dub a shorter range (start/dur).")


class _Pcm16:
    """Read-only float32 view over a PCM16 wav's samples, backed by a memory map: slicing converts only the slice, so
    the planner never holds the whole decoded track as float32. Supports len(), .size,
    slicing and to_array()."""

    def __init__(self, mm):
        self._mm = mm
        self.size = int(len(mm))
        self.shape = (self.size,)

    def __len__(self):
        return self.size

    def __getitem__(self, k):
        if isinstance(k, slice):
            return np.asarray(self._mm[k], dtype=np.float32) / 32768.0
        return float(self._mm[k]) / 32768.0

    def to_array(self) -> np.ndarray:
        return self[0:self.size]


def _as_array(y):
    return y.to_array() if hasattr(y, "to_array") else np.asarray(y, dtype=np.float32)


def _wav_data_span(path: str):
    """(offset, nbytes) of the RIFF 'data' chunk (clamped to the real file size); raises on a non-16k-mono-PCM16 file."""
    import wave
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
            raise RuntimeError(f"source16k.wav is not 16kHz mono PCM16: {path}")
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise RuntimeError(f"not a RIFF/WAVE file: {path}")
        pos = 12
        while pos + 8 <= fsize:
            f.seek(pos)
            cid, csz = f.read(4), int.from_bytes(f.read(4), "little")
            if cid == b"data":
                off = pos + 8
                return off, max(0, min(csz, fsize - off))
            pos += 8 + csz + (csz & 1)
    raise RuntimeError(f"no data chunk in {path}")


def _read_mono16k(path: str):
    """Mono 16 kHz PCM16 wav as a lazy float32 view (np.memmap underneath). Same interface the callers used
    (len, slicing, .size)."""
    off, nbytes = _wav_data_span(path)
    n = nbytes // 2
    if n <= 0:
        return _Pcm16(np.zeros(0, dtype=np.int16))
    return _Pcm16(np.memmap(path, dtype="<i2", mode="r", offset=off, shape=(n,)))


# --------------------------------------------------------------------------- #
# Stage 1 — fetch (yt_dlp download / local copy) + meta.json + source.mp4 cut
# --------------------------------------------------------------------------- #

def _is_bot_check(text) -> bool:
    """yt-dlp's exact 'Sign in to confirm you're not a bot' diagnostic (apostrophe variants allowed). Age-gate / private-video / generic
    'use --cookies' messages do NOT match."""
    global _BOT_RE
    import re
    if _BOT_RE is None:
        _BOT_RE = re.compile("sign in to confirm you['\u2019\u2018\u02bc`]?re not a bot", re.I)
    return bool(_BOT_RE.search(str(text or "")))


_YT_COOKIES_DEFAULT = ""          # no cookies unless YTDLP_COOKIES names a mode-600 cookie file
_YT_COOKIES_TMPDIR = os.environ.get("YTDLP_COOKIES_TMPDIR") or os.path.join(__import__("tempfile").gettempdir(), "sunwai_yt_cookies")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _cookie_private_copy():
    """Returns the path of a private mode-600 COPY of the YouTube cookie file, or None (then no cookies are used and behaviour is exactly as before).
    The source is env YTDLP_COOKIES (default: none): opened ONCE with O_NOFOLLOW, then validated on the
    open descriptor (regular file, owned by us, mode exactly 0600) and copied from that descriptor (no stat/open TOCTOU, no symlinks).
    yt-dlp rewrites its jar, so the master is never given to it. The caller deletes the copy. Contents are never read into Python strings, printed or logged."""
    import stat
    import tempfile
    src = (os.environ.get("YTDLP_COOKIES") or "").strip() or _YT_COOKIES_DEFAULT
    try:
        sfd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))      # O_NONBLOCK: a FIFO cannot hang us
    except OSError:
        return None
    tmp = None
    try:
        st = os.fstat(sfd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            return None
        if stat.S_IMODE(st.st_mode) != 0o600:
            log.warning("[1] YouTube cookie file ignored: its mode is %o, it must be exactly 600", stat.S_IMODE(st.st_mode))
            return None
        d = _YT_COOKIES_TMPDIR
        dfd = None
        name = None
        try:
            os.makedirs(d, mode=0o700, exist_ok=True)
            dfd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))      # the dir itself must not be a symlink
            ds = os.fstat(dfd)
            if ds.st_uid != os.getuid():
                log.warning("[1] cookie temp dir is not owned by us: downloading without cookies")
                return None
            if stat.S_IMODE(ds.st_mode) != 0o700:
                os.fchmod(dfd, 0o700)
            now = time.time()                    # sweep copies of planners that were killed: dead pid AND > 1 h old (all ops relative to the held dir fd)
            for nm in os.listdir(dfd):
                parts = nm.split("_")
                if not (nm.startswith("ytc_") and len(parts) > 2 and parts[1].isdigit() and not _pid_alive(int(parts[1]))):
                    continue
                try:
                    es = os.stat(nm, dir_fd=dfd, follow_symlinks=False)
                    if stat.S_ISREG(es.st_mode) and now - es.st_mtime > 3600:
                        os.unlink(nm, dir_fd=dfd)
                except OSError:
                    pass
            import secrets
            name = f"ytc_{os.getpid()}_{secrets.token_hex(8)}.txt"
            tfd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=dfd)
            os.fchmod(tfd, 0o600)
            with os.fdopen(tfd, "wb") as w, os.fdopen(sfd, "rb", closefd=False) as r:
                shutil.copyfileobj(r, w)
            return os.path.join(d, name)
        except BaseException as e:
            if name is not None and dfd is not None:
                try:
                    os.unlink(name, dir_fd=dfd)          # whatever went wrong, a half-written secret copy never stays
                except OSError:
                    pass
            if not isinstance(e, Exception):
                raise
            log.warning("[1] cannot make a private cookie copy (%s): downloading without cookies", type(e).__name__)
            return None
        finally:
            if dfd is not None:
                os.close(dfd)
    finally:
        os.close(sfd)


class _YDLLog:
    """Forwards EVERY yt-dlp message to the planner log (yt-dlp's documented level pattern: '[debug] ' => debug, other output => info) and redacts the
    bot-check text on every level: the planner reports it as a clean PLANNER_ERROR instead."""
    @staticmethod
    def _emit(level, msg):
        if _is_bot_check(msg):
            log.warning("[1] yt-dlp: YouTube asked for sign-in (bot check)")
        else:
            log.log(level, "[yt-dlp] %s", msg)

    def debug(self, msg):
        self._emit(logging.DEBUG if str(msg).startswith("[debug] ") else logging.INFO, msg)

    def info(self, msg):
        self._emit(logging.INFO, msg)

    def warning(self, msg):
        self._emit(logging.WARNING, msg)

    def error(self, msg):
        self._emit(logging.ERROR, msg)


def _yt_cache_path(url: str, out_dir: str):
    """YouTube blocks cluster IPs (bot check). A copy downloaded elsewhere lives in <DUB_ROOT>/uploads/yt_<id>.mp4
    (see fetch_yt_local.sh); use it instead of downloading."""
    import re
    from urllib.parse import urlparse
    _h = (urlparse(url or "").hostname or "").lower()
    if not (_h == "youtu.be" or _h == "youtube.com" or _h.endswith(".youtube.com")):
        return None
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})", url or "")
    if not m:
        return None
    cand = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "uploads", f"yt_{m.group(1)}.mp4")
    return cand if os.path.isfile(cand) and os.path.getsize(cand) > 0 else None


def _sweep_tmp(out_dir: str) -> None:
    """Remove leftovers of a crashed earlier attempt (temp downloads/encodes are only ever named .g066tmp_*)."""
    try:
        now = time.time()
        for nm in os.listdir(out_dir):
            if nm.startswith(".g066tmp_"):
                p = os.path.join(out_dir, nm)
                try:
                    if now - os.path.getmtime(p) < 1800:     # may belong to a live planner on this dir: leave it
                        continue
                except OSError:
                    continue
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else _rm(p)
    except OSError:
        pass


def _input_sig(path: str):
    try:
        st = os.stat(_norm_path(path))
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def stage_fetch(url: str, out_dir: str, start: float, dur: float):
    _c = _yt_cache_path(url, out_dir) if _is_url(url) else None
    _orig_url = url
    if _c:
        log.info("[1] using pre-fetched local copy %s instead of downloading", _c)
        url = _c
    raw_path = os.path.join(out_dir, "raw.mp4")
    meta_path = os.path.join(out_dir, "meta.json")
    source_path = os.path.join(out_dir, "source.mp4")
    _sweep_tmp(out_dir)
    meta = _json_or_none(meta_path)
    meta = meta if isinstance(meta, dict) else {}

    # ---- raw.mp4: reuse only if it is a complete, decodable video
    need_fetch = True
    _want_url = _orig_url if _is_url(_orig_url) else _norm_path(_orig_url)
    _sig = _input_sig(url) if not _is_url(url) else None        # size+mtime of a local input
    if os.path.exists(raw_path):
        if not meta.get("url"):
            ok, why = False, "meta.json missing/corrupt: cannot prove which input raw.mp4 belongs to"
        elif meta["url"] != _want_url:
            ok, why = False, f"belongs to another input ({meta['url']})"
        elif not _is_url(url) and (not _sig or meta.get("src_sig") != _sig):
            ok, why = False, "local input: no matching size/mtime signature in meta.json (legacy or changed file)"
        else:
            ok, why = _check_video(raw_path, expect_dur=float(meta.get("duration") or 0.0) or None)
        if ok:
            need_fetch = False
        else:
            log.warning("[1] raw.mp4 failed validation (%s): discarding it and fetching again", why)
            _rm(raw_path)

    if need_fetch:
        tmpd = os.path.join(out_dir, f".g066tmp_dl{os.getpid()}")
        shutil.rmtree(tmpd, ignore_errors=True)
        os.makedirs(tmpd)
        try:
            tmp_raw = os.path.join(tmpd, "raw.mp4")
            if _is_url(url):
                log.info("[1] downloading via yt_dlp ...")
                import yt_dlp
                max_raw = _env_float("PLANNER_MAX_RAW_S", 12 * 3600.0, lo=60.0)

                def _too_long(info, *, incomplete=False):
                    d = info.get("duration")
                    if d and d > max_raw:
                        return f"video is {d / 3600:.1f} h, longer than the {max_raw / 3600:.0f} h download ceiling"
                    return None

                ydl_opts = {
                    "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
                    "merge_output_format": "mp4",
                    "outtmpl": os.path.join(tmpd, "raw.%(ext)s"),
                    "quiet": True,
                    "noprogress": True,
                    "no_warnings": True,
                    "match_filter": _too_long,
                    "socket_timeout": 60,
                    "retries": 5,
                    "logger": _YDLLog(),
                }
                _ck_tmp = None
                try:
                    _ck_tmp = _cookie_private_copy()
                    if _ck_tmp:
                        ydl_opts["cookiefile"] = _ck_tmp
                        log.info("[1] using the YouTube cookie file (private copy)")
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=True)
                    except Exception as _e:
                        if "download ceiling" in str(_e):
                            raise ResourceLimitError(str(_e).replace("ERROR: ", ""))
                        if _is_bot_check(_e):
                            raise DownloadBlockedError(YT_BLOCKED_MSG)
                        # last-resort: accept any downloadable mp4 stream.
                        log.warning("primary format failed; retrying with best single mp4")
                        ydl_opts["format"] = "b[ext=mp4]/b"
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(url, download=True)
                        except Exception as _e2:
                            if "download ceiling" in str(_e2):
                                raise ResourceLimitError(str(_e2).replace("ERROR: ", ""))
                            if _is_bot_check(_e2):
                                raise DownloadBlockedError(YT_BLOCKED_MSG)
                            raise
                finally:
                    if _ck_tmp:
                        _rm(_ck_tmp)           # the private copy never outlives the download
                if not info:
                    raise ResourceLimitError("download refused: video is longer than the download ceiling "
                                             "(PLANNER_MAX_RAW_S) or yt_dlp returned nothing")
                downloaded = info.get("requested_downloads") or []
                got = (downloaded[0].get("filepath") if downloaded else None) or tmp_raw
                got = _norm_path(got)
                if not os.path.exists(got):
                    cands = [os.path.join(tmpd, n) for n in os.listdir(tmpd) if n.startswith("raw.") and not n.endswith(".part")]
                    if not cands:
                        raise MediaError("yt_dlp finished but produced no file")
                    got = cands[0]
                if got != tmp_raw:
                    os.replace(got, tmp_raw)
                meta = {
                    "title": info.get("title"),
                    "uploader": info.get("uploader"),
                    "duration": info.get("duration"),
                    "license": info.get("license"),
                    "url": url,
                }
            else:
                log.info("[1] local file -> copy to raw.mp4")
                src = _norm_path(url)
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                shutil.copy2(src, tmp_raw)
                try:
                    import yt_dlp
                    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True,
                                           "no_warnings": True,
                                           "simulate": False, "skip_download": True}) as ydl:
                        info = ydl.extract_info(src, download=False)
                    meta = {
                        "title": info.get("title") or os.path.basename(src),
                        "uploader": info.get("uploader"),
                        "duration": info.get("duration"),
                        "license": info.get("license"),
                        "url": src,
                    }
                except Exception:
                    meta = {
                        "title": os.path.basename(src),
                        "uploader": None,
                        "duration": _ffprobe_duration(src),
                        "license": None,
                        "url": src,
                    }

            ok, why = _check_video(tmp_raw, expect_dur=float(meta.get("duration") or 0.0) or None)
            if not ok:
                raise MediaError(f"fetched video is not usable: {why}")
            os.replace(tmp_raw, raw_path)          # atomic publish: raw.mp4 only ever exists complete
            _rm_derived(out_dir, ("source.mp4", "source16k.wav") + _DERIVED_FROM_WAV)    # everything built from a previous raw is stale
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)

        if _c:   # keep the source URL; title/uploader/license from the sidecar written by fetch_yt_local.sh
            meta["url"] = _orig_url
            try:
                _sc = json.load(open(_c[:-4] + ".json", encoding="utf-8"))
                for _k in ("title", "uploader", "license", "duration"):
                    if _sc.get(_k) is not None:
                        meta[_k] = _sc[_k]
            except Exception:
                pass
        if _sig:
            meta["src_sig"] = _sig
        _atomic_json(meta_path, meta, indent=2)
        log.info("[1] wrote meta.json")
    else:
        raw_path = _norm_path(raw_path)

    raw_path = _norm_path(raw_path)
    cut_needed = (start > 0.0) or (dur > 0.0)
    raw_dur = _ffprobe_duration(raw_path)
    src_dur = float(meta.get("duration") or 0.0) or raw_dur
    s = max(0.0, float(start)) if cut_needed else 0.0
    if cut_needed and (raw_dur or src_dur) > 0 and s >= (raw_dur or src_dur) - 0.05:
        raise MediaError(f"start={s:.1f}s is beyond the end of the video ({(raw_dur or src_dur):.1f}s)")
    d = (float(dur) if dur and dur > 0 else max(0.0, src_dur - s)) if cut_needed else None
    expect = None
    if raw_dur > 0:
        expect = min(d, raw_dur - s) if cut_needed else raw_dur
    if os.path.exists(source_path):
        if cut_needed and (abs(float(meta.get("start", -1)) - s) > 1e-3 or abs(float(meta.get("dur", -1)) - d) > 1e-3):
            ok, why = False, "cut range differs from the requested start/dur"
        elif not cut_needed and ("start" in meta or "dur" in meta):
            ok, why = False, "was cut for a different range"
        else:
            ok, why = _check_video(source_path, expect_dur=expect, tail=True)
            _sd = _ffprobe_duration(source_path) if ok else 0.0
            if ok and cut_needed and expect and _sd > expect + max(2.0, 0.02 * expect):
                ok, why = False, f"duration {_sd:.1f}s is longer than the requested cut {expect:.1f}s"
        if not ok:
            log.warning("[1] source.mp4 failed validation (%s): discarding it and re-cutting", why)
            _rm(source_path)
    if not os.path.exists(source_path):
        log.info("[1] producing source.mp4 ...")
        tmp_src = os.path.join(out_dir, f".g066tmp_source{os.getpid()}.mp4")
        try:
            if cut_needed:
                log.info("[1] cutting [%.3f, %.3f]s from raw", s, s + d)
                # input-side -ss (accurate seek with re-encode) instead of output-side -ss, which decoded from t=0.
                # Canary: first V/A PTS, duration and frame count identical on 2 clips. Kill switch: PLANNER_INPUT_SEEK=0.
                _iseek = os.environ.get("PLANNER_INPUT_SEEK", "1").strip().lower() not in ("0", "false", "no", "off", "n", "f")
                _run_ffmpeg((["-ss", f"{s:.3f}", "-i", raw_path, "-t", f"{d:.3f}"] if _iseek else
                             ["-i", raw_path, "-ss", f"{s:.3f}", "-t", f"{d:.3f}"]) + [
                             "-map", "0",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                             "-c:a", "aac", "-b:a", "128k",
                             "-avoid_negative_ts", "make_zero", "-y", tmp_src],
                            step=1)
                meta["start"] = s
                meta["dur"] = d
            else:
                log.info("[1] remux/re-encode full -> source.mp4")
                _run_ffmpeg(["-i", raw_path,
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                             "-c:a", "aac", "-b:a", "128k",
                             "-avoid_negative_ts", "make_zero", "-y", tmp_src],
                            step=1)
            ok, why = _check_video(tmp_src, expect_dur=expect)
            if not ok:
                raise MediaError(f"source.mp4 encode is not usable: {why}")
            _rm_derived(out_dir, ("source16k.wav",) + _DERIVED_FROM_WAV)      # audio and plan were built from the previous source
            os.replace(tmp_src, source_path)
            if cut_needed:     # meta AFTER the replace: a crash in between only costs a re-cut, never a mislabelled source
                _atomic_json(meta_path, meta, indent=2)
        finally:
            _rm(tmp_src)
    log.info("[1] ready: %s", _norm_path(source_path))


def _run_ffmpeg(args, step: int = 0, timeout: float = None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"] + args
    log.debug("[ffmpeg s%d] %s", step, " ".join(cmd))
    tmo = _ffmpeg_timeout() if timeout is None else timeout
    try:
        rc, _so, se = _run_cmd(cmd, tmo)
    except subprocess.TimeoutExpired:
        raise SubprocessTimeout(f"ffmpeg step {step} timed out after {tmo:.0f}s (PLANNER_FFMPEG_TIMEOUT)")
    if rc != 0:
        tail = (se or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg failed (rc=%d): %s" % (rc, "\n".join(tail)))


def _ffprobe_duration(path: str) -> float:
    info = _probe(path)
    return info["duration"] if info else 0.0


def _ffprobe_streams(path: str) -> tuple[int, bool]:
    """(num_video_streams, has_audio). Raises MediaError when the file cannot be probed (the old version assumed
    'audio present' on any probe failure, hiding the problem)."""
    info = _probe(path)
    if info is None:
        raise MediaError(f"cannot probe {os.path.basename(path)} (unreadable or ffprobe timed out)")
    return info["n_video"], info["n_audio"] > 0


# --------------------------------------------------------------------------- #
# Stage 2 — audio -> source16k.wav (16 kHz mono PCM16)
# --------------------------------------------------------------------------- #

def _wav_ok(path: str, expect_dur: float = None) -> bool:
    """16 kHz mono PCM16 wav whose header parses, has samples, and is about as long as the source."""
    import wave
    try:
        if os.path.getsize(path) < 44:
            return False
        with wave.open(path, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
                return False
            hdr_n = w.getnframes()
        _off, nbytes = _wav_data_span(path)
        n = nbytes // 2
        if n <= 0 or n < hdr_n:          # header promises more samples than the file holds = truncated
            return False
        if expect_dur and expect_dur > 0 and n / 16000.0 < expect_dur - max(2.0, 0.02 * expect_dur):
            return False
        return True
    except Exception:
        return False


_DERIVED_FROM_WAV = ("vad.json", "plan.json", "spkturns.json", "spkbackend.json", "spkgender.json", "speakers_global.json")


def _rm_derived(out_dir: str, names) -> None:
    """Delete stale derived products, EXCEPT the speaker map of a job that carries a voice_plan (rollout marker 1 or a plan in
    the file): a voice plan is immutable, so every invalidation path keeps it and the later stage_write hard-fails on a changed map."""
    keep_map = False
    m = _json_or_none(os.path.join(out_dir, "voice_rollout.json"))
    sg = _json_or_none(os.path.join(out_dir, "speakers_global.json"))
    if (isinstance(m, dict) and m.get("voice_plan") == 1) or (isinstance(sg, dict) and isinstance(sg.get("voice_plan"), dict)):
        keep_map = True
    for nm in names:
        if keep_map and nm == "speakers_global.json":
            log.warning("[voice] keeping speakers_global.json (immutable voice plan) through invalidation of %s", nm)
            continue
        _rm(os.path.join(out_dir, nm))


def stage_audio(out_dir: str):
    wav_path = os.path.join(out_dir, "source16k.wav")
    source = os.path.join(out_dir, "source.mp4")
    info = _probe(source)
    if info is None:
        raise MediaError("source.mp4 is missing or unreadable")
    if info["n_audio"] < 1:
        raise NoAudioError("this video has no audio track, so there is nothing to dub. "
                           "Pick a video with speech (or a range that contains audio).")
    # expected length = the track that gets extracted (0:a:0), not the longest of several audio tracks; None = unknown
    a_dur = info.get("a0_dur") or None      # unknown -> no length expectation; the real wav length is measured instead
    if a_dur:
        _check_resources(a_dur, "audio")          # the SELECTED track only; also for a cached wav (not bypassable)
    if os.path.exists(wav_path):
        if _wav_ok(wav_path, expect_dur=a_dur):
            if not a_dur:                  # cached wav, selected track length unknown: ceiling on the wav's real length
                _check_resources(_wav_data_span(wav_path)[1] / 32000.0, "audio")
            log.info("[2] cached")
            return
        log.warning("[2] source16k.wav failed validation (partial/short): re-extracting")
        _rm(wav_path)
        _rm_derived(out_dir, _DERIVED_FROM_WAV)      # products computed from the bad wav are stale (a voice plan is kept)
    log.info("[2] extracting audio (%d video stream(s), %d audio stream(s), %.1fs)", info["n_video"], info["n_audio"],
             info["duration"])
    tmp_wav = os.path.join(out_dir, f".g066tmp_a{os.getpid()}.wav")
    try:
        _run_ffmpeg(["-i", source, "-vn", "-map", "0:a:0",
                     "-ac", "1", "-ar", "16000",
                     "-c:a", "pcm_s16le",
                     "-af", "aresample=async=1:first_pts=0",
                     "-y", tmp_wav], step=2)
        if not _wav_ok(tmp_wav, expect_dur=a_dur):
            raise MediaError("audio extraction produced an empty or truncated file")
        if not a_dur:                      # selected track's length unknown: ceiling on the real extracted length
            _off, _nb = _wav_data_span(tmp_wav)
            _check_resources(_nb / 32000.0, "audio")
        os.replace(tmp_wav, wav_path)
    finally:
        _rm(tmp_wav)


# --------------------------------------------------------------------------- #
# Stage 3 — VAD (silero) -> vad.json, processing in 10-minute blocks.
# --------------------------------------------------------------------------- #

def stage_vad(out_dir: str):
    vad_path = os.path.join(out_dir, "vad.json")
    _cached = _json_or_none(vad_path)
    if isinstance(_cached, list):
        log.info("[3] cached")
        return _cached

    wav_path = os.path.join(out_dir, "source16k.wav")
    y = _read_mono16k(wav_path)
    total = len(y) / 16000.0
    log.info("[3] running silero VAD on %.3fs (%d samples)", total, len(y))

    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()

    def _ts(seg):
        # NEVER pass `padding`; pad_ms=0. return_seconds gives block-local seconds.
        return get_speech_timestamps(
            seg, model,
            sampling_rate=16000,
            threshold=0.5,
            min_silence_duration_ms=100,
            min_speech_duration_ms=100,
            speech_pad_ms=0,
            return_seconds=True,
        )

    intervals = _blocked_vad(y, total, _ts)

    merged = _merge_intervals(intervals)
    vad = [[round(s, 3), round(e, 3)] for (s, e) in merged]
    _atomic_json(vad_path, vad, indent=2)
    log.info("[3] wrote vad.json (%d speech intervals)", len(vad))
    return vad


def _blocked_vad(y, total, get_ts, sr=16000, BLOCK=600.0, OVERLAP=0.5):
    """Run get_ts(block_samples) -> [{start,end}] (block-LOCAL seconds) over 10-minute blocks with a small overlap.
    fix: block-local timestamps were treated as global whenever start <= t0 (almost always true for
    blocks after the first), collapsing later speech back into the first ten minutes. The block offset t0 is now ALWAYS
    added; the overlap duplicates are removed by _merge_intervals."""
    intervals = []
    t0 = 0.0
    while t0 < total:
        t1 = min(total, t0 + BLOCK)
        seg = y[int(round(t0 * sr)):int(round(t1 * sr))]
        if seg.size < int(0.05 * sr):
            break
        for it in get_ts(seg):
            intervals.append([t0 + float(it["start"]), t0 + float(it["end"])])
        if t1 >= total:
            break
        t0 = max(t0 + 0.1, t1 - OVERLAP)
    return intervals


def _merge_intervals(ints):
    if not ints:
        return []
    ints = sorted((float(a), float(b)) for a, b in ints if b > a)
    out = [list(ints[0])]
    for a, b in ints[1:]:
        if a <= out[-1][1] + 1e-3:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# --------------------------------------------------------------------------- #
# Stage 4 — chunks -> plan.json
# --------------------------------------------------------------------------- #

def stage_chunks(out_dir: str, chunk: float):
    plan_path = os.path.join(out_dir, "plan.json")
    vad = _load_json(os.path.join(out_dir, "vad.json"))
    y = _read_mono16k(os.path.join(out_dir, "source16k.wav"))
    sr = 16000
    duration = len(y) / sr

    plan = _json_or_none(plan_path)
    if not isinstance(plan, dict):
        plan = {"duration": duration, "chunk_target": chunk, "chunks": []}

    # recompute chunk list only when not already present (idempotent).
    if not plan.get("chunks"):
        cuts = _make_cuts(vad, duration, chunk, sr)
        plan["chunks"] = cuts
        plan["n_chunks"] = len(cuts)
        _atomic_json(plan_path, plan, indent=2)
    log.info("[4] plan: %d chunks (target %.0fs) -> plan.json",
             len(plan["chunks"]), plan.get("chunk_target", chunk))
    return plan


def _make_cuts(vad, duration, chunk, sr):
    """Walk from 0. Each cut point = midpoint of the LONGEST vad gap whose
    midpoint lies in [prev+0.75*chunk, prev+1.25*chunk]; else cut at the
    lowest-RMS 20ms frame in that window. Last remainder <0.5*chunk merges back."""
    eps = 1e-6
    windows = _rms_windows(vad, duration, sr, frame=0.02)  # list[(s,e,rms)]

    boundaries: list[float] = []
    prev = 0.0
    guard = 0
    while prev < duration - eps:
        guard += 1
        if guard > 100000:
            break
        # a short first part so playback can start quickly
        # ramp: chunk sizes grow 20 -> 25 -> 30 -> 40 -> chunk(60) so chunk n+1 is ready before chunk n finishes playing
        try:
            _ramp = [float(x) for x in os.environ.get("CHUNK_RAMP", os.environ.get("FIRST_CHUNK_S", "20") + ",25,30,40").split(",") if x.strip()]
            assert all(0.5 < v < 1e4 for v in _ramp)
        except Exception:
            _ramp = [20.0, 25.0, 30.0, 40.0]
        c = min(_ramp[len(boundaries)], chunk) if len(boundaries) < len(_ramp) else chunk
        lo = prev + 0.75 * c
        hi = prev + 1.25 * c
        if lo >= duration - 1e-3:
            break

        # candidate gaps: vad gaps and file edges whose midpoint falls in window
        cands: list[tuple[float, float, float]] = []  # (mid, gap_len, is_gap)

        # gap = [end_i, start_{i+1}] for consecutive vad intervals (gap len >= eps)
        pts = _gap_points(vad, duration)  # (a, b) sorted gap boundaries inside [0, duration]
        # leading gap (0..first start) and trailing gap (last end..duration) handled below.
        for (a, b) in pts:
            if b - a < eps:
                continue
            mid = (a + b) / 2.0
            if lo <= mid <= hi:
                cands.append((mid, b - a, 1.0))
        # also allow a mid exactly at lo/hi boundary edge via small search? not needed.

        if cands:
            cands.sort(key=lambda x: (-x[1], x[0]))
            cut = cands[0][0]
        else:
            cut = _lowest_rms_cut(windows, lo, hi)
            if cut is None:
                cut = min(hi, duration)

        if not (prev < cut < duration - eps):
            break
        if cut - prev < 1e-3:
            break
        boundaries.append(round(cut, 3))
        prev = cut

    # build chunk list [start, dur]
    edges = [0.0] + boundaries + [duration]
    chunks = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        d = e - s
        if d <= eps:
            continue
        chunks.append({"idx": len(chunks), "start": round(s, 3), "dur": round(d, 3)})

    # merge tiny tail into previous if < 0.5*chunk
    if len(chunks) >= 2:
        tail = chunks[-1]["dur"]
        if tail < 0.5 * chunk:
            last = chunks.pop()
            prev_c = chunks[-1]
            new_end = prev_c["start"] + prev_c["dur"] + last["dur"]
            prev_c["dur"] = round(new_end - prev_c["start"], 3)
            chunks[-1]["idx"] = len(chunks) - 1
    # re-index
    for i, c in enumerate(chunks):
        c["idx"] = i
    return chunks


def _gap_points(vad, duration=None):
    """Silence gaps (a, b) between MERGED speech intervals: [end_i, start_{i+1}], plus genuine leading [0, first_start]
    and trailing [last_end, duration] silence. fix: the old version paired every consecutive sorted
    boundary, so speech intervals themselves were returned as 'gaps' and chunks could be cut mid-sentence."""
    if not vad:
        return []
    ints = _merge_intervals([(float(a), float(b)) for (a, b) in vad])
    out = []
    if ints and ints[0][0] > 1e-6:
        out.append((0.0, ints[0][0]))
    for i in range(len(ints) - 1):
        a, b = ints[i][1], ints[i + 1][0]
        if b - a > 1e-6:
            out.append((a, b))
    if duration is not None and ints and duration - ints[-1][1] > 1e-6:
        out.append((ints[-1][1], float(duration)))
    return out


def _rms_windows(vad, duration, sr, frame=0.02):
    """Per-20ms-frame RMS, computed block-wise so a long track is never materialised as float32 at once."""
    y = _read_mono16k_cached(sr)
    n = len(y)
    win = max(1, int(frame * sr))
    rms = []
    blk = win * 3000                       # 60 s of samples per block (multiple of the frame length)
    for b0 in range(0, n, blk):
        seg = y[b0:b0 + blk]
        m = (len(seg) // win) * win
        if m:
            fr = np.asarray(seg[:m], dtype=np.float32).reshape(-1, win)
            rms.extend(np.sqrt(np.mean(np.square(fr), axis=1)).astype(float).tolist())
        if len(seg) > m:
            rms.append(float(np.sqrt(np.mean(np.square(seg[m:])))))
    return rms, win


def _lowest_rms_cut(windows, lo, hi):
    rms, win = windows
    i0 = max(0, int(lo * (len(rms) / (len(rms) * (0.02 * 16000) / (0.02 * 16000)))) )
    # simpler: frame duration = 0.02s
    f0 = int(lo / 0.02)
    f1 = max(f0 + 1, int(hi / 0.02) + 1)
    best_i, best_v = None, None
    for i in range(f0, min(f1, len(rms))):
        v = rms[i]
        if v is None:
            continue
        if v != v:  # nan
            continue
        if best_v is None or v < best_v:
            best_v, best_i = v, i
    if best_i is None:
        return None
    t = (best_i + 0.5) * 0.02
    # snap to nearest vad boundary if very close (<=0.05s)
    return t


def _read_mono16k_cached(sr):
    global _Y_CACHE
    try:
        return _Y_CACHE
    except NameError:
        raise


# --------------------------------------------------------------------------- #
# Stage 5 — diarization (whole file, once) -> pyannote or ECAPA fallback.
# --------------------------------------------------------------------------- #

def stage_diarize(out_dir: str):
    vad = _load_json(os.path.join(out_dir, "vad.json"))
    turns_path = os.path.join(out_dir, "spkturns.json")
    backend_path = os.path.join(out_dir, "spkbackend.json")

    _b, _t = _json_or_none(backend_path), _json_or_none(turns_path)
    if isinstance(_b, dict) and "backend" in _b and isinstance(_t, list):
        log.info("[5] cached")
        _vet = _json_or_none(os.path.join(out_dir, "spkvet.json"))      # only jobs vetted at planning time stay vetted
        _set_vetted(isinstance(_vet, dict) and _vet.get("vetted") is True)
        return _b["backend"], _t
    _set_vetted(False)

    global _FULL_DIAR
    _FULL_DIAR = None          # never reuse a stale non-exclusive annotation
    _rm(backend_path)         # the completion marker must not outlive the data it describes
    _rm(os.path.join(out_dir, "spkvet.json")); _rm(os.path.join(out_dir, "overlap_spans.json"))
    y = _read_mono16k(os.path.join(out_dir, "source16k.wav"))
    sr = 16000

    degraded, reason = False, None
    backend, turns, why = _diarize_pyannote(y, sr, out_dir)
    if not backend:
        #: the weaker ECAPA clustering must never be silent. It is flagged degraded in
        # spkbackend.json, speakers_global.json and plan.json.
        degraded, reason = True, f"pyannote failed: {why}"
        log.warning("[5] DEGRADED diarization: %s -> ECAPA agglomerative fallback", reason)
        backend, turns = _diarize_ecapa(y, sr, vad, out_dir)

    # drop turns < 0.3s, relabel 0..N-1 by first appearance
    _all_turns = list(turns)
    turns = [(s, e, spk) for (s, e, spk) in turns if (e - s) >= 0.3]
    turns.sort(key=lambda x: (x[0], x[1]))
    relabel = {}
    out_turns = []
    for (s, e, spk) in turns:
        key = str(spk)
        if key not in relabel:
            relabel[key] = len(relabel)
        out_turns.append({"start": round(s, 3), "end": round(e, 3),
                          "speaker": relabel[key]})

    # identity-evidence speaker vetting + overlap/short-turn spans (pyannote only; fails open to the legacy path)
    if not degraded and _FULL_DIAR is not None and _g90_on("SPK_EVIDENCE"):
        try:
            out_turns = _g90_vet(out_dir, y, sr, out_turns, relabel, _all_turns, _FULL_DIAR)
        except Exception as e:  # noqa: BLE001
            log.warning("[5] vetting failed -> legacy speaker handling: %s: %s", type(e).__name__, e)
            _rm(os.path.join(out_dir, "spkvet.json")); _rm(os.path.join(out_dir, "overlap_spans.json"))
            _set_vetted(False)
            out_turns = [t for t in out_turns]
    relabel = {t["speaker"]: 1 for t in out_turns}

    backend = {"backend": backend, "degraded": degraded, "degraded_reason": reason}
    _atomic_json(turns_path, out_turns, indent=2)
    _atomic_json(backend_path, backend, indent=2)      # written LAST: it is the "stage complete" marker
    log.info("[5] diarized (%s) -> %d turns, %d speakers",
             backend["backend"], len(out_turns), len(relabel))
    return backend["backend"], out_turns


_FULL_DIAR = None      # non-exclusive pyannote segments (start, end, label) of the last _diarize_pyannote call


def _diarize_pyannote(y, sr, out_dir):
    try:
        import torch
        from pyannote.audio import Pipeline
        token = os.environ.get("HF_TOKEN")
        p = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=token)
        if torch.cuda.is_available():
            p.to(torch.device("cuda"))
        out = p({"waveform": torch.from_numpy(_as_array(y))[None], "sample_rate": sr})
        sub = None
        for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
            sub = getattr(out, attr, None)
            if sub is not None:
                break
        if sub is None:
            sub = out
        turns = []
        for seg, _trk, lab in sub.itertracks(yield_label=True):
            turns.append((float(seg.start), float(seg.end), lab))
        # (#11): keep the ordinary (overlap-aware) annotation next to the exclusive one; stage_diarize publishes
        # overlap_spans.json from it. Never fails the diarization.
        global _FULL_DIAR
        _FULL_DIAR = None
        try:
            full = getattr(out, "speaker_diarization", None)
            if full is not None:
                _FULL_DIAR = [(float(s.start), float(s.end), str(l)) for s, _t, l in full.itertracks(yield_label=True)]
        except Exception as e:  # noqa: BLE001
            log.warning("[5] overlap annotation unavailable: %s: %s", type(e).__name__, e)
        return "pyannote-community-1", turns, None
    except Exception as e:  # noqa: BLE001 - must fall back on ANY error
        log.warning("[5] pyannote failed -> ECAPA fallback: %s: %s", type(e).__name__, e)
        return None, None, f"{type(e).__name__}: {str(e)[:300]}"


def _diarize_ecapa(y, sr, vad, out_dir):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import AgglomerativeClustering

    savedir = os.path.join(out_dir, ".ecapa")
    os.makedirs(savedir, exist_ok=True)
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    hop, win = 0.75, 1.5
    feats, tstarts = [], []
    with torch.no_grad():
        for (a, b) in vad:
            a, b = float(a), float(b)
            if b - a < 0.4:
                continue
            pos = a
            while pos + win <= b + 1e-3:
                i0, i1 = int(pos * sr), int((pos + win) * sr)
                x = torch.from_numpy(y[i0:i1])[None, :]
                emb = clf.encode_batch(x)  # [1,1,D]
                feats.append(emb[0, 0, :].cpu().numpy().astype(np.float32))
                tstarts.append(pos)
                pos += hop
        if feats:
            feats = np.vstack(feats)
            # normalize for cosine
            norm = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / (norm + 1e-9)
            model = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=0.6)
            if len(feats) < 2:                    # sklearn cannot cluster 1 sample -> one speaker
                labels = np.zeros(len(feats), dtype=int)
            else:
                labels = model.fit_predict(feats)
        else:
            labels = np.array([], dtype=int)

    # merge consecutive same-label windows into turns
    turns = []
    cur_lab = None
    cur_start = cur_end = None
    n = len(tstarts)
    for i in range(n):
        if labels[i] != cur_lab:
            if cur_lab is not None:
                turns.append((cur_start, cur_end, cur_lab))
            cur_lab, cur_start = labels[i], tstarts[i]
            cur_end = tstarts[i] + win
        else:
            cur_end = max(cur_end, tstarts[i] + win)
    if cur_lab is not None:
        turns.append((cur_start, cur_end, cur_lab))

    # normalize cosine distance threshold: sklearn 'cosine' distance = 1-cos;
    # 0.6 corresponds to cos>0.4 (reasonable). Keep as-is per spec.
    return "ecapa-agglomerative", turns


# --------------------------------------------------------------------------- #
# Stage 6 — gender per speaker -> per-speaker mean p_female.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# — identity-evidence speaker vetting (#4 short speakers, over-split) + overlap spans (#11)
# --------------------------------------------------------------------------- #

_VETTED = False        # True only while the speaker turns in memory were vetted by _g90_vet (this run or a cached vetted run)


def _set_vetted(v: bool) -> None:
    global _VETTED
    _VETTED = bool(v)


def _g90_on(name: str, default: bool = True) -> bool:
    """Kill switch parsed robustly: blank -> default; 0/false/no/off/n/f -> off; anything else -> on."""
    v = os.environ.get(name, "").strip().lower()
    return default if v == "" else v not in ("0", "false", "no", "off", "n", "f")


# Calibrated on the 4 demo jobs (see vault note): same-speaker split null vs different-speaker cosines.
G90_COS_SHORT = 0.60    # ECAPA centroid cosine needed to fold a < 5 s speaker into a main speaker as the SAME person
G90_COS_MERGE = 0.72    # ... to merge two established speakers (over-split repair); stricter
G90_F0_TOL = 0.25       # max |delta median ln F0| (about 4 semitones) for a merge
G90_SIM_MAX_S = 0.5     # max simultaneous-speech seconds between two labels that are claimed to be one person
G90_MIN_OWN_S = 2.0     # below this much speech a speaker cannot carry an identity: legacy temporal fold
G90_MIN_CLEAN_S = 1.0   # minimum clean (non-overlapped) speech to compute an embedding at all
G90_MIN_TURNS = 2       # over-split repair: each side needs this many turns with clean speech


def _g90_clear_vet(out_dir: str) -> None:
    """Hook for stage_write's smap_hash hard-fail path: drop the derived caches so a retry recomputes them.
    Never touches speakers_global.json."""
    _rm(os.path.join(out_dir, "spkvet.json")); _rm(os.path.join(out_dir, "overlap_spans.json"))


def _g90_consts():
    return {
        "cos_short": _env_float("SPK_COS_SHORT", G90_COS_SHORT, lo=-1.0, hi=1.0),
        "cos_merge": _env_float("SPK_COS_MERGE", G90_COS_MERGE, lo=-1.0, hi=1.0),
        "f0_tol": _env_float("SPK_F0_TOL", G90_F0_TOL, lo=0.0, hi=3.0),
        "sim_max_s": _env_float("SPK_SIM_MAX_S", G90_SIM_MAX_S, lo=0.0),
        "min_own_s": _env_float("SPK_MIN_OWN_S", G90_MIN_OWN_S, lo=0.0),
        "min_clean_s": _env_float("SPK_MIN_CLEAN_S", G90_MIN_CLEAN_S, lo=0.2),
        "min_turns": int(_env_float("SPK_MIN_TURNS", G90_MIN_TURNS, lo=1.0)),
        "overlap_on": _g90_on("SPK_OVERLAP_SPANS"),
        "overlap_merge_gap": 0.15, "overlap_min": 0.1,
    }


def _sub_intervals(segs, cuts):
    """segs minus the union of cuts; both lists of (a, b)."""
    out = []
    for (x, z) in segs:
        cur = [(x, z)]
        for (a, b) in cuts:
            nxt = []
            for (u, v) in cur:
                if b <= u or a >= v:
                    nxt.append((u, v))
                else:
                    if a > u:
                        nxt.append((u, a))
                    if b < v:
                        nxt.append((b, v))
            cur = nxt
        out += cur
    return out


def _g90_regions(full):
    """Sweep the non-exclusive annotation. Returns (overlap_regions [(a, b, frozenset(labels))], pair_s {(la, lb): seconds}).
    Overlap = two or more labels active at once."""
    ev = []
    for (a, b, l) in full:
        if b > a:
            ev.append((a, 1, l)); ev.append((b, -1, l))
    ev.sort(key=lambda e: (e[0], e[1]))
    active, regs, pair, last = {}, [], {}, None
    for (t, d, l) in ev:
        if last is not None and t > last and len(active) >= 2:
            labs = frozenset(active)
            regs.append((last, t, labs))
            ls = sorted(labs)
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    pair[(ls[i], ls[j])] = pair.get((ls[i], ls[j]), 0.0) + (t - last)
        if d == 1:
            active[l] = active.get(l, 0) + 1
        else:
            active[l] = active.get(l, 0) - 1
            if active[l] <= 0:
                active.pop(l, None)
        last = t
    return regs, pair


def _g90_ecapa(savedir):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    os.makedirs(savedir, exist_ok=True)
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
                                          run_opts={"device": "cpu"})


def _g90_embed(clf, seg):
    import torch
    with torch.no_grad():
        e = clf.encode_batch(torch.from_numpy(np.ascontiguousarray(seg, dtype="float32"))[None]).squeeze().cpu().numpy()
    return (e / (np.linalg.norm(e) + 1e-9)).astype(np.float32)


def _g90_logf0(seg):
    try:
        import librosa
        f0, vf, _ = librosa.pyin(np.asarray(seg, dtype="float32"), fmin=60, fmax=450, sr=16000, frame_length=1024)
        f0 = f0[vf & ~np.isnan(f0)]
        return float(np.median(np.log(f0))) if len(f0) >= 20 else None
    except Exception:  # noqa: BLE001
        return None


def _g90_pieces(turns, spk, overlap_ints, min_len, max_piece=8.0, max_total=40.0):
    """Clean (never simultaneous with another speaker) pieces of one speaker, split to <= max_piece, longest first,
    capped at max_total seconds. Returns [(a, b)]."""
    base = [(t["start"], t["end"]) for t in turns if t["speaker"] == spk]
    cl = [(a, b) for (a, b) in _sub_intervals(base, overlap_ints) if b - a >= min_len]
    pcs = []
    for (a, b) in cl:
        while b - a > max_piece * 1.5:
            pcs.append((a, a + max_piece)); a += max_piece
        pcs.append((a, b))
    pcs.sort(key=lambda p: (-(p[1] - p[0]), p[0]))
    out, tot = [], 0.0
    for p in pcs:
        if tot >= max_total:
            break
        out.append(p); tot += p[1] - p[0]
    return sorted(out)


def _g90_evidence(y, sr, turns, overlap_ints, savedir, c):
    """per speaker: {'pieces': [(a, b, emb)], 'clean_s', 'n_clean_turns', 'logf0'}.
    Clean speech is CUMULATIVE: every clean interval counts (also those shorter than min_clean_s); the embeddings come from
    the concatenated clean audio (longest intervals first, <= 40 s, cut into <= 8 s chunks). A speaker whose cumulative
    clean speech is < min_clean_s gets pieces == [] (legacy fold). Speakers with < SPK_MIN_OWN_S total speech are also
    folded by the legacy rule in _g90_decide (ruling: they cannot carry an identity)."""
    clf = _g90_ecapa(savedir)
    ev = {}
    for spk in sorted({t["speaker"] for t in turns}):
        base = [(t["start"], t["end"]) for t in turns if t["speaker"] == spk]
        cl = [(a, b) for (a, b) in _sub_intervals(base, overlap_ints) if b - a >= 0.05]     # fragments < 50 ms are deliberately discarded
        tot = sum(b - a for (a, b) in cl)
        n_turns = sum(1 for t in turns if t["speaker"] == spk and any(a < t["end"] and b > t["start"] and b - a >= 0.2 for (a, b) in cl))
        pieces, f0 = [], None
        if tot >= c["min_clean_s"]:
            cl.sort(key=lambda p: (-(p[1] - p[0]), p[0]))
            used, acc = 0.0, []
            for (a, b) in cl:
                if used >= 40.0:
                    break
                if used + (b - a) > 40.0:
                    b = a + (40.0 - used)          # truncate the last interval: embedded audio is capped at 40 s
                acc.append((a, b)); used += b - a
            acc.sort()
            audio = np.concatenate([y[int(a * sr):int(b * sr)] for (a, b) in acc])
            t0 = acc[0][0]
            step = int(8.0 * sr)
            for i in range(0, len(audio), step):
                seg = audio[i:i + step]
                if len(seg) < int(0.5 * sr) and pieces:
                    break
                pieces.append((t0 + i / sr, t0 + (i + len(seg)) / sr, _g90_embed(clf, seg)))
            f0 = _g90_logf0(audio[:sr * 20])
        ev[spk] = {"pieces": pieces, "clean_s": tot, "n_clean_turns": n_turns, "logf0": f0}
    return ev


def _g90_centroid(pieces):
    if not pieces:
        return None
    w = np.array([b - a for (a, b, _e) in pieces], dtype="float64")
    c = np.average(np.stack([e for (_a, _b, e) in pieces]), axis=0, weights=w)
    return (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32)


def _g90_cos(a, b):
    return float(np.dot(a, b))


def _g90_decide(turns, ev, pair_s, c, min_s=5.0):
    """Pure decision logic (unit-testable). Returns (mapping {spk: spk'}, log [dict]).
    1. Short (< min_s) speakers with enough clean speech: merge into the MOST SIMILAR main speaker iff cosine >= cos_short,
       F0-compatible and not simultaneous with it; otherwise they stay their own speaker. Without evidence (< min_own_s
       of speech or < min_clean_s clean) -> legacy temporal fold.
    2. Over-split repair among speakers that remain: greedy highest-cosine pair with cosine >= cos_merge, F0 compatible,
       simultaneous speech <= sim_max_s, and >= min_turns clean turns each; centroid pooled after each merge."""
    tot = {}
    for t in turns:
        tot[t["speaker"]] = tot.get(t["speaker"], 0.0) + t["end"] - t["start"]
    if not tot:
        return {}, []
    big = {k for k, v in tot.items() if v >= min_s} or {max(tot, key=tot.get)}
    mapping = {k: k for k in tot}
    cent = {k: _g90_centroid(ev.get(k, {}).get("pieces", [])) for k in tot}
    logf0 = {k: ev.get(k, {}).get("logf0") for k in tot}
    lg = []

    def sim(a, b):
        return pair_s.get((a, b), pair_s.get((b, a), 0.0))

    def f0_ok(a, b):
        return logf0[a] is None or logf0[b] is None or abs(logf0[a] - logf0[b]) <= c["f0_tol"]

    bigs = [t for t in turns if t["speaker"] in big]
    for k in sorted(tot, key=lambda k: tot[k]):
        if k in big:
            continue
        have = cent[k] is not None and tot[k] >= c["min_own_s"]
        cand = [(b, _g90_cos(cent[k], cent[b])) for b in big if b != k and cent.get(b) is not None] if have else []
        rec = {"speaker": k, "total_s": round(tot[k], 2), "rule": "short"}
        if cand:
            b, cs = max(cand, key=lambda x: x[1])
            rec.update(best=b, cos=round(cs, 4))
            if cs >= c["cos_short"] and f0_ok(k, b) and sim(k, b) <= c["sim_max_s"]:
                mapping[k] = b; rec["action"] = "merge_identity"
            else:
                rec["action"] = "keep_own"
        else:
            # legacy: fold into the temporally nearest main speaker
            ks = [t for t in turns if t["speaker"] == k]
            cm = (ks[0]["start"] + ks[0]["end"]) / 2 if ks else 0.0
            near = min(bigs, key=lambda b: 0.0 if b["start"] <= cm <= b["end"] else min(abs(b["start"] - cm), abs(b["end"] - cm)))
            mapping[k] = near["speaker"]; rec["action"] = "fold_no_evidence"
        lg.append(rec)

    # over-split repair among the survivors
    alive = sorted({mapping[k] for k in mapping})
    pooled = {k: list(ev.get(k, {}).get("pieces", [])) for k in alive}
    members = {k: {k} for k in alive}
    for k in tot:
        if mapping[k] != k:
            members[mapping[k]].add(k)
    nclean = {k: ev.get(k, {}).get("n_clean_turns", 0) for k in alive}
    while True:
        best = None
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                ca, cb = _g90_centroid(pooled[a]), _g90_centroid(pooled[b])
                if ca is None or cb is None or nclean[a] < c["min_turns"] or nclean[b] < c["min_turns"]:
                    continue
                cs = _g90_cos(ca, cb)
                simult = sum(sim(x, z) for x in members[a] for z in members[b])
                if cs >= c["cos_merge"] and simult <= c["sim_max_s"] and f0_ok(a, b) and (best is None or cs > best[0]):
                    best = (cs, a, b, simult)
        if best is None:
            break
        cs, a, b, simult = best
        keep, gone = (a, b) if tot_of(members, tot, a) >= tot_of(members, tot, b) else (b, a)
        lg.append({"rule": "oversplit", "keep": keep, "merged": gone, "cos": round(cs, 4), "simultaneous_s": round(simult, 3),
                   "action": "merge_identity"})
        for k in members[gone]:
            mapping[k] = keep
        members[keep] |= members[gone]; pooled[keep] += pooled[gone]; nclean[keep] += nclean[gone]
        alive.remove(gone); members.pop(gone); pooled.pop(gone)
    return mapping, lg


def tot_of(members, tot, k):
    return sum(tot[m] for m in members[k])


def _g90_overlap_spans(all_turns, full, relabel, c):
    """Published to <job>/overlap_spans.json. kind 'overlap': >= 2 speakers at once (non-exclusive pyannote output).
    kind 'short_turn': exclusive turn < 0.3 s that stage_diarize drops. Times in seconds on the job's source16k timeline."""
    regs, _pair = _g90_regions(full) if full else ([], {})
    spans = []
    for (a, b, labs) in regs:
        ids = sorted({relabel[l] for l in labs if l in relabel})
        if spans and spans[-1]["kind"] == "overlap" and a - spans[-1]["end"] <= c["overlap_merge_gap"]:
            spans[-1]["end"] = b
            spans[-1]["speakers"] = sorted(set(spans[-1]["speakers"]) | set(ids))
        else:
            spans.append({"start": a, "end": b, "kind": "overlap", "speakers": ids})
    spans = [s for s in spans if s["end"] - s["start"] >= c["overlap_min"]]
    for (a, b, l) in all_turns:
        if (b - a) < 0.3 and b > a:
            spans.append({"start": a, "end": b, "kind": "short_turn", "speakers": [relabel[str(l)]] if str(l) in relabel else []})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    for s in spans:
        s["start"] = round(s["start"], 3); s["end"] = round(s["end"], 3)
    return spans


def _g90_vet(out_dir, y, sr, out_turns, relabel, all_turns, full):
    """Returns the vetted turns (final labels 0..N-1 by first appearance). Writes overlap_spans.json and spkvet.json.
    `relabel` maps str(pyannote label) -> id used in out_turns."""
    if not full:
        raise RuntimeError("no non-exclusive diarization: legacy path")
    c = _g90_consts()
    regs, pair_lab = _g90_regions(full)
    inv = {v: k for k, v in relabel.items()}
    pair_s = {}
    for (la, lb), s in pair_lab.items():
        if la in relabel and lb in relabel:
            pair_s[(relabel[la], relabel[lb])] = s
    overlap_ints = [(a, b) for (a, b, _l) in regs]
    ev = _g90_evidence(y, sr, out_turns, overlap_ints, os.path.join(out_dir, ".ecapa"), c)
    mapping, lg = _g90_decide(out_turns, ev, pair_s, c)
    turns = [dict(t, speaker=mapping.get(t["speaker"], t["speaker"])) for t in out_turns]
    order = []
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in order:
            order.append(t["speaker"])
    fin = {k: order.index(k) for k in order}
    turns = [dict(t, speaker=fin[t["speaker"]]) for t in turns]
    final_of_pya = {lab: fin[mapping.get(i, i)] for lab, i in relabel.items() if mapping.get(i, i) in fin}
    if c["overlap_on"] and full is not None:
        spans = _g90_overlap_spans(all_turns, full, final_of_pya, c)
        doc = {"version": 1, "source": "pyannote-community-1 speaker_diarization (non-exclusive)",
               "duration": round(len(y) / float(sr), 3),
               "overlap_s": round(sum(s["end"] - s["start"] for s in spans if s["kind"] == "overlap"), 3),
               "short_turn_s": round(sum(s["end"] - s["start"] for s in spans if s["kind"] == "short_turn"), 3),
               "spans": spans}
        _atomic_json(os.path.join(out_dir, "overlap_spans.json"), doc, indent=1)
    vet = {"vetted": True, "version": 1, "constants": {k: v for k, v in c.items()},
           "mapping": {str(k): v for k, v in mapping.items()}, "decisions": lg,
           "evidence": {str(k): {"clean_s": round(v["clean_s"], 2), "n_clean_turns": v["n_clean_turns"],
                                 "logf0": v["logf0"]} for k, v in ev.items()}}
    _atomic_json(os.path.join(out_dir, "spkvet.json"), vet, indent=1)
    _set_vetted(True)
    n_before, n_after = len(set(mapping)), len(order)
    log.info("[5] vetting: %d -> %d speakers; %s", n_before, n_after,
             [(d.get("speaker", d.get("merged")), d["action"]) for d in lg])
    return turns


def _merge_small_speakers(turns, min_s=5.0):
    """Fold speakers with < min_s of speech into the temporally nearest main speaker, relabel 0..N-1 (in place).
    when stage_diarize already vetted the speakers by identity evidence (_VETTED), nothing is folded here
    (short speakers that are acoustically distinct stay their own speaker); only the relabel runs."""
    tot = {}
    for t in turns:
        tot[t["speaker"]] = tot.get(t["speaker"], 0.0) + t["end"] - t["start"]
    if not tot:
        return turns
    big = {k for k, v in tot.items() if v >= min_s} or {max(tot, key=tot.get)}
    if _VETTED:
        big = set(tot)
    bigs = [t for t in turns if t["speaker"] in big]
    for t in turns:
        if t["speaker"] not in big:
            c = (t["start"] + t["end"]) / 2
            t["speaker"] = min(bigs, key=lambda b: 0.0 if b["start"] <= c <= b["end"] else min(abs(b["start"] - c), abs(b["end"] - c)))["speaker"]
    order = []
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in order:
            order.append(t["speaker"])
    for t in turns:
        t["speaker"] = order.index(t["speaker"])
    return turns


def _norm_gender_keys(g):
    """JSON round-trips int speaker ids to str keys; every lookup uses int ids. Normalise to int keys (kill switch
    GENDER_KEYFIX=0 restores the old behaviour). Non-numeric keys / non-dict values are kept as-is (never raises)."""
    if os.environ.get("GENDER_KEYFIX", "1") != "1" or not isinstance(g, dict):
        return g
    out = {}
    for k, v in g.items():
        try:
            k2 = int(k)
        except (TypeError, ValueError):
            k2 = k
        out[k2] = v
    return out


def stage_gender(out_dir: str, turns):
    _merge_small_speakers(turns)
    cache_path = os.path.join(out_dir, "spkgender.json")
    _g = _json_or_none(cache_path)
    if isinstance(_g, dict):
        return _norm_gender_keys(_g)

    from model import ECAPA_gender
    import torch

    m = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
    m.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    y = _read_mono16k(os.path.join(out_dir, "source16k.wav"))
    sr = 16000

    by_spk = {}
    for t in turns:
        by_spk.setdefault(t["speaker"], []).append(t)

    gender = {}
    for spk, tl in by_spk.items():
        tl = sorted(tl, key=lambda x: (x["end"] - x["start"]), reverse=True)
        longest = tl[:10]
        probs = []
        for t in longest:
            dur = t["end"] - t["start"]
            if dur < 1.0:
                continue
            i0, i1 = int(t["start"] * sr), int(t["end"] * sr)
            x = torch.from_numpy(y[i0:i1])[None, :].to(device)
            with torch.no_grad():
                logits = m.forward(x)
                probs.append(float(np.exp(1.0) * 0))  # placeholder to avoid unused
                # compute softmax properly
                p = torch.softmax(logits, dim=-1)[0]
                probs.append(float(p[1]))  # index 1 = female (per spec)
            probs.pop(-2) if len(probs) > 1 and probs[-2] == 0.0 else None
        # robust: recompute if placeholder left
        probs = [p for p in probs if isinstance(p, float) and 0.0 <= p <= 1.0]
        mean_p = float(np.mean(probs)) if probs else 0.5
        gender[spk] = {"mean_p": mean_p, "probs": probs,
                       "total_s": sum(t["end"] - t["start"] for t in tl),
                       "n_turns": len(tl)}

    _atomic_json(cache_path, gender, indent=2)
    return gender


_GENDER_WARNED = set()


def _gender_band():
    """(HI, LO) from GENDER_HI / GENDER_LO. Invalid values (non-numeric, NaN/inf, outside [0,1]) fall back to the
    validated defaults 0.6 / 0.4 with a warning; an inverted pair (LO > HI) would classify every speaker as female, so
    the whole pair falls back to the defaults."""
    hi = _env_float("GENDER_HI", 0.6, lo=0.0, hi=1.0)
    lo = _env_float("GENDER_LO", 0.4, lo=0.0, hi=1.0)
    if lo > hi:
        key = (hi, lo)
        if key not in _GENDER_WARNED:
            _GENDER_WARNED.add(key)
            log.warning("GENDER_LO=%s > GENDER_HI=%s is inverted; using defaults 0.4/0.6", lo, hi)
        return 0.6, 0.4
    return hi, lo


def _assign_gender(mean_p, total_s, n_turns):
    # Band validated on labelled CV17+Kathbath (speaker-disjoint DEV; see vault RESULT - Speaker gender validation):
    # old 0.9/0.1 abstained on 15.6% of TEST speakers (unknown -> parity voice = coin flip); 0.6/0.4 lowers expected mismatch.
    GENDER_HI, GENDER_LO = _gender_band()
    if mean_p != mean_p:                      # NaN probability: no evidence
        return "unknown"
    if mean_p >= GENDER_HI and total_s >= 8.0 and (n_turns >= 2 or total_s >= 20.0):
        return "female"
    if mean_p <= GENDER_LO and total_s >= 8.0 and (n_turns >= 2 or total_s >= 20.0):
        return "male"
    return "unknown"


# --------------------------------------------------------------------------- #
# Stage 7 — assemble speakers_global.json + write timings into plan.json
# --------------------------------------------------------------------------- #

def stage_write(out_dir: str, backend, turns, gender, degraded=None, degraded_reason=None, plan_fn=None):
    by_spk = {}
    for t in turns:
        by_spk.setdefault(t["speaker"], []).append(t)

    speakers = []
    for spk in sorted(by_spk.keys()):
        tl = by_spk[spk]
        total_s = sum(t["end"] - t["start"] for t in tl)
        n_turns = len(tl)
        g = gender.get(spk)
        if g is None and os.environ.get("GENDER_KEYFIX", "1") == "1":
            g = gender.get(str(spk))                 # tolerate str-keyed (JSON-reloaded) dicts from any caller
        if not isinstance(g, dict):
            g = {"mean_p": 0.5}
        mean_p = g.get("mean_p", 0.5)
        label = _assign_gender(mean_p, total_s, n_turns)
        speakers.append({
            "id": spk,
            "gender": label,
            "p_female": round(mean_p, 4),
            "total_s": round(total_s, 3),
            "n_turns": n_turns,
        })

    if degraded is None:
        degraded = str(backend).startswith("ecapa")
    data = {
        "backend": backend,
        "degraded": bool(degraded),
        "degraded_reason": degraded_reason if degraded else None,
        "speakers": sorted(speakers, key=lambda s: s["id"]),
        "turns": turns,
    }
    sp_path = os.path.join(out_dir, "speakers_global.json")
    # ONE atomic publish of the complete contract. For a rollout=1 job the voice plan is computed BEFORE the file is
    # written and is part of the same write; an existing plan is immutable (a changed map hard-fails, files untouched); a
    # planning failure FAILS the planner (never a silent downgrade to legacy). All under a job-level flock.
    with _job_lock(out_dir):
        _prev = _json_or_none(sp_path)
        if plan_fn is not None and _prev is None and os.path.exists(sp_path):
            raise PlannerError("speakers_global.json exists but is unreadable on a rollout=1 job; refusing to overwrite it")
        if plan_fn is not None:
            from sunwai.planning import voices as _VA
            pv = _prev.get("voice_plan") if isinstance(_prev, dict) else None
            if isinstance(pv, dict):
                if pv.get("smap_hash") != _VA.smap_hash(data):
                    raise PlannerError("voice plan is immutable but the speaker map changed on retry (smap_hash "
                                       f"{pv.get('smap_hash')} -> {_VA.smap_hash(data)}); old plan and files kept")
                data["voice_plan"] = pv
            else:
                try:
                    data["voice_plan"] = plan_fn(data)
                except Exception as e:
                    raise PlannerError(f"voice plan failed for a rollout=1 job: {type(e).__name__}: {e}")
        _atomic_json(sp_path, data, indent=2, ensure_ascii=False)
        if "voice_plan" in data:
            log.info("[voice] plan v%s degraded=%s canonical=%s warnings=%s", data["voice_plan"].get("version"),
                     data["voice_plan"].get("distinctness_degraded"), data["voice_plan"].get("canonical"), data["voice_plan"].get("warnings"))

    # timings already tracked into plan.json during run (caller updates it).
    return data


# --------------------------------------------------------------------------- #
# Stage 8 — voice plan: computed ONCE for NEW jobs, embedded in speakers_global.json, read-only for workers.
# --------------------------------------------------------------------------- #

def _env_on(name, default=True):
    v = os.environ.get(name, "").strip().lower()
    return default if v == "" else v not in ("0", "false", "no", "off", "n", "f")


class _job_lock:
    """Job-level exclusive flock (planner retries / concurrent planners serialise on it)."""
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, ".planner.lock")
    def __enter__(self):
        import fcntl
        self.f = open(self.path, "a+")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self
    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def decide_voice_rollout(out_dir: str) -> bool:
    """Persisted at the FIRST planner run of a job. rollout=1 ONLY if, at that moment, the job dir holds no speakers_global.json,
    no plan.json from an earlier run and no language/chunk output directory (any name length, incl. _prep); anything
    pre-existing = legacy for good. Retries read the marker and never re-decide."""
    mp = os.path.join(out_dir, "voice_rollout.json")
    with _job_lock(out_dir):
        m = _json_or_none(mp)
        if isinstance(m, dict) and "voice_plan" in m:
            return m["voice_plan"] == 1
        _cfg = _json_or_none(str(DATA_DIR / "languages.json")) or {}
        _codes = {t["code"] for t in _cfg.get("targets", [])}
        def _has_out(d):          # any pre-existing language dir or _prep, even EMPTY, or any dir holding c* entries
            full = os.path.join(out_dir, d)
            return os.path.isdir(full) and not d.startswith(".") and (d in _codes or d == "_prep" or any(c.startswith("c") for c in os.listdir(full)))
        old = os.path.exists(os.path.join(out_dir, "speakers_global.json")) or os.path.exists(os.path.join(out_dir, "plan.json")) \
            or any(_has_out(d) for d in os.listdir(out_dir))
        new = (not old) and _env_on("VOICE_PLAN") and _env_on("VOICE_XLING")
        _atomic_json(mp, {"voice_plan": 1 if new else 0, "decided": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "preexisting": bool(old)}, indent=2)
        log.info("[voice] rollout decided for this job: %s", "voice_plan" if new else "legacy")
        return new


def make_plan_fn(out_dir: str):
    """closure that computes the voice plan for the finished speaker map (called inside stage_write, before the single write)."""
    def fn(data):
        from sunwai.planning import voices as VA
        y = _as_array(_read_mono16k(os.path.join(out_dir, "source16k.wav")))
        feats = VA.speaker_features(y, data["turns"], os.path.join(out_dir, ".ecapa"))
        cfg = _json_or_none(str(DATA_DIR / "languages.json")) or {}
        langs = [t["code"] for t in cfg.get("targets", [])]
        job = _json_or_none(os.path.join(out_dir, "job.json")) or {}
        opt = [l for l in (job.get("langs") or []) if l in langs] or langs
        return VA.plan_job(data, langs, feats, opt_langs=opt)
    return fn


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _record_timing(plan_path, name, secs):
    try:
        plan = _load_json(plan_path)
    except Exception:
        plan = {}
    plan.setdefault("timings", {})[name] = round(float(secs), 3)
    _atomic_json(plan_path, plan, indent=2)


def _update_plan_plan(plan_path, plan):
    existing = {}
    try:
        existing = _load_json(plan_path)
    except Exception:
        existing = {}
    existing["duration"] = plan["duration"]
    existing["chunk_target"] = plan["chunk_target"]
    existing["n_chunks"] = plan["n_chunks"]
    existing["chunks"] = plan["chunks"]
    existing.setdefault("timings", {})
    _atomic_json(plan_path, existing, indent=2)


def main(argv=None):
    """CLI entry. Clean failures (no audio, too long, bad media, timeouts) write <out>/planner_error.json, print a final
    'PLANNER_ERROR <code>: <message>' line on stderr (server2 shows the log tail) and return 3."""
    try:
        return _main(argv)
    except PlannerError as e:
        return _report_error(argv, e.code, str(e))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("planner crashed")
        _report_error(argv, "internal", f"{type(e).__name__}: {e}")
        return 1


def _report_error(argv, code, msg):
    out = None
    try:
        a = list(argv if argv is not None else sys.argv[1:])
        out = _norm_path(a[a.index("--out") + 1])
        _atomic_json(os.path.join(out, "planner_error.json"), {"error": code, "message": msg, "ts": time.time()}, indent=2)
    except Exception:
        pass
    print(f"PLANNER_ERROR {code}: {msg}", file=sys.stderr, flush=True)
    return 3


def _main(argv=None):
    global _Y_CACHE
    import argparse
    setup_logging()
    ap = argparse.ArgumentParser(
        description="Prepare a video for chunked progressive dubbing.")
    ap.add_argument("--url", required=True, help="URL or local file")
    ap.add_argument("--out", required=True, help="JOB_DIR")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=0.0, help="0 = to end")
    ap.add_argument("--chunk", type=float, default=60.0)
    args = ap.parse_args(argv)

    out_dir = _norm_path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    _rm(os.path.join(out_dir, "planner_error.json"))
    _voice_new = decide_voice_rollout(out_dir)
    if not (args.chunk > 1.0 and math.isfinite(args.chunk)):
        raise PlannerError(f"--chunk must be a finite number > 1 s, got {args.chunk}")

    # Pre-cache the mono wav into a module global for chunk/RMS speedups.
    try:
        _Y_CACHE  # noqa
    except NameError:
        pass

    t_total = time.time()
    timings = {}

    def _run(name, fn):
        t0 = time.time()
        r = fn()
        sec = time.time() - t0
        timings[name] = round(sec, 3)
        log.info("[%s] done in %.3fs", name, sec)
        return r

    _run("fetch", lambda: stage_fetch(args.url, out_dir, args.start, args.dur))
    _run("audio", lambda: stage_audio(out_dir))

    # Pre-load mono samples once; share with VAD/chunks/diarize to avoid re-read.
    _Y_CACHE = _read_mono16k(os.path.join(out_dir, "source16k.wav"))

    plan_path = os.path.join(out_dir, "plan.json")
    vad = _run("vad", lambda: stage_vad(out_dir))
    # VAD may have created vad.json; reload if needed
    vad = _load_json(os.path.join(out_dir, "vad.json")) if vad is None else vad

    plan = _run("chunks", lambda: stage_chunks(out_dir, args.chunk))
    _update_plan_plan(plan_path, plan)

    backend, turns = _run("diarize", lambda: stage_diarize(out_dir))
    gender = _run("gender", lambda: stage_gender(out_dir, turns))
    _bi0 = _json_or_none(os.path.join(out_dir, "spkbackend.json")) or {}
    _run("write", lambda: stage_write(out_dir, backend, turns, gender,
                                      degraded=_bi0.get("degraded"), degraded_reason=_bi0.get("degraded_reason"),
                                      plan_fn=make_plan_fn(out_dir) if _voice_new else None))

    # Finalize timings in plan.json
    try:
        plan = _load_json(plan_path)
    except Exception:
        plan = {"duration": len(_Y_CACHE) / 16000.0,
                "chunk_target": args.chunk, "chunks": []}
    plan["timings"] = timings
    _bi = _json_or_none(os.path.join(out_dir, "spkbackend.json")) or {}
    _deg = bool(_bi.get("degraded", str(backend).startswith("ecapa")))
    plan["diarization"] = {"backend": backend, "degraded": _deg, "reason": _bi.get("degraded_reason")}
    _atomic_json(plan_path, plan, indent=2, ensure_ascii=False)

    # summary
    sp = _load_json(os.path.join(out_dir, "speakers_global.json"))
    log.info("=" * 60)
    log.info("SUMMARY  duration=%.3fs  n_chunks=%d  backend=%s%s  total=%.1fs",
             plan.get("duration", 0.0), len(plan.get("chunks", [])),
             sp["backend"], " (DEGRADED)" if sp.get("degraded") else "", time.time() - t_total)
    for s in sp["speakers"]:
        log.info("   speaker %-3s %-8s p_female=%.3f total=%.2fs n=%d",
                 s["id"], s["gender"], s["p_female"], s["total_s"], s["n_turns"])
    log.info("outputs: %s", out_dir)
    print("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
