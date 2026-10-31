import sys
import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pydantic import BaseModel, Field  # noqa: F401

from sunwai.server.schemas import FocusRequest, JobRequest, TextRequest  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
from sunwai.paths import DATA_DIR, PACKAGE_SRC, WEB_DIR  # noqa: E402

APP_DIR = DATA_DIR
DUB_ROOT = Path(os.environ.get("DUB_ROOT", "./jobs")).resolve()
PLANNER_MODULE = "sunwai.planning.planner"        # run as `python -m <module>` in a child process
PIPELINE_MODULE = "sunwai.pipeline.engine"
LANGUAGES_FILE = DATA_DIR / "languages.json"

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "3"))
PIPELINE_TIMEOUT = 3600     # seconds, TOTAL wall time per chunk subprocess (not per output line)
PLANNER_TIMEOUT = 3600
RESYNTH_TIMEOUT = float(os.environ.get("DUB_RESYNTH_TIMEOUT", "900"))
EDIT_GATE_TIMEOUT = float(os.environ.get("DUB_EDIT_GATE_TIMEOUT", "240"))

# Admission control. ON by default; DUB_ADMISSION=0 restores unlimited mode.
ADMISSION = os.environ.get("DUB_ADMISSION", "1") != "0"
ADM_MAX = int(os.environ.get("DUB_MAX_UNCACHED_JOBS", "1"))     # uncached jobs running at once; the rest are queued
MAX_TARGETS = int(os.environ.get("DUB_MAX_TARGETS", "3"))
MAX_START_S = float(os.environ.get("DUB_MAX_START_S", "1800"))
MAX_DUR_S = float(os.environ.get("DUB_MAX_DUR_S", "1800"))
MAX_VIDEO_S = float(os.environ.get("DUB_MAX_VIDEO_S", "1800"))  # whole-video jobs: checked right after planning
MAX_EDIT_CHARS = int(os.environ.get("DUB_MAX_EDIT_CHARS", "600"))
MAX_QUEUED_JOBS = int(os.environ.get("DUB_MAX_QUEUED_JOBS", "4"))   # new uncached jobs allowed to WAIT behind the running ones (demo ceiling)
MAX_WORK_S = float(os.environ.get("DUB_MAX_WORK_S", str(MAX_TARGETS * MAX_VIDEO_S)))   # per job: dubbed seconds x languages
FANOUT = max(1, int(os.environ.get("DUB_FANOUT", str(4 * MAX_CONCURRENT))))   # chunk x language tasks alive at once per job
CORS_ORIGINS = [o.strip() for o in os.environ.get("DUB_CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
DUB_TOKEN = os.environ.get("DUB_TOKEN", "")                     # optional shared secret for mutating /api/ calls; OFF when empty

# Only these may be passed to the CLI as --lang / --out.
LANG_RE = re.compile(r"^[a-z]{2,3}$")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")
SEG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
                 "youtu.be", "www.youtu.be", "music.youtube.com", "studio.youtube.com"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_id_of(url: str, start: float, dur: float) -> str:
    key = f"{url}|{float(start)}|{float(dur)}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:12]


def resolve_job(job_id: str) -> Optional[Path]:
    m = JOB_RE.match(job_id or "")
    if not m:
        return None
    p = (DUB_ROOT / m.group(0)).resolve()
    return p if p.parent == DUB_ROOT else None


def resolve_lang(job_dir: Path, lang: str) -> Optional[Path]:
    if not LANG_RE.match(lang or ""):
        return None
    p = (job_dir / lang).resolve()
    return p if p.parent == job_dir else None


def resolve_chunk_dir(job_dir: Path, lang: str, chunk_idx: int) -> Optional[Path]:
    if not LANG_RE.match(lang or ""):
        return None
    if chunk_idx < 0:
        return None
    p = (job_dir / lang / f"c{chunk_idx:04d}").resolve()
    # Ensure it stays within lang dir
    if p.parent != (job_dir / lang).resolve():
        return None
    return p


def valid_source(url: str) -> Optional[str]:
    """Return a safe local path string, or raise HTTPException(422)."""
    if not url or len(url) > 2048:
        raise HTTPException(422, "url is required")
    low = url.strip().lower()
    if low.startswith("file://"):
        return _uploads_path(low[len("file://"):])
    if not (low.startswith("http://") or low.startswith("https://")):
        # A bare filesystem path must live under uploads/.
        return _uploads_path(url)
    try:
        from urllib.parse import urlparse
        u = urlparse(url.strip())
    except Exception:
        raise HTTPException(422, "invalid url")
    if u.scheme not in ("http", "https") or not u.netloc:
        raise HTTPException(422, "invalid url")
    host = u.netloc.split(":")[0].lower()
    if host not in ALLOWED_HOSTS:
        raise HTTPException(422, "only youtube.com / youtu.be links are supported")
    return url.strip()


def _uploads_path(raw: str) -> str:
    p = Path(raw)
    if p.is_absolute():
        candidate = p.resolve()
    else:
        candidate = (DUB_ROOT / "uploads" / raw).resolve()
    uploads = (DUB_ROOT / "uploads").resolve()
    if not (candidate == uploads or uploads in candidate.parents):
        raise HTTPException(422, "file path must live under uploads/")
    if not candidate.is_file():
        raise HTTPException(422, f"file not found: {raw}")
    return str(candidate)


def stage_from_log(text: str) -> str:
    last = "starting"
    for line in text.splitlines():
        m = re.search(r"Stage\s+([A-Za-z0-9_\-]+)", line)
        if m:
            last = m.group(1)
    return last


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# --------------------------------------------------------------------------
# App / shared state
# --------------------------------------------------------------------------
app = FastAPI(title="Dub Pipeline API")


@app.middleware("http")
async def _token_guard(request: Request, call_next):
    """Optional shared secret on mutating endpoints (env DUB_TOKEN; OFF when unset)."""
    if DUB_TOKEN and request.method in ("POST", "PUT", "PATCH", "DELETE") and request.url.path.startswith("/api/"):
        got = request.headers.get("x-dub-token") or request.query_params.get("token") or ""
        if not hmac.compare_digest(got.encode("utf-8", "replace"), DUB_TOKEN.encode("utf-8")):
            return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
    return await call_next(request)


# added AFTER the token guard so CORS is the outermost layer (401s carry CORS headers too)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Atomic file helpers: readers never see a half-written JSON
# --------------------------------------------------------------------------
def atomic_write(path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}_{uuid.uuid4().hex[:6]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}_{uuid.uuid4().hex[:6]}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj, indent=2) -> None:
    atomic_write(path, json.dumps(obj, indent=indent, ensure_ascii=False))


# --------------------------------------------------------------------------
# Viewer-aware scheduling: the chunk the viewer is about to watch, in the language they are watching, runs first.
# Focus is also persisted to DUB_ROOT/focus.json for the lip-sync workers.
# --------------------------------------------------------------------------
FOCUS: Dict[str, tuple] = {}   # job_id -> (lang, chunk_idx, ts), reported by the player
FOCUS_TTL = 1800.0
FOCUS_FILE = DUB_ROOT / "focus.json"
_focus_gen = 0


def _priority(job_id: str, lang: str, idx: int, edit: bool = False):
    if edit:
        return (-1, 0, lang)                       # a user edit always jumps the queue
    f = FOCUS.get(job_id)
    if not f or time.time() - f[2] > FOCUS_TTL:
        return (0, idx, lang)                      # no (fresh) viewer: chunk-major order
    flang, fidx = f[0], f[1]
    ahead = idx - fidx
    return (0 if lang == flang else 1, ahead if ahead >= 0 else 1000 - ahead, lang)


class GateBusy(Exception):
    pass


class PriorityGate:
    """Semaphore whose free slot goes to the highest-priority waiter instead of the oldest.
    Bounded waiters; acquisition may time out; a cancelled/timed-out waiter never leaks a slot."""

    def __init__(self, n: int, max_waiters: int = 4000):
        self.free, self.waiters, self.max_waiters = n, [], max_waiters

    async def acquire(self, job_id: str, lang: str, idx: int, edit: bool = False, timeout: Optional[float] = None):
        if self.free > 0 and not self.waiters:
            self.free -= 1
            return
        if len(self.waiters) >= self.max_waiters:
            raise GateBusy("too many queued tasks")
        fut = asyncio.get_running_loop().create_future()
        w = (job_id, lang, idx, fut, edit)
        self.waiters.append(w)
        try:
            if timeout:
                await asyncio.wait_for(fut, timeout)
            else:
                await fut
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release()                      # slot was already handed to us: pass it on
            elif w in self.waiters:
                self.waiters.remove(w)
            raise

    def release(self):
        live = [w for w in self.waiters if not w[3].done()]
        self.waiters = live
        if live:
            best = min(live, key=lambda w: _priority(w[0], w[1], w[2], w[4]))
            self.waiters.remove(best)
            best[3].set_result(None)
        else:
            self.free += 1


GATE = PriorityGate(MAX_CONCURRENT)
locks: Dict[str, asyncio.Lock] = {}


LOCK_PRUNE_AT = 2000


def _prune_locks() -> None:
    for k in [k for k, v in locks.items() if not v.locked() and not getattr(v, "_waiters", None)]:
        locks.pop(k, None)


def get_lock(key: str) -> asyncio.Lock:
    """Callers always `async with get_lock(k)` straight away (no await between lookup and acquire), so dropping a
    free, waiter-less lock is safe; bounds memory on a long-lived public service."""
    lk = locks.get(key)
    if lk is None:
        if len(locks) >= LOCK_PRUNE_AT:
            _prune_locks()
        lk = asyncio.Lock()
        locks[key] = lk
    return lk


# --------------------------------------------------------------------------
# Job admission control: at most ADM_MAX uncached jobs run at once, the rest are QUEUED (not rejected)
# --------------------------------------------------------------------------
class JobAdmission:
    def __init__(self, n: int):
        self.n = max(1, n)
        self.running: List[str] = []
        self.queue: List[str] = []
        self._cv = None

    def _cond(self):
        if self._cv is None:
            self._cv = asyncio.Condition()
        return self._cv

    async def admit(self, jid: str):
        cv = self._cond()
        async with cv:
            self.queue.append(jid)
            try:
                await cv.wait_for(lambda: self.queue and self.queue[0] == jid and len(self.running) < self.n)
            except BaseException:
                if jid in self.queue:
                    self.queue.remove(jid)
                cv.notify_all()
                raise
            self.queue.pop(0)
            self.running.append(jid)

    async def release(self, jid: str):
        cv = self._cond()
        async with cv:
            if jid in self.running:
                self.running.remove(jid)
            if jid in self.queue:
                self.queue.remove(jid)
            cv.notify_all()

    def position(self, jid: str) -> Optional[int]:
        if jid in self.running:
            return 0
        if jid in self.queue:
            return self.queue.index(jid) + 1
        return None


ADM = JobAdmission(ADM_MAX if ADMISSION else 10 ** 6)
JOB_TASKS: Dict[str, "asyncio.Task"] = {}


# --------------------------------------------------------------------------
# Status file helpers (single event-loop writer per dir; atomic replace)
# --------------------------------------------------------------------------
def _status_path(out_dir: Path) -> Path:
    return out_dir / "status.json"


def _write_status(out_dir: Path, status: dict) -> None:
    status["updated"] = utcnow()
    atomic_write_json(_status_path(out_dir), status)


def status_set(out_dir: Path, **fields) -> dict:
    s = read_json(_status_path(out_dir))
    s.update(fields)
    _write_status(out_dir, s)
    return s


def bump_version(out_dir: Path, version: Optional[int]) -> int:
    v = version if isinstance(version, int) else read_json(_status_path(out_dir)).get("version", 0)
    v = int(v) + 1
    status_set(out_dir, version=v)
    return v


# --------------------------------------------------------------------------
# Subprocess runner (never shell=True; env inherited only).
# Own session -> the whole process group can be killed; total (not per-line) timeout; output goes to the log
# file (no unbounded in-memory buffer); every child is registered in DUB_ROOT/.procs so a restarted server can
# kill orphans of a dead predecessor.
# --------------------------------------------------------------------------
PROCS_DIR = DUB_ROOT / ".procs"


def _tail_since(path: Path, off: int, nbytes: int = 4000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(off, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _kill_pgrp(pgid: int, grace: float = 5.0) -> None:
    """SIGTERM the group, then SIGKILL whatever is left after `grace` seconds (blocking, short)."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    t0 = time.time()
    while time.time() - t0 < grace:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def sweep_orphans() -> int:
    """At startup nothing is our child yet: any pipeline/planner process working under DUB_ROOT belongs to a dead
    predecessor and would race the retry. Kill its process group. Returns number of groups killed."""
    killed = set()
    me = os.getpid()
    scripts = (PIPELINE_MODULE, PLANNER_MODULE)
    root = str(DUB_ROOT)
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        pids = []
    for pid in pids:
        if pid == me:
            continue
        cl = _proc_cmdline(pid)
        if cl and any(s in cl for s in scripts) and root in cl:
            try:
                pg = os.getpgid(pid)
            except OSError:
                continue
            if pg not in killed and pg != os.getpgid(0):
                print(f"[sweep] killing orphan pid={pid} pgid={pg}: {cl[:160]}", flush=True)
                _kill_pgrp(pg)
                killed.add(pg)
    try:
        for f in PROCS_DIR.glob("*.json"):
            f.unlink()
    except OSError:
        pass
    return len(killed)


async def run_script(module: str, args: list, cwd: Path, log_path: Path, extra_env: dict = None,
                     timeout: float = PIPELINE_TIMEOUT, kind: str = "job") -> (int, str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_SRC), env.get("PYTHONPATH", "")]))
    if extra_env:
        env.update(extra_env)
    try:
        logf = open(log_path, "ab")
    except OSError as e:
        return 1, f"cannot open log: {e!r}"
    off = logf.seek(0, 2)
    logf.write(f"\n[server {utcnow()}] start {kind}: {module} {' '.join(args)[:300]}\n".encode("utf-8", "replace"))
    logf.flush()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", module, *args,
            cwd=str(cwd), stdin=asyncio.subprocess.DEVNULL, stdout=logf, stderr=asyncio.subprocess.STDOUT,
            env=env, start_new_session=True)
    except FileNotFoundError:
        logf.close()
        return 127, "Python executable not found"
    except Exception as e:
        logf.close()
        return 1, f"spawn error: {e!r}"
    reg = PROCS_DIR / f"{proc.pid}.json"
    try:
        PROCS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(reg, {"pid": proc.pid, "pgid": proc.pid, "kind": kind, "script": module,
                                "cwd": str(cwd), "started": utcnow(), "server_pid": os.getpid()})
    except OSError:
        pass
    note = ""
    rc = None
    try:
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            note = f"\n[server] {kind} timed out after {timeout:.0f}s; process group killed\n"
            await asyncio.get_running_loop().run_in_executor(None, _kill_pgrp, proc.pid, 5.0)
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except Exception:
                pass
            rc = 124
    except BaseException:
        # server task cancelled / shutting down: never leave the child group behind
        await asyncio.get_running_loop().run_in_executor(None, _kill_pgrp, proc.pid, 2.0)
        raise
    finally:
        try:
            reg.unlink()
        except OSError:
            pass
        if note:
            try:
                logf.write(note.encode("utf-8"))
                logf.flush()
            except OSError:
                pass
        logf.close()
    # reap any stragglers the child left in its group (e.g. ffmpeg grandchildren holding GPU/files)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return (rc if rc is not None else 1), _tail_since(log_path, off, 6000) + note


# --------------------------------------------------------------------------
# Plan / chunk / job artefact checks (completion is derived from artifacts, not from marker files)
# --------------------------------------------------------------------------
def _plan_valid(job_dir: Path, need_source: bool = True) -> bool:
    plan = read_json(job_dir / "plan.json")
    ch = plan.get("chunks") if isinstance(plan, dict) else None
    if not isinstance(ch, list) or not ch:
        return False
    try:
        for c in ch:
            int(c["idx"])
            if not (math.isfinite(float(c["start"])) and float(c["start"]) >= 0
                    and math.isfinite(float(c["dur"])) and float(c["dur"]) > 0):
                return False
    except Exception:
        return False
    if need_source:
        try:
            if (job_dir / "source.mp4").stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _chunk_done(out_dir: Optional[Path], lang: str) -> bool:
    if not out_dir or not out_dir.is_dir():
        return False
    if read_json(_status_path(out_dir)).get("state") != "done":
        return False
    try:
        return (out_dir / f"dub_{lang}.mp4").stat().st_size > 0
    except OSError:
        return False


def _missing_pairs(job_dir: Path, langs: List[str], plan: dict) -> list:
    out = []
    for chunk in plan.get("chunks", []):
        for lang in langs:
            if not _chunk_done(resolve_chunk_dir(job_dir, lang, chunk["idx"]), lang):
                out.append((lang, chunk))
    return out


def _source_file(job_dir: Path, langs: List[str], idx: int) -> Optional[Path]:
    """The ONE per-chunk original-video file: first non-empty src.mp4 (any language) that has been stable for 3 s.
    Used by both readiness (has_source) and the media route so they can never disagree."""
    for lg in langs:
        d = resolve_chunk_dir(job_dir, lg, idx)
        try:
            if d:
                st = (d / "src.mp4").stat()
                if st.st_size > 0 and time.time() - st.st_mtime > 3:
                    return d / "src.mp4"
        except OSError:
            pass
    return None


def _job_meta(job_dir: Path) -> dict:
    m = read_json(job_dir / "job.json")
    return m if isinstance(m, dict) and m.get("langs") else {}


# --------------------------------------------------------------------------
# Background workers
# --------------------------------------------------------------------------
# planner products that a killed planner may leave PARTIAL (the planner reuses any existing file: crash -> corrupt reuse)
def _plan_sig(plan) -> Optional[str]:
    """Fingerprint of a plan's chunk layout; None when there is no usable plan."""
    try:
        ch = plan["chunks"]
        if not isinstance(ch, list) or not ch:
            return None
        return json.dumps([[int(c["idx"]), float(c["start"]), float(c["dur"])] for c in ch])
    except Exception:
        return None


def _purge_chunk_dirs(job_dir: Path, langs: List[str]) -> int:
    """Remove every chunk dir under the job's language dirs. Returns the number of dirs that could NOT be removed."""
    failed = 0
    for lg in langs:
        if not LANG_RE.match(lg):
            continue
        for d in (job_dir / lg).glob("c[0-9]*"):
            if d.is_symlink() or not d.is_dir():
                continue
            shutil.rmtree(d, ignore_errors=True)
            if d.exists():
                failed += 1
    return failed


def _read_sig(job_dir: Path) -> Optional[str]:
    try:
        return (job_dir / "plan.sig").read_text() or None
    except OSError:
        return None


def _write_sig(job_dir: Path, sig: Optional[str]) -> bool:
    """Persist plan.sig; a failure only counts as harmless when the file already holds exactly this value."""
    try:
        atomic_write(job_dir / "plan.sig", sig or "")
        return True
    except OSError:
        return _read_sig(job_dir) == (sig or None)


async def _apply_plan(job_dir: Path, meta: dict, old_sig: Optional[str]) -> bool:
    """Make the chunk dirs consistent with plan.json. plan.sig = the layout the chunk dirs were built for; it is
    committed ONLY after a purge succeeded and MUST be persisted before any chunk starts. The purge runs in a worker
    thread (the event loop stays responsive) and a cancelled job WAITS for it, so it can never outlive the attempt."""
    new_sig = _plan_sig(read_json(job_dir / "plan.json"))
    if new_sig != old_sig:
        fut = asyncio.get_running_loop().run_in_executor(None, _purge_chunk_dirs, job_dir, list(meta["langs"]))
        cancelled = False
        while not fut.done():
            try:
                await asyncio.wait({fut})
            except asyncio.CancelledError:
                cancelled = True                      # keep waiting for the purge, re-raise afterwards
        if cancelled:
            raise asyncio.CancelledError()
        failed = fut.result()
        if failed:
            try:
                (job_dir / "plan.json").unlink()      # force a re-plan + re-purge next time; do not commit the new signature
            except OSError:
                pass
            status_set(job_dir, state="error", stage="planning", finished=utcnow(),
                       error_tail=f"cannot remove {failed} chunk dirs of the previous plan; please retry")
            return False
        print(f"[replan] {job_dir.name}: plan layout changed, stale chunk dirs purged", flush=True)
    if not _write_sig(job_dir, new_sig):
        status_set(job_dir, state="error", stage="planning", finished=utcnow(),
                   error_tail="cannot persist the plan signature (plan.sig); please retry")
        return False
    return True


_PLANNER_DERIVED = ("source.mp4", "source16k.wav", "vad.json", "plan.json", "speakers_global.json",
                    "spkturns.json", "spkgender.json", "spkbackend.json", "lid.json")


def _planner_purge_names(job_dir: Path):
    """the recovery purge must keep a published speakers_global.json that carries an immutable voice_plan (or any
    rollout=1 job that already has a map): deleting it would let the rerun publish a DIFFERENT plan (old- and new-voice chunks)."""
    keep = False
    try:
        sg_p = job_dir / "speakers_global.json"
        if sg_p.exists():
            m = read_json(job_dir / "voice_rollout.json")
            sg = read_json(sg_p)
            keep = (isinstance(sg, dict) and isinstance(sg.get("voice_plan"), dict)) or (isinstance(m, dict) and m.get("voice_plan") == 1)
    except Exception:
        keep = (job_dir / "voice_rollout.json").exists() and (job_dir / "speakers_global.json").exists()   # unreadable: keep, never delete
    return [nm for nm in _PLANNER_DERIVED if not (keep and nm == "speakers_global.json")]


async def _run_planner(job_dir: Path, meta: dict) -> bool:
    aid = uuid.uuid4().hex[:8]
    old_sig = _read_sig(job_dir) or _plan_sig(read_json(job_dir / "plan.json"))   # plan.sig = last APPLIED layout, wins over a plan.json that may be newer
    for nm in _planner_purge_names(job_dir):          # never trust derived files of an unfinished plan
        try:
            (job_dir / nm).unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            status_set(job_dir, state="error", stage="planning", finished=utcnow(),
                       error_tail=f"cannot remove stale {nm}: {e!r}")
            return False
    status_set(job_dir, state="planning", stage="planning", started=utcnow(), finished=None,
               error_tail=None, attempt=aid)
    args = ["--url", meta["url"], "--out", str(job_dir), "--start", str(meta["start"]), "--dur", str(meta["dur"])]
    rc, tail = await run_script(PLANNER_MODULE, args, DUB_ROOT, job_dir / "planner.log",
                                timeout=PLANNER_TIMEOUT, kind="planner")
    if rc == 0 and _plan_valid(job_dir):
        if not await _apply_plan(job_dir, meta, old_sig):
            return False
        status_set(job_dir, state="planning", stage="planned", error_tail=None)
        return True
    why = "Planner exited 0 but plan.json/source.mp4 missing or invalid" if rc == 0 else (tail[-2000:] or f"exit {rc}")
    status_set(job_dir, state="error", stage="planning", finished=utcnow(), error_tail=why)
    return False


async def worker_chunk(job_dir: Path, url: str, lang: str, chunk: dict):
    chunk_idx = chunk["idx"]
    out_dir = resolve_chunk_dir(job_dir, lang, chunk_idx)
    if not out_dir:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if _chunk_done(out_dir, lang):
        return
    aid = uuid.uuid4().hex[:8]                      # attempt id: only the owning attempt may publish a final state
    prev = read_json(_status_path(out_dir))
    status_set(out_dir, state="queued", stage="queued", started=None, finished=None, error_tail=None,
               version=int(prev.get("version", 0) or 0), attempt=aid, pid=None, editing=False)
    args = ["--url", url, "--lang", lang, "--out", str(out_dir),
            "--start", str(chunk["start"]), "--dur", str(chunk["dur"])]
    env_extra = {"SPEAKER_MAP": str(job_dir / "speakers_global.json"), "CHUNK_START": str(chunk["start"])}

    await GATE.acquire(job_dir.name, lang, chunk_idx)
    try:
        status_set(out_dir, state="running", stage="running", started=utcnow())
        try:
            rc, tail = await run_script(PIPELINE_MODULE, args, DUB_ROOT, out_dir / "pipeline.log",
                                        extra_env=env_extra, timeout=PIPELINE_TIMEOUT, kind="chunk")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            rc, tail = 1, f"spawn error: {e!r}"
    finally:
        GATE.release()

    if read_json(_status_path(out_dir)).get("attempt") != aid:
        return                                       # superseded by a newer attempt: do not clobber it
    stage = stage_from_log(tail)
    mp4 = out_dir / f"dub_{lang}.mp4"
    if rc == 0 and mp4.is_file() and mp4.stat().st_size > 0:
        status_set(out_dir, state="done", stage=stage or "done", finished=utcnow(), error_tail=None,
                   version=int(read_json(_status_path(out_dir)).get("version", 0) or 0) + 1)
    elif rc == 0:
        status_set(out_dir, state="error", stage=stage or "done", finished=utcnow(),
                   error_tail="exited 0 but dub mp4 was not produced or empty\n" + (tail[-2000:] or ""))
    else:
        status_set(out_dir, state="error", stage=stage or "error", finished=utcnow(),
                   error_tail=(tail[-2000:] or f"exit {rc}"))


async def enqueue_chunks(job_id: str, job_dir: Path, pairs: list):
    """Run every missing (lang, chunk) with at most FANOUT tasks alive at once (bounded memory/queue). Whenever a slot
    frees, the next task is the one with the best CURRENT priority (viewer focus first); the shared PriorityGate
    still decides which alive task gets a pipeline slot."""
    source_file = job_dir / "source.mp4"
    if not source_file.exists():
        return
    pending = sorted(pairs, key=lambda p: (p[1]["idx"], p[0]))      # chunk-major: ties resolve in playback order
    alive: set = set()
    try:
        while pending or alive:
            while pending and len(alive) < FANOUT:
                best = min(pending, key=lambda p: _priority(job_id, p[0], p[1]["idx"]))
                pending.remove(best)
                alive.add(asyncio.ensure_future(worker_chunk(job_dir, str(source_file), best[0], best[1])))
            done, _ = await asyncio.wait(alive, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                alive.discard(t)
                if not t.cancelled() and t.exception() is not None:
                    print(f"[enqueue_chunks] {job_id}: task error {t.exception()!r}", flush=True)
    except BaseException:
        for t in alive:
            t.cancel()
        await asyncio.gather(*alive, return_exceptions=True)
        raise


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Helper: Load Languages Config
# --------------------------------------------------------------------------
def get_languages_config() -> dict:
    if LANGUAGES_FILE.exists():
        return read_json(LANGUAGES_FILE)
    # Fallback if missing
    return {"targets": [], "sources": []}

def get_target_codes() -> set:
    cfg = get_languages_config()
    return {t["code"] for t in cfg.get("targets", [])}


# --------------------------------------------------------------------------
# Job orchestration
# --------------------------------------------------------------------------
async def run_job(jid: str):
    """One task per job: wait for an admission slot, (re)plan if needed, dub every missing chunk x lang of the
    CURRENT job.json languages (re-read after each pass, so languages added meanwhile are picked up), derive the
    final state from artifacts. Idempotent: safe to call again for the same job after any crash."""
    job_dir = DUB_ROOT / jid
    admitted = False
    try:
        await ADM.admit(jid)
        admitted = True
        for _pass in range(12):
            meta = _job_meta(job_dir)
            if not meta:
                status_set(job_dir, state="error", stage="error", finished=utcnow(), error_tail="job.json missing")
                return
            langs = list(meta["langs"])
            if not _plan_valid(job_dir):
                if not await _run_planner(job_dir, meta):
                    return
            applied = _read_sig(job_dir)
            if applied is None:
                if not _write_sig(job_dir, _plan_sig(read_json(job_dir / "plan.json"))):          # legacy job: adopt
                    status_set(job_dir, state="error", stage="planning", finished=utcnow(),
                               error_tail="cannot persist the plan signature (plan.sig); please retry")
                    return
            elif applied != _plan_sig(read_json(job_dir / "plan.json")):    # crash between plan.json and purge: finish the purge now
                if not await _apply_plan(job_dir, meta, applied):
                    return
            plan = read_json(job_dir / "plan.json")
            try:
                ends = [float(c["start"]) + float(c["dur"]) for c in plan["chunks"]]
                plan_len = max(ends) if all(math.isfinite(e) and e >= 0 for e in ends) else float("inf")
            except Exception:
                plan_len = float("inf")
            if ADMISSION and math.isfinite(plan_len) and plan_len <= MAX_VIDEO_S and plan_len * len(langs) > MAX_WORK_S:
                status_set(job_dir, state="error", stage="planning", finished=utcnow(),
                           error_tail=f"this demo dubs at most {MAX_WORK_S:.0f} s of video x languages per job; "
                                      f"{plan_len:.0f} s x {len(langs)} languages is too much. Pick a shorter range or fewer languages.")
                return
            if not math.isfinite(plan_len) or (ADMISSION and plan_len > MAX_VIDEO_S):
                status_set(job_dir, state="error", stage="planning", finished=utcnow(),
                           error_tail=f"video is {plan_len:.0f}s; the limit is {MAX_VIDEO_S:.0f}s. "
                                      f"Pick a start/end range.")
                return
            pairs = _missing_pairs(job_dir, langs, plan)
            if pairs:
                status_set(job_dir, state="running", stage="dubbing", finished=None, error_tail=None)
                await enqueue_chunks(jid, job_dir, pairs)
            meta2 = _job_meta(job_dir)
            if meta2 and set(meta2["langs"]) != set(langs):
                continue                              # languages were added while we worked
            break
        meta = _job_meta(job_dir)
        plan = read_json(job_dir / "plan.json")
        left = _missing_pairs(job_dir, meta.get("langs", []), plan)
        if not left:
            status_set(job_dir, state="done", stage="done", finished=utcnow(), error_tail=None)
        else:
            errs = []
            for lang, chunk in left:
                od = resolve_chunk_dir(job_dir, lang, chunk["idx"])
                s = read_json(_status_path(od)) if od else {}
                if s.get("state") == "error":
                    errs.append(f"{lang}/c{chunk['idx']:04d}")
            hint = await _down_hint()
            status_set(job_dir, state="error", stage="dubbing", finished=utcnow(),
                       error_tail=(("chunks failed: " + ", ".join(errs[:12])) if errs else "Incomplete state") + hint)
    except asyncio.CancelledError:
        raise                                         # shutdown: state stays resumable, reconcile picks it up
    except Exception as e:
        traceback.print_exc()
        try:
            status_set(job_dir, state="error", stage="error", finished=utcnow(), error_tail=f"server error: {e!r}"[:1500])
        except Exception:
            pass
    finally:
        JOB_TASKS.pop(jid, None)
        await ADM.release(jid)


def _job_needs_work(job_dir: Path) -> bool:
    meta = _job_meta(job_dir)
    if not meta:
        return False
    if not _plan_valid(job_dir, need_source=False):
        return True
    plan = read_json(job_dir / "plan.json")
    applied = _read_sig(job_dir)
    if applied and applied != _plan_sig(plan):          # plan.json changed but the purge was never applied (crash window)
        return True
    return bool(_missing_pairs(job_dir, meta["langs"], plan))


def ensure_job_running(jid: str) -> str:
    """Start (or keep) the single run_job task for this job. Returns 'running' | 'cached'."""
    t = JOB_TASKS.get(jid)
    if t is not None and not t.done():
        return "running"
    job_dir = DUB_ROOT / jid
    if not _job_needs_work(job_dir):
        s = read_json(job_dir / "status.json")
        if s.get("state") != "done":
            status_set(job_dir, state="done", stage="done", finished=utcnow(), error_tail=None)
        return "cached"
    status_set(job_dir, state="queued", stage="queued", queued_at=utcnow(), finished=None, error_tail=None)
    JOB_TASKS[jid] = asyncio.create_task(run_job(jid))
    return "running"


def _restore_interrupted_edits(job_dir: Path) -> int:
    """An edit killed mid-flight left segments.json.pre_edit: put the pre-edit segments back."""
    n = 0
    for pre in job_dir.glob("*/c*/segments.json.pre_edit"):
        try:
            os.replace(pre, pre.with_name("segments.json"))
            status_set(pre.parent, editing=False)
            n += 1
        except OSError:
            pass
    return n


async def _reconcile():
    """Startup: finish what a dead predecessor left half-done. Only jobs that were IN FLIGHT (queued/planning/running,
    or with a queued/running chunk) are resumed; merely errored old jobs wait for an explicit re-POST."""
    if os.environ.get("DUB_RECONCILE", "1") != "1":
        return
    cand = []
    try:
        dirs = sorted(p for p in DUB_ROOT.iterdir() if p.is_dir() and JOB_RE.match(p.name))
    except OSError:
        return
    for jd in dirs:
        try:
            if not (jd / "job.json").is_file():
                continue
            restored = _restore_interrupted_edits(jd)
            st = read_json(jd / "status.json")
            inflight = (not st) or st.get("state") in ("queued", "planning", "running")   # no status.json = died right after job.json
            if not inflight:
                for sp in jd.glob("*/c*/status.json"):
                    if read_json(sp).get("state") in ("queued", "running"):
                        inflight = True
                        break
            if inflight:
                cand.append((st.get("queued_at") or st.get("started") or "", jd.name))
            print(f"[reconcile] {jd.name} state={st.get('state')} inflight={inflight} restored_edits={restored}", flush=True)
        except Exception as e:
            print(f"[reconcile] {jd.name} error {e!r}", flush=True)
    for _, jid in sorted(cand):
        try:
            print(f"[reconcile] resuming {jid}: {ensure_job_running(jid)}", flush=True)
        except Exception as e:
            print(f"[reconcile] resume {jid} failed {e!r}", flush=True)


@app.on_event("startup")
async def _startup():
    DUB_ROOT.mkdir(parents=True, exist_ok=True)
    n = await asyncio.get_running_loop().run_in_executor(None, sweep_orphans)
    print(f"[startup] pid={os.getpid()} root={DUB_ROOT} admission={ADMISSION} max_jobs={ADM_MAX} orphans_killed={n}", flush=True)
    global _focus_gen
    try:
        fd = read_json(FOCUS_FILE)
        _focus_gen = int(fd.get("gen", 0))
        if time.time() - float(fd.get("ts", 0)) > 900:
            FOCUS_FILE.unlink()
    except Exception:
        pass
    asyncio.create_task(_reconcile())


@app.on_event("shutdown")
async def _shutdown():
    for t in list(JOB_TASKS.values()):
        t.cancel()                                    # run_script's cancel path kills the child process groups
    await asyncio.gather(*JOB_TASKS.values(), return_exceptions=True)


# --------------------------------------------------------------------------
# Job endpoints
# --------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_job(body: JobRequest):
    url = valid_source(body.url)

    langs = list(dict.fromkeys(l.strip().lower() for l in body.langs))
    langs = [l for l in langs if LANG_RE.match(l)]
    if not langs:
        raise HTTPException(422, "no valid languages")
    allowed_codes = get_target_codes()
    if allowed_codes:
        invalid = [l for l in langs if l not in allowed_codes]
        if invalid:
            raise HTTPException(422, f"Unsupported languages: {invalid}")
    if ADMISSION and len(langs) > MAX_TARGETS:
        raise HTTPException(422, f"at most {MAX_TARGETS} target languages per request")

    try:
        start = float(body.start)
        dur = float(body.dur)
    except (TypeError, ValueError):
        raise HTTPException(422, "start/dur must be numbers")
    if not (math.isfinite(start) and math.isfinite(dur)):
        raise HTTPException(422, "start/dur must be finite numbers")
    if start < 0:
        raise HTTPException(422, "start must be >= 0")
    if dur < 0:
        raise HTTPException(422, "dur must be >= 0")
    if start > MAX_START_S:
        raise HTTPException(422, f"start must be at most {MAX_START_S:.0f} s")
    if dur > MAX_DUR_S:
        raise HTTPException(422, f"dur must be at most {MAX_DUR_S:.0f} s")

    jid = job_id_of(url, start, dur)
    job_dir = DUB_ROOT / jid
    async with get_lock(f"job:{jid}"):
        old = _job_meta(job_dir)
        merged = list(dict.fromkeys(list(old.get("langs", [])) + langs))   # new languages MERGE into the job
        if ADMISSION and len(merged) > max(MAX_TARGETS, len(old.get("langs", []))):
            raise HTTPException(422, f"a job may have at most {MAX_TARGETS} target languages "
                                     f"(it already has {old.get('langs')})")
        if ADMISSION and dur > 0 and dur * len(merged) > MAX_WORK_S and dur * len(merged) > dur * len(old.get("langs", [])):
            raise HTTPException(422, f"too much work: {dur:.0f} s x {len(merged)} languages; this demo allows at most "
                                     f"{MAX_WORK_S:.0f} s of video x languages per job. Pick a shorter range or fewer languages.")
        busy = len(ADM.running) + len(ADM.queue)
        newwork = (not old) or bool(set(langs) - set(old.get("langs", []))) or _job_needs_work(job_dir)
        if ADMISSION and newwork and jid not in JOB_TASKS and busy >= ADM_MAX + MAX_QUEUED_JOBS:
            raise HTTPException(429, f"The demo is busy: {busy} dubbing jobs are already running or waiting "
                                     f"(limit {ADM_MAX + MAX_QUEUED_JOBS}). Please try again in a few minutes.",
                                headers={"Retry-After": "60"})
        _fresh = old == {} and (not job_dir.exists() or not any(job_dir.iterdir()))
        job_dir.mkdir(parents=True, exist_ok=True)
        # the voice-plan rollout is decided HERE, before any language dir exists, for NEW jobs only; the planner trusts it.
        if _fresh and not (job_dir / "voice_rollout.json").exists() and \
                all(os.environ.get(k, "").strip().lower() not in ("0", "false", "no", "off", "n", "f") for k in ("VOICE_PLAN", "VOICE_XLING")):
            atomic_write_json(job_dir / "voice_rollout.json", {"voice_plan": 1, "decided": utcnow(), "by": "create_job"})

        _given = "lipsync" in (getattr(body, "model_fields_set", None) if getattr(body, "model_fields_set", None) is not None else getattr(body, "__fields_set__", set()))
        lip = bool(body.lipsync) if _given else (old.get("lipsync", True) is not False)     # absent -> keep the job's setting, new jobs default ON
        meta = {"job_id": jid, "url": url, "langs": merged, "start": start, "dur": dur, "lipsync": lip,
                "created": old.get("created") or utcnow(), "updated": utcnow()}
        atomic_write_json(job_dir / "job.json", meta)
        for lang in merged:
            (job_dir / lang).mkdir(parents=True, exist_ok=True)
        mode = ensure_job_running(jid)
    await asyncio.sleep(0.05)   # let run_job enter the admission queue so the position is real
    st = read_json(job_dir / "status.json")
    return {"job_id": jid, "langs": merged, "mode": mode, "state": st.get("state"),
            "queue_position": ADM.position(jid), "queue_len": len(ADM.queue)}



def _chunk_sync(out_dir: Path, lang: str) -> dict:
    """Per-chunk sync-quality metrics from segments.json (+ lipsync file freshness)."""
    segs = _load_segments(out_dir)
    n = len(segs)
    ovf = [x for x in segs if x.get("overflow")]
    def _f(v, d=0.0):
        try:
            return float(v)
        except Exception:
            return d
    cut = sum(_f(x.get("overflow_s")) for x in segs)
    refit = sum(1 for x in segs if x.get("refit"))
    tempos = [_f(x.get("tempo"), 1.0) for x in segs if x.get("tempo") is not None]
    dub_p = out_dir / f"dub_{lang}.mp4"
    ls_p = out_dir / f"dub_{lang}_lipsync.mp4"
    ls = False
    try:
        if ls_p.exists() and ls_p.stat().st_size > 0:
            ls = (not dub_p.exists()) or ls_p.stat().st_mtime > dub_p.stat().st_mtime
    except OSError:
        ls = False
    return {
        "segments": n,
        "overflowed": len(ovf),
        "cut_s": round(cut, 2),
        "refit": refit,
        "tempo_mean": round(sum(tempos) / len(tempos), 3) if tempos else None,
        "tempo_max": round(max(tempos), 3) if tempos else None,
        "tempo_n": len(tempos),
        "lipsync": ls,
    }


def _lang_sync(chunk_syncs: list, done: int) -> dict:
    """Aggregate over chunks (only chunks that have segments)."""
    have = [c for c in chunk_syncs if c]
    nseg = sum(c["segments"] for c in have)
    tw = [(c["tempo_mean"], c.get("tempo_n", c["segments"])) for c in have if c["tempo_mean"] is not None]
    tmax = [c["tempo_max"] for c in have if c["tempo_max"] is not None]
    return {
        "chunks_measured": len(have),
        "segments": nseg,
        "overflowed": sum(c["overflowed"] for c in have),
        "cut_s": round(sum(c["cut_s"] for c in have), 2),
        "refit": sum(c["refit"] for c in have),
        "tempo_mean": round(sum(m * w for m, w in tw) / max(1, sum(w for _, w in tw)), 3) if tw else None,
        "tempo_max": max(tmax) if tmax else None,
        "lipsync_chunks": sum(1 for c in have if c["lipsync"]),
        "chunks_done": done,
        "lipsync_coverage": round(sum(1 for c in have if c["lipsync"]) / done, 3) if done else 0.0,
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job_dir = resolve_job(job_id)
    if job_dir is None:
        raise HTTPException(404, "unknown job")

    job_meta = read_json(job_dir / "job.json")
    if not job_meta:
        raise HTTPException(404, "unknown job")

    job_status = read_json(job_dir / "status.json")
    state = job_status.get("state", "unknown")
    qpos = ADM.position(job_id)
    if qpos is not None and qpos > 0:
        state = "queued"                      # waiting for an admission slot

    plan = read_json(job_dir / "plan.json")
    duration = plan.get("duration", 0)
    chunks_plan = plan.get("chunks", [])

    langs_info = {}
    target_langs = job_meta.get("langs", [])

    for lang in target_langs:
        done_count = 0
        total_count = 0
        chunk_states = []
        chunk_syncs = []

        for c in chunks_plan:
            idx = c["idx"]
            out_dir = resolve_chunk_dir(job_dir, lang, idx)
            if out_dir and out_dir.exists():
                s = read_json(_status_path(out_dir))
                st = s.get("state", "unknown")
                has_media = (out_dir / f"dub_{lang}.mp4").exists()
                csync = None
                if st == "done":
                    try:
                        csync = _chunk_sync(out_dir, lang)
                    except Exception:
                        csync = None
                chunk_syncs.append(csync)
                chunk_states.append({
                    "idx": idx,
                    "state": st,
                    "stage": s.get("stage", ""),
                    "version": s.get("version", 0),
                    "has_media": has_media,
                    "sync": csync,
                    "started": s.get("started"),      # ISO UTC; used by the player's adaptive start-buffer ETA
                    "finished": s.get("finished"),
                    "editing": bool(s.get("editing")),
                })
                if st == "done":
                    done_count += 1
            else:
                chunk_states.append({
                    "idx": idx,
                    "state": "queued", # or missing
                    "stage": "",
                    "version": 0,
                    "has_media": False
                })
            total_count += 1

        langs_info[lang] = {
            "done": done_count,
            "total": total_count,
            "chunks": chunk_states,
            "sync": _lang_sync(chunk_syncs, done_count)
        }

    speakers = plan.get("speakers", [])
    # If speakers are in separate file
    spk_file = job_dir / "speakers_global.json"
    if spk_file.exists():
        spk_data = read_json(spk_file)
        if "speakers" in spk_data:
            speakers = spk_data["speakers"]

    def _has_src(idx):
        return _source_file(job_dir, target_langs, idx) is not None

    return {
        "job_id": job_id,
        "url": job_meta.get("url"),
        "state": state,
        "queue_position": qpos,
        "queue_len": len(ADM.queue),
        "error_tail": (job_status.get("error_tail") if state == "error" else None),
        "services_down": (await _down_hint()).strip(" ()") if state in ("running", "queued", "planning", "error") else "",
        "duration": duration,
        "lipsync": job_meta.get("lipsync", True) is not False,
        "chunks": [dict(c, has_source=_has_src(c["idx"])) for c in chunks_plan],
        "speakers": speakers,
        "langs": langs_info,
        "now": utcnow(),
    }


@app.get("/api/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Sync-quality metrics per language and per chunk (same numbers as in GET /api/jobs/{id})."""
    data = await get_job(job_id)
    out = {}
    for lang, info in data["langs"].items():
        out[lang] = {"total": info["sync"],
                     "chunks": {c["idx"]: c["sync"] for c in info["chunks"] if c.get("sync")}}
    return {"job_id": job_id, "metrics": out,
            "null": "cut_s=0 and overflowed=0 means nothing was truncated; tempo 1.0 = no speed-up"}


# --------------------------------------------------------------------------
# Segment Endpoints
# --------------------------------------------------------------------------

def _load_segments(chunk_dir: Path) -> List[dict]:
    # Assume dub_pipeline writes segments.json in the chunk output dir
    path = chunk_dir / "segments.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "segments" in data:
            return data["segments"]
        if isinstance(data, list):
            return data
        return []
    except:
        return []

def _save_segments(chunk_dir: Path, segs: List[dict]):
    atomic_write_json(chunk_dir / "segments.json", segs)

def _find_segment(segs: list, seg_id: str):
    for i, s in enumerate(segs):
        if str(s.get("id")) == seg_id:
            return i, s
    return -1, None


@app.get("/api/jobs/{job_id}/{lang}/segments")
async def get_segments(job_id: str, lang: str):
    job_dir = resolve_job(job_id)
    if not job_dir:
        raise HTTPException(404, "unknown job")

    # Validate lang
    if not LANG_RE.match(lang):
        raise HTTPException(404, "unknown lang")

    job_meta = read_json(job_dir / "job.json")
    if not job_meta or lang not in job_meta.get("langs", []):
        raise HTTPException(404, "unknown job/lang")

    plan = read_json(job_dir / "plan.json")
    chunks = plan.get("chunks", [])

    all_segments = []

    for c in chunks:
        idx = c["idx"]
        chunk_start = c["start"]
        out_dir = resolve_chunk_dir(job_dir, lang, idx)

        if not out_dir or not out_dir.exists():
            continue

        s_status = read_json(_status_path(out_dir))
        if s_status.get("state") != "done":
            continue

        segs = _load_segments(out_dir)
        for s in segs:
            # Calculate global times
            loc_start = s.get("start", 0)
            loc_end = s.get("end", 0)

            global_seg = {
                "chunk": idx,
                "id": s.get("id"),
                "start": chunk_start + loc_start,
                "end": chunk_start + loc_end,
                "speaker": s.get("speaker"),
                "gender": s.get("gender"),
                "src_text": s.get("src_text"),
                "mt_text": s.get("mt_text"),
                "tgt_text": s.get("tgt_text"),
                "edited": s.get("edited", False),
                "fallback": bool(s.get("fallback", False)),
                "fallback_reason": s.get("fallback_reason"),
                "history_len": len(s.get("history", [])) if isinstance(s.get("history"), list) else 0
            }
            all_segments.append(global_seg)

    # Sort by start time
    all_segments.sort(key=lambda x: x["start"])

    return {"segments": all_segments}


async def _resynth_segment(job_dir: Path, url: str, lang: str, chunk_idx: int, chunk_data: dict, seg_id: str, text: str) -> None:
    out_dir = resolve_chunk_dir(job_dir, lang, chunk_idx)
    if not out_dir:
        raise HTTPException(404, "chunk dir not found")
    args = ["--url", url, "--lang", lang, "--out", str(out_dir),
            "--resynth", seg_id, "--text", text]
    env_extra = {
        "SPEAKER_MAP": str(job_dir / "speakers_global.json"),
        "CHUNK_START": str(chunk_data["start"])
    }
    rc, logs = await run_script(PIPELINE_MODULE, args, DUB_ROOT, out_dir / "pipeline.log",
                                extra_env=env_extra, timeout=RESYNTH_TIMEOUT, kind="resynth")
    if rc != 0:
        raise HTTPException(500, f"resynth failed (exit {rc}): {logs[-500:]}")


def _edit_prelude(job_id: str, lang: str, c: int, seg_id: str):
    job_dir = resolve_job(job_id)
    if not job_dir:
        raise HTTPException(404, "unknown job")
    if not LANG_RE.match(lang):
        raise HTTPException(404, "unknown lang")
    if not SEG_RE.match(seg_id):
        raise HTTPException(400, "invalid seg id")
    job_meta = read_json(job_dir / "job.json")
    if not job_meta or lang not in job_meta.get("langs", []):
        raise HTTPException(404, "unknown job/lang")
    plan = read_json(job_dir / "plan.json")
    chunk_data = next((ch for ch in plan.get("chunks", []) if ch["idx"] == c), None)
    if not chunk_data:
        raise HTTPException(404, "chunk not found in plan")
    out_dir = resolve_chunk_dir(job_dir, lang, c)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "chunk dir not found")
    return job_dir, job_meta, chunk_data, out_dir


async def _edit_flow(job_id: str, lang: str, c: int, seg_id: str, make_change, after=None):
    """Shared edit/undo path. Under the chunk lock: take the PriorityGate (edit priority), snapshot segments.json
    byte-for-byte, apply the change, re-synthesise; on ANY failure restore the snapshot (atomic) so text, history and
    the 'edited' flag are exactly as before. The published mp4 stays the previous version until the pipeline swaps it."""
    job_dir, job_meta, chunk_data, out_dir = _edit_prelude(job_id, lang, c, seg_id)
    async with get_lock(f"{job_id}:{lang}:{c}"):
        s_status = read_json(_status_path(out_dir))
        if s_status.get("state") != "done":
            raise HTTPException(409, f"chunk not done (state={s_status.get('state')})")
        segs_path = out_dir / "segments.json"
        try:
            raw_before = segs_path.read_bytes()
        except OSError:
            raise HTTPException(404, "segments not found")
        segs = _load_segments(out_dir)
        _i, seg = _find_segment(segs, seg_id)
        if seg is None:
            raise HTTPException(404, "segment not found")
        new_text = make_change(seg)                   # may raise 409 before anything is touched
        try:
            await GATE.acquire(job_id, lang, c, edit=True, timeout=EDIT_GATE_TIMEOUT)
        except (asyncio.TimeoutError, GateBusy):
            raise HTTPException(503, "the server is busy dubbing other parts; please retry the edit in a minute")
        try:
            try:
                atomic_write_bytes(out_dir / "segments.json.pre_edit", raw_before)
                status_set(out_dir, editing=True)
                _save_segments(out_dir, segs)
                source_file = job_dir / "source.mp4"
                url_arg = str(source_file) if source_file.exists() else job_meta.get("url", "")
                await _resynth_segment(job_dir, url_arg, lang, c, chunk_data, seg_id, new_text)
                try:                                  # media is published: the edit is COMMITTED, drop the rollback snapshot now
                    (out_dir / "segments.json.pre_edit").unlink()
                except OSError:
                    pass
                if after is not None:
                    segs_after = _load_segments(out_dir)
                    _, seg_after = _find_segment(segs_after, seg_id)
                    if seg_after is not None:
                        after(seg_after)
                        _save_segments(out_dir, segs_after)
            except BaseException:
                try:
                    atomic_write_bytes(segs_path, raw_before)
                finally:
                    try:
                        (out_dir / "segments.json.pre_edit").unlink()
                    except OSError:
                        pass
                    try:
                        status_set(out_dir, editing=False)
                    except OSError:
                        pass
                raise
        finally:
            GATE.release()
        try:
            (out_dir / "segments.json.pre_edit").unlink()
        except OSError:
            pass
        cur = read_json(_status_path(out_dir))
        version = int(cur.get("version", 0) or 0) + 1
        status_set(out_dir, version=version, editing=False)
        segs_updated = _load_segments(out_dir)
        _, seg_updated = _find_segment(segs_updated, seg_id)
    return {"ok": True, "version": version, "segment": seg_updated}


@app.post("/api/jobs/{job_id}/{lang}/chunks/{c}/segments/{seg_id}")
async def edit_segment(job_id: str, lang: str, c: int, seg_id: str, body: TextRequest):
    text = body.text
    if not text or not text.strip():
        raise HTTPException(422, "text must not be empty")
    if len(text) > MAX_EDIT_CHARS:
        raise HTTPException(422, f"text too long ({len(text)} chars; the limit is {MAX_EDIT_CHARS})")

    def change(seg):
        hist = seg.get("history")
        if not isinstance(hist, list):
            hist = []
        hist.append(seg.get("tgt_text", ""))
        seg["tgt_text"] = text
        seg["history"] = hist
        seg["edited"] = True
        return text

    return await _edit_flow(job_id, lang, c, seg_id, change)


@app.post("/api/jobs/{job_id}/{lang}/chunks/{c}/segments/{seg_id}/undo")
async def undo_segment(job_id: str, lang: str, c: int, seg_id: str):
    holder = {}

    def change(seg):
        hist = seg.get("history")
        if not isinstance(hist, list) or not hist:
            raise HTTPException(409, "nothing to undo")
        item = hist.pop()
        # history entries are plain strings (this server) or dicts (older servers)
        new_text = (item.get("tgt_text") or item.get("text") or "") if isinstance(item, dict) else str(item)
        seg["tgt_text"] = new_text
        seg["history"] = hist
        seg["edited"] = bool(hist)
        holder["hist"] = hist
        return new_text

    def after(seg_after):
        # the pipeline must not own history: force our popped list back
        seg_after["history"] = holder["hist"]
        seg_after["edited"] = bool(holder["hist"])

    return await _edit_flow(job_id, lang, c, seg_id, change, after)


# --------------------------------------------------------------------------
# Media endpoints
# --------------------------------------------------------------------------
def _file_or_404(path: Optional[Path]):
    if path is None or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(path), media_type="video/mp4")


@app.get("/media/{job_id}/original/{c}.mp4")
async def media_original_chunk(job_id: str, c: int, request: Request):
    job_dir = resolve_job(job_id)
    if not job_dir:
        raise HTTPException(404, "unknown job")

    # The prompt says: GET /media/{id}/original/{c}.mp4 -> src.mp4 from any language's chunk dir
    # But planner writes JOB_DIR/source.mp4.
    # The chunks are processed from source.mp4.
    # Is there a per-chunk source? The prompt says "planner... writes JOB_DIR/source.mp4".
    # It also says "GET /media/{id}/original/{c}.mp4 -> src.mp4 from any language's chunk dir"
    # This implies the pipeline might have cut it out, or we just serve the global source?
    # "src.mp4 from any language's chunk dir" suggests the pipeline creates a src.mp4 in the chunk folder.

    # Let's check any lang's chunk dir for src.mp4
    job_meta = read_json(job_dir / "job.json")
    langs = job_meta.get("langs", [])

    target_path = _source_file(job_dir, langs, c)

    # Fallback to global source if per-chunk not found?
    # The prompt specifically says "from any language's chunk dir".
    # If none found, maybe 404.

    if target_path:
        return _file_or_404(target_path)

    # Fallback: maybe the global source.mp4 is expected if chunks aren't split?
    # But we are in chunked mode. Let's stick to the prompt.
    raise HTTPException(404, "chunk source not found")


@app.get("/media/{job_id}/{lang}/chunks/{c}.mp4")
async def media_chunk(job_id: str, lang: str, c: int, request: Request, lip: int = 1):
    job_dir = resolve_job(job_id)
    if not job_dir:
        raise HTTPException(404, "unknown job")
    if not LANG_RE.match(lang):
        raise HTTPException(404, "unknown lang")

    out_dir = resolve_chunk_dir(job_dir, lang, c)
    if not out_dir:
        raise HTTPException(404, "chunk dir invalid")

    if read_json(_status_path(out_dir)).get("state") != "done":
        raise HTTPException(404, "video not ready")      # never serve a chunk that is queued/running/errored (partial file)
    dub_path = out_dir / f"dub_{lang}.mp4"
    ls_path = out_dir / f"dub_{lang}_lipsync.mp4"

    # Lipsync preference rule: serve lipsync if non-empty AND newer than dub
    final_path = dub_path

    _jm = read_json(job_dir / "job.json")
    _want_lip = lip != 0 and _jm.get("lipsync", True) is not False        # ?lip=0 or a lipsync=false job: aligned-audio video only
    if _want_lip and ls_path.exists() and ls_path.stat().st_size > 0:
        if dub_path.exists():
            if ls_path.stat().st_mtime > dub_path.stat().st_mtime:
                final_path = ls_path
        else:
            # If dub doesn't exist but lipsync does, serve lipsync
            final_path = ls_path
    elif ((not dub_path.exists()) or dub_path.stat().st_size == 0) and ls_path.exists() and ls_path.stat().st_size > 0:
        final_path = ls_path                       # aligned requested but only the lip render exists: better than a 404

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise HTTPException(404, "video not ready")

    return _file_or_404(final_path)


# --------------------------------------------------------------------------
# Languages endpoint
# --------------------------------------------------------------------------
@app.get("/api/languages")
async def get_languages():
    if not LANGUAGES_FILE.exists():
        return {"targets": [], "sources": []}
    return read_json(LANGUAGES_FILE)


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles  # noqa: E402

app.mount("/static", StaticFiles(directory=str(WEB_DIR), check_dir=False), name="static")   # styles.css and js/*


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return FileResponse(str(WEB_DIR / "index.html"), media_type="text/html")


# --------------------------------------------------------------------------
# Health: this process + every dependency the pipeline needs
# --------------------------------------------------------------------------
def _valid_worker_ids(lw) -> bool:
    return (isinstance(lw, list) and bool(lw) and all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.\-]+", x) for x in lw)
            and len(set(lw)) == len(lw))


def load_services() -> dict:
    """Dependency endpoints as {name: ["host:port", ...]}.  Source: file named by env DUB_SERVICES (JSON), else the
    built-in port lists on host env DEP_HOST (default 127.0.0.1).  Never hard-codes a node, so a service migration
    only needs a services.json (optionally with "lip_mode": "proc"|"none" and "lip_workers": ["wA","wB"])."""
    host = os.environ.get("DEP_HOST", "127.0.0.1")
    ports = {"asr": [18931], "mt": [18933, 18981, 18982, 18983, 18984], "it2": [18936],
             "tts": list(range(18990, 19000)), "sep": [18952]}
    cfg = {k: [f"{host}:{p}" for p in v] for k, v in ports.items()}
    f = os.environ.get("DUB_SERVICES", "")
    if f:
        try:
            j = json.loads(Path(f).read_text(encoding="utf-8"))
            for k, v in j.items():
                if isinstance(v, list) and k not in ("lip_workers",):
                    if v and all(":" in str(x) for x in v):
                        cfg[k] = [str(x) for x in v]
                    else:
                        print(f"[services] ignoring empty/invalid list for {k}", flush=True)
            lm, lw = j.get("lip_mode"), j.get("lip_workers")      # absent -> fall through to env LIP_MODE / LIP_WORKERS
            if lm is not None:
                if lm in ("proc", "none"):
                    cfg["lip_mode"] = lm
                else:
                    print(f"[services] invalid lip_mode {lm!r}, using env/default", flush=True)
            if lw is not None:
                if _valid_worker_ids(lw):
                    cfg["lip_workers"] = lw
                else:
                    print(f"[services] invalid lip_workers {lw!r} (need unique non-empty ids), using env/default", flush=True)
        except Exception as e:
            print(f"[services] cannot read {f}: {e!r}", flush=True)
    if "lip_mode" not in cfg:
        lm = os.environ.get("LIP_MODE", "proc")
        cfg["lip_mode"] = lm if lm in ("proc", "none") else "proc"
    if "lip_workers" not in cfg:
        envw = [w.strip() for w in os.environ.get("LIP_WORKERS", "wA,wB").split(",") if w.strip()]
        if not _valid_worker_ids(envw):
            print(f"[services] invalid env LIP_WORKERS {envw!r} (need unique ids), using wA,wB", flush=True)
            envw = ["wA", "wB"]
        cfg["lip_workers"] = envw
    return cfg


SERVICES = load_services()
LIP_WORKERS = SERVICES["lip_workers"]
LIP_MODE = SERVICES["lip_mode"]
DEP_NAMES = [k for k in SERVICES if k not in ("lip_mode", "lip_workers")]
_health_cache = {"t": 0.0, "v": None}


async def _probe(hp: str, path: str = "/health", timeout: float = 1.5) -> bool:
    host, _, port = hp.rpartition(":")
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        try:
            w.write(f"GET {path} HTTP/1.0\r\nHost: x\r\n\r\n".encode())
            await w.drain()
            head = await asyncio.wait_for(r.read(64), timeout)
        finally:
            w.close()
        return head.startswith(b"HTTP/1.") and b" 200" in head.split(b"\r\n", 1)[0]
    except Exception:
        return False


def _lip_worker_pids() -> Dict[str, Optional[int]]:
    found: Dict[str, Optional[int]] = {w: None for w in LIP_WORKERS}
    if LIP_MODE != "proc":
        return found
    try:
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            cl = _proc_cmdline(int(p))
            if "sunwai.workers.lipsync" not in cl:
                continue
            for w in LIP_WORKERS:
                if f"--id {w} " in cl + " ":
                    found[w] = int(p)
    except OSError:
        pass
    return found


async def deps_status() -> dict:
    if time.time() - _health_cache["t"] < 3.0 and _health_cache["v"] is not None:
        return _health_cache["v"]
    jobs = [(n, p) for n in DEP_NAMES for p in SERVICES[n]]
    res = await asyncio.gather(*[_probe(p) for _, p in jobs])
    deps = {n: {"up": 0, "total": len(SERVICES[n]), "down": []} for n in DEP_NAMES}
    for (n, p), ok in zip(jobs, res):
        if ok:
            deps[n]["up"] += 1
        else:
            deps[n]["down"].append(p)
    lip = await asyncio.get_running_loop().run_in_executor(None, _lip_worker_pids)
    v = {"deps": deps, "lip": lip}
    _health_cache.update(t=time.time(), v=v)
    return v


async def _down_hint() -> str:
    """' (services down: tts 9/10, ...)' for error messages; '' when all deps are up or the probe fails."""
    try:
        d = await deps_status()
        bits = [f"{n} {x['up']}/{x['total']}" for n, x in d["deps"].items() if x["up"] < x["total"]]
        return f" (services unavailable: {', '.join(bits)})" if bits else ""
    except Exception:
        return ""


@app.get("/health")
async def health():
    d = await deps_status()
    degraded = []
    for n, x in d["deps"].items():
        if x["up"] < x["total"]:
            degraded.append(f"{n}:{x['up']}/{x['total']} down={x['down']}")
    for w, pid in d["lip"].items():
        if pid is None and LIP_MODE == "proc":
            degraded.append(f"lip:{w} down")
    return {"ok": True, "ready": not degraded, "degraded": degraded,
            "root": str(DUB_ROOT), "pipeline": True, "max_concurrent": MAX_CONCURRENT,
            "admission": {"on": ADMISSION, "max_jobs": ADM_MAX, "running": list(ADM.running), "queued": list(ADM.queue),
                          "targets_max": MAX_TARGETS, "max_queued": MAX_QUEUED_JOBS, "max_dur_s": MAX_DUR_S,
                          "max_work_s": MAX_WORK_S, "fanout": FANOUT, "token_required": bool(DUB_TOKEN)},
            "deps": d["deps"], "lip": d["lip"], "pid": os.getpid(), "time": utcnow()}


@app.exception_handler(500)
async def _unhandled(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse({"detail": "internal error"}, status_code=500)


@app.post("/api/jobs/{job_id}/focus")
async def set_focus(job_id: str, body: FocusRequest):
    """Viewer focus: drives PriorityGate order AND is persisted atomically to DUB_ROOT/focus.json for the lip workers."""
    jd = resolve_job(job_id)
    if jd is None or not (jd / "job.json").is_file():
        raise HTTPException(404, "job not found")
    lang = (body.lang or "").strip().lower()
    if not LANG_RE.match(lang) or lang not in _job_meta(jd).get("langs", []):
        raise HTTPException(422, "language is not part of this job")
    plan = read_json(jd / "plan.json")
    try:
        idxs = {int(c["idx"]) for c in plan.get("chunks", [])}
    except Exception:
        idxs = set()
    if int(body.chunk) not in idxs:
        raise HTTPException(422, "unknown chunk")
    global _focus_gen
    _focus_gen += 1
    ts = time.time()
    for k in [k for k, v in FOCUS.items() if ts - v[2] > FOCUS_TTL]:
        FOCUS.pop(k, None)                            # expired entries never accumulate
    FOCUS[job_id] = (lang, int(body.chunk), ts)
    try:
        atomic_write_json(FOCUS_FILE, {"job": job_id, "lang": lang, "chunk": int(body.chunk), "ts": ts,
                                       "gen": _focus_gen, "server_pid": os.getpid()}, indent=None)
    except OSError as e:
        print(f"[focus] persist failed {e!r}", flush=True)
    return {"ok": True, "gen": _focus_gen}


# action/revision/chat/events routes. OFF unless DUB_ACTIONS=1 or app/ctl_enabled exists; a failure here must never stop the server.
try:
    from sunwai.server.routers import control as ctl_api
    if ctl_api.register(app, globals()):
        print
except Exception as _ctl_e:
    print


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("sunwai.server.app:create_app", factory=True, host="127.0.0.1", port=int(os.environ.get("PORT", "18941")))
