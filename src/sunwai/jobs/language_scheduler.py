"""add-language scheduler + source-span (code-switching) records.

Pure module: does NOT import server2. Server seams are injected via Hooks. Ships disabled (DUB_ADD_LANG).
State (job dir): tracks.json, source_spans.json, source_lang_overrides.json -- all written atomically.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

LANG_RE = re.compile(r"^[a-z]{2,3}$")
from sunwai.paths import DATA_DIR  # noqa: E402

APP_DIR = DATA_DIR
_LOCK = threading.RLock()
_OFF = ("0", "false", "no", "off", "n", "f", "")


def _env_int(name: str, default: int, lo: int = 0, hi: int = 10 ** 9) -> int:
    try:
        v = int(float(os.environ.get(name, "").strip()))
        return v if lo <= v <= hi else default
    except Exception:
        return default


def enabled() -> bool:
    """Kill switch, parsed robustly. DEFAULT ON; DUB_ADD_LANG=0/false/off/no turns it off."""
    return os.environ.get("DUB_ADD_LANG", "1").strip().lower() not in _OFF


def lead_ms() -> int:
    return _env_int("DUB_LANG_LEAD_MS", 20000, 0, 600000)


_INFLIGHT: Dict[Tuple[str, str], float] = {}   # (job_dir, lang) -> start; added-language tracks still preparing (all jobs)


def max_inflight() -> int:
    return _env_int("DUB_MAX_ADDLANG_INFLIGHT", 1, 1, 16)


_RECONCILE: Dict[Tuple[str, str], dict] = {}   # in-memory fallback when the marker file cannot be written either


def _mark_reconcile(job_dir: Path, lang: str, why: str) -> dict:
    rec = {"lang": lang, "reason": why, "ts": time.time(), "action": "re-request the language to persist the track"}
    _RECONCILE[(str(job_dir), lang)] = rec
    try:
        p = Path(job_dir) / "tracks.reconcile.json"
        d = _read_json(p)
        d[lang] = rec
        _atomic_json(p, d)
    except Exception:
        pass
    return rec


def _clear_reconcile(job_dir: Path, lang: str) -> None:
    _RECONCILE.pop((str(job_dir), lang), None)
    p = Path(job_dir) / "tracks.reconcile.json"
    if p.exists():
        d = _read_json(p)
        if d.pop(lang, None) is not None:
            _atomic_json(p, d)


def _get_reconcile(job_dir: Path, lang: str) -> Optional[dict]:
    r = _RECONCILE.get((str(job_dir), lang)) or _read_json(Path(job_dir) / "tracks.reconcile.json").get(lang)
    if r:
        return r
    tr = (_read_json(Path(job_dir) / "tracks.json").get("tracks") or {}).get(lang)
    if isinstance(tr, dict) and tr.get("phase") == "scheduling":      # write-ahead intent never committed (crash or failed commit write)
        return {"lang": lang, "reason": "scheduling intent not committed", "action": "re-request the language (recovery rolls it forward)"}
    return None


def _releases_cap(job_dir: Path, lang: str, tr: dict, hooks) -> bool:
    """True when this (non ready/error) track no longer counts toward the global cap.
    * phase 'abandoned' (durable marker from release_track): frees only once the job's run_job task is NOT alive (hook job_active), so work
      that starts after the marker is still counted (no TOCTOU) and a refused/ordered-late release needs no retry. No hook -> frees.
    * phase 'scheduled' with no live watcher for a long time: frees only if the job task is not alive AND the track is older than
      DUB_ADDLANG_ORPHAN_S (default 1800 s). Unknown (no hook) -> keeps counting.
    * phase 'scheduling' (uncommitted) never frees: it must be recovered."""
    ph = tr.get("phase")
    ja = getattr(hooks, "job_active", None)
    jid = Path(job_dir).name
    try:
        active = bool(ja(jid)) if ja else None
    except Exception:
        active = None
    if ph == "abandoned":
        return active is not True
    if ph == "scheduled" and active is False:
        return time.time() - float(tr.get("created") or 0) > _env_int("DUB_ADDLANG_ORPHAN_S", 1800, 60, 86400)
    return False


def release_track(job_dir: Path, lang: str, reason: str = "") -> bool:
    """Durably mark a track abandoned (operation failed/cancelled/timed out). It stops counting toward the cap only once the job's run_job
    task is dead (see _releases_cap). A later request for the same lang re-schedules it."""
    job_dir = Path(job_dir)
    with _LOCK:
        tpath = job_dir / "tracks.json"
        st = _read_json(tpath)
        tr = (st.get("tracks") or {}).get(lang) if isinstance(st.get("tracks"), dict) else None
        if not isinstance(tr, dict) or tr.get("phase") == "abandoned":
            return False
        tr["phase"], tr["abandoned_reason"], tr["abandoned_ts"] = "abandoned", reason[:200], time.time()
        _atomic_json(tpath, st)
    _INFLIGHT.pop((str(job_dir), lang), None)
    return True


def _inflight_disk(root: Path, exclude=None, hooks=None) -> set:
    """Added-language tracks still preparing, derived from DISK (tracks.json phase scheduling|scheduled, not yet
    ready/error). Survives a restart, so the global cap cannot be bypassed by a crash; the in-memory dict is only a hint."""
    out = set()
    try:
        files = list(Path(root).glob("*/tracks.json"))
    except OSError:
        return out
    for tp in files:
        jd = tp.parent
        st = _read_json(tp)
        for lg, tr in (st.get("tracks") or {}).items() if isinstance(st.get("tracks"), dict) else []:
            if not isinstance(tr, dict) or tr.get("phase") not in ("scheduling", "scheduled", "abandoned") or (str(jd), lg) == exclude:
                continue
            if track_readiness(jd, lg)["audio_state"] in ("ready", "error", "absent"):
                continue
            if _releases_cap(jd, lg, tr, hooks):
                continue
            out.add((str(jd), lg))
    return out


def recover_job(job_dir: Path, hooks: "Hooks") -> List[dict]:
    """Crash recovery for tracks left in phase 'scheduling' (process died between the write-ahead intent and the commit).
    lang already merged into job.json -> ROLL FORWARD: ensure_running (idempotent) and commit phase=scheduled.
    lang never merged (died before job.json) -> drop the intent. Server2's own startup reconcile only resumes jobs whose
    status is queued/running, so a finished job with a newly merged language would otherwise never be dubbed."""
    job_dir = Path(job_dir)
    acts: List[dict] = []
    if not enabled():
        return acts                       # kill switch: intents stay intact, nothing is (re)started
    with _LOCK:
        tpath = job_dir / "tracks.json"
        st = _read_json(tpath)
        trs = st.get("tracks") if isinstance(st.get("tracks"), dict) else {}
        pend = [lg for lg, t in trs.items() if isinstance(t, dict) and t.get("phase") == "scheduling"]
        if not pend:
            return acts
        meta = _read_json(job_dir / "job.json")
        for lg in pend:
            if lg in (meta.get("langs") or []):
                try:
                    if not hooks.ensure_running:
                        raise RuntimeError("no ensure_running hook: cannot schedule")
                    hooks.ensure_running(meta.get("job_id") or job_dir.name)
                    trs[lg]["phase"] = "scheduled"
                    acts.append({"lang": lg, "action": "rolled_forward"})
                except Exception as e:
                    acts.append({"lang": lg, "action": "roll_forward_failed", "error": f"{type(e).__name__}: {e}"})
            else:
                trs.pop(lg, None)
                acts.append({"lang": lg, "action": "intent_dropped"})
        try:
            _atomic_json(tpath, st)
            for a in acts:
                if a["action"] != "roll_forward_failed":
                    _clear_reconcile(job_dir, a["lang"])
        except Exception as e:
            acts.append({"action": "persist_failed", "error": f"{type(e).__name__}: {e}"})
    return acts


def recover_all(root: Path, hooks: "Hooks") -> List[dict]:
    out = []
    if not enabled():
        return out
    try:
        for tp in sorted(Path(root).glob("*/tracks.json")):
            for a in recover_job(tp.parent, hooks):
                out.append({"job": tp.parent.name, **a})
    except Exception as e:
        out.append({"action": "recover_all_failed", "error": f"{type(e).__name__}: {e}"})
    return out


class LangSchedError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def as_dict(self):
        return {"status": self.status, "code": self.code, "detail": self.message}


@dataclass
class Hooks:
    ensure_running: Optional[Callable[[str], object]] = None          # server2.ensure_job_running
    set_focus: Optional[Callable[[str, str, int], object]] = None     # fn(job_id, lang, chunk_idx) -> same effect as POST /focus
    target_codes: Optional[Callable[[], set]] = None                   # server2.get_target_codes
    max_targets: int = 3                                              # server2.MAX_TARGETS (DUB_MAX_TARGETS)
    merge_lang: Optional[Callable[[Path, str], List[str]]] = None     # default: atomic job.json writer below
    busy: Optional[Callable[[], bool]] = None                          # e.g. lambda: len(ADM.running)+len(ADM.queue) >= ADM_MAX+MAX_QUEUED_JOBS
    job_active: Optional[Callable[[str], bool]] = None                 # job_id -> True while server2's run_job task for it is alive (authoritative "may still schedule/run chunks")
    lipsync_hash_ok: Optional[Callable[[Path, str, int], Optional[bool]]] = None   # (job_dir, lang, idx) -> True/False/None(unknown); wire expression lipsync_complete


# ---------------------------------------------------------------------------------------------- io helpers
def _read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _atomic_json(p: Path, obj) -> None:
    p = Path(p)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _chunks(job_dir: Path) -> List[dict]:
    pl = _read_json(Path(job_dir) / "plan.json")
    out = []
    for c in pl.get("chunks", []) if isinstance(pl, dict) else []:
        try:
            out.append({"idx": int(c["idx"]), "start": float(c["start"]), "dur": float(c["dur"])})
        except Exception:
            continue
    return sorted(out, key=lambda c: c["idx"])


def _chunk_dir(job_dir: Path, lang: str, idx: int) -> Path:
    return Path(job_dir) / lang / f"c{idx:04d}"


def _chunk_done(job_dir: Path, lang: str, idx: int) -> bool:
    d = _chunk_dir(job_dir, lang, idx)
    if _read_json(d / "status.json").get("state") != "done":
        return False
    try:
        return (d / f"dub_{lang}.mp4").stat().st_size > 0
    except OSError:
        return False


def _sha(p: Path) -> Optional[str]:
    import hashlib
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _chunk_lipsync(job_dir: Path, lang: str, idx: int, hook=None) -> Tuple[bool, str]:
    """(ready, basis). mtime freshness is necessary; then the audio-hash check: hook if given, else
    lipsync.json's chunk_audio_hash vs sha256(dub.wav) when recorded. A False hash result is NOT ready. basis 'mtime' means
    the hash could not be verified (reported as such, never as verified)."""
    d = _chunk_dir(job_dir, lang, idx)
    dub, ls = d / f"dub_{lang}.mp4", d / f"dub_{lang}_lipsync.mp4"
    try:
        if not (ls.stat().st_size > 0 and dub.exists() and ls.stat().st_mtime > dub.stat().st_mtime):
            return False, "stale_or_missing"
    except OSError:
        return False, "stale_or_missing"
    ok = None
    if hook is not None:
        try:
            ok = hook(Path(job_dir), lang, idx)
        except Exception:
            ok = None
    if ok is None:
        want = _read_json(d / "lipsync.json").get("chunk_audio_hash")
        if want:
            ok = (_sha(d / "dub.wav") == want)
    if ok is False:
        return False, "hash_mismatch"
    return True, "hash" if ok else "mtime"


# ---------------------------------------------------------------------------------------------- scheduling
def chunk_for_ms(chunks: List[dict], ms: float) -> int:
    """Index of the chunk containing ms (clamped to first/last)."""
    if not chunks:
        return 0
    t = ms / 1000.0
    for c in chunks:
        if c["start"] <= t < c["start"] + c["dur"]:
            return c["idx"]
    return chunks[0]["idx"] if t < chunks[0]["start"] else chunks[-1]["idx"]


def runway(chunks: List[dict], from_ms: float, lead: Optional[int] = None) -> List[int]:
    """The containing chunk plus every chunk starting within `lead` ms after from_ms."""
    lead = lead_ms() if lead is None else lead
    if not chunks:
        return []
    k = chunk_for_ms(chunks, from_ms)
    hi = (from_ms + lead) / 1000.0
    return [k] + [c["idx"] for c in chunks if c["idx"] > k and c["start"] < hi]


def plan_order(chunks: List[dict], from_ms: float, lead: Optional[int] = None) -> List[int]:
    """Containing chunk, then chunks overlapping [from, from+lead] (the playable runway), then the rest AHEAD in order,
    then BEHIND nearest-first (a viewer who seeks back sees the closest earlier chunk first)."""
    lead = lead_ms() if lead is None else lead
    if not chunks:
        return []
    k = chunk_for_ms(chunks, from_ms)
    ahead = [c["idx"] for c in chunks if c["idx"] > k]
    run = runway(chunks, from_ms, lead)[1:]
    rest = [i for i in ahead if i not in run]
    behind = sorted((c["idx"] for c in chunks if c["idx"] < k), reverse=True)
    return [k] + run + rest + behind


def _merge(iv: List[List[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def track_readiness(job_dir: Path, lang: str, hooks: Optional[Hooks] = None) -> dict:
    job_dir = Path(job_dir)
    meta = _read_json(job_dir / "job.json")
    chunks = _chunks(job_dir)
    if lang not in (meta.get("langs") or []):
        return {"lang": lang, "ready_intervals": [], "audio_state": "absent", "lipsync_state": "none",
                "lipsync_intervals": [], "chunks": []}
    a_iv, l_iv, rows = [], [], []
    n_done = n_ls = n_run = n_err = 0
    for c in chunks:
        d = _chunk_dir(job_dir, lang, c["idx"])
        st = _read_json(d / "status.json").get("state")
        done = _chunk_done(job_dir, lang, c["idx"])
        ls, basis = _chunk_lipsync(job_dir, lang, c["idx"], hooks.lipsync_hash_ok if hooks else None) if done else (False, "no_audio")
        iv = [int(round(c["start"] * 1000)), int(round((c["start"] + c["dur"]) * 1000))]
        n_done += done
        n_ls += ls
        n_run += (not done and st in ("running", "queued"))
        n_err += (not done and st == "error")
        if done:
            a_iv.append(iv)
        if ls:
            l_iv.append(iv)
        rows.append({"chunk": c["idx"], "state": "done" if done else (st or "pending"), "lipsync": bool(ls), "lipsync_basis": basis})
    n = len(chunks)
    if n and n_done == n:
        audio = "ready"
    elif n_done:
        audio = "partial"
    elif n_err and not n_run:
        audio = "error"
    elif n_run:
        audio = "preparing"
    else:
        audio = "preparing"        # contract enum is absent|preparing|partial|ready|error: 'queued' is reported as preparing
    lipsync = "none" if not n_ls else ("ready" if n_ls == n else "partial")
    return {"lang": lang, "ready_intervals": _merge(a_iv), "audio_state": audio, "lipsync_state": lipsync,
            "lipsync_intervals": _merge(l_iv), "reconcile": _get_reconcile(job_dir, lang), "lipsync_verified": bool(n_ls) and all(r["lipsync_basis"] == "hash" for r in rows if r["lipsync"]),
            "chunks": rows}


def focus_adapter(job_dir: Path, job_id: str, watching_lang: Optional[str], lang: str, from_ms: float,
                  hooks: Hooks) -> dict:
    """Map the new track's from_ms onto the existing single-slot /focus priority (lang, chunk). The viewer's own language
    keeps priority until its runway is buffered; after that the new language is focused at the containing chunk.
    PriorityGate/_priority are untouched: with focus=(lang, k) they order (lang==focus first, then chunk distance ahead)."""
    chunks = _chunks(job_dir)
    k = chunk_for_ms(chunks, from_ms)
    if hooks.set_focus is None:
        return {"applied": False, "chunk": k, "reason": "no_focus_hook"}
    if watching_lang and watching_lang != lang:
        run = runway(chunks, from_ms)
        if not all(_chunk_done(job_dir, watching_lang, i) for i in run):
            return {"applied": False, "chunk": k, "reason": "watching_lang_runway_not_buffered"}
    try:
        hooks.set_focus(job_id, lang, k)
    except Exception as e:  # focus is advisory; never fail the request
        return {"applied": False, "chunk": k, "reason": f"set_focus_failed:{type(e).__name__}"}
    return {"applied": True, "chunk": k, "reason": "ok"}


# ---------------------------------------------------------------------------------------------- voice plan
def check_voice_plan(job_dir: Path, lang: str) -> Tuple[bool, str]:
    """New languages use the frozen speaker identities/voices: a planned job accepts only voice_plan.langs minus
    unavailable_langs. A legacy job (no voice_plan) is accepted (old pick_voice per language). Never mutates the plan."""
    p = Path(job_dir) / "speakers_global.json"
    if not p.is_file():
        return False, "planner_not_ready"
    g = _read_json(p)
    vp = g.get("voice_plan") if isinstance(g, dict) else None
    if vp is None:
        return True, "legacy"
    if not isinstance(vp, dict):
        return False, "voice_plan_invalid"
    if lang in (vp.get("unavailable_langs") or {}):
        return False, f"voice plan marks {lang} unavailable: {vp['unavailable_langs'][lang]}"
    if lang not in (vp.get("langs") or []) or lang not in (vp.get("voices") or {}):
        return False, f"{lang} is not in this job's immutable voice plan"
    return True, "planned"


def _default_merge(job_dir: Path, lang: str) -> List[str]:
    meta = _read_json(Path(job_dir) / "job.json")
    langs = list(meta.get("langs") or [])
    if lang not in langs:
        langs.append(lang)
        meta["langs"] = langs
        meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        _atomic_json(Path(job_dir) / "job.json", meta)
    (Path(job_dir) / lang).mkdir(parents=True, exist_ok=True)
    return langs


def request_track(job_dir: Path, lang: str, from_ms, playback_generation, *, watching_lang: Optional[str] = None,
                  hooks: Optional[Hooks] = None) -> dict:
    """Idempotent track request. Same (job, lang) again -> created=False and only re-prioritises (newer/equal generation).
    A generation OLDER than the newest seen is rejected (409 obsolete_generation): stale switch responses never act."""
    hooks = hooks or Hooks()
    if not enabled():
        raise LangSchedError(503, "disabled", "add-language is disabled (DUB_ADD_LANG)")
    job_dir = Path(job_dir)
    meta = _read_json(job_dir / "job.json")
    if not meta.get("langs"):
        raise LangSchedError(404, "job_not_found", "job not found")
    lang = (lang or "").strip().lower() if isinstance(lang, str) else ""
    if not LANG_RE.match(lang):
        raise LangSchedError(422, "bad_lang", "invalid language code")
    try:
        from_ms = float(from_ms)
        gen = int(playback_generation)
    except (TypeError, ValueError):
        raise LangSchedError(422, "bad_args", "from_ms must be a number and playback_generation an integer")
    if not math.isfinite(from_ms) or from_ms < 0 or gen < 0:
        raise LangSchedError(422, "bad_args", "from_ms >= 0 and playback_generation >= 0 required")
    from_ms = int(from_ms)
    codes = hooks.target_codes() if hooks.target_codes else {t.get("code") for t in _read_json(APP_DIR / "languages.json").get("targets", [])}
    if codes and lang not in codes:
        raise LangSchedError(422, "unsupported", f"unsupported language {lang}")
    recover_job(job_dir, hooks)                 # a previous crash/failed commit for this job is rolled forward first
    with _LOCK:
        meta = _read_json(job_dir / "job.json")
        langs = list(meta["langs"])
        tpath = job_dir / "tracks.json"
        st = _read_json(tpath)
        if not isinstance(st.get("tracks"), dict):
            st = {"gen": 0, "tracks": {}}
        # ---- all validation BEFORE any mutation ----
        if gen < int(st.get("gen", 0)):
            raise LangSchedError(409, "obsolete_generation", f"playback_generation {gen} < {st['gen']}")
        is_new = lang not in langs
        key = (str(job_dir), lang)
        if not is_new and (st["tracks"].get(lang) or {}).get("phase") == "scheduling":
            raise LangSchedError(503, "recovery_pending", f"{lang} was merged but could not be scheduled yet; retry shortly")
        retry = (not is_new) and (st["tracks"].get(lang) or {}).get("phase") == "abandoned"
        if is_new or retry:
            if is_new and len(langs) + 1 > hooks.max_targets:
                raise LangSchedError(422, "max_targets", f"a job may have at most {hooks.max_targets} target languages (has {langs})")
            ok, why = check_voice_plan(job_dir, lang) if is_new else (True, "retry")
            if not ok:
                code = "planner_not_ready" if why == "planner_not_ready" else "voice_plan_excludes_lang"
                msg = ("planner has not finished; retry shortly" if code == "planner_not_ready" else
                       f"{why}. Voices are frozen per job: start a NEW job with this language to get a voice plan for it.")
                raise LangSchedError(409 if code == "planner_not_ready" else 422, code, msg)
            for k in list(_INFLIGHT):        # in-memory hints: drop finished ones
                tr0 = (_read_json(Path(k[0]) / "tracks.json").get("tracks") or {}).get(k[1]) or {}
                if track_readiness(Path(k[0]), k[1])["audio_state"] in ("ready", "error", "absent") or _releases_cap(Path(k[0]), k[1], tr0, hooks):
                    _INFLIGHT.pop(k, None)
            others = (_inflight_disk(job_dir.parent, exclude=key, hooks=hooks) | set(_INFLIGHT)) - {key}
            if len(others) >= max_inflight():
                raise LangSchedError(429, "busy", f"{len(others)} added-language track(s) already preparing "
                                                   f"(limit {max_inflight()}); try again when it finishes")
            if hooks.busy is not None and hooks.busy():
                raise LangSchedError(429, "busy", "the demo is busy with other dubbing jobs; try again in a few minutes")
            # ---- transaction (write-ahead): 1 durable intent -> 2 job.json merge -> 3 ensure_running -> 4 commit.
            # A crash anywhere leaves phase=scheduling on disk, which counts toward the cap and which recover_job rolls forward.
            jp = job_dir / "job.json"
            snap = jp.read_bytes()
            tsnap = tpath.read_bytes() if tpath.exists() else None
            lang_dir_existed = (job_dir / lang).exists()
            prev0 = st["tracks"].get(lang)
            st["tracks"][lang] = {"from_ms": from_ms, "generation": gen, "created": time.time(), "requests": 1,
                                  "watching_lang": watching_lang, "phase": "scheduling", "prev_langs": langs}
            st["gen"] = max(int(st.get("gen", 0)), gen)
            try:
                _atomic_json(tpath, st)
            except Exception as e:           # nothing mutated yet
                raise LangSchedError(500, "persist_failed", f"could not persist the scheduling intent: {type(e).__name__}: {e}")
            try:
                if is_new:
                    (hooks.merge_lang or _default_merge)(job_dir, lang)
                if hooks.ensure_running:
                    hooks.ensure_running(meta.get("job_id") or job_dir.name)
            except BaseException as e:
                try:
                    if is_new:
                        tmp = jp.with_name(jp.name + f".rb{os.getpid()}")
                        tmp.write_bytes(snap); os.replace(tmp, jp)
                    if is_new and not lang_dir_existed:
                        try:
                            (job_dir / lang).rmdir()
                        except OSError:
                            pass
                    if tsnap is None:
                        tpath.unlink(missing_ok=True)
                    else:
                        t2 = tpath.with_name(tpath.name + f".rb{os.getpid()}")
                        t2.write_bytes(tsnap); os.replace(t2, tpath)
                except Exception:
                    pass                     # intent stays; recover_job drops it (lang not in job.json) -- never silent
                _INFLIGHT.pop(key, None)
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                if isinstance(e, LangSchedError):
                    raise
                raise LangSchedError(502, "schedule_failed", f"could not schedule {lang}: {type(e).__name__}: {e}; no changes were kept")
            _INFLIGHT[key] = time.time()
            st["tracks"][lang]["phase"] = "scheduled"
            created = prev0 is None or retry
        else:
            prev = st["tracks"].get(lang)
            created = prev is None
            st["gen"] = max(int(st.get("gen", 0)), gen)
            st["tracks"][lang] = {"from_ms": from_ms, "generation": gen,
                                  "created": (prev or {}).get("created") or time.time(),
                                  "requests": int((prev or {}).get("requests", 0)) + 1,
                                  "watching_lang": watching_lang, **({"phase": prev["phase"]} if prev and prev.get("phase") else {})}
        reconcile = None
        persisted = False
        try:
            _atomic_json(tpath, st)
            persisted = True
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if not (is_new or retry):      # nothing was scheduled by this call: plain structured failure
                raise LangSchedError(500, "persist_failed", f"could not persist track state: {type(e).__name__}: {e}")
            # work is scheduled; the durable intent (phase=scheduling) is still on disk: report it, recovery rolls it forward.
            reconcile = _mark_reconcile(job_dir, lang, f"tracks.json commit failed: {type(e).__name__}: {e}")
        if persisted:
            try:
                _clear_reconcile(job_dir, lang)
            except Exception:
                pass
    chunks = _chunks(job_dir)
    jid = meta.get("job_id") or job_dir.name
    focus = focus_adapter(job_dir, jid, watching_lang, lang, from_ms, hooks)
    return {"lang": lang, "created": created, "reconcile": reconcile, "from_ms": from_ms, "generation": gen,
            "order": plan_order(chunks, from_ms), "focus": focus, "readiness": track_readiness(job_dir, lang, hooks)}


# ---------------------------------------------------------------------------------------------- source LID / spans
_SCRIPT_LANG = {  # unicode script -> candidate source languages (first = default when no job LID agrees)
    "DEVANAGARI": ["hi", "mr", "ne", "sa", "kok", "mai", "doi", "brx"], "TAMIL": ["ta"], "BENGALI": ["bn", "as"],
    "GUJARATI": ["gu"], "GURMUKHI": ["pa"], "ORIYA": ["or"], "TELUGU": ["te"], "KANNADA": ["kn"],
    "MALAYALAM": ["ml"], "ARABIC": ["ur", "sd", "ks"], "LATIN": ["en"],
}
_TOKEN = re.compile(r"\S+")
_SCRIPT_CACHE: Dict[str, str] = {}


def _script_of(ch: str) -> Optional[str]:
    if not (ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc")):   # Indic vowel signs are marks, not letters
        return None
    s = _SCRIPT_CACHE.get(ch)
    if s is None:
        try:
            s = unicodedata.name(ch).split()[0]
        except ValueError:
            s = "OTHER"
        _SCRIPT_CACHE[ch] = s
    return s


def script_profile(text: str) -> Dict[str, int]:
    prof: Dict[str, int] = {}
    for ch in text or "":
        s = _script_of(ch)
        if s:
            prof[s] = prof.get(s, 0) + 1
    return prof


def token_profile(text: str) -> Dict[str, int]:
    """Words per script (a word counts for its majority script). Token-level, so a few long Latin words do not outvote
    the Indic-script words around them."""
    prof: Dict[str, int] = {}
    for m in _TOKEN.finditer(text or ""):
        t = _token_script(m.group(0))
        if t:
            prof[t] = prof.get(t, 0) + 1
    return prof


def classify_text(text: str, job_lid: Optional[str], min_letters: int = 3) -> dict:
    """Cheap per-segment source LID from script. Same-script ambiguity (e.g. Devanagari hi/mr/ne) is resolved by the
    video-level LID only when it lies in that script's candidate set (method script+job_lid, lower confidence); Latin text
    in a job whose LID is a non-Latin language is English (ASR emits native script for native speech) -- but flagged lower.
    Too few letters or a conflict -> lang 'und' (unresolved), never a guess."""
    prof = token_profile(text)
    n = sum(script_profile(text).values())
    if n < min_letters or not prof:
        return {"lang": "und", "confidence": 0.0, "method": "too_short", "mixed": False, "embedded": []}
    dom = max(prof, key=lambda k: (prof[k], k != "LATIN"))      # tie -> the non-Latin script
    frac = prof[dom] / sum(prof.values())
    cands = _SCRIPT_LANG.get(dom)
    if not cands:
        return {"lang": "und", "confidence": 0.0, "method": "script_unknown", "mixed": False, "embedded": []}
    if dom == "ARABIC" and job_lid in ("hi", "ur"):
        lang, conf, meth = "ur", 0.7, "script+job_lid_hindustani"   # script says Urdu; same spoken language as hi (lang_group)
    elif dom == "DEVANAGARI" and job_lid == "ur":
        lang, conf, meth = "hi", 0.7, "script+job_lid_hindustani"
    elif len(cands) == 1:
        lang, conf, meth = cands[0], 0.95, "script"
    elif job_lid in cands:
        lang, conf, meth = job_lid, 0.7, "script+job_lid"
    else:
        lang, conf, meth = cands[0], 0.4, "script_ambiguous"
    if dom == "LATIN" and job_lid and job_lid != "en":
        conf = min(conf, 0.6)            # could be romanised native speech; not resolvable from script
    mixed = frac < 0.9
    conf = round(conf * (frac if mixed else 1.0), 3)
    emb = [m.group(0) for m in _TOKEN.finditer(text or "") if _token_script(m.group(0)) not in (None, dom)]
    return {"lang": lang, "confidence": conf, "method": meth, "mixed": mixed, "embedded": emb}


def _token_script(tok: str) -> Optional[str]:
    p = script_profile(tok)
    return max(p, key=p.get) if p else None


_EQUIV = {"hi": "hi-ur", "ur": "hi-ur"}   # Hindustani: one spoken language, ASR flips between Devanagari and Arabic script


def lang_group(lang: str) -> str:
    """Spoken-language group for scoring/routing: hi and ur share one spoken language (Hindustani)."""
    return _EQUIV.get(lang, lang)


def _seg_key(chunk: int, seg_id) -> str:
    return f"{int(chunk)}:{seg_id}"


def _job_lid(job_dir: Path) -> Optional[str]:
    d = _read_json(Path(job_dir) / "lid.json")
    code = d.get("code") or (str(d.get("label", "")).split(":")[0].strip() or None)
    return code or None


def _source_segments(job_dir: Path, idx: int, langs: List[str]) -> List[dict]:
    """Cached source ASR: shared _prep first, else any language dir's segments.json (src_text is language-independent)."""
    cands = [Path(job_dir) / "_prep" / f"c{idx:04d}" / "segments.json"] + [_chunk_dir(job_dir, l, idx) / "segments.json" for l in langs]
    for p in cands:
        s = _read_json(p, default=[])
        if isinstance(s, dict):
            s = s.get("segments", [])
        if isinstance(s, list) and s:
            return s
    return []


def build_source_spans(job_dir: Path) -> List[dict]:
    job_dir = Path(job_dir)
    meta = _read_json(job_dir / "job.json")
    langs = list(meta.get("langs") or [])
    lid = _job_lid(job_dir)
    ovr = _read_json(job_dir / "source_lang_overrides.json")
    spans = []
    for c in _chunks(job_dir):
        for sg in _source_segments(job_dir, c["idx"], langs):
            try:
                a = int(round((c["start"] + float(sg["start"])) * 1000))
                b = int(round((c["start"] + float(sg["end"])) * 1000))
            except Exception:
                continue
            r = classify_text(sg.get("src_text") or "", lid)
            o = ovr.get(_seg_key(c["idx"], sg.get("id")))
            if o:
                r.update(lang=o["lang"], confidence=1.0, method="override")
            spans.append({"start_ms": a, "end_ms": b, "lang": r["lang"], "confidence": r["confidence"],
                          "method": r["method"], "chunk": c["idx"], "seg_id": sg.get("id"),
                          "mixed": r["mixed"], "embedded": r["embedded"]})
    return spans


def persist_source_spans(job_dir: Path) -> List[dict]:
    spans = build_source_spans(job_dir)
    with _LOCK:
        _atomic_json(Path(job_dir) / "source_spans.json", {"job_lid": _job_lid(job_dir), "spans": spans})
    return spans


def load_source_spans(job_dir: Path) -> List[dict]:
    d = _read_json(Path(job_dir) / "source_spans.json")
    return d.get("spans", []) if isinstance(d, dict) else []


def set_segment_lang_override(job_dir: Path, chunk: int, seg_id, lang: str, reviewer: str = "") -> List[dict]:
    if not isinstance(lang, str) or not (LANG_RE.match(lang) or lang == "und"):
        raise LangSchedError(422, "bad_lang", "invalid language code")
    with _LOCK:
        p = Path(job_dir) / "source_lang_overrides.json"
        o = _read_json(p)
        o[_seg_key(chunk, seg_id)] = {"lang": lang, "by": reviewer, "ts": time.time()}
        _atomic_json(p, o)
    return persist_source_spans(job_dir)


def clear_segment_lang_override(job_dir: Path, chunk: int, seg_id) -> List[dict]:
    with _LOCK:
        p = Path(job_dir) / "source_lang_overrides.json"
        o = _read_json(p)
        o.pop(_seg_key(chunk, seg_id), None)
        _atomic_json(p, o)
    return persist_source_spans(job_dir)


def route_segment(span: dict, target_lang: str, it2_targets=("mni", "sd")) -> dict:
    """Translation route for one source span (mirrors dub_pipeline.routed_mt). Already-in-target text is not translated;
    unresolved ('und') spans use the job path unchanged (never a guess). Homogeneous spans -> local indic-translate."""
    src = span.get("lang") or "und"
    if src == target_lang and src != "und" and span.get("confidence", 0) >= 0.5 and not span.get("mixed"):
        return {"route": "passthrough", "src": src}
    if target_lang == "en":
        return {"route": "gemini-en", "src": src}
    if target_lang in it2_targets:
        return {"route": "it2", "src": src}
    return {"route": "indic-translate", "src": src}


def mask_foreign_terms(text: str, dominant_script: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
    """Replace tokens whose script differs from the segment's dominant script (brand names, English terms) with
    placeholders so MT cannot mangle them. unmask_terms() reports missing placeholders; caller then falls back."""
    prof = token_profile(text)
    dom = dominant_script or (max(prof, key=lambda k: (prof[k], k != "LATIN")) if prof else None)
    mapping: Dict[str, str] = {}

    def sub(m):
        tok = m.group(0)
        core = tok.strip(".,;:!?\"'()[]{}।|-")
        s = _token_script(core)
        if core and s and dom and s != dom:
            key = f"QX{len(mapping)}Z"
            mapping[key] = core
            return tok.replace(core, key)
        return tok
    return _TOKEN.sub(sub, text or ""), mapping


def unmask_terms(text: str, mapping: Dict[str, str]) -> Tuple[str, List[str]]:
    missing = [k for k in mapping if k not in text]
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text, missing
