"""adapters from the ctl_hooks seams to expression (expression.py) and language scheduler (lang_sched.py). Installed by ctl_api.register only when
those modules import and are enabled; otherwise the 501 stubs stay. Sync expression calls run in a thread (they call local TTS/ffmpeg)."""
import asyncio, time
from pathlib import Path

from sunwai.control import hooks as ctl_hooks
from sunwai.control.expression import revisions as ex_rev
from sunwai.control.expression.fsutil import read_json, sha256_file
from sunwai.control.store import CtlError


def install(ns, log=print):
    installed = []
    try:
        from sunwai.control import expression as ex
        if ex.enabled():
            def cdir(jd, lang, chunk):
                return ns["resolve_chunk_dir"](jd, lang, int(chunk))

            async def expression_resynth(job_dir, lang, chunk, segment_id, intent, base_text, tempo_factor, out_candidate_dir):
                if intent is None:
                    return {"ok": False, "error": "retime-only previews are not supported by the expression pipeline (tempo is a fit mechanism, not a control)"}
                od = cdir(job_dir, lang, chunk)
                def run():
                    head = ex.head_revision(od, lang, segment_id)
                    return ex.do_expression_resynth(od, lang, segment_id, intent, head["revision"], f"ctl-{time.time_ns()}", job_id=Path(job_dir).name)
                try:
                    c = await asyncio.to_thread(run)
                except ex.ExprError as e:
                    return {"ok": False, "error": f"{e.status}: {e.msg}"}
                if c.get("status") != "ready":
                    return {"ok": False, "error": c.get("reason") or c.get("status")}
                return {"ok": True, "audio_hash": c["audio_hash"], "spoken_text": c["spoken_text"], "style": c["style"],
                        "tempo_factor": c["tempo_factor"], "duration_ms": int(1000 * c.get("fitted_s", 0)), "m2_candidate_id": c["candidate_id"]}

            async def expression_apply(job_dir, lang, chunk, segment_id, candidate):
                od = cdir(job_dir, lang, chunk)
                def run():
                    head = ex.head_revision(od, lang, segment_id)
                    return ex.apply_candidate(od, segment_id, candidate["m2_candidate_id"], head["revision"], lang=lang)
                try:
                    rec = await asyncio.to_thread(run)          # the caller (ActionService) already holds the legacy chunk lock
                except ex.ExprError as e:
                    return {"ok": False, "error": f"{e.status}: {e.msg}"}
                return {"ok": True, "audio_hash": rec.get("audio_hash") or candidate["audio_hash"], "m2_revision": rec["revision"], "chunk_audio_hash": rec.get("chunk_audio_hash")}

            async def expression_undo(job_dir, lang, chunk, segment_id):
                od = cdir(job_dir, lang, chunk)
                def run():
                    head = ex.head_revision(od, lang, segment_id)
                    return ex.undo_revision(od, segment_id, head["revision"], lang=lang)
                try:
                    rec = await asyncio.to_thread(run)          # chunk lock held by the caller
                except ex.ExprError as e:
                    return {"ok": False, "error": f"{e.status}: {e.msg}"}
                return {"ok": True, "audio_hash": rec.get("audio_hash"), "m2_revision": rec["revision"], "chunk_audio_hash": rec.get("chunk_audio_hash")}

            async def media_path(job_dir, lang, chunk, segment_id, kind, m2_candidate_id, is_head=True, m2_revision=None):
                import re as _re
                od = Path(cdir(job_dir, lang, chunk)).resolve()
                if kind == "cand":
                    if not m2_candidate_id or not _re.match(r"^[A-Za-z0-9_\-]{1,40}$", m2_candidate_id):
                        raise FileNotFoundError("no candidate")
                    rec = read_json(Path(ex_rev.expr_dir(od)) / "cands" / f"{m2_candidate_id}.json") or {}
                    p = Path(rec.get("audio_path") or "")
                elif is_head:
                    p = Path(ex_rev.active_wav(od, segment_id))
                elif m2_revision is not None:                       # an older revision: serve THAT revision's snapshot, never the newer live audio
                    p = Path(ex_rev.revision_wav(od, segment_id, m2_revision))
                else:
                    raise FileNotFoundError("old version gone")
                p = p.resolve()
                if od not in p.parents or p.suffix != ".wav" or not p.is_file():     # job/chunk-scoped: never outside the chunk dir
                    raise FileNotFoundError("media not found")
                return p
            ctl_hooks.media_path = media_path

            def _guard(fn):
                try:
                    return fn()
                except ex.ExprError as e:
                    return {"ok": False, "error": f"{e.status}: {e.msg}"}

            def expression_apply_start(job_dir, lang, chunk, segment_id, candidate):
                od = cdir(job_dir, lang, chunk)
                def run():
                    def go():
                        head = ex.head_revision(od, lang, segment_id)
                        rec = ex.apply_candidate(od, segment_id, candidate["m2_candidate_id"], head["revision"], lang=lang)
                        return {"ok": True, "audio_hash": rec.get("audio_hash") or candidate["audio_hash"], "m2_revision": rec["revision"], "chunk_audio_hash": rec.get("chunk_audio_hash")}
                    return _guard(go)
                return ctl_hooks.EXEC.submit(run)

            def expression_undo_start(job_dir, lang, chunk, segment_id):
                od = cdir(job_dir, lang, chunk)
                def run():
                    def go():
                        head = ex.head_revision(od, lang, segment_id)
                        rec = ex.undo_revision(od, segment_id, head["revision"], lang=lang)
                        return {"ok": True, "audio_hash": rec.get("audio_hash"), "m2_revision": rec["revision"], "chunk_audio_hash": rec.get("chunk_audio_hash")}
                    return _guard(go)
                return ctl_hooks.EXEC.submit(run)
            ctl_hooks.expression_apply_start, ctl_hooks.expression_undo_start = expression_apply_start, expression_undo_start

            def live_state(job_dir, lang, chunk, segment_id):
                try:
                    return ex.live_state(cdir(job_dir, lang, chunk), segment_id)
                except Exception:
                    return {"m2_head": None, "owned": True, "in_flight": True}      # fail CLOSED: unknown => do not reconcile into a legacy revision
            ctl_hooks.live_state = live_state
            ctl_hooks.expression_resynth, ctl_hooks.expression_apply, ctl_hooks.expression_undo = expression_resynth, expression_apply, expression_undo
            installed.append("expression")
    except ImportError:
        pass
    try:
        from sunwai.jobs import language_scheduler as ls
        if ls.enabled():
            def set_focus(job_id, lang, chunk):
                ns["_focus_gen"] = ns.get("_focus_gen", 0) + 1
                ns["FOCUS"][job_id] = (lang, int(chunk), time.time())
            def busy():
                adm = ns.get("ADM")
                return bool(adm) and (len(adm.running) + len(adm.queue) >= ns.get("ADM_MAX", 1) + ns.get("MAX_QUEUED_JOBS", 4))

            def lipsync_hash_ok(job_dir, lang, idx):
                """True/False when expression recorded a chunk_audio_hash for this chunk (compare with the dub.wav now); None = unknown (mtime rule only)."""
                try:
                    od = ns["resolve_chunk_dir"](Path(job_dir), lang, int(idx))
                    xd = Path(ex_rev.expr_dir(od))
                    best = None
                    for f in xd.glob("seg*.revs.jsonl"):
                        for rec in ex_rev.read_revisions(od, f.name[3:-len(".revs.jsonl")]):
                            if rec.get("chunk_audio_hash"):
                                best = rec["chunk_audio_hash"]
                    if best is None or not (od / "dub.wav").exists():
                        return None
                    return sha256_file(od / "dub.wav") == best
                except Exception:
                    return None
            def job_active(job_id):
                t = ns.get("JOB_TASKS", {}).get(job_id)
                return t is not None and not t.done()
            hooks = ls.Hooks(ensure_running=ns["ensure_job_running"], set_focus=set_focus, target_codes=ns["get_target_codes"], max_targets=ns.get("MAX_TARGETS", 3),
                             busy=busy, lipsync_hash_ok=lipsync_hash_ok, job_active=job_active)

            async def request_language(job_id, lang, from_ms, generation):
                jd = ns["resolve_job"](job_id)
                try:
                    async with ns["get_lock"](f"job:{job_id}"):
                        r = ls.request_track(jd, lang, from_ms, generation, hooks=hooks)
                except ls.LangSchedError as e:
                    if e.code == "obsolete_generation":
                        return {"track": None, "obsolete": True}
                    raise CtlError(e.status, e.code, e.message)
                tr = ls.track_readiness(jd, lang, hooks)
                track = {k: tr.get(k) for k in ("lang", "ready_intervals", "audio_state", "lipsync_state")}
                if track["audio_state"] == "queued":          # contract enum: absent|preparing|partial|ready|error
                    track["audio_state"] = "preparing"
                return {"duplicate": not r.get("created", True), "track": track}
            async def language_status(job_id, lang):
                jd = ns["resolve_job"](job_id)
                tr = await asyncio.to_thread(ls.track_readiness, jd, lang, hooks)
                out = {k: tr.get(k) for k in ("lang", "ready_intervals", "audio_state", "lipsync_state")}
                if out["audio_state"] == "queued":
                    out["audio_state"] = "preparing"
                return out
            def release_language(job_id, lang, reason):
                jd = ns["resolve_job"](job_id)
                if jd is not None:
                    ls.release_track(jd, lang, reason)       # marker only; the cap frees when the job task is dead (hooks.job_active)
            ctl_hooks.release_language = release_language
            ctl_hooks.request_language = request_language
            ctl_hooks.language_status = language_status

            async def _addlang_recover():
                """server2's startup reconcile only resumes jobs that were queued/running; a finished job whose new language was merged
                just before a crash would stay un-dubbed. Roll every stranded (phase=scheduling) track forward."""
                try:
                    from sunwai.control.settings import addlang_enabled
                    if not addlang_enabled():          # kill switch: leave stranded intents untouched
                        return
                    acts = ls.recover_all(ns["DUB_ROOT"], hooks)
                    if acts:
                        log(f"[ctl] add-language recovery: {acts}")
                except Exception as e:
                    log(f"[ctl] add-language recovery failed: {e!r}")
            try:
                ns["app"].on_event("startup")(_addlang_recover)
            except Exception as e:
                log(f"[ctl] add-language recovery hook NOT installed: {e!r}")
            installed.append("language_scheduler")
    except ImportError:
        pass
    log(f"[ctl] adapters installed: {installed or 'none (stubs -> 501)'}")
    return installed
