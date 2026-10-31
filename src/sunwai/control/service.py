"""the single validated action service. Buttons, transcript edits and chat all call ActionService.submit()."""
import asyncio, contextlib, hashlib, hmac, logging, os, secrets, time, uuid
from pathlib import Path

from sunwai.control import hooks as ctl_hooks
from sunwai.control.hooks import NotImplementedHook
from sunwai.control.store import CtlError, canon, sha, utcnow, seg_key, store_for
from sunwai.control.tools import validate_args, BadArgs, CONFIRM_TOOLS, INTENTS

SERVER_TYPES = {"edit_line", "undo_line", "redo_line", "change_expression", "retime", "regenerate_chunk", "change_voice", "add_language", "report_issue"}
SEG_TYPES = {"edit_line", "undo_line", "redo_line", "change_expression", "retime"}
COMMIT_TYPES = {"edit_line", "undo_line", "redo_line", "add_language", "regenerate_chunk", "change_voice"}
PREVIEW_TYPES = {"change_expression", "retime"}


def _f(name, default):
    return float(os.environ.get(name, default))


def limits():
    return {"max_synth": int(_f("DUB_CTL_MAX_SYNTH", 1)), "previews_per_min": int(_f("DUB_CTL_PREVIEWS_PER_MIN", 5)),
            "spend_cap_usd": _f("DUB_CTL_SPEND_CAP_USD", 0.50), "synth_cap_s": _f("DUB_CTL_SYNTH_CAP_S", 600),
            "confirm_ttl_s": int(_f("DUB_CTL_CONFIRM_TTL", 300)), "max_edit_chars": int(_f("DUB_MAX_EDIT_CHARS", 600))}


class Confirmer:
    """Server-issued tokens bound to sha256(canonical{job,type,args,expected_revision}); TTL; consumed only when the confirmed work SUCCEEDS
    (in-use while running so it cannot be double-spent; released again on failure so a transient error can be retried)."""
    def __init__(self):
        self.secret = secrets.token_bytes(32)
        self.live = {}                 # token -> [bind, expiry, in_use]

    @staticmethod
    def bind(job_id, typ, args, expected):
        return sha(canon({"job": job_id, "type": typ, "args": args, "expected": expected}))

    def issue(self, job_id, typ, args, expected, ttl):
        b = self.bind(job_id, typ, args, expected)
        tok = hmac.new(self.secret, (b + secrets.token_hex(8)).encode(), hashlib.sha256).hexdigest()[:40]
        now = time.time()
        for k in [k for k, v in self.live.items() if v[1] < now]:
            self.live.pop(k, None)
        self.live[tok] = [b, now + ttl, False]
        return tok, now + ttl

    def hold(self, tok, job_id, typ, args, expected):
        ent = self.live.get(tok or "")
        if not ent or ent[1] < time.time() or ent[2] or not hmac.compare_digest(ent[0], self.bind(job_id, typ, args, expected)):
            raise CtlError(403, "bad_confirmation", "confirmation token missing, expired, in use, or not bound to this exact proposal")
        ent[2] = True

    def done(self, tok, ok):
        if not tok or tok not in self.live:
            return
        if ok:
            self.live.pop(tok, None)
        else:
            self.live[tok][2] = False


def addlang_enabled():
    """add-language is ON by default; DUB_CTL_ADDLANG=0/false/no/off turns it off (read at call time). An unset or blank var = on."""
    v = os.environ.get("DUB_CTL_ADDLANG")
    return True if v is None or not v.strip() else v.strip().lower() not in ("0", "false", "no", "off", "n", "f")


def _spoken(seg):
    """The actual spoken text: the expression service's tts_text when it belongs to the current display text, else the display text."""
    t = seg.get("tgt_text", "")
    return seg["tts_text"] if seg.get("tts_text") and seg.get("tts_src") == t else t


def _wait_worker(cfut):
    """Runs in an executor thread: blocks until the worker future finishes (never raises)."""
    try:
        cfut.result()
    except BaseException:
        pass


class _NullLock:
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class ActionService:
    def __init__(self, deps, guard=None):
        self.guard = guard
        self.d = deps                  # see ctl_api.ServerDeps / eval FakeDeps
        self.confirm = Confirmer()
        self.tasks = set()

    # -------- helpers
    def _ctx(self, job_id):
        jd = self.d.job_dir(job_id)
        if jd is None or not (jd / "job.json").is_file():
            raise CtlError(404, "unknown_job", "job not found")
        return jd, store_for(jd)

    def _seg(self, jd, job_id, lang, chunk, seg_id):
        if lang not in self.d.job_langs(jd):
            raise CtlError(404, "unknown_lang", "language is not part of this job")
        if int(chunk) not in self.d.chunk_ids(jd):
            raise CtlError(404, "unknown_chunk", "unknown chunk")
        segs, cstart = self.d.segments(jd, lang, int(chunk))
        for s in segs:
            if str(s.get("id")) == seg_id:
                return s, cstart
        raise CtlError(404, "unknown_segment", "segment not found")

    def head_view(self, job_id, lang, chunk, seg_id):
        """Reconcile the live segment into the revision log (rev 0 base, or a 'legacy' revision after an old-route edit)."""
        jd, st = self._ctx(job_id)
        seg, cstart = self._seg(jd, job_id, lang, chunk, seg_id)
        key = seg_key(lang, chunk, seg_id)
        text = seg.get("tgt_text", "")
        hl = len(seg["history"]) if isinstance(seg.get("history"), list) else 0
        with st.lock:
            h = st.head(key)
            if h is None or h["display_text"] != text or h.get("spoken_text") != _spoken(seg) or (h.get("live_hist") is not None and h["live_hist"] != hl):
                if h is not None:
                    # NEVER reconcile a divergence into a 'legacy' revision while (a) an op of ours has this line claimed, (b) an expression apply/undo holds the line right now, or (c) the live line equals
                    # the expression service's own head state. this race produced the spurious revision 3 and "committed but not recorded".
                    ls_ = ctl_hooks.live_state(jd, lang, chunk, seg_id)
                    if key in st.running or ls_.get("in_flight") or ls_.get("owned"):
                        return h, seg, cstart, st, key
                prov = "base" if h is None else "legacy"
                rev = {"job_id": job_id, "lang": lang, "chunk_id": int(chunk), "segment_id": seg_id,
                       "parent_revision": h["revision"] if h else None, "source_revision": 0,
                       "speaker_id": seg.get("speaker"), "voice_plan_id": None,
                       "slot_start_ms": int((cstart + seg.get("start", 0)) * 1000), "slot_end_ms": int((cstart + seg.get("end", 0)) * 1000),
                       "raw_source_text": seg.get("src_text", ""), "display_text": text, "spoken_text": _spoken(seg), "live_hist": hl,
                       "expression": {"intent": None, "emotion": None, "confidence": None, "provenance": prov},
                       "style": None, "tempo_factor": 1.0, "audio_hash": self.d.audio_hash(jd, lang, chunk, text),
                       "subtitle_revision": (h or {}).get("subtitle_revision", 0), "status": "active", "ts": utcnow()}
                h = st.add_revision(key, rev)
        return h, seg, cstart, st, key

    # -------- public
    async def submit(self, job_id, body):
        """Returns (http_status, json_body). Raises CtlError for 4xx/5xx. Everything up to the idempotency reservation is synchronous (atomic on the loop)."""
        jd, st = self._ctx(job_id)
        typ = body.get("type")
        if typ not in SERVER_TYPES:
            raise CtlError(422, "invalid_args", f"unknown action type {typ!r}")
        if typ == "add_language" and not addlang_enabled():
            raise CtlError(403, "feature_disabled", "Adding a language while watching is not available yet.")
        allowed_env = {"action_id", "expected_revision", "idempotency_key", "type", "args", "preview_only", "confirmation_token"}
        if set(body) - allowed_env:
            raise CtlError(422, "invalid_args", f"unexpected fields: {sorted(set(body) - allowed_env)}")
        args = dict(body.get("args") or {})
        if not isinstance(body.get("args", {}), dict):
            raise CtlError(422, "invalid_args", "args must be an object")
        apply_cand = args.pop("apply_candidate", None)
        if isinstance(apply_cand, str) and apply_cand.isdigit():        # tolerate numeric strings from older clients
            apply_cand = int(apply_cand)
        if apply_cand is not None and typ != "change_expression":
            raise CtlError(422, "invalid_args", "apply_candidate is only valid for change_expression")
        pgen = args.pop("playback_generation", None) if typ == "add_language" else None
        lang = chunk = None
        if typ in SEG_TYPES or typ == "regenerate_chunk":
            lang = args.pop("lang", None); chunk = args.pop("chunk", None)
            if isinstance(chunk, str) and chunk.isdigit():
                chunk = int(chunk)
            if typ == "regenerate_chunk":
                args = {"chunk_id": chunk}
        try:
            core = validate_args(typ, args)
            if apply_cand is not None and (isinstance(apply_cand, bool) or not isinstance(apply_cand, int) or not (0 <= apply_cand <= 10**6)):
                raise BadArgs("apply_candidate must be an integer revision (0..1000000)")
            if pgen is not None and (isinstance(pgen, bool) or not isinstance(pgen, int) or pgen < 0):
                raise BadArgs("playback_generation must be a non-negative integer")
            if typ in SEG_TYPES or typ == "regenerate_chunk":
                if not isinstance(lang, str) or not isinstance(chunk, int) or isinstance(chunk, bool) or not (0 <= chunk <= 9999):
                    raise BadArgs("args.lang (string) and args.chunk (int 0..9999) are required")
                from sunwai.control.tools import LANG_RE
                if not LANG_RE.match(lang):
                    raise BadArgs("invalid lang")
        except BadArgs as e:
            raise CtlError(422, "invalid_args", str(e))
        if typ == "add_language":
            lang = core["lang"]; core = {**core, "playback_generation": pgen or 0}
        idem = body.get("idempotency_key")
        if not isinstance(idem, str) or not (8 <= len(idem) <= 80):
            raise CtlError(422, "invalid_args", "idempotency_key (8..80 chars) is required")
        exp = body.get("expected_revision")
        preview_only = typ in PREVIEW_TYPES and apply_cand is None      # server-derived: a confirmed apply always applies; the caller's flag is ignored
        ph = sha(canon({"t": typ, "a": core if typ != "add_language" else {"lang": core["lang"]}, "l": lang, "c": chunk, "x": exp, "p": preview_only, "ac": apply_cand}))
        if typ == "add_language":           # idempotent per (job, lang): a FAILED earlier attempt may be retried under the same key
            old = st.actions.get(st.idem.get(idem, (None, None))[1]) if idem in st.idem else None
            oid = ((old or {}).get("response") or {}).get("operation") or {}
            if old and st.ops.get(oid.get("operation_id"), {}).get("state") == "failed":
                st.idem.pop(idem, None)
        dup = st.idem_lookup(idem, ph)
        if dup:
            return 200, {**dup["response"], "status": "duplicate"}
        h = key = None
        if typ in SEG_TYPES:
            if not isinstance(exp, int) or isinstance(exp, bool):
                raise CtlError(422, "invalid_args", "segment actions need args.lang, args.chunk and an integer expected_revision")
            h, seg, cstart, st, key = self.head_view(job_id, lang, int(chunk), core["segment_id"])
            if typ == "redo_line":
                src = next((r for r in st.all_revs(key) if r["revision"] == h.get("undo_of")), None) if (h.get("undo_of") is not None and (h.get("expression") or {}).get("provenance") == "undo_line") else None
                if src is None:
                    raise CtlError(409, "nothing_to_redo", "there is no undone edit to re-apply on this line")
            if h["revision"] != exp:
                raise CtlError(409, "stale_revision", "segment changed since you loaded it", expected=exp, current=h["revision"])
        L = limits()
        bind_args = {**core, "lang": lang, "chunk": chunk, "apply_candidate": apply_cand}      # the token binds the FULL payload incl. which candidate
        tok = body.get("confirmation_token")
        # SERVER-ENFORCED: every action that commits, spends GPU or forks needs a valid token no matter what the caller sends.
        # Only previews (change_expression / retime without apply_candidate) and report_issue run without one.
        need_confirm = typ in COMMIT_TYPES or (typ in PREVIEW_TYPES and apply_cand is not None)
        if need_confirm:
            if not tok:
                t, expiry = self.confirm.issue(job_id, typ, bind_args, exp, L["confirm_ttl_s"])
                kind = "fork_proposal" if typ == "change_voice" else "needs_confirmation"
                prop = {"type": typ, "args": bind_args, "expected_revision": exp, "expires_at": expiry, "summary": self._summary(typ, bind_args)}
                if typ == "change_voice":
                    prop["fork"] = {"from_job": job_id, "speaker_id": core["speaker_id"], "voice": core["voice"],
                                    "note": "voice plans are immutable; confirming creates a NEW job with a copied, modified plan"}
                return 200, {"status": kind, "confirmation_token": t, "proposal": prop}
            self.confirm.hold(tok, job_id, typ, bind_args, exp)
        else:
            tok = None
        aid = "act_" + uuid.uuid4().hex[:12]                  # server-generated only: a caller-supplied action_id is ignored
        claimed = []
        try:
            need = 0.0
            if typ in SEG_TYPES:
                need = max(0.0, (h["slot_end_ms"] - h["slot_start_ms"]) / 1000.0)
            elif typ == "regenerate_chunk":
                need = 30.0
            elif typ == "add_language":
                need = float(os.environ.get("DUB_CTL_ADDLANG_S", "120"))         # GPU work: counted against job + daily synthesis budgets
            if need:
                st.check_caps(L["spend_cap_usd"], L["synth_cap_s"], need)
                if self.guard:
                    self.guard.check_daily(need_synth_s=need)
                if typ in PREVIEW_TYPES and apply_cand is None:
                    st.check_preview_rate(L["previews_per_min"])
                ckey = key if typ in SEG_TYPES else (f"lang/{core['lang']}" if typ == "add_language" else f"chunk/{chunk}")
                self._claim_synth(st, job_id, ckey, L)
                claimed.append(ckey)
            prev, inprog = st.idem_reserve(idem, ph, aid)
            if inprog:
                for k in claimed: self._unclaim(st, k)
                claimed = []
                if tok: self.confirm.done(tok, False)
                return 200, {**(prev["response"] if prev else {"action_id": inprog}), "status": "duplicate"}
            if need:                                           # charge the attempt now: failed and retried attempts cost too
                st.add_spend(synth_s=need)
                if self.guard: self.guard.charge(synth_s=need)
            return await self._dispatch(jd, st, job_id, aid, idem, ph, typ, core, lang, chunk, exp, preview_only, apply_cand, h, key, tok)
        except BaseException:
            for k in claimed: self._unclaim(st, k)
            st.idem_release(idem, aid)
            if tok: self.confirm.done(tok, False)
            raise

    def _claim_synth(self, st, job_id, key, L):
        with st.lock:
            if key in st.running:
                raise CtlError(409, "busy", "this line already has an operation running")
            if len(st.running) >= L["max_synth"]:
                raise CtlError(429, "one_synthesis_per_job", "another synthesis is running for this job; try again when it finishes", retry_after=5)
            if self.guard:
                self.guard.synth_acquire()
            st.running[key] = "pending"

    def _unclaim(self, st, key):
        with st.lock:
            if st.running.pop(key, None) is not None and self.guard:
                self.guard.synth_release()

    def _summary(self, typ, a):
        if typ == "regenerate_chunk":
            return f"Regenerate chunk {a.get('chunk')} in {a.get('lang')}: re-dubs the whole chunk (takes minutes) and replaces its audio (undoable)"
        if typ == "change_voice":
            return f"Create a NEW job with speaker {a['speaker_id']} voiced as {a['voice']}; this job is unchanged"
        if typ == "add_language":
            return f"Start dubbing {a['lang']} from {a['from_ms']} ms (uses GPU time; this job already has two or more languages)"
        if typ == "edit_line":
            return f"Replace line {a['segment_id']} (chunk {a.get('chunk')}) with: {a['text'][:120]}"
        if typ == "undo_line":
            return f"Undo the last edit of line {a['segment_id']} (chunk {a.get('chunk')})"
        if typ == "redo_line":
            return f"Re-apply the edit of line {a['segment_id']} (chunk {a.get('chunk')}) that was just undone"
        return typ

    async def _dispatch(self, jd, st, job_id, aid, idem, ph, typ, core, lang, chunk, exp, preview_only, apply_cand, h, key, tok):
        rec = {"action_id": aid, "job_id": job_id, "idempotency_key": idem, "payload_hash": ph, "type": typ,
               "args": {**core, "lang": lang, "chunk": chunk}, "expected_revision": exp, "preview_only": preview_only, "ts": utcnow()}
        if typ == "report_issue":
            with open(st.dir / "issues.jsonl", "a", encoding="utf-8") as f:
                f.write(canon({"ts": utcnow(), "action_id": aid, "details": core["details"]}) + "\n")
            resp = {"status": "accepted", "action_id": aid, "operation": None, "revision": None}
            rec["response"] = resp; st.add_action(rec)
            return 200, resp
        if typ == "add_language":
            op = {"operation_id": "op_" + uuid.uuid4().hex[:12], "action_id": aid, "state": "running", "stage": "queued",
                  "revision": None, "audio_ready": False, "lipsync_ready": False, "error": None}
            st.put_op(op)
            lkey = f"lang/{core['lang']}"
            try:
                res = await self.request_language(job_id, core["lang"], core["from_ms"], int(core.get("playback_generation", 0)))
            except BaseException:
                op.update(state="failed", stage="error", error="language request failed"); st.put_op(op)
                self._unclaim(st, lkey)
                raise
            gen = int(core.get("playback_generation", 0))
            if res.get("obsolete"):
                if tok: self.confirm.done(tok, True)
                self._unclaim(st, lkey)
                op.update(state="done", stage="complete"); st.put_op(op)
                resp = {"status": "obsolete", "action_id": aid, "operation": op, "revision": None, "track": None, "generation": gen}
                rec["response"] = resp; st.add_action(rec)
                return 200, resp
            limit = float(os.environ.get("DUB_CTL_LANG_MAX", "900"))
            op.update(state="running", stage="preparing", kind="add_language", lang=core["lang"], deadline=time.time() + limit); st.put_op(op)       # the slot stays CLAIMED until ready / failed / deadline
            resp = {"status": "accepted", "action_id": aid, "operation": op, "revision": None, "track": res.get("track"), "generation": gen}
            rec["response"] = resp; st.add_action(rec)
            t = asyncio.create_task(self._watch_track(st, op, job_id, core["lang"], lkey, tok))
            self.tasks.add(t); t.add_done_callback(self.tasks.discard)
            return 202, resp
        if typ == "regenerate_chunk":
            self._unclaim(st, f"chunk/{chunk}")
            raise CtlError(501, "not_implemented", "Regenerating a whole chunk is not available yet. You can edit or re-express individual lines instead.")
        if typ == "change_voice":
            raise CtlError(501, "not_implemented", "Voice changes create a new job; confirming is not wired to job creation yet.")
        op = {"operation_id": "op_" + uuid.uuid4().hex[:12], "action_id": aid, "state": "queued", "stage": "queued",
              "revision": None, "audio_ready": False, "lipsync_ready": False, "error": None}
        st.running[key] = op["operation_id"]
        st.put_op(op)
        st.emit("op.queued", job_id=job_id, operation_id=op["operation_id"], action_id=aid, action_type=typ, lang=lang, chunk=int(chunk), segment_id=core["segment_id"],
                head_revision=h["revision"] if h else None)
        resp = {"status": "accepted", "action_id": aid, "operation": op, "revision": None, "head_revision": h["revision"] if h else None,
                "lang": lang, "chunk": int(chunk), "segment_id": core["segment_id"]}
        rec["response"] = resp; st.add_action(rec)
        t = asyncio.create_task(self._run(jd, st, job_id, op, typ, core, lang, int(chunk), exp, preview_only, apply_cand, h, key, tok))
        self.tasks.add(t); t.add_done_callback(self.tasks.discard)
        return 202, resp

    @staticmethod
    def _safe_put(st, op):
        """Persist the op; retry once. Returns True when persisted."""
        for _ in range(2):
            try:
                st.put_op(op); return True
            except Exception:
                time.sleep(0.01)
        return False

    def _terminal(self, st, op, ok, err, job_id, events=()):
        """Drive the op to a TERMINAL persisted state even if persistence/event writes misbehave: persist (retry once), then best-effort events."""
        if ok:
            op.update(state="done", stage="complete", error=None)
        else:
            op.update(state="failed", stage="error", error=err)
        persisted = self._safe_put(st, op)
        if not persisted and ok:                                    # could not persist success: record the failure instead (also retried once)
            op.update(state="failed", stage="error", error="completed but the completion record could not be written")
            persisted = self._safe_put(st, op); ok = False
        for typ, kw in (list(events) if ok else [("op.error", {"operation_id": op["operation_id"], "action_id": op["action_id"], "error": op.get("error")})]):
            try:
                st.emit(typ, job_id=job_id, **kw)
            except Exception:
                pass
        return ok

    async def _watch_track(self, st, op, job_id, lang, lkey, tok=None, deadline=None):
        """Hold the synthesis slot while the language is being dubbed. Done when track_readiness says ready (language.ready + complete),
        failed when it says error / the status call keeps failing, or released at the op's deadline (default 15 min from the request;
        after a restart the ORIGINAL deadline is kept)."""
        limit = float(os.environ.get("DUB_CTL_LANG_MAX", "900")); poll = float(os.environ.get("DUB_CTL_LANG_POLL", "10"))
        t0 = time.time()
        deadline = deadline or op.get("deadline") or (t0 + limit)
        max_err = int(os.environ.get("DUB_CTL_LANG_MAX_ERRS", "5")); max_err_s = float(os.environ.get("DUB_CTL_LANG_MAX_ERR_S", "60"))
        errs = 0; first_err = None
        ok = False
        try:
            while time.time() < deadline:
                try:
                    tr = await ctl_hooks.language_status(job_id, lang)
                    errs = 0; first_err = None
                except NotImplementedHook as e:
                    self._terminal(st, op, False, f"not_implemented: {e}", job_id); return
                except Exception as e:
                    errs += 1; first_err = first_err or time.time()
                    if errs >= max_err or time.time() - first_err >= max_err_s:
                        self._terminal(st, op, False, f"language status unavailable ({errs} consecutive errors: {type(e).__name__})", job_id); return
                    await asyncio.sleep(poll); continue
                if tr.get("audio_state") == "ready":
                    op.update(audio_ready=True, lipsync_ready=tr.get("lipsync_state") == "ready")
                    ok = self._terminal(st, op, True, None, job_id, [
                        ("language.ready", {"lang": lang, "ready_intervals": tr.get("ready_intervals"), "audio_state": "ready", "lipsync_state": tr.get("lipsync_state")}),
                        ("op.done", {"operation_id": op["operation_id"], "action_id": op["action_id"], "elapsed_s": round(time.time() - t0, 2)})])
                    return
                if tr.get("audio_state") == "error":
                    self._terminal(st, op, False, "language preparation failed", job_id); return
                await asyncio.sleep(poll)
            self._terminal(st, op, False, f"timed out waiting for the track (deadline passed; it may still finish in the background)", job_id)
        except asyncio.CancelledError:
            self._terminal(st, op, False, "cancelled", job_id); raise
        except BaseException as e:
            self._terminal(st, op, False, f"watcher crashed: {type(e).__name__}", job_id)
        finally:
            self._unclaim(st, lkey)
            if not ok:
                self._release_lang(job_id, lang, op.get("error") or "operation did not complete")
            if tok:
                self.confirm.done(tok, ok)          # consumed only on terminal SUCCESS; a terminal failure releases it so the same confirmation can retry

    @staticmethod
    def _release_lang(job_id, lang, reason):
        """durable abandonment marker for a failed/cancelled add-language op (the adapter refuses while the language still has queued/running chunks)."""
        try:
            fn = getattr(ctl_hooks, "release_language", None)
            if fn:
                fn(job_id, lang, str(reason))
        except Exception:
            pass

    def resume_all(self, root):
        """Called at startup (inside the loop): re-claim the slots of add_language ops that survived a restart and re-attach their watchers
        with the remaining time of the ORIGINAL deadline. Over-cap / unclaimable ones are failed (terminal)."""
        n = 0
        for d in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
            if not (d / "ctl" / "operations.jsonl").is_file():
                continue
            st = store_for(d)
            for op in list(st.resumable):
                st.resumable.remove(op)
                lkey = f"lang/{op.get('lang')}"
                if not addlang_enabled():
                    self._terminal(st, op, False, "add-language is disabled on this server", d.name); continue
                if float(op.get("deadline") or 0) <= time.time():        # check BEFORE claiming anything
                    self._terminal(st, op, False, "language deadline passed while the server was down", d.name); self._release_lang(d.name, op.get("lang"), "deadline passed while down"); continue
                if len(st.running) >= limits()["max_synth"]:              # per-job cap
                    self._terminal(st, op, False, "job already has a running synthesis; not resuming", d.name); self._release_lang(d.name, op.get("lang"), "not resumed: running synthesis"); continue
                try:
                    with st.lock:
                        if lkey in st.running:
                            raise CtlError(409, "busy", "slot taken")
                        if self.guard:
                            self.guard.synth_acquire()
                        st.running[lkey] = op["operation_id"]
                except CtlError:
                    self._terminal(st, op, False, "could not re-claim the synthesis slot after restart", d.name); self._release_lang(d.name, op.get("lang"), "could not re-claim slot")
                    continue
                t = asyncio.create_task(self._watch_track(st, op, d.name, op["lang"], lkey, None, float(op["deadline"])))
                self.tasks.add(t); t.add_done_callback(self.tasks.discard)
                n += 1
        return n

    async def request_language(self, job_id, lang, from_ms, gen):
        jd, st = self._ctx(job_id)
        with st.lock:
            if gen < st.lang_gen.get(lang, 0):
                return {"track": None, "obsolete": True}
            st.lang_gen[lang] = gen
        try:
            res = await ctl_hooks.request_language(job_id, lang, from_ms, gen)
        except NotImplementedHook as e:
            raise CtlError(501, "not_implemented", str(e))
        st.emit("language.requested", job_id=job_id, lang=lang, from_ms=from_ms, generation=gen)
        return res

    def _stage(self, st, op, stage, **kw):
        op.update(stage=stage, state="running" if stage not in ("complete", "error") else ("done" if stage == "complete" else "failed"), **kw)
        st.put_op(op)

    @staticmethod
    def _m2_keep(h, out):
        return {"m2_applied": True, "m2_revision": out.get("m2_revision"), "chunk_audio_hash": out.get("chunk_audio_hash"), "m2_candidate_id": h.get("m2_candidate_id")}

    async def _run(self, jd, st, job_id, op, typ, core, lang, chunk, exp, preview_only, apply_cand, h, key, tok):
        cf = {"v": False, "recorded": False}
        await self._run_inner(jd, st, job_id, op, typ, core, lang, chunk, exp, preview_only, apply_cand, h, key, tok, cf)
        if cf["v"]:
            raise asyncio.CancelledError()          # cancellation is re-raised only AFTER the expression result was awaited and recorded

    async def _run_inner(self, jd, st, job_id, op, typ, core, lang, chunk, exp, preview_only, apply_cand, h, key, tok, cf):
        seg_id = core["segment_id"]
        t0 = time.time()
        ok = False
        released = False
        stack = contextlib.AsyncExitStack()        # holds the shared chunk lock across the expression call AND our revision recording
        try:
            self._stage(st, op, "synthesizing")
            st.emit("op.stage", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], stage="synthesizing")
            expect = {"text": h["display_text"], "hist": h.get("live_hist")}
            extra = {}; keep = {}
            if typ in ("edit_line", "undo_line", "redo_line"):
                if typ == "redo_line":
                    src = next(r for r in st.all_revs(key) if r["revision"] == h["undo_of"])
                if typ in ("undo_line", "redo_line") and h.get("m2_applied") and (typ == "undo_line" or h.get("undo_of") is not None):
                    await self._lock_chunk(stack, job_id, lang, chunk)     # a legacy edit can never land between the expression service's commit and our record
                    out = await self._m2_call("expression_undo_start", st, op, job_id, cf, jd, lang, chunk, seg_id)      # expression history: undo of an undo is its redo
                    if not out.get("ok"):
                        raise RuntimeError(out.get("error") or "undo failed")
                    segs_now, _cs = self.d.segments(jd, lang, chunk)
                    res = {"segment": next((x for x in segs_now if str(x.get("id")) == seg_id), {})}
                    keep = self._m2_keep(h, out); keep["audio_hash"] = out.get("audio_hash") or h["audio_hash"]
                    extra = {"undo_of": h["revision"]} if typ == "undo_line" else {"redo_of": h.get("undo_of")}
                elif typ == "redo_line":
                    res = await self.d.edit(job_id, lang, chunk, seg_id, src["display_text"], expect)
                    extra = {"redo_of": src["revision"]}
                elif typ == "undo_line":
                    res = await self.d.undo(job_id, lang, chunk, seg_id, expect)
                    extra = {"undo_of": h["revision"]}
                else:
                    res = await self.d.edit(job_id, lang, chunk, seg_id, core["text"], expect)
                seg_after = res.get("segment") or {}
                newtext = seg_after.get("tgt_text", core.get("text", h["display_text"]))
                rev = dict(h)
                for k_ in ("m2_applied", "m2_candidate_id", "m2_revision", "chunk_audio_hash"):
                    rev.pop(k_, None)
                reuse = next((r for r in reversed(st.all_revs(key)) if r["display_text"] == newtext and r["revision"] != h["revision"] and r["status"] in ("active", "superseded")), None)
                rev.update(parent_revision=h["revision"], display_text=newtext, spoken_text=(_spoken(seg_after) if keep and seg_after else newtext), status="active", ts=utcnow(),
                           live_hist=len(seg_after["history"]) if isinstance(seg_after.get("history"), list) else None,
                           audio_hash=(reuse["audio_hash"] if (typ == "undo_line" and reuse) else self.d.audio_hash(jd, lang, chunk, newtext)),
                           expression={"intent": None, "emotion": None, "confidence": None, "provenance": typ},
                           subtitle_revision=h.get("subtitle_revision", 0) + 1)
                rev["undo_of"] = None; rev.update(extra); rev.update(keep)
                try:
                    rv = st.add_revision(key, rev, expect_head=exp, check=True)
                except BaseException as e:           # host/expression already committed: never leave it orphaned
                    self._reconcile(st, op, job_id, f"committed but not recorded ({type(e).__name__})")
                    return
                cf["recorded"] = True
                self._finish(st, op, rv, t0, job_id, jd, lang, chunk, undo=(typ == "undo_line"))
                ok = True
                return
            if apply_cand is not None and not preview_only:
                crev = next((r for r in st.all_revs(key) if r["revision"] == apply_cand and r["status"] == "candidate"), None)
                if crev is None or crev["parent_revision"] != exp:
                    raise CtlError(409, "stale_revision", "candidate is no longer based on the active revision", expected=exp, current=h["revision"])
                await self._lock_chunk(stack, job_id, lang, chunk)
                out = await self._m2_call("expression_apply_start", st, op, job_id, cf, jd, lang, chunk, seg_id, crev)
                if not out.get("ok"):
                    raise RuntimeError(out.get("error") or "apply failed")
                segs_now, _cs = self.d.segments(jd, lang, chunk)
                seg_now = next((x for x in segs_now if str(x.get("id")) == seg_id), {})
                now_txt = seg_now.get("tgt_text", crev["display_text"])
                new = dict(crev); new.pop("undo_of", None); new.pop("redo_of", None)
                new.update(undo_of=None, parent_revision=h["revision"], status="active", audio_hash=out.get("audio_hash", crev["audio_hash"]),
                                             display_text=now_txt, spoken_text=_spoken(seg_now) if seg_now else crev["spoken_text"], m2_applied=True, m2_revision=out.get("m2_revision"),
                                             live_hist=len(seg_now["history"]) if isinstance(seg_now.get("history"), list) else None,
                                             chunk_audio_hash=out.get("chunk_audio_hash"), ts=utcnow())
                try:
                    rv = st.add_revision(key, new, expect_head=exp, check=True)
                    st.set_status(key, crev["revision"], "superseded")
                except BaseException as e:
                    self._reconcile(st, op, job_id, f"applied but not recorded ({type(e).__name__})")
                    return
                cf["recorded"] = True
                self._finish(st, op, rv, t0, job_id, jd, lang, chunk)
                ok = True
                return
            intent = core.get("intent"); tempo = core.get("tempo_factor")
            out = await ctl_hooks.expression_resynth(jd, lang, chunk, seg_id, intent, h["display_text"], tempo, st.dir / "candidates")
            if not out.get("ok"):
                raise RuntimeError(out.get("error") or "resynth failed")
            rev = dict(h)
            for k_ in ("m2_applied", "undo_of", "redo_of", "m2_revision", "chunk_audio_hash"):
                rev.pop(k_, None)                      # a candidate never inherits undo/redo bookkeeping from the head it was derived from
            rev.update({k: v for k, v in out.items() if k.startswith("m2_")})
            rev.update(parent_revision=h["revision"], status="candidate", ts=utcnow(),
                       spoken_text=out.get("spoken_text", h["spoken_text"]), style=out.get("style"),
                       tempo_factor=out.get("tempo_factor", tempo or 1.0), audio_hash=out["audio_hash"],
                       expression={"intent": intent, "emotion": None, "confidence": None, "provenance": "user_request"})
            rv = st.add_revision(key, rev)
            self._finish(st, op, rv, t0, job_id, jd, lang, chunk, candidate=True)
            ok = True
        except NotImplementedHook as e:
            self._fail(st, op, f"not_implemented: {e}", job_id)
        except CtlError as e:
            self._fail(st, op, f"{e.code}: {e.detail}", job_id)
        except asyncio.CancelledError:
            cf["v"] = True                       #
            if not cf["recorded"]:
                self._fail(st, op, "cancelled", job_id)
        except BaseException as e:      # incl. HTTPException from the legacy flow (state is rolled back by _edit_flow)
            self._fail(st, op, f"{type(e).__name__}: {getattr(e, 'detail', e)}"[:300], job_id)
        finally:
            try:
                await stack.aclose()
            finally:
                if not released:
                    self._unclaim(st, key)
                if tok:
                    self.confirm.done(tok, ok)

    async def _lock_chunk(self, stack, job_id, lang, chunk):
        """Acquire the shared chunk lock, but never wait forever: ~20 s then `busy`."""
        wait = float(os.environ.get("DUB_CTL_LOCK_WAIT_S", "20"))
        try:
            await asyncio.wait_for(stack.enter_async_context(self._chunk_lock(job_id, lang, chunk)), wait)
        except asyncio.TimeoutError:
            raise CtlError(409, "busy", "this chunk is being modified by another operation; try again in a moment")

    @staticmethod
    async def _drain(cfut):
        """Wait for the REAL worker (a concurrent.futures.Future) to finish, no matter how many times this task or any wrapper future is
        cancelled. NEVER blocks the event loop: a non-blocking poll of cfut.done() every 0.1 s; every CancelledError is remembered (not
        propagated) and the wait continues. Returns only when cfut.done(). -> (result_dict, was_cancelled); the caller re-raises the cancel
        after recording and unlocking."""
        was = False
        while not cfut.done():
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                was = True
        try:
            return cfut.result(), was
        except BaseException as e:
            return {"ok": False, "error": f"worker raised {type(e).__name__}: {e}"}, was

    async def _m2_call(self, hook_name, st, op, job_id, cf, *args):
        """Run an expression mutation (apply/undo) WITH the chunk lock held by the caller. The worker is a concurrent.futures.Future; the lock is only
        released and the outcome only recorded after cfut.done(): cancellation is deferred (cf['v']), a timeout marks the op `stuck` and keeps waiting."""
        timeout = float(os.environ.get("DUB_CTL_M2_TIMEOUT_S", "180"))
        starter = getattr(ctl_hooks, hook_name)
        cfut = starter(*args)
        try:
            return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(cfut)), timeout)
        except asyncio.TimeoutError:
            msg = f"[ctl][STUCK] {hook_name} on job {job_id} exceeded {timeout:.0f}s; keeping the chunk lock until the worker ends"
            print(msg, flush=True); logging.getLogger("ctl").error(msg)
            try:
                op.update; self._stage(st, op, "stuck")
                st.emit("op.stuck", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], after_s=timeout)
            except Exception:
                pass
            out, was = await self._drain(cfut)
            cf["v"] = cf["v"] or was
            if isinstance(out, dict): out = {**out, "stuck": True}
            op.update(error=None)
            return out
        except asyncio.CancelledError:
            out, _was = await self._drain(cfut)        # wrapper / task / shutdown cancelled: the worker thread still owns live files
            cf["v"] = True
            return out
        except BaseException as e:                     # the worker itself raised
            if not cfut.done():
                out, _ = await self._drain(cfut)
                return out
            return {"ok": False, "error": f"worker raised {type(e).__name__}: {e}"}

    def _chunk_lock(self, job_id, lang, chunk):
        f = getattr(self.d, "chunk_lock", None)
        return f(job_id, lang, chunk) if f else _NullLock()

    def _reconcile(self, st, op, job_id, msg):
        try:
            with open(st.dir / "reconcile.jsonl", "a", encoding="utf-8") as f:
                f.write(canon({"ts": utcnow(), "operation_id": op["operation_id"], "action_id": op["action_id"], "note": msg}) + "\n")
        except OSError:
            pass
        op.update(error="reconcile: " + msg)
        self._stage(st, op, "error")
        try:
            st.emit("op.error", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], error=op["error"], reconcile=True)
        except Exception:
            pass

    def _finish(self, st, op, rv, t0, job_id, jd, lang, chunk, candidate=False, undo=False):
        """Audio is ready. lipsync_ready is NEVER set here: a watcher sets it from the real lip-sync file."""
        op.update(revision=rv["revision"], audio_ready=True, lipsync_ready=False)
        self._stage(st, op, "audio_ready")
        key = seg_key(rv["lang"], rv["chunk_id"], rv["segment_id"])
        head = st.head(key)
        base = f"/api/jobs/{job_id}/ctl/media/%s/{rv['lang']}/{rv['chunk_id']}/{rv['segment_id']}/{rv['revision']}"
        st.emit("op.audio_ready", operation_id=op["operation_id"], action_id=op["action_id"], lang=rv["lang"], chunk=rv["chunk_id"], segment_id=rv["segment_id"],
                revision=rv["revision"], candidate=candidate, candidate_revision=rv["revision"] if candidate else None,
                head_revision=head["revision"] if head else None, audio_hash=rv["audio_hash"],
                audio_url=base % ("cand" if candidate else "base"), preview_url=(base % "cand") if candidate else None,
                base_audio_url=(f"/api/jobs/{job_id}/ctl/media/base/{rv['lang']}/{rv['chunk_id']}/{rv['segment_id']}/{head['revision'] if head else 0}"))
        if not candidate:
            st.emit("revision.undone" if undo else "revision.committed", job_id=job_id, revision=rv, head_revision=rv["revision"])
            st.emit("subtitle.revision", job_id=job_id, lang=rv["lang"], chunk=rv["chunk_id"], segment_id=rv["segment_id"], subtitle_revision=rv["subtitle_revision"])
            self._stage(st, op, "lip_syncing")
            t = asyncio.create_task(self._watch_lipsync(st, op, job_id, jd, lang, chunk, rv, t0))
            self.tasks.add(t); t.add_done_callback(self.tasks.discard)
        else:
            self._stage(st, op, "complete")      # a candidate is a preview: no lip-sync is attempted
            st.emit("op.done", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], revision=rv["revision"], elapsed_s=round(time.time() - t0, 2))

    async def _watch_lipsync(self, st, op, job_id, jd, lang, chunk, rv, t0):
        """lipsync_worker3 already re-lips any chunk whose dub_<lang>.mp4 is newer than its dub_<lang>_lipsync.mp4 (3 s settle; needs dub.wav, segments.json, src.mp4), so an Apply/edit that remixes the dub is picked up automatically: nothing to trigger, we only watch.
        Lip-sync is ready only when dub_<lang>_lipsync.mp4 exists, is non-empty and is NEWER than the dub."""
        limit = float(os.environ.get("DUB_CTL_LS_WAIT", "900")); poll = float(os.environ.get("DUB_CTL_LS_POLL", "5"))
        t1 = time.time()
        check = getattr(self.d, "lipsync_ready", None)
        while check and time.time() - t1 < limit:
            try:
                if check(jd, lang, chunk, rv):
                    op.update(lipsync_ready=True); self._stage(st, op, "complete")
                    st.emit("op.lipsync_ready", job_id=job_id, operation_id=op["operation_id"], revision=rv["revision"])
                    st.emit("op.done", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], revision=rv["revision"], elapsed_s=round(time.time() - t0, 2))
                    return
            except Exception:
                pass
            await asyncio.sleep(poll)
        op.update(lipsync_ready=False)
        op.update(stage="audio_ready", state="done"); st.put_op(op)         # audio is live; mouth movements still the original frames
        st.emit("op.lipsync_pending", job_id=job_id, operation_id=op["operation_id"], revision=rv["revision"])
        st.emit("op.done", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], revision=rv["revision"], elapsed_s=round(time.time() - t0, 2), lipsync_ready=False)

    def _fail(self, st, op, err, job_id):
        op.update(error=err)
        self._stage(st, op, "error")
        st.emit("op.error", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], error=err)
