"""routes. Registered from server2.py by: `import ctl_api; ctl_api.register(app, globals())` behind DUB_ACTIONS=1."""
import asyncio, json, os, re, time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sunwai.control import hooks as ctl_hooks, tools as ctl_tools
from sunwai.control.chat import ChatRouter
from sunwai.control.service import ActionService, limits, addlang_enabled
from sunwai.control.store import CtlError, canon, sha, store_for, utcnow, seg_key

from sunwai.control.guard import JOB_RE, LANG_RE, SEG_RE, OP_RE


class ServerDeps:
    """Thin adapters over server2 globals (ns = server2 globals()). Nothing in server2 is modified."""
    def __init__(self, ns):
        self.ns = ns
        self.dry = os.environ.get("DUB_CTL_DRYRUN", "0") == "1"

    def chunk_lock(self, job_id, lang, chunk):
        return self.ns["get_lock"](f"{job_id}:{lang}:{int(chunk)}")

    def job_dir(self, job_id):
        return self.ns["resolve_job"](job_id)

    def job_langs(self, jd):
        return list(self.ns["read_json"](jd / "job.json").get("langs", []))

    def chunk_ids(self, jd):
        return {int(c["idx"]) for c in self.ns["read_json"](jd / "plan.json").get("chunks", [])}

    def _cdir(self, jd, lang, chunk):
        return self.ns["resolve_chunk_dir"](jd, lang, int(chunk))

    def segments(self, jd, lang, chunk):
        plan = self.ns["read_json"](jd / "plan.json")
        cd = next((c for c in plan.get("chunks", []) if int(c["idx"]) == int(chunk)), None)
        od = self._cdir(jd, lang, chunk)
        if cd is None or od is None or not od.exists():
            return [], 0.0
        return self.ns["_load_segments"](od), float(cd["start"])

    def all_segments(self, jd, lang):
        out = []
        for c in self.ns["read_json"](jd / "plan.json").get("chunks", []):
            segs, cs = self.segments(jd, lang, c["idx"])
            for s in segs:
                out.append({"chunk": int(c["idx"]), "id": str(s.get("id")), "start": cs + s.get("start", 0), "end": cs + s.get("end", 0),
                            "speaker": s.get("speaker"), "tgt_text": s.get("tgt_text"), "src_text": s.get("src_text")})
        return out

    def audio_hash(self, jd, lang, chunk, text):
        od = self._cdir(jd, lang, chunk)
        v = 0
        if od is not None:
            try:
                v = int(self.ns["read_json"](self.ns["_status_path"](od)).get("version", 0) or 0)
            except Exception:
                pass
        return "v1:" + sha(f"{v}|{text}")[:24]       # proxy until expression publishes real chunk-audio hashes

    @staticmethod
    def _cas(seg, expect):
        """Runs INSIDE the legacy chunk lock before anything is touched: a mismatch raises and nothing is mutated."""
        from fastapi import HTTPException
        if expect is None:
            return
        hist = seg.get("history") if isinstance(seg.get("history"), list) else []
        if seg.get("tgt_text", "") != expect.get("text") or (expect.get("hist") is not None and len(hist) != expect["hist"]):
            raise HTTPException(409, "stale_revision: the line changed since the revision was read")

    async def edit(self, job_id, lang, chunk, seg_id, text, expect=None):
        def change(seg):
            self._cas(seg, expect)
            hist = seg.get("history") if isinstance(seg.get("history"), list) else []
            hist.append(seg.get("tgt_text", "")); seg["tgt_text"] = text; seg["history"] = hist; seg["edited"] = True
            return text
        if self.dry:
            return await self._dry_flow(job_id, lang, chunk, seg_id, change)
        return await self.ns["_edit_flow"](job_id, lang, chunk, seg_id, change)      # same lock/gate/rollback as the legacy endpoint

    async def undo(self, job_id, lang, chunk, seg_id, expect=None):
        from fastapi import HTTPException
        holder = {}
        def change(seg):
            self._cas(seg, expect)
            hist = seg.get("history")
            if not isinstance(hist, list) or not hist:
                raise HTTPException(409, "nothing to undo")
            item = hist.pop()
            new_text = (item.get("tgt_text") or item.get("text") or "") if isinstance(item, dict) else str(item)
            seg["tgt_text"] = new_text; seg["history"] = hist; seg["edited"] = bool(hist); holder["hist"] = hist
            return new_text
        def after(seg_after):
            seg_after["history"] = holder["hist"]; seg_after["edited"] = bool(holder["hist"])
        if self.dry:
            return await self._dry_flow(job_id, lang, chunk, seg_id, change)
        return await self.ns["_edit_flow"](job_id, lang, chunk, seg_id, change, after)

    async def _dry_flow(self, job_id, lang, chunk, seg_id, change):
        """DEV ONLY (DUB_CTL_DRYRUN=1): same make_change under the same chunk lock, no synthesis, on the COPIED segments.json."""
        from fastapi import HTTPException
        jd = self.job_dir(job_id); od = self._cdir(jd, lang, chunk)
        async with self.ns["get_lock"](f"{job_id}:{lang}:{chunk}"):
            await asyncio.sleep(float(os.environ.get("DUB_CTL_DRYRUN_DELAY", "0")))
            segs = self.ns["_load_segments"](od)
            i, seg = self.ns["_find_segment"](segs, seg_id)
            if seg is None:
                raise HTTPException(404, "segment not found")
            change(seg)
            self.ns["_save_segments"](od, segs)
            return {"ok": True, "segment": seg}

    def lipsync_ready(self, jd, lang, chunk, rv):
        """Real file evidence only: lip-synced mp4 exists, non-empty, and is NEWER than the dub mp4 (same rule as server2._chunk_sync)."""
        od = self._cdir(jd, lang, chunk)
        if od is None:
            return False
        dub, ls = od / f"dub_{lang}.mp4", od / f"dub_{lang}_lipsync.mp4"
        try:
            return ls.exists() and ls.stat().st_size > 0 and dub.exists() and ls.stat().st_mtime > dub.stat().st_mtime
        except OSError:
            return False


def _err(e: CtlError):
    h = {"Retry-After": str(e.extra["retry_after"])} if "retry_after" in e.extra else None
    body = {"detail": e.detail, "code": e.code, **{k: v for k, v in e.extra.items() if k != "retry_after"}}
    return JSONResponse(body, status_code=e.status, headers=h)


def sse(ev):
    return f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"


READ_RL = int(os.environ.get("DUB_CTL_IP_READ_PER_MIN", "120"))
POLL_RL = int(os.environ.get("DUB_CTL_IP_POLL_PER_MIN", "240"))      # its own bucket: 3 tabs polling at 1.5 s must not starve media/state reads
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Content-Encoding": "identity"}
SSE_PAD = ": " + "." * 2048 + "\n\n"          # >=2 KB comment first: proxies that buffer small bodies flush it
CTX_KEYS = {"lang", "chunk", "segment_id", "playhead_ms", "subtitle_mode", "playback_generation", "line_index"}


def validate_context(c):
    """Strict, structured: unknown keys / wrong types / out-of-range values -> CtlError 422 (never an exception inside the SSE generator)."""
    if not isinstance(c, dict):
        raise CtlError(422, "invalid_context", "context must be an object")
    extra = set(c) - CTX_KEYS
    if extra:
        raise CtlError(422, "invalid_context", f"unexpected context fields: {sorted(extra)}")
    out = {}
    def num(k, lo, hi, integer):
        v = c.get(k)
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lstrip("-").replace(".", "", 1).isdigit():       # tolerate numeric strings (dataset attributes)
            v = float(v) if not integer else (int(v) if v.strip().lstrip("-").isdigit() else v)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not float(v).is_integer()) or not (lo <= v <= hi) or v != v:
            raise CtlError(422, "invalid_context", f"context.{k} must be a {'integer' if integer else 'number'} in [{lo}, {hi}]")
        return int(v) if integer else float(v)
    if c.get("lang") is not None:
        if not isinstance(c["lang"], str) or not LANG_RE.match(c["lang"]):
            raise CtlError(422, "invalid_context", "context.lang invalid")
        out["lang"] = c["lang"]
    if c.get("segment_id") is not None:
        v = c["segment_id"]
        v = str(v) if isinstance(v, int) and not isinstance(v, bool) else v
        if not isinstance(v, str) or not SEG_RE.match(v):
            raise CtlError(422, "invalid_context", "context.segment_id invalid")
        out["segment_id"] = v
    for k, lo, hi, integer in (("chunk", 0, 9999, True), ("playhead_ms", 0, 10**9, False), ("playback_generation", 0, 10**9, True), ("line_index", 0, 100000, True)):
        v = num(k, lo, hi, integer)
        if v is not None:
            out[k] = v
    if c.get("subtitle_mode") is not None:
        if c["subtitle_mode"] not in ctl_tools.SUBMODES:
            raise CtlError(422, "invalid_context", "context.subtitle_mode invalid")
        out["subtitle_mode"] = c["subtitle_mode"]
    return out


def _membership(deps, jd, c):
    """lang / chunk / segment from the client must belong to THIS job before anything is planned."""
    langs = deps.job_langs(jd)
    if c.get("lang") is not None and c["lang"] not in langs:
        raise CtlError(422, "invalid_context", "context.lang is not part of this job")
    if c.get("chunk") is not None and c["chunk"] not in deps.chunk_ids(jd):
        raise CtlError(422, "invalid_context", "context.chunk is not part of this job")
    if c.get("segment_id") is not None:
        if c.get("chunk") is None:
            raise CtlError(422, "invalid_context", "context.segment_id needs context.chunk")
        lang = c.get("lang") or (langs[0] if langs else None)
        segs, _ = deps.segments(jd, lang, c["chunk"])
        if not any(str(x.get("id")) == c["segment_id"] for x in segs):
            raise CtlError(422, "invalid_context", "context.segment_id is not part of this job")


def make_router(svc, deps, guard):
    r = APIRouter()
    chat = ChatRouter(svc, deps)

    def pre(request, kind, per_min, job_id=None):
        guard.require(request)
        guard.ip_limit(request, kind, per_min)
        if job_id is not None and not JOB_RE.match(job_id or ""):
            raise CtlError(404, "unknown_job", "job not found")

    def ctx(job_id):
        if not JOB_RE.match(job_id or ""):
            raise CtlError(404, "unknown_job", "job not found")
        jd = deps.job_dir(job_id)
        if jd is None or not (jd / "job.json").is_file():
            raise CtlError(404, "unknown_job", "job not found")
        return jd, store_for(jd)

    async def body_of(request, limit=20000):
        raw = await request.body()
        if len(raw) > limit:
            raise CtlError(413, "too_large", "request too large")
        try:
            d = json.loads(raw or b"{}")
        except Exception:
            raise CtlError(422, "invalid_args", "body must be JSON")
        if not isinstance(d, dict):
            raise CtlError(422, "invalid_args", "body must be an object")
        return d

    @r.get("/api/ctl/config")
    async def cfg(request: Request):
        try:
            if not guard.authed(request):
                # the page probes this on load; an anonymous visitor gets "control plane off" (200), not a 401 console error.
                # Nothing else is disclosed (no tools/limits/hints). Still IP rate-limited; every other ctl endpoint keeps requiring the token.
                guard.ip_limit(request, "read", READ_RL)
                return {"enabled": False}
            pre(request, "read", READ_RL)
        except CtlError as e:
            return _err(e)
        av, un = chat.available()
        return {"enabled": True, "chat": os.environ.get("DUB_CHAT", "1") != "0", "limits": limits(), "tools": sorted(t_ for t_ in ctl_tools.TOOLS if t_ != "add_language" or addlang_enabled()),
                "unavailable": un, "dryrun": deps.dry, "addlang": addlang_enabled()}

    @r.post("/api/jobs/{job_id}/actions")
    async def post_action(job_id: str, request: Request):
        try:
            pre(request, "action", int(os.environ.get("DUB_CTL_IP_ACTIONS_PER_MIN", "30")), job_id)
            code, resp = await svc.submit(job_id, await body_of(request))
            return JSONResponse(resp, status_code=code)
        except CtlError as e:
            return _err(e)
        except ctl_hooks.NotImplementedHook as e:
            return _err(CtlError(501, "not_implemented", str(e)))

    @r.get("/api/jobs/{job_id}/actions/{action_id}")
    async def get_action(job_id: str, action_id: str, request: Request):
        try:
            pre(request, "read", READ_RL, job_id)
            if not OP_RE.match(action_id):
                raise CtlError(404, "unknown_action", "action not found")
            jd, st = ctx(job_id)
        except CtlError as e:
            return _err(e)
        a = st.actions.get(action_id)
        if not a:
            return _err(CtlError(404, "unknown_action", "action not found"))
        ops = [o for o in st.ops.values() if o["action_id"] == action_id]
        a = {k: v for k, v in a.items() if k not in ("payload_hash",)}
        return {"action": a, "operations": ops}

    @r.get("/api/jobs/{job_id}/operations/{operation_id}")
    async def get_op(job_id: str, operation_id: str, request: Request):
        try:
            pre(request, "read", READ_RL, job_id)
            if not OP_RE.match(operation_id):
                raise CtlError(404, "unknown_operation", "operation not found")
            jd, st = ctx(job_id)
        except CtlError as e:
            return _err(e)
        o = st.ops.get(operation_id)
        return o if o else _err(CtlError(404, "unknown_operation", "operation not found"))

    @r.get("/api/jobs/{job_id}/revisions")
    async def get_revs(job_id: str, request: Request, lang: str, chunk: int, segment: str):
        try:
            pre(request, "read", READ_RL, job_id)
            if not LANG_RE.match(lang) or not (0 <= chunk <= 9999) or not SEG_RE.match(segment):
                raise CtlError(422, "invalid_args", "bad lang/chunk/segment")
            jd, st = ctx(job_id)
            svc.head_view(job_id, lang, chunk, segment)
        except CtlError as e:
            return _err(e)
        revs = st.all_revs(seg_key(lang, chunk, segment))
        head = st.head(seg_key(lang, chunk, segment))
        return {"revisions": revs, "head_revision": head["revision"] if head else None,
                "candidate_revisions": [r["revision"] for r in revs if r.get("status") == "candidate"]}

    @r.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request, after: int = -1):
        try:
            pre(request, "sse", 20, job_id)
            jd, st = ctx(job_id)
            guard.sse_open(job_id)
        except CtlError as e:
            return _err(e)
        lei = request.headers.get("last-event-id")
        if after < 0:
            try:
                after = int(lei) if lei else 0
            except ValueError:
                after = 0
        after = max(0, min(after, 10**12))

        async def gen():
            try:
                yield SSE_PAD
                cur = after
                if cur > st.last_seq:                          # client is ahead of the log (e.g. job copy / reset)
                    yield f"id: {st.last_seq}\nevent: reset\ndata: {json.dumps({'type': 'reset', 'job_id': job_id, 'seq': st.last_seq})}\n\n"
                    cur = st.last_seq
                first = True
                while True:
                    waiter = st.waiter()
                    batch, reset = st.events_after(cur)
                    if reset and first:
                        yield f"id: {st.last_seq}\nevent: reset\ndata: {json.dumps({'type': 'reset', 'job_id': job_id, 'seq': st.last_seq})}\n\n"
                    first = False
                    for ev in batch:
                        yield sse(ev); cur = ev["seq"]
                    if await request.is_disconnected():
                        return
                    if not batch:
                        try:
                            await asyncio.wait_for(waiter.wait(), 10)
                        except asyncio.TimeoutError:
                            yield ": hb\n\n"
            finally:
                guard.sse_close(job_id)
        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    @r.get("/api/jobs/{job_id}/events/poll")
    async def events_poll(job_id: str, request: Request, after: int = 0):
        """Same events as /events, as JSON (for clients whose SSE is buffered by a proxy). Bounded: at most the last 100 events after `after`."""
        try:
            pre(request, "poll", POLL_RL, job_id)
            jd, st = ctx(job_id)
        except CtlError as e:
            return _err(e)
        after = max(0, min(after, 10**12))
        if after > st.last_seq:
            return {"events": [], "last_seq": st.last_seq, "reset": True}
        evs, reset = st.events_after(after, limit=100)
        return JSONResponse({"events": evs, "last_seq": st.last_seq, "reset": bool(reset)}, headers={"Cache-Control": "no-store"})

    @r.get("/api/jobs/{job_id}/ctl/media/{kind}/{lang}/{chunk}/{segment}/{rev}")
    async def media(job_id: str, kind: str, lang: str, chunk: int, segment: str, rev: int, request: Request):
        """A/B audio for the control plane. Token-checked, same-origin (cookie), job-scoped: ids are validated, the file must resolve inside the job's chunk dir."""
        from fastapi.responses import FileResponse
        try:
            pre(request, "read", READ_RL, job_id)
            if kind not in ("cand", "base") or not LANG_RE.match(lang) or not SEG_RE.match(segment) or not (0 <= chunk <= 9999) or not (0 <= rev <= 10**6):
                raise CtlError(404, "unknown_media", "media not found")
            jd, st = ctx(job_id)
            key = seg_key(lang, chunk, segment)
            rv = next((x for x in st.all_revs(key) if x["revision"] == rev), None)
            if rv is None:
                raise CtlError(404, "unknown_media", "media not found")
            try:
                hd_ = st.head(key)
                p = await ctl_hooks.media_path(jd, lang, chunk, segment, kind, rv.get("m2_candidate_id"), bool(hd_ and hd_["revision"] == rev), rv.get("m2_revision"))
            except ctl_hooks.NotImplementedHook:
                raise CtlError(404, "unknown_media", "media not found")
            except FileNotFoundError:
                raise CtlError(410 if kind == "base" else 404, "gone" if kind == "base" else "unknown_media", "that version of the audio is no longer available")
        except CtlError as e:
            return _err(e)
        return FileResponse(str(p), media_type="audio/wav", headers={"Cache-Control": "private, no-store"})

    @r.post("/api/jobs/{job_id}/chat")
    async def post_chat(job_id: str, request: Request):
        try:
            pre(request, "chat", int(os.environ.get("DUB_CTL_IP_CHAT_PER_MIN", "8")), job_id)
            jd, st = ctx(job_id)
            if os.environ.get("DUB_CHAT", "1") == "0":
                raise CtlError(503, "chat_disabled", "chat is disabled")
            b = await body_of(request, 12000)
            if set(b) - {"message", "history", "context", "tools"}:
                raise CtlError(422, "invalid_args", "unexpected fields")
            msg = b.get("message")
            if not isinstance(msg, str) or not msg.strip() or len(msg) > 1000:
                raise CtlError(422, "invalid_args", "message must be 1..1000 chars")
            context = validate_context(b.get("context") or {})
            _membership(deps, jd, context)
        except CtlError as e:
            return _err(e)

        async def gen():
            try:
                async for evn, data in chat.turn(job_id, msg, [], context, tools_on=not (deps.dry and b.get("tools") is False)):
                    yield f"event: {evn}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            except CtlError as e:
                yield f"event: error\ndata: {json.dumps({'code': e.code, 'detail': e.detail})}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'code': 'internal', 'detail': type(e).__name__})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    @r.post("/api/jobs/{job_id}/languages")
    async def post_lang(job_id: str, request: Request):
        """Same handshake as /actions add_language (it IS that action): first call -> needs_confirmation + token; resubmit with the token."""
        try:
            pre(request, "action", int(os.environ.get("DUB_CTL_IP_ACTIONS_PER_MIN", "30")), job_id)
            if not addlang_enabled():
                raise CtlError(403, "feature_disabled", "Adding a language while watching is not available yet.")
            jd, st = ctx(job_id)
            b = await body_of(request)
            if set(b) - {"lang", "from_ms", "playback_generation", "idempotency_key", "confirmation_token"}:
                raise CtlError(422, "invalid_args", "unexpected fields")
            lang, fm, gen = b.get("lang"), b.get("from_ms"), b.get("playback_generation")
            if not isinstance(lang, str) or not ctl_tools.LANG_RE.match(lang):
                raise CtlError(422, "invalid_args", "lang")
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 10**9 for x in (fm, gen)):
                raise CtlError(422, "invalid_args", "from_ms and playback_generation must be non-negative integers")
            codes = deps.ns["get_target_codes"]()
            if codes and lang not in codes:
                raise CtlError(422, "unsupported_lang", f"unsupported language {lang}")
            have = deps.job_langs(jd)
            if lang not in have and len(have) + 1 > deps.ns.get("MAX_TARGETS", 3):
                raise CtlError(422, "too_many_languages", "this job already has the maximum number of languages")
            action = {"type": "add_language", "args": {"lang": lang, "from_ms": fm, "playback_generation": gen},
                      "idempotency_key": b.get("idempotency_key") or "lang-" + sha(canon([job_id, lang]))[:24]}      # contract: idempotent per (job, lang)
            if b.get("confirmation_token"):
                action["confirmation_token"] = b["confirmation_token"]
            code, resp = await svc.submit(job_id, action)
            return JSONResponse(resp, status_code=code)
        except CtlError as e:
            return _err(e)

    return r


def register(app, ns):
    """Minimal registration; no-op unless DUB_ACTIONS is truthy. Parsed robustly (0/false/no/off/empty = off)."""
    on = os.environ.get("DUB_ACTIONS", "0").strip().lower() not in ("", "0", "false", "no", "off", "n", "f")
    if not on:
        return False
    deps = ServerDeps(ns)
    if os.environ.get("DUB_CTL_DEVHOOKS", "0") == "1":
        raise RuntimeError("DUB_CTL_DEVHOOKS is a test-only switch and is not available in this distribution")
    else:
        from sunwai.control import adapters as ctl_adapters; ctl_adapters.install(ns)
    from sunwai.control.guard import Guard, COOKIE
    state_dir = Path(ns.get("DUB_ROOT", "."))
    guard = Guard(state_dir, state_dir)
    svc = ActionService(deps, guard)
    app.state.ctl_svc = svc
    app.include_router(make_router(svc, deps, guard))

    @app.on_event("startup")
    async def _ctl_resume():
        app.state.ctl_loop = asyncio.get_running_loop()
        try:
            n = svc.resume_all(state_dir)
            if n:
                print(f"[ctl] resumed {n} add_language watcher(s) after restart", flush=True)
        except Exception as e:
            print(f"[ctl] resume failed: {e!r}", flush=True)

    SEG_RE_PATH = re.compile(r"^/api/jobs/([0-9a-f]{12})/([a-z]{2,3})/segments$")

    def _annotate(job_id, lang, payload):
        """Add `expr_applied` (+ ctl_head_revision) to each line of the legacy segments payload so the UI can show Undo after a reload."""
        jd = deps.job_dir(job_id)
        if jd is None or not (jd / "ctl").is_dir():
            return payload
        st = store_for(jd)
        for s in payload.get("segments", []):
            try:
                h = st.head(seg_key(lang, s.get("chunk"), str(s.get("id"))))
            except Exception:
                h = None
            s["expr_applied"] = bool(h and h.get("status") == "active" and h.get("m2_applied") and (h.get("expression") or {}).get("provenance") != "undo_line")      # provenance decides (a stale undo_of on old data cannot hide Undo)
            s["ctl_head_revision"] = h["revision"] if h else None
        return payload

    @app.middleware("http")
    async def _ctl_cookie(request, call_next):
        """The UI is served from this origin: when it loads '/', hand it the control token as an HttpOnly SameSite=Strict cookie, which the
        browser sends on same-origin fetch and EventSource automatically (no UI change). Other clients send header x-dub-token."""
        resp = await call_next(request)
        m_ = SEG_RE_PATH.match(request.url.path) if request.method == "GET" else None
        if m_ and resp.status_code == 200 and guard.authed(request):
            from fastapi.responses import Response as _R
            body = b"".join([chunk async for chunk in resp.body_iterator])
            hdrs = {k: v for k, v in resp.headers.items() if k.lower() not in ("content-length", "content-type")}
            try:
                body = json.dumps(_annotate(m_.group(1), m_.group(2), json.loads(body)), ensure_ascii=False).encode()
            except Exception:
                pass                                   # annotation is best-effort: never break the legacy payload
            return _R(content=body, status_code=200, headers=hdrs, media_type="application/json")
        if request.method == "GET" and request.url.path == "/" and resp.status_code == 200:
            https = request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"
            resp.set_cookie(COOKIE, guard.token, httponly=True, samesite="strict", secure=https, max_age=86400)
        return resp
    return True
