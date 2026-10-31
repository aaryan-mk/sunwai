"""reversible per-line expression (candidate revisions). Self-contained; no edit to live files.

Entry points (all return plain dicts, raise ExprError(status, msg) on 409/422/...):
  do_expression_resynth(out_dir, lang, seg_id, intent, expected_revision, idempotency_key, ...) -> candidate (A/B preview)
  apply_candidate(out_dir, seg_id, cand_id, expected_revision, remix=...)  -> commits (CAS), remix, new SegmentRevision
  undo_revision(out_dir, seg_id, expected_revision, remix=...)             -> new revision restoring prior assets
  lipsync_complete(out_dir, seg_id, revision, chunk_audio_hash)            -> 'accepted' | 'stale'
Kill switch: EXPRESSION_ENABLED=0 (parsed robustly) -> ExprError(503).
"""
import hashlib, json, logging, os, re, shutil, subprocess, sys, time, unicodedata, uuid
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

INTENT_STYLE = {"angrier": "anger", "sadder": "sad", "excited": "happy", "calmer": None}
VALID_STYLES = {"anger", "sad", "happy", "fear", "surprise", "disgust"}
TEMPO_LO, TEMPO_HI = 0.92, 1.08
TERM = "।.!?…۔"
_TERM_RE = re.compile(r"[" + re.escape(TERM) + r"\s]+$")
DEFAULT_STOP = {"hi": "।", "mr": "।", "ur": "۔", "bn": "।", "ta": ".", "te": ".", "ml": ".", "kn": ".", "gu": "।"}


class ExprError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status, self.msg = status, msg


def enabled():
    """ON only if EXPRESSION_ENABLED is truthy (default OFF)."""
    v = (os.environ.get("EXPRESSION_ENABLED") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return False                                                                                            # default OFF


# ----------------------------------------------------------------- planner (rules)
def _split_term(text):
    t = (text or "").strip()
    m = _TERM_RE.search(t)
    body = t[:m.start()].rstrip() if m else t
    return body, (m.group(0).strip() if m else "")


def plan_expression(text, lang, intent, *, compact=False, wording_patch=None):
    """-> {style, punctuation_patch, spoken_text, wording_patch, rationale}. Style/punctuation only unless wording_patch."""
    intent = (intent or "").strip().lower()
    if intent not in INTENT_STYLE:
        raise ExprError(422, f"unknown intent {intent!r}; one of {sorted(INTENT_STYLE)}")
    base = text
    if wording_patch:
        ok, why = check_wording(text, wording_patch, lang)
        if not ok:
            raise ExprError(422, "wording patch rejected: " + "; ".join(why))
        base = wording_patch
    body, term = _split_term(base)
    body = re.sub(r"\.{2,}|…", ",", body) if intent != "sadder" else body
    stop = DEFAULT_STOP.get(lang, ".")
    q = "?" in term
    if compact:
        new_term, style, why = ("?" if q else stop), INTENT_STYLE[intent], "compact repair: restrained punctuation"
        body = re.sub(r"\s*(\.{2,}|…)\s*", " ", body)
    elif intent == "angrier":
        new_term, style, why = ("?!" if q else "!"), "anger", "terminal '!' + style anger"
    elif intent == "excited":
        new_term, style, why = ("?!" if q else "!"), "happy", "terminal '!' + style happy (auditioned)"
    elif intent == "sadder":
        new_term, style, why = ("?" if q else "..."), "sad", "trailing ellipsis + style sad"
    else:  # calmer: no style tag, restrained punctuation
        new_term, style, why = ("?" if q else stop), None, "style omitted, restrained punctuation"
    spoken = body + new_term
    patch = [{"op": "terminal", "from": term, "to": new_term}] if term != new_term else []
    return {"style": style, "punctuation_patch": patch, "spoken_text": spoken,
            "wording_patch": wording_patch, "rationale": why, "intent": intent}


_NEG = {"hi": ["नहीं", "मत", "न "], "mr": ["नाही", "नको"], "ur": ["نہیں", "مت"], "ta": ["இல்லை", "வேண்டாம்", "மாட்டேன்", "முடியாது", "அல்ல"],
        "te": ["లేదు", "వద్దు"], "bn": ["না", "নেই"]}


def _digits(s):
    out = []
    for ch in s:
        try:
            out.append(str(unicodedata.digit(ch)))
        except ValueError:
            out.append(" " if not ch.isalnum() else ch)
    return re.findall(r"\d+", "".join(out))


def check_wording(old, new, lang):
    """entity (Latin tokens), number and negation check. Cannot guarantee semantic fidelity."""
    why = []
    if sorted(_digits(old)) != sorted(_digits(new)):
        why.append("numbers differ")
    if sorted(re.findall(r"[A-Za-z][A-Za-z0-9&.-]+", old)) != sorted(re.findall(r"[A-Za-z][A-Za-z0-9&.-]+", new)):
        why.append("Latin/entity tokens differ")
    negs = _NEG.get(lang, [])
    if sum(old.count(n) for n in negs) != sum(new.count(n) for n in negs):
        why.append("negation count differs")
    return (not why), why


# ----------------------------------------------------------------- storage helpers
def _x(out_dir):
    d = Path(out_dir) / "expr"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Lock:
    def __init__(self, path):
        self.path = str(path)

    def __enter__(self):
        self.f = open(self.path, "a+")
        if fcntl:
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        if fcntl:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _jw(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}_{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _wb(path, data):
    tmp = Path(str(path) + f".tmp{os.getpid()}_{uuid.uuid4().hex[:6]}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _jr(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _segs(out_dir):
    segs = _jr(Path(out_dir) / "segments.json")
    if segs is None:
        raise ExprError(404, "segments.json not found")
    return segs


def _seg(segs, seg_id):
    s = next((s for s in segs if str(s["id"]) == str(seg_id)), None)
    if s is None:
        raise ExprError(404, f"segment {seg_id} not found")
    return s


def _active_wav(out_dir, seg_id):
    return Path(out_dir) / "tts" / f"{seg_id}.wav"


def _state_fp(out_dir, seg):
    p = _active_wav(out_dir, seg["id"])
    return hashlib.sha1(json.dumps([seg.get("tgt_text"), seg.get("tts_text"), seg.get("style"), _sha(p) if p.exists() else None]).encode()).hexdigest()


def _revs_path(out_dir, seg_id):
    return _x(out_dir) / f"seg{seg_id}.revs.jsonl"


def _read_revs(out_dir, seg_id):
    """Append-only log. A crash can leave a torn (unparseable) FINAL line: skipped and logged. A bad line elsewhere is corruption."""
    p = _revs_path(out_dir, seg_id)
    if not p.exists():
        return []
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    out = []
    for i, l in enumerate(lines):
        try:
            out.append(json.loads(l))
        except ValueError:
            if i == len(lines) - 1:
                logging.getLogger("expression").warning("torn final revision line skipped (%s)", p)
                break
            raise
    return out


def _append_rev(out_dir, seg_id, rec):
    p = _revs_path(out_dir, seg_id)
    if p.exists():
        raw = p.read_bytes()
        if raw and not raw.endswith(b"\n"):                  # heal a torn tail so the new record starts on its own line
            with open(p, "ab") as f: f.truncate(raw.rfind(b"\n") + 1)
    size = p.stat().st_size if p.exists() else 0
    try:
        with open(p, "a", encoding="utf-8") as f:   # append-only; the commit point of apply/undo
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
    except BaseException:
        try:
            with open(p, "ab") as f: f.truncate(size)       # never leave a torn/partial record
        except OSError:
            pass
        raise


def _rev_wav(out_dir, seg_id, n):
    d = _x(out_dir) / "rev"
    d.mkdir(exist_ok=True)
    return d / f"{seg_id}_r{n}.wav"


def _mkrec(out_dir, seg, lang, revision, parent, source, extra=None):
    wav = _active_wav(out_dir, seg["id"])
    r = {"job_id": None, "lang": lang, "chunk_id": Path(out_dir).name, "segment_id": seg["id"], "revision": revision,
         "parent_revision": parent, "source_revision": None, "speaker_id": seg.get("speaker"), "voice_plan_id": None,
         "slot_start_ms": int(seg["start"] * 1000), "slot_end_ms": int(seg["end"] * 1000),
         "raw_source_text": seg.get("src_text"), "display_text": seg.get("tgt_text"), "spoken_text": seg.get("tts_text") if (seg.get("tts_text") and seg.get("tts_src") == seg.get("tgt_text")) else seg.get("tgt_text"),
         "tts_text": seg.get("tts_text"), "tts_src": seg.get("tts_src"), "tts_by": seg.get("tts_by"),
         "expression": seg.get("expression") or {"intent": None, "emotion": None, "confidence": None, "provenance": "none"},
         "style": seg.get("style"), "style_src": seg.get("style_src"), "tempo_factor": seg.get("tempo"), "audio_hash": _sha(wav) if wav.exists() else None,
         "subtitle_revision": revision, "status": "active", "state_fp": _state_fp(out_dir, seg), "origin": source,
         "chunk_audio_hash": None, "ts": time.time()}
    if extra:
        r.update(extra)
    return r


def head_revision(out_dir, lang, seg_id):
    """Head SegmentRevision; auto-appends an 'external' revision if the segment changed outside this module (text edit)."""
    segs = _segs(out_dir)
    seg = _seg(segs, seg_id)
    with _Lock(_x(out_dir) / f"seg{seg_id}.lock"):
        revs = _read_revs(out_dir, seg_id)
        fp = _state_fp(out_dir, seg)
        if not revs:
            rec = _mkrec(out_dir, seg, lang, 0, None, "initial")
            _append_rev(out_dir, seg_id, rec)
            _snap(out_dir, seg_id, 0)
            return rec
        if revs[-1]["state_fp"] != fp:
            n = revs[-1]["revision"] + 1
            rec = _mkrec(out_dir, seg, lang, n, revs[-1]["revision"], "external")
            _append_rev(out_dir, seg_id, rec)
            _snap(out_dir, seg_id, n)
            return rec
        return revs[-1]


def _snap(out_dir, seg_id, n):
    w = _active_wav(out_dir, seg_id)
    if w.exists() and not _rev_wav(out_dir, seg_id, n).exists():
        shutil.copyfile(w, _rev_wav(out_dir, seg_id, n))


def _op(out_dir, op_id, **kw):
    p = _x(out_dir) / "ops"
    p.mkdir(exist_ok=True)
    cur = _jr(p / f"{op_id}.json", {"operation_id": op_id, "action_id": None, "state": "queued", "stage": "queued",
                                    "revision": None, "audio_ready": False, "lipsync_ready": False, "error": None})
    cur.update(kw)
    _jw(p / f"{op_id}.json", cur)
    return cur


# ----------------------------------------------------------------- audio fitting
def _default_tts(text, lang, voice, style):
    from sunwai.pipeline import engine as dp
    return dp.bodhan_tts(dp.thread_session(), dp.load_keys()["tts"], text, lang, voice, style=style)


def resolve_voice(out_dir, lang, seg):
    """Immutable voice plan first. Planned job (voice_rollout.json voice_plan==1, or speakers_global.json carries voice_plan):
    the plan's voice for this speaker, NO legacy fallback (error instead). Legacy job: seg['voice'] else dub_pipeline.pick_voice."""
    jd = Path(out_dir).resolve().parent.parent
    smap = os.environ.get("SPEAKER_MAP") or str(jd / "speakers_global.json")
    g = _jr(smap)
    ro = _jr(jd / "voice_rollout.json") or {}
    planned = (isinstance(ro, dict) and ro.get("voice_plan") == 1) or (isinstance(g, dict) and "voice_plan" in g)
    if planned:
        if not isinstance(g, dict) or "voice_plan" not in g:
            raise ExprError(422, "planned job but speakers_global.json has no voice_plan; refusing legacy voice")
        try:
            from sunwai.planning import voices as VA
            v = VA.validate_plan(g, lang)["voices"][lang].get(str(seg.get("speaker")))
        except Exception as e:
            raise ExprError(422, f"voice plan invalid: {e}")
        if not v:
            raise ExprError(422, f"voice plan has no voice for speaker {seg.get('speaker')}")
        return v
    if seg.get("voice"):
        return seg["voice"]
    from sunwai.pipeline import engine as dp
    return dp.pick_voice(lang, seg["gender"], seg["speaker"])


def _default_voice(args_lang, seg):          # kept for the voice_fn injection point (tests)
    from sunwai.pipeline import engine as dp
    return dp.pick_voice(args_lang, seg["gender"], seg["speaker"])


def _wav_dur(path):
    import soundfile as sf
    i = sf.info(str(path))
    return i.frames / float(i.samplerate)


def _default_trim(raw, dst):
    from sunwai.pipeline import engine as dp
    return dp._pretrim_wav(raw, dst)


def _default_tempo(src, dst, tempo):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-filter:a", f"atempo={tempo:.6f}",
                    "-ac", "1", "-ar", "24000", str(dst)], check=True, timeout=float(os.environ.get("EXPR_FFMPEG_TIMEOUT", "120")))      # r11: every external call is time-bounded


def _pad_to(path, dur):
    import soundfile as sf, numpy as np
    y, sr = sf.read(str(path), dtype="float32")
    n = int(round(dur * sr))
    if len(y) < n:
        sf.write(str(path), np.concatenate([y, np.zeros(n - len(y), dtype="float32")]), sr, subtype="PCM_16")


def _synth_fit(out_dir, seg, lang, text, style, voice, cand_dir, tag, tts, trim, tempo_fn, allowed, slot):
    wb = tts(text, lang, voice, style)
    raw = cand_dir / f"{tag}_raw.wav"
    raw.write_bytes(wb)
    fit_src, dur = trim(raw, cand_dir / f"{tag}_trim.wav")
    ratio = dur / slot
    tempo = 1.0 if ratio <= 1.0 else min(max(ratio, TEMPO_LO), TEMPO_HI)   # overruns only; underruns are padded
    outp = cand_dir / f"{tag}.wav"
    tempo_fn(fit_src, outp, tempo)
    fdur = _wav_dur(outp)
    return outp, dur, tempo, fdur, fdur > allowed + 1e-3


def do_expression_resynth(out_dir, lang, seg_id, intent, expected_revision, idempotency_key, *, job_id=None,
                          action_id=None, wording_patch=None, tts=None, voice_fn=None, trim=None, tempo_fn=None):
    """Synthesise a CANDIDATE revision (never touches active audio). Returns the candidate record (status ready|rejected|superseded)."""
    if not enabled():
        raise ExprError(503, "expression disabled (EXPRESSION_ENABLED=0)")
    if not idempotency_key:
        raise ExprError(422, "idempotency_key required")
    tts, trim, tempo_fn = tts or _default_tts, trim or _default_trim, tempo_fn or _default_tempo
    cost = {"tts_calls": 0, "tts_chars": 0, "attempts": []}
    _tts0 = tts
    def tts(text, lang_, voice_, style_):          # every attempt is charged, success or not
        cost["tts_calls"] += 1; cost["tts_chars"] += len(text); cost["attempts"].append({"chars": len(text), "style": style_})
        return _tts0(text, lang_, voice_, style_)
    out_dir = Path(out_dir)
    idem = _x(out_dir) / "idem"
    idem.mkdir(exist_ok=True)
    ik = idem / (hashlib.sha1(f"{seg_id}|{idempotency_key}".encode()).hexdigest() + ".json")
    with _Lock(str(ik) + ".lock"):                      # duplicate submissions serialise, 2nd returns the 1st result
        prev = _jr(ik)
        if prev:
            prev["duplicate"] = True
            prev["cost_this_call"] = {"tts_calls": 0, "tts_chars": 0}      # replay of a stored result: nothing new to charge
            return prev
        head = head_revision(out_dir, lang, seg_id)
        if head["revision"] != int(expected_revision):
            raise ExprError(409, f"stale revision: expected {expected_revision}, head {head['revision']}")
        segs = _segs(out_dir)
        seg = _seg(segs, seg_id)
        if not (seg.get("tgt_text") or "").strip():
            raise ExprError(422, "segment has no spoken text")
        cid = "c" + uuid.uuid4().hex[:10]
        op = _op(out_dir, cid, action_id=action_id, state="running", stage="synthesizing", revision=head["revision"])
        plan = plan_expression(seg["tgt_text"], lang, intent, wording_patch=wording_patch)
        slot = max(seg["end"] - seg["start"], 1e-6)
        nxt = next((s for s in segs if s["id"] == seg["id"] + 1), None) if isinstance(seg["id"], int) else None
        allowed = max(slot, nxt["start"] - seg["start"]) if nxt else slot
        cdir = _x(out_dir) / "cands"
        cdir.mkdir(exist_ok=True)
        voice = voice_fn(out_dir, lang, seg) if voice_fn else resolve_voice(out_dir, lang, seg)   # voice plan respected, never changed
        rec = {"candidate_id": cid, "operation_id": cid, "action_id": action_id, "job_id": job_id, "lang": lang,
               "segment_id": seg["id"], "base_revision": head["revision"], "intent": plan["intent"], "style": plan["style"],
               "spoken_text": plan["spoken_text"], "plan": plan, "voice": voice, "slot_s": slot, "allowed_s": allowed,
               "repair_used": False, "status": "running", "ts": time.time(), "cost": cost}
        try:
            outp, dur, tempo, fdur, over = _synth_fit(out_dir, seg, lang, plan["spoken_text"], plan["style"], voice, cdir, cid,
                                                      tts, trim, tempo_fn, allowed, slot)
            if over:                                                  # ONE duration-informed repair
                rplan = plan_expression(seg["tgt_text"], lang, intent, compact=True, wording_patch=wording_patch)
                outp, dur, tempo, fdur, over = _synth_fit(out_dir, seg, lang, rplan["spoken_text"], rplan["style"], voice, cdir,
                                                          cid + "r", tts, trim, tempo_fn, allowed, slot)
                rec.update(repair_used=True, spoken_text=rplan["spoken_text"], style=rplan["style"], plan_repair=rplan)
            if over:
                rec.update(status="rejected", reason=f"overrun: {fdur:.2f}s > {allowed:.2f}s after repair", fitted_s=fdur, tempo_factor=tempo)
                _op(out_dir, cid, state="error", stage="rejected", error=rec["reason"])
            else:
                if fdur < slot:
                    _pad_to(outp, slot)
                rec.update(audio_path=str(outp), audio_hash=_sha(outp), raw_s=dur, fitted_s=_wav_dur(outp), tempo_factor=tempo, status="ready")
                # undo/edit during generation: if the head moved, this candidate is obsolete
                if head_revision(out_dir, lang, seg_id)["revision"] != head["revision"]:
                    rec["status"] = "superseded"
                    _op(out_dir, cid, state="cancelled", stage="superseded")
                else:
                    _op(out_dir, cid, state="running", stage="audio ready", audio_ready=True)
        except ExprError:
            raise
        except Exception as e:
            rec.update(status="error", reason=str(e))
            _op(out_dir, cid, state="error", stage="error", error=str(e))
        rec["cost_this_call"] = {"tts_calls": cost["tts_calls"], "tts_chars": cost["tts_chars"]}
        _jw(cdir / f"{cid}.json", rec)
        _jw(ik, rec)
        return rec


def _mix_files(out_dir):
    """every file stage_mix publishes/overwrites in the chunk dir (dub.wav, dub_<lang>.mp4 for any lang, tts_placed.*)."""
    d = Path(out_dir)
    names = {"dub.wav", "tts_placed.json", "tts_placed.wav"} | {p.name for p in d.glob("dub_*.mp4") if "lipsync" not in p.name}
    return sorted(names)


def snapshot_mix(out_dir):
    bd = _x(out_dir) / "mixbak" / uuid.uuid4().hex[:8]
    bd.mkdir(parents=True)
    snap = {}
    for n in _mix_files(out_dir):
        p = Path(out_dir) / n
        if p.exists():
            shutil.copy2(p, bd / n)            # copy2 keeps mtime: lip-sync freshness (lipsync newer than dub) must not flip
            snap[n] = str(bd / n)
        else:
            snap[n] = None
    return {"dir": str(bd), "files": snap}


def restore_mix(out_dir, snap):
    d = Path(out_dir)
    for n, b in snap["files"].items():
        p = d / n
        if b is None:
            try: p.unlink()
            except OSError: pass
        else:
            tmp = d / f".restore_{uuid.uuid4().hex[:6]}_{n}"
            shutil.copy2(b, tmp); os.replace(tmp, p)
    for p in list(d.glob("dub_*.mp4")) + list(d.glob("tts_placed.*")):
        if "lipsync" not in p.name and p.name not in snap["files"]:      # created by the failed remix
            try: p.unlink()
            except OSError: pass
    for t in d.glob(".dubtmp_*"):
        try: t.unlink()
        except OSError: pass
    for t in d.glob(".stemtmp_*"):
        try: t.unlink()
        except OSError: pass
    shutil.rmtree(snap["dir"], ignore_errors=True)


def _default_remix(out_dir, lang):
    from sunwai.pipeline import engine as dp
    class A: pass
    a = A(); a.lang = lang
    segs = _jr(Path(out_dir) / "segments.json")
    return dp.stage_mix(a, Path(out_dir), segs)


def apply_candidate(out_dir, seg_id, cand_id, expected_revision, *, lang=None, remix=None):
    if not enabled():
        raise ExprError(503, "expression disabled")
    out_dir = Path(out_dir)
    cand = _jr(_x(out_dir) / "cands" / f"{cand_id}.json")
    if not cand:
        raise ExprError(404, "candidate not found")
    lang = lang or cand["lang"]
    remix = remix or (lambda: _default_remix(out_dir, lang))
    head = head_revision(out_dir, lang, seg_id)
    with _Lock(_x(out_dir) / f"seg{seg_id}.lock"):
        revs = _read_revs(out_dir, seg_id)
        head = revs[-1]
        segs = _segs(out_dir); seg = _seg(segs, seg_id)
        if head["revision"] != int(expected_revision) or cand["base_revision"] != head["revision"] or head["state_fp"] != _state_fp(out_dir, seg):
            raise ExprError(409, f"stale: head {head['revision']}, expected {expected_revision}, candidate base {cand['base_revision']}")
        if cand["status"] != "ready":
            raise ExprError(409, f"candidate not ready ({cand['status']})")
        if cand.get("applied"):
            raise ExprError(409, "candidate already applied")
        live = _active_wav(out_dir, seg_id)
        seg_bytes = (out_dir / "segments.json").read_bytes()      # rollback snapshots (exact bytes)
        live_bak = _x(out_dir) / f"seg{seg_id}.rollback.wav"
        if live.exists():
            shutil.copyfile(live, live_bak)
        msnap = snapshot_mix(out_dir)
        try:
            tmp = live.with_name(f".{live.stem}.expr{os.getpid()}.wav")
            shutil.copyfile(cand["audio_path"], tmp)
            os.replace(tmp, live)
            # subtitles/display stay faithful: tgt_text is untouched (unless an explicit, validated wording patch); the spoken text lives in tts_text
            disp = (cand.get("plan") or {}).get("wording_patch") or seg["tgt_text"]
            seg.update(tgt_text=disp, tts_text=cand["spoken_text"], tts_src=disp, tts_by="m2",
                       style=cand["style"], style_src=("m2" if cand["style"] else None), tempo=cand["tempo_factor"], edited=True,
                       overflow=False, overflow_s=0.0,
                       expression={"intent": cand["intent"], "emotion": cand["style"], "confidence": None, "provenance": "button"})
            _jw(out_dir / "segments.json", segs)
            remix()
            # bookkeeping: everything below must commit before the snapshot is dropped; _append_rev is the commit point
            n = head["revision"] + 1
            chunk_hash = _sha(out_dir / "dub.wav") if (out_dir / "dub.wav").exists() else None
            rec = _mkrec(out_dir, seg, lang, n, head["revision"], "expression",
                         {"chunk_audio_hash": chunk_hash, "undo_target": head["revision"], "candidate_id": cand_id,
                          "job_id": cand.get("job_id")})
            _snap(out_dir, seg_id, n)
            cand_p = _x(out_dir) / "cands" / f"{cand_id}.json"
            cand_new = dict(cand, applied=n)
            _jw(cand_p, cand_new)
            _op(out_dir, cand_id, state="running", stage="lip-syncing", revision=n, audio_ready=True, lipsync_ready=False)
            _append_rev(out_dir, seg_id, rec)
        except BaseException:
            if live_bak.exists():
                os.replace(live_bak, live)
            else:
                try: live.unlink()                    # no active wav existed before: don't leave the newly installed one
                except OSError: pass
            _wb(out_dir / "segments.json", seg_bytes)
            restore_mix(out_dir, msnap)
            for _f in (_rev_wav(out_dir, seg_id, head["revision"] + 1),):
                try: _f.unlink()
                except OSError: pass
            try: _jw(_x(out_dir) / "cands" / f"{cand_id}.json", cand)
            except Exception: pass
            try: _op(out_dir, cand_id, state="error", stage="rolled back", lipsync_ready=False, error="apply rolled back")
            except Exception: pass
            raise
        shutil.rmtree(msnap["dir"], ignore_errors=True)
        try:
            live_bak.unlink()
        except OSError:
            pass
        return rec


def undo_revision(out_dir, seg_id, expected_revision, *, lang=None, remix=None):
    """Creates a NEW revision that restores the audio/text/style of head.undo_target. Undo of an undo = redo."""
    out_dir = Path(out_dir)
    lang = lang or "xx"
    remix = remix or (lambda: _default_remix(out_dir, lang))
    head_revision(out_dir, lang, seg_id)
    with _Lock(_x(out_dir) / f"seg{seg_id}.lock"):
        revs = _read_revs(out_dir, seg_id)
        head = revs[-1]
        if head["revision"] != int(expected_revision):
            raise ExprError(409, f"stale: head {head['revision']}, expected {expected_revision}")
        tgt = head.get("undo_target")
        if tgt is None:
            raise ExprError(409, "nothing to undo")
        trec = next(r for r in revs if r["revision"] == tgt)
        src = _rev_wav(out_dir, seg_id, tgt)
        if not src.exists():
            raise ExprError(409, "prior assets missing")
        segs = _segs(out_dir); seg = _seg(segs, seg_id)
        live = _active_wav(out_dir, seg_id)
        bak = _x(out_dir) / f"seg{seg_id}.rollback.wav"
        if live.exists():
            shutil.copyfile(live, bak)
        sb = (out_dir / "segments.json").read_bytes()
        msnap = snapshot_mix(out_dir)
        try:
            tmp = live.with_name(f".{live.stem}.undo{os.getpid()}.wav")
            shutil.copyfile(src, tmp); os.replace(tmp, live)
            seg.pop("tts_text", None); seg.pop("tts_src", None); seg.pop("tts_by", None)
            seg.update({k: trec[k] for k in ("tts_text", "tts_src", "tts_by") if trec.get(k)})
            seg.update(tgt_text=trec["display_text"], style=trec["style"], style_src=trec.get("style_src"), tempo=trec["tempo_factor"], expression=trec["expression"])
            _jw(out_dir / "segments.json", segs)
            remix()
            n = head["revision"] + 1
            chunk_hash = _sha(out_dir / "dub.wav") if (out_dir / "dub.wav").exists() else None
            rec = _mkrec(out_dir, seg, lang, n, head["revision"], "undo",
                         {"chunk_audio_hash": chunk_hash, "undo_target": head["revision"], "restores": tgt})
            _snap(out_dir, seg_id, n)
            _append_rev(out_dir, seg_id, rec)
        except BaseException:
            if bak.exists():
                os.replace(bak, live)
            else:
                try: live.unlink()
                except OSError: pass
            _wb(out_dir / "segments.json", sb); restore_mix(out_dir, msnap)
            try: _rev_wav(out_dir, seg_id, head["revision"] + 1).unlink()
            except OSError: pass
            raise
        shutil.rmtree(msnap["dir"], ignore_errors=True)
        try:
            bak.unlink()
        except OSError:
            pass
        return rec


def lipsync_complete(out_dir, seg_id, revision, chunk_audio_hash):
    """Worker calls this when LatentSync finishes. Stale (revision moved on, or chunk audio hash differs from the head's AND
    from the dub.wav on disk) -> 'stale' and nothing is recorded. 'accepted' records the binding used by lipsync_state()."""
    revs = _read_revs(out_dir, seg_id)
    if not revs:
        return "stale"
    h = revs[-1]
    cur = _sha(Path(out_dir) / "dub.wav") if (Path(out_dir) / "dub.wav").exists() else None
    if h["revision"] != int(revision) or (h.get("chunk_audio_hash") and h["chunk_audio_hash"] != chunk_audio_hash) \
            or (cur is not None and cur != chunk_audio_hash):
        return "stale"
    _jw(_x(out_dir) / f"seg{seg_id}.lipsync.json", {"revision": int(revision), "chunk_audio_hash": chunk_audio_hash, "ts": time.time()})
    return "accepted"


def lipsync_state(out_dir, lang, seg_id):
    """Honest readiness: 'ready' only if dub_<lang>_lipsync.mp4 is newer than dub_<lang>.mp4 AND (no expression revision has been
    applied on this segment OR a lipsync_complete() binding matches the head revision and the current dub.wav hash).
    Never 'ready' merely because Apply succeeded: right after apply it is 'pending'."""
    d = Path(out_dir)
    dub, ls = d / f"dub_{lang}.mp4", d / f"dub_{lang}_lipsync.mp4"
    try:
        fresh = ls.stat().st_size > 0 and dub.exists() and ls.stat().st_mtime > dub.stat().st_mtime
    except OSError:
        fresh = False
    revs = _read_revs(out_dir, seg_id)
    if revs and any(r.get("origin") in ("expression", "undo") for r in revs):
        b = _jr(_x(out_dir) / f"seg{seg_id}.lipsync.json")
        cur = _sha(d / "dub.wav") if (d / "dub.wav").exists() else None
        bound = bool(b) and b["revision"] == revs[-1]["revision"] and b["chunk_audio_hash"] == cur
        return "ready" if (fresh and bound) else "pending"
    return "ready" if fresh else "pending"


def live_state(out_dir, seg_id):
    """READ-ONLY view for control plane's head_view (never appends a revision, never blocks):
    {"m2_head": int|None, "owned": live segment still equals the expression head state, "in_flight": an apply/undo currently holds the segment lock}.
    control plane must NOT reconcile a divergence into a 'legacy' revision while in_flight, nor when owned.
    FAILS CLOSED on EVERY error path (r10): busy lock => immediate in_flight; no fcntl, unopenable lock file, any read error, an unreadable /
    missing segment (ExprError) => owned+in_flight, so control plane never invents a legacy revision from a state it could not inspect."""
    closed = {"m2_head": None, "owned": True, "in_flight": True}
    out = {"m2_head": None, "owned": False, "in_flight": False}
    try:
        if fcntl is None:
            return closed                                    # cannot probe the lock: unknown => closed
        lk = _x(out_dir) / f"seg{seg_id}.lock"
        try:
            with open(lk, "a+") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN)
                except OSError:
                    out["in_flight"] = True
                    return out                               # busy: report immediately, read nothing else
        except OSError:
            return closed                                    # lock file cannot be opened: unknown => closed
        revs = _read_revs(out_dir, seg_id)
        if revs:
            out["m2_head"] = revs[-1]["revision"]
            try:
                out["owned"] = revs[-1]["state_fp"] == _state_fp(out_dir, _seg(_segs(out_dir), seg_id))
            except ExprError:
                out["owned"] = True; out["in_flight"] = True    # segment/state unreadable: unknown => closed, do not reconcile
        return out
    except Exception:
        return {"m2_head": out.get("m2_head"), "owned": True, "in_flight": True}


def can_undo(out_dir, seg_id):
    """control plane: route undo/redo here whenever this is True (asset history lives in expr/); never drop the markers after an undo."""
    revs = _read_revs(out_dir, seg_id)
    return bool(revs) and revs[-1].get("undo_target") is not None


if __name__ == "__main__":      # CLI for server2 run_script: last stdout line = JSON
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "apply", "undo", "head"])
    ap.add_argument("--out", required=True); ap.add_argument("--lang", required=True); ap.add_argument("--seg", required=True)
    ap.add_argument("--intent"); ap.add_argument("--expected-revision", type=int); ap.add_argument("--idem")
    ap.add_argument("--cand"); ap.add_argument("--job-id")
    a = ap.parse_args()
    try:
        if a.cmd == "preview":
            r = do_expression_resynth(a.out, a.lang, a.seg, a.intent, a.expected_revision, a.idem, job_id=a.job_id)
        elif a.cmd == "apply":
            r = apply_candidate(a.out, a.seg, a.cand, a.expected_revision, lang=a.lang)
        elif a.cmd == "undo":
            r = undo_revision(a.out, a.seg, a.expected_revision, lang=a.lang)
        else:
            r = head_revision(a.out, a.lang, a.seg)
        print(json.dumps({"ok": True, "result": r}, ensure_ascii=False))
    except ExprError as e:
        print(json.dumps({"ok": False, "status": e.status, "error": e.msg})); sys.exit(0)
