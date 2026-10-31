import io, json, os, sys, tempfile, threading, shutil
from pathlib import Path
import numpy as np, soundfile as sf
from sunwai.control import expression as E
from sunwai.control.expression.fitting import wav_duration
from sunwai.control.expression.fsutil import sha256_file
from sunwai.control.expression.revisions import read_revisions

os.environ["EXPRESSION_ENABLED"] = "1"
F = []


def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        F.append(name)


SR = 24000


def wav(sec):
    b = io.BytesIO()
    sf.write(b, (0.1 * np.sin(np.arange(int(sec * SR)) * 0.05)).astype("float32"), SR, format="WAV", subtype="PCM_16")
    return b.getvalue()


def mk(slot=2.0, nxt=False):
    d = Path(tempfile.mkdtemp())
    (d / "tts").mkdir()
    segs = [
        {
            "id": 1,
            "start": 0.0,
            "end": slot,
            "src_text": "x",
            "tgt_text": "तुमने झूठ बोला।",
            "speaker": 0,
            "gender": "f",
            "voice": "Kavya",
            "style": None,
            "tempo": 1.0,
        }
    ]
    if nxt:
        segs.append({"id": 2, "start": slot, "end": slot + 1, "tgt_text": "ok", "speaker": 0, "gender": "f"})
    (d / "segments.json").write_text(json.dumps(segs))
    (d / "tts" / "1.wav").write_bytes(wav(1.5))
    (d / "dub.wav").write_bytes(wav(3))
    return d


def tempo_fn(src, dst, t):
    y, sr = sf.read(str(src), dtype="float32")
    n = int(len(y) / t)
    sf.write(
        str(dst), np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype("float32"), sr, subtype="PCM_16"
    )


trim = lambda raw, dst: (raw, wav_duration(raw))


def kw(tts, **k):
    return dict(tts=tts, voice_fn=lambda d, l, s: "V", trim=trim, tempo_fn=tempo_fn, **k)


def remix_for(d, calls=None):
    def r():
        if calls is not None:
            calls.append(1)
        (d / "dub.wav").write_bytes(
            wav(3) + bytes([len(calls or [])]) if False else wav(3.0 + 0.01 * len(calls if calls is not None else [0]))
        )

    return r


# planner
p = E.plan_expression("तुमने झूठ बोला।", "hi", "angrier")
chk("angrier->anger,!", p["style"] == "anger" and p["spoken_text"].endswith("!"))
chk("excited->happy", E.plan_expression("हम जीत गए।", "hi", "excited")["style"] == "happy")
c = E.plan_expression("सब ठीक है!!", "hi", "calmer")
chk(
    "calmer: no style, restrained",
    c["style"] is None and c["spoken_text"].endswith("।") and "!" not in c["spoken_text"],
)
chk("no invented calm tag", all(E.INTENT_STYLE[i] in (None, "anger", "sad", "happy") for i in E.INTENT_STYLE))
chk("wording: number change rejected", not E.check_wording("मुझे 5 रुपये दो", "मुझे 6 रुपये दो", "hi")[0])
chk("wording: negation rejected", not E.check_wording("यह सही नहीं है", "यह सही है", "hi")[0])
try:
    E.plan_expression("x", "hi", "furious")
    chk("bad intent 422", False)
except E.ExprError as e:
    chk("bad intent 422", e.status == 422)

# happy path + never overwrites active
d = mk()
h = E.head_revision(d, "hi", 1)
before = sha256_file(d / "tts" / "1.wav")
c = E.do_expression_resynth(d, "hi", 1, "angrier", 0, "k1", **kw(lambda *a: wav(1.5)))
chk(
    "candidate ready, active untouched",
    c["status"] == "ready" and sha256_file(d / "tts" / "1.wav") == before and c["audio_path"] != str(d / "tts" / "1.wav"),
)
chk("underrun padded to slot", abs(wav_duration(c["audio_path"]) - 2.0) < 0.01)
# duplicate submission
n = {"n": 0}


def counting(*a):
    n["n"] += 1
    return wav(1.5)


c2 = E.do_expression_resynth(d, "hi", 1, "angrier", 0, "k1", **kw(counting))
chk("duplicate submission returns same candidate, no new TTS", c2["candidate_id"] == c["candidate_id"] and n["n"] == 0)
# tempo bound + repair + reject
d2 = mk(slot=2.0)
calls = []


def long_then_short(text, lang, voice, style):
    calls.append(text)
    return wav(2.3 if len(calls) == 1 else 1.9)


r = E.do_expression_resynth(d2, "hi", 1, "sadder", 0, "k", **kw(long_then_short))
chk(
    "overrun -> one repair succeeds",
    r["status"] == "ready" and r["repair_used"] and len(calls) == 2 and 0.92 <= r["tempo_factor"] <= 1.08,
)
d3 = mk(slot=2.0)
k3 = []
r = E.do_expression_resynth(d3, "hi", 1, "angrier", 0, "k", **kw(lambda *a: (k3.append(1), wav(3.0))[1]))
chk(
    "repair failure -> rejected, exactly 2 TTS calls, no audio",
    r["status"] == "rejected" and len(k3) == 2 and "audio_path" not in r,
)
try:
    E.apply_candidate(d3, 1, r["candidate_id"], 0, lang="hi", remix=lambda: None)
    chk("apply rejected cand 409", False)
except E.ExprError as e:
    chk("apply rejected cand 409", e.status == 409)
# apply / CAS / undo / redo
d4 = mk()
E.head_revision(d4, "hi", 1)
c = E.do_expression_resynth(d4, "hi", 1, "angrier", 0, "a", **kw(lambda *a: wav(1.8)))
orig = sha256_file(d4 / "tts" / "1.wav")
rm = []
rev = E.apply_candidate(
    d4, 1, c["candidate_id"], 0, lang="hi", remix=lambda: (rm.append(1), (d4 / "dub.wav").write_bytes(wav(3.2)))
)
chk(
    "apply: rev1, remix ran, style stored, audio changed",
    rev["revision"] == 1 and rm and rev["style"] == "anger" and sha256_file(d4 / "tts" / "1.wav") != orig,
)
chk("apply: chunk hash recorded", rev["chunk_audio_hash"] == sha256_file(d4 / "dub.wav"))
try:
    E.apply_candidate(d4, 1, c["candidate_id"], 0, lang="hi", remix=lambda: None)
    chk("double apply 409", False)
except E.ExprError as e:
    chk("double apply 409", e.status == 409)
u = E.undo_revision(d4, 1, 1, lang="hi", remix=lambda: (d4 / "dub.wav").write_bytes(wav(3.0)))
chk(
    "undo: new revision 2 restoring original audio+text",
    u["revision"] == 2 and sha256_file(d4 / "tts" / "1.wav") == orig and u["style"] is None and u["restores"] == 0,
)
chk("revisions append-only (3 records)", len(read_revisions(d4, 1)) == 3)
r3 = E.undo_revision(d4, 1, 2, lang="hi", remix=lambda: None)
chk("undo of undo = redo", r3["revision"] == 3 and r3["style"] == "anger")
# remix failure rolls back
d5 = mk()
E.head_revision(d5, "hi", 1)
c = E.do_expression_resynth(d5, "hi", 1, "excited", 0, "z", **kw(lambda *a: wav(1.8)))
o5 = sha256_file(d5 / "tts" / "1.wav")


def boom():
    raise RuntimeError("mix failed")


try:
    E.apply_candidate(d5, 1, c["candidate_id"], 0, lang="hi", remix=boom)
except RuntimeError:
    pass
chk(
    "remix failure restores active audio + segments + no revision",
    sha256_file(d5 / "tts" / "1.wav") == o5
    and json.loads((d5 / "segments.json").read_text())[0]["style"] is None
    and len(read_revisions(d5, 1)) == 1,
)
# concurrent edit: external text edit between preview and apply
d6 = mk()
E.head_revision(d6, "hi", 1)
c = E.do_expression_resynth(d6, "hi", 1, "angrier", 0, "c", **kw(lambda *a: wav(1.8)))
segs = json.loads((d6 / "segments.json").read_text())
segs[0]["tgt_text"] = "नया पाठ।"
(d6 / "segments.json").write_text(json.dumps(segs))
try:
    E.apply_candidate(d6, 1, c["candidate_id"], 0, lang="hi", remix=lambda: None)
    chk("concurrent text edit -> 409", False)
except E.ExprError as e:
    chk("concurrent text edit -> 409", e.status == 409)
try:
    E.do_expression_resynth(d6, "hi", 1, "sadder", 0, "c2", **kw(lambda *a: wav(1.8)))
    chk("preview with stale expected_revision 409", False)
except E.ExprError as e:
    chk("preview with stale expected_revision 409", e.status == 409)
# two concurrent applies of two candidates: exactly one wins
d7 = mk()
E.head_revision(d7, "hi", 1)
cs = [E.do_expression_resynth(d7, "hi", 1, i, 0, i, **kw(lambda *a: wav(1.8))) for i in ("angrier", "sadder")]
res = []


def go(c):
    try:
        E.apply_candidate(d7, 1, c["candidate_id"], 0, lang="hi", remix=lambda: None)
        res.append("ok")
    except E.ExprError as e:
        res.append(e.status)


ts = [threading.Thread(target=go, args=(c,)) for c in cs]
[t.start() for t in ts]
[t.join() for t in ts]
chk("concurrent applies: one ok, one 409", sorted(map(str, res)) == ["409", "ok"])
# undo during generation: candidate becomes superseded
d8 = mk()
E.head_revision(d8, "hi", 1)
c0 = E.do_expression_resynth(d8, "hi", 1, "angrier", 0, "p", **kw(lambda *a: wav(1.8)))
E.apply_candidate(d8, 1, c0["candidate_id"], 0, lang="hi", remix=lambda: None)


def slow_tts(*a):
    E.undo_revision(d8, 1, 1, lang="hi", remix=lambda: None)  # undo lands while synthesis is running
    return wav(1.8)


c = E.do_expression_resynth(d8, "hi", 1, "sadder", 1, "q", **kw(slow_tts))
chk("undo during generation -> candidate superseded, not appliable", c["status"] == "superseded")
try:
    E.apply_candidate(d8, 1, c["candidate_id"], 2, lang="hi", remix=lambda: None)
    chk("superseded apply 409", False)
except E.ExprError as e:
    chk("superseded apply 409", e.status == 409)
# stale lip-sync completion
hd = read_revisions(d4, 1)[-1]
chk("lipsync current accepted", E.lipsync_complete(d4, 1, hd["revision"], hd["chunk_audio_hash"]) == "accepted")
chk("lipsync old revision stale", E.lipsync_complete(d4, 1, 1, "whatever") == "stale")
chk("lipsync wrong chunk hash stale", E.lipsync_complete(d4, 1, hd["revision"], "deadbeef") == "stale")
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_expression_extra.py")).read())
# kill switch
os.environ["EXPRESSION_ENABLED"] = "0"
try:
    E.do_expression_resynth(mk(), "hi", 1, "angrier", 0, "x", **kw(lambda *a: wav(1)))
    chk("kill switch 503", False)
except E.ExprError as e:
    chk("kill switch 503", e.status == 503)
print("FAILS:", len(F))
sys.exit(1 if F else 0)
