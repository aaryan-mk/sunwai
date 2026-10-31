# review items (exec'd inside test_expression.py namespace)
d9 = mk()
(d9 / "dub_hi.mp4").write_bytes(b"MP4ORIG")
(d9 / "tts_placed.json").write_text("{}")
E.head_revision(d9, "hi", 1)
c = E.do_expression_resynth(d9, "hi", 1, "angrier", 0, "m", **kw(lambda *a: wav(1.8)))


def snap(d):
    return {
        n: ((d / n).read_bytes(), (d / n).stat().st_mtime_ns if n.startswith(("dub", "tts_placed")) else 0)
        for n in ("dub.wav", "dub_hi.mp4", "tts_placed.json", "tts/1.wav", "segments.json")
    }


os.utime(d9 / "dub_hi.mp4", ns=(10**18, 10**18))
b0 = snap(d9)


def partial_mix():
    (d9 / "dub.wav").write_bytes(b"PARTIAL")
    (d9 / "dub_hi.mp4").write_bytes(b"BROKEN")
    (d9 / "tts_placed.json").write_text("junk")
    (d9 / "tts_placed.wav").write_bytes(b"x")
    (d9 / ".dubtmp_a.wav").write_bytes(b"t")
    raise RuntimeError("mix died")


try:
    E.apply_candidate(d9, 1, c["candidate_id"], 0, lang="hi", remix=partial_mix)
except RuntimeError:
    pass
chk(
    "apply remix failure: playback files byte+mtime identical, no stray files",
    snap(d9) == b0 and not (d9 / "tts_placed.wav").exists() and not (d9 / ".dubtmp_a.wav").exists(),
)
c = E.do_expression_resynth(d9, "hi", 1, "angrier", 0, "m2", **kw(lambda *a: wav(1.8)))
E.apply_candidate(d9, 1, c["candidate_id"], 0, lang="hi", remix=lambda: (d9 / "dub_hi.mp4").write_bytes(b"NEW"))
b1 = snap(d9)
try:
    E.undo_revision(d9, 1, 1, lang="hi", remix=partial_mix)
except RuntimeError:
    pass
chk("undo remix failure: byte+mtime identical", snap(d9) == b1)
d10 = mk()
(d10 / "dub_hi.mp4").write_bytes(b"A")
E.head_revision(d10, "hi", 1)
c = E.do_expression_resynth(d10, "hi", 1, "angrier", 0, "l", **kw(lambda *a: wav(1.8)))


def rm10():
    (d10 / "dub.wav").write_bytes(wav(3.3))
    (d10 / "dub_hi.mp4").write_bytes(b"B")


rv = E.apply_candidate(d10, 1, c["candidate_id"], 0, lang="hi", remix=rm10)
chk("right after apply lipsync is pending", E.lipsync_state(d10, "hi", 1) == "pending")
(d10 / "dub_hi_lipsync.mp4").write_bytes(b"L")
_m = (d10 / "dub_hi.mp4").stat().st_mtime_ns + 10**9
os.utime(d10 / "dub_hi_lipsync.mp4", ns=(_m, _m))
chk("lipsync file alone (no hash binding) not ready", E.lipsync_state(d10, "hi", 1) == "pending")
chk(
    "lipsync_complete wrong hash -> stale, still pending",
    E.lipsync_complete(d10, 1, rv["revision"], "bad") == "stale" and E.lipsync_state(d10, "hi", 1) == "pending",
)
chk(
    "lipsync_complete matching -> accepted + ready",
    E.lipsync_complete(d10, 1, rv["revision"], E._sha(d10 / "dub.wav")) == "accepted"
    and E.lipsync_state(d10, "hi", 1) == "ready",
)
os.utime(d10 / "dub_hi_lipsync.mp4", ns=(1, 1))
chk("older lipsync file -> pending", E.lipsync_state(d10, "hi", 1) == "pending")
d11 = Path(tempfile.mkdtemp())
ch = d11 / "hi" / "c0000"
(ch / "tts").mkdir(parents=True)
segs0 = [{"id": 1, "start": 0.0, "end": 2.0, "tgt_text": "अ।", "speaker": 1, "gender": "female"}]
(ch / "segments.json").write_text(json.dumps(segs0))
(ch / "tts" / "1.wav").write_bytes(wav(1))
(d11 / "speakers_global.json").write_text(json.dumps({"voice_plan": {"bogus": 1}}))
try:
    E.resolve_voice(ch, "hi", segs0[0])
    chk("planned job, invalid plan: error not legacy", False)
except E.ExprError as e:
    chk("planned job, invalid plan: error not legacy", e.status == 422)
(d11 / "speakers_global.json").unlink()
(d11 / "voice_rollout.json").write_text('{"voice_plan": 1}')
try:
    E.resolve_voice(ch, "hi", segs0[0])
    chk("rollout marker but no map: error", False)
except E.ExprError as e:
    chk("rollout marker but no map: error", e.status == 422)
(d11 / "voice_rollout.json").unlink()
chk("legacy job uses seg voice", E.resolve_voice(ch, "hi", dict(segs0[0], voice="Kavya")) == "Kavya")
import types as _ty
from sunwai.planning import voices as _va

_va.validate_plan = lambda g, lang: {"voices": {lang: {"1": "PlanVoice"}}}
(d11 / "speakers_global.json").write_text(json.dumps({"voice_plan": {"x": 1}}))
chk(
    "planned job: plan voice wins, seg voice absent or present",
    E.resolve_voice(ch, "hi", segs0[0]) == "PlanVoice"
    and E.resolve_voice(ch, "hi", dict(segs0[0], voice="Other")) == "PlanVoice",
)
d12 = mk()
E.head_revision(d12, "hi", 1)
r = E.do_expression_resynth(d12, "hi", 1, "angrier", 0, "ch", **kw(lambda *a: wav(3.0)))
chk(
    "rejected attempt reports cost (2 TTS calls incl. repair)",
    r["status"] == "rejected" and r["cost_this_call"]["tts_calls"] == 2 and r["cost_this_call"]["tts_chars"] > 0,
)


def bad(*a):
    raise RuntimeError("tts down")


r = E.do_expression_resynth(d12, "hi", 1, "sadder", 0, "ch2", **kw(bad))
chk("failed attempt reports cost", r["status"] == "error" and r["cost_this_call"]["tts_calls"] == 1)
r2 = E.do_expression_resynth(d12, "hi", 1, "sadder", 0, "ch2", **kw(bad))
chk("duplicate replay costs 0", r2["cost_this_call"]["tts_calls"] == 0)
chk("can_undo after apply/undo/redo; false on fresh", E.can_undo(d4, 1) and not E.can_undo(mk(), 1))
sg = json.loads((d4 / "segments.json").read_text())[0]
chk("apply sets style_src m2 with style", sg.get("style_src") == "m2" and sg["style"] == "anger")
dc = mk()
E.head_revision(dc, "hi", 1)
cc = E.do_expression_resynth(dc, "hi", 1, "calmer", 0, "cc", **kw(lambda *a: wav(1.8)))
E.apply_candidate(dc, 1, cc["candidate_id"], 0, lang="hi", remix=lambda: None)
chk("calmer: style and style_src None", json.loads((dc / "segments.json").read_text())[0].get("style_src") is None)
_e = os.environ.pop("EXPRESSION_ENABLED")
chk("default OFF", not E.enabled())
os.environ["EXPRESSION_ENABLED"] = _e


# ---- r2: bookkeeping failure after a successful remix must restore playback (apply + undo, each write)
def snapall(d):
    return {
        n: (d / n).read_bytes() for n in ("dub.wav", "dub_hi.mp4", "tts_placed.json", "tts/1.wav", "segments.json")
    } | {"revs": E._read_revs(d, 1).__len__(), "files": sorted(p.name for p in d.glob("dub_*.mp4"))}


def good_mix(d):
    def r():
        (d / "dub.wav").write_bytes(wav(3.4))
        (d / "dub_hi.mp4").write_bytes(b"NEWMP4")
        (d / "dub_zz.mp4").write_bytes(b"NEWLANG")
        (d / "tts_placed.json").write_text("new")

    return r


_orig = {k: getattr(E, k) for k in ("_snap", "_op", "_jw", "_append_rev")}


def failing(name, only=None):
    f = _orig[name]

    def w(*a, **k):
        if only is None or only in str(a[0]):
            raise OSError(f"forced {name} failure")
        return f(*a, **k)

    return w


for what, only in (("_snap", None), ("_op", None), ("_jw", "cands"), ("_append_rev", None)):
    dd = mk()
    (dd / "dub_hi.mp4").write_bytes(b"MP4ORIG")
    (dd / "tts_placed.json").write_text("{}")
    E.head_revision(dd, "hi", 1)
    c = E.do_expression_resynth(dd, "hi", 1, "angrier", 0, "bk" + what, **kw(lambda *a: wav(1.8)))
    b = snapall(dd)
    setattr(E, what, failing(what, only))
    try:
        E.apply_candidate(dd, 1, c["candidate_id"], 0, lang="hi", remix=good_mix(dd))
        ok = False
    except OSError:
        ok = True
    finally:
        setattr(E, what, _orig[what])
    chk(
        f"apply: {what} failure after remix -> playback restored (incl. new dub_zz.mp4 removed), no revision, candidate unapplied",
        ok and snapall(dd) == b and not E._jr(dd / "expr" / "cands" / f"{c['candidate_id']}.json").get("applied"),
    )
    c2 = E.apply_candidate(dd, 1, c["candidate_id"], 0, lang="hi", remix=good_mix(dd))
    chk(f"apply retry after {what} rollback succeeds", c2["revision"] == 1)
    if what == "_op":
        continue
    b = snapall(dd)
    setattr(E, what, failing(what, only if what != "_jw" else "NOPE"))
    if what == "_jw":
        setattr(E, what, _orig[what])
    if what in ("_snap", "_append_rev"):
        try:
            E.undo_revision(dd, 1, 1, lang="hi", remix=good_mix(dd))
            ok = False
        except OSError:
            ok = True
        finally:
            setattr(E, what, _orig[what])
        chk(f"undo: {what} failure after remix -> playback restored, no revision", ok and snapall(dd) == b)
# torn append: truncated back
dt = mk()
E.head_revision(dt, "hi", 1)
p = E._revs_path(dt, 1)
sz = p.stat().st_size
import builtins


class Boom(Exception):
    pass


_ow = E.os.fsync
E.os.fsync = lambda fd: (_ for _ in ()).throw(OSError("fsync boom"))
try:
    E._append_rev(dt, 1, {"revision": 99})
except OSError:
    pass
E.os.fsync = _ow
chk("torn append truncated back to prior size", p.stat().st_size == sz)

# ---- r3
dr = mk()
E.head_revision(dr, "hi", 1)
p = E._revs_path(dr, 1)
with open(p, "a") as f:
    f.write('{"revision": 5, "torn')
chk("torn final line skipped on load", len(E._read_revs(dr, 1)) == 1 and E.head_revision(dr, "hi", 1)["revision"] == 0)
cr = E.do_expression_resynth(dr, "hi", 1, "angrier", 0, "t1", **kw(lambda *a: wav(1.8)))
rv = E.apply_candidate(dr, 1, cr["candidate_id"], 0, lang="hi", remix=lambda: None)
chk(
    "append after torn tail heals it (2 clean records, parse ok)",
    [r["revision"] for r in E._read_revs(dr, 1)] == [0, 1] and all(json.loads(l) for l in p.read_text().splitlines()),
)
with open(p, "w") as f:
    f.write("garbage\n" + json.dumps({"revision": 0}) + "\n")
try:
    E._read_revs(dr, 1)
    chk("corrupt middle line still raises", False)
except ValueError:
    chk("corrupt middle line still raises", True)
d13 = mk()
E.head_revision(d13, "hi", 1)
orig_txt = json.loads((d13 / "segments.json").read_text())[0]["tgt_text"]
c = E.do_expression_resynth(d13, "hi", 1, "angrier", 0, "sub", **kw(lambda *a: wav(1.8)))
E.apply_candidate(d13, 1, c["candidate_id"], 0, lang="hi", remix=lambda: None)
sg = json.loads((d13 / "segments.json").read_text())[0]
chk(
    "apply: tgt_text (subtitles) untouched; spoken text in tts_text/tts_src/tts_by",
    sg["tgt_text"] == orig_txt and sg["tts_text"].endswith("!") and sg["tts_src"] == orig_txt and sg["tts_by"] == "m2",
)
E.undo_revision(d13, 1, 1, lang="hi", remix=lambda: None)
sg = json.loads((d13 / "segments.json").read_text())[0]
chk("undo: tts_text cleared, tgt_text same", "tts_text" not in sg and sg["tgt_text"] == orig_txt)
E.undo_revision(d13, 1, 2, lang="hi", remix=lambda: None)
chk("redo restores tts_text", json.loads((d13 / "segments.json").read_text())[0].get("tts_text", "").endswith("!"))
d14 = mk()
E.head_revision(d14, "hi", 1)
c = E.do_expression_resynth(d14, "hi", 1, "angrier", 0, "nw", **kw(lambda *a: wav(1.8)))
(d14 / "tts" / "1.wav").unlink()
E.head_revision  # active wav absent: apply must roll back by deleting the newly installed wav


def fail_mix():
    raise RuntimeError("x")


try:
    E.apply_candidate(d14, 1, c["candidate_id"], 0, lang="hi", remix=fail_mix)
except RuntimeError:
    pass
except E.ExprError:
    pass
chk("rollback with no prior active wav leaves none", not (d14 / "tts" / "1.wav").exists())

_e = os.environ.pop("EXPRESSION_ENABLED", None)
try:
    chk("module: no env OFF", not E.enabled())
    os.environ["EXPRESSION_ENABLED"] = "1"
    chk("module: explicit 1 ON", E.enabled())
    os.environ["EXPRESSION_ENABLED"] = "0"
    chk("module: explicit 0 OFF", not E.enabled())
finally:
    os.environ["EXPRESSION_ENABLED"] = _e or "1"

# ---- live chat-undo sequence (preview -> apply -> legacy view of segments.json -> undo)
import threading as _th

dl = mk()
(dl / "dub_hi.mp4").write_bytes(b"A")
h0 = E.head_revision(dl, "hi", 1)
c = E.do_expression_resynth(dl, "hi", 1, "angrier", h0["revision"], "live1", **kw(lambda *a: wav(1.8)))
ap = E.apply_candidate(dl, 1, c["candidate_id"], 0, lang="hi", remix=lambda: (dl / "dub_hi.mp4").write_bytes(b"B"))


def legacy_view(d):  # what control plane.head_view does: reads segments.json only, no expression call
    sg = json.loads((d / "segments.json").read_text())[0]
    return (sg["tgt_text"], sg.get("tts_text"), sg.get("style"), len(sg.get("history") or []))


v_applied = legacy_view(dl)
st = E.live_state(dl, 1)
chk("after apply: live_state owned, not in flight, head 1", st == {"m2_head": 1, "owned": True, "in_flight": False})
seen = []


def undo_with_probe():
    def rm():
        seen.append((E.live_state(dl, 1), legacy_view(dl)))  # a concurrent reader mid-undo sees the half-way state
        (dl / "dub_hi.mp4").write_bytes(b"C")

    return E.undo_revision(dl, 1, 1, lang="hi", remix=rm)


res = {}


def probe():
    res["st"] = E.live_state(dl, 1)


u = undo_with_probe()
chk(
    "mid-undo reader: in_flight True (control plane must not create a legacy revision then)",
    seen and seen[0][0]["in_flight"] is True and not seen[0][0]["owned"],
)
chk(
    "undo: new expression revision 2 restoring rev 0, tts_text/style cleared consistently",
    u["revision"] == 2 and u["restores"] == 0 and legacy_view(dl)[1:3] == (None, None) and u["style"] is None,
)
chk(
    "after undo: owned, not in flight, head 2", E.live_state(dl, 1) == {"m2_head": 2, "owned": True, "in_flight": False}
)
chk(
    "no spurious external/legacy revision in expression log",
    [r["origin"] for r in E._read_revs(dl, 1)] == ["initial", "expression", "undo"],
)
chk(
    "undo-of-undo (redo) restores the expression, markers back",
    E.undo_revision(dl, 1, 2, lang="hi", remix=lambda: None)["revision"] == 3
    and legacy_view(dl)[2] == "anger"
    and E.live_state(dl, 1)["owned"],
)
