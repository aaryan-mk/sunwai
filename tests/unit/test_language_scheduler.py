import json
import threading
import time
from pathlib import Path

import pytest

from sunwai.jobs import language_scheduler as L

CH = [(0, 0.0, 18.05), (1, 18.05, 24.2), (2, 42.25, 37.5), (3, 79.75, 37.25), (4, 117.0, 63.0)]


def mkjob(tmp, langs=("hi",), voice_plan=None, planner=True, done=(), segs=None, lid="hi"):
    j = tmp / "aaaaaaaaaaaa"
    j.mkdir()
    (j / "job.json").write_text(json.dumps({"job_id": j.name, "langs": list(langs)}))
    (j / "plan.json").write_text(json.dumps({"chunks": [{"idx": i, "start": s, "dur": d} for i, s, d in CH]}))
    (j / "lid.json").write_text(json.dumps({"label": f"{lid}: x", "prob": .9, "code": lid}))
    if planner:
        g = {"speakers": []}
        if voice_plan is not None:
            g["voice_plan"] = voice_plan
        (j / "speakers_global.json").write_text(json.dumps(g))
    for lg, i in done:
        d = j / lg / f"c{i:04d}"
        d.mkdir(parents=True)
        (d / "status.json").write_text('{"state":"done"}')
        (d / f"dub_{lg}.mp4").write_bytes(b"x")
    if segs:
        for i, ss in segs.items():
            d = j / "_prep" / f"c{i:04d}"
            d.mkdir(parents=True)
            (d / "segments.json").write_text(json.dumps(ss, ensure_ascii=False))
    return j


@pytest.fixture(autouse=True)
def on(monkeypatch):
    monkeypatch.setenv("DUB_ADD_LANG", "1")
    L._INFLIGHT.clear()


H = lambda **k: L.Hooks(target_codes=lambda: {"hi", "ta", "bn", "te", "en"}, **k)


def test_default_on_kill_switch_off(monkeypatch, tmp_path):
    monkeypatch.delenv("DUB_ADD_LANG")
    assert L.enabled()                                        # default ON
    monkeypatch.setenv("DUB_ADD_LANG", "0")
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(mkjob(tmp_path), "ta", 0, 1, hooks=H())
    assert e.value.status == 503


@pytest.mark.parametrize("v", ["0", "false", "OFF", " no ", ""])
def test_kill_switch_parse(monkeypatch, v):
    monkeypatch.setenv("DUB_ADD_LANG", v)
    assert not L.enabled()


def test_duplicate_language_creation_idempotent(tmp_path):
    j = mkjob(tmp_path)
    r1 = L.request_track(j, "ta", 30000, 1, hooks=H())
    r2 = L.request_track(j, "ta", 30000, 1, hooks=H())
    assert r1["created"] and not r2["created"]
    assert json.loads((j / "job.json").read_text())["langs"] == ["hi", "ta"]
    assert json.loads((j / "tracks.json").read_text())["tracks"]["ta"]["requests"] == 2


def test_duplicate_concurrent(tmp_path):
    j = mkjob(tmp_path)
    res = []
    ts = [threading.Thread(target=lambda: res.append(L.request_track(j, "ta", 0, 1, hooks=H()))) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sum(r["created"] for r in res) == 1
    assert json.loads((j / "job.json").read_text())["langs"] == ["hi", "ta"]


def test_existing_job_lang_is_not_new(tmp_path):
    j = mkjob(tmp_path, langs=("hi", "ta"))
    called = []
    r = L.request_track(j, "ta", 0, 1, hooks=H(ensure_running=lambda x: called.append(x)))
    assert r["created"] and not called       # track created, no relaunch of an already-running language


def test_rapid_switches_obsolete_generation(tmp_path):
    j = mkjob(tmp_path, langs=("hi", "ta", "bn"))
    L.request_track(j, "ta", 10000, 5, hooks=H())
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "bn", 10000, 4, hooks=H())       # stale response from an older switch
    assert e.value.status == 409 and e.value.code == "obsolete_generation"
    L.request_track(j, "bn", 10000, 5, hooks=H())           # same generation allowed
    L.request_track(j, "ta", 90000, 9, hooks=H())           # newer re-prioritises
    st = json.loads((j / "tracks.json").read_text())
    assert st["gen"] == 9 and st["tracks"]["ta"]["from_ms"] == 90000
    for g in range(10, 40):                                 # rapid flip-flop never corrupts state
        L.request_track(j, ["ta", "bn"][g % 2], g * 1000, g, hooks=H())
    assert json.loads((j / "tracks.json").read_text())["gen"] == 39


def test_max_targets(tmp_path):
    j = mkjob(tmp_path, langs=("hi", "ta", "bn"))
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "te", 0, 1, hooks=H())
    assert e.value.code == "max_targets"
    assert L.request_track(j, "ta", 0, 1, hooks=H())["created"]   # existing one still fine


def test_voice_plan_rejects_unavailable_and_unplanned(tmp_path):
    vp = {"langs": ["hi", "ta", "te"], "voices": {"hi": {}, "ta": {}, "te": {}}, "unavailable_langs": {"te": "capacity"}}
    j = mkjob(tmp_path, voice_plan=vp)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "te", 0, 1, hooks=H())
    assert e.value.code == "voice_plan_excludes_lang" and "NEW job" in e.value.message
    with pytest.raises(L.LangSchedError):
        L.request_track(j, "bn", 0, 1, hooks=H())            # not in plan.langs
    assert L.request_track(j, "ta", 0, 1, hooks=H())["created"]
    assert json.loads((j / "speakers_global.json").read_text())["voice_plan"] == vp   # plan never mutated


def test_planner_not_ready(tmp_path):
    j = mkjob(tmp_path, planner=False)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "ta", 0, 1, hooks=H())
    assert e.value.status == 409 and e.value.code == "planner_not_ready"
    assert json.loads((j / "job.json").read_text())["langs"] == ["hi"]


@pytest.mark.parametrize("bad", [("ta", -1, 1), ("ta", float("nan"), 1), ("ta", 0, -1), ("t", 0, 1), ("ta", "x", 1), ("zz", 0, 1)])
def test_bad_args(tmp_path, bad):
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(mkjob(tmp_path), *bad, hooks=H())
    assert e.value.status == 422


def test_seek_order_and_runway():
    ch = [{"idx": i, "start": s, "dur": d} for i, s, d in CH]
    assert L.chunk_for_ms(ch, 0) == 0 and L.chunk_for_ms(ch, 60000) == 2 and L.chunk_for_ms(ch, 10 ** 9) == 4
    assert L.plan_order(ch, 0) == [0, 1, 2, 3, 4]
    assert L.plan_order(ch, 60000) == [2, 3, 4, 1, 0]
    # from_ms 70s -> +20s = 90s reaches chunk 3 (starts 79.75): runway [2,3]
    assert L.runway(ch, 70000) == [2, 3]
    assert sorted(L.plan_order(ch, 70000)) == [0, 1, 2, 3, 4]
    assert L.plan_order([], 5) == []


def test_focus_adapter_waits_for_watching_runway(tmp_path):
    j = mkjob(tmp_path, langs=("hi",), done=[("hi", 2)])
    calls = []
    h = H(set_focus=lambda *a: calls.append(a))
    r = L.request_track(j, "ta", 70000, 1, watching_lang="hi", hooks=h)   # runway [2,3]; hi chunk3 not done
    assert not r["focus"]["applied"] and not calls
    L._INFLIGHT.clear()
    j2 = mkjob(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, langs=("hi",), done=[("hi", 2), ("hi", 3)])
    r = L.request_track(j2, "ta", 70000, 1, watching_lang="hi", hooks=h)
    assert r["focus"]["applied"] and calls == [("aaaaaaaaaaaa", "ta", 2)]


def test_focus_hook_failure_is_not_fatal(tmp_path):
    def boom(*a): raise RuntimeError("x")
    r = L.request_track(mkjob(tmp_path), "ta", 0, 1, hooks=H(set_focus=boom))
    assert r["created"] and not r["focus"]["applied"]


def test_readiness(tmp_path):
    j = mkjob(tmp_path, langs=("hi", "ta"), done=[("ta", 0), ("ta", 1), ("ta", 3)])
    r = L.track_readiness(j, "ta")
    assert r["audio_state"] == "partial" and r["lipsync_state"] == "none"
    assert r["ready_intervals"] == [[0, 42250], [79750, 117000]]
    assert L.track_readiness(j, "bn")["audio_state"] == "absent"
    # lipsync freshness
    d = j / "ta" / "c0000"
    time.sleep(0.02); (d / "dub_ta_lipsync.mp4").write_bytes(b"y")
    r = L.track_readiness(j, "ta")
    assert r["lipsync_state"] == "partial" and r["lipsync_intervals"] == [[0, 18050]]
    # a re-dub makes the old lipsync stale (never show mismatched mouth)
    time.sleep(0.02); (d / "dub_ta.mp4").write_bytes(b"z")
    assert L.track_readiness(j, "ta")["lipsync_intervals"] == []


def test_readiness_all_ready_and_error(tmp_path):
    j = mkjob(tmp_path, langs=("hi", "ta"), done=[("ta", i) for i in range(5)])
    assert L.track_readiness(j, "ta")["audio_state"] == "ready"
    j2 = mkjob(tmp_path / "e" if (tmp_path / "e").mkdir() is None else tmp_path, langs=("hi", "ta"))
    d = j2 / "ta" / "c0000"; d.mkdir(parents=True); (d / "status.json").write_text('{"state":"error"}')
    assert L.track_readiness(j2, "ta")["audio_state"] == "error"


# ---- code-switching fixture (hand-labelled) ----
FIX = [  # (text, true spoken lang, true embedded foreign tokens)
    ("तो हम लोगों ने कल मीटिंग की", "hi", []),
    ("हमने WhatsApp पर message भेजा", "hi", ["WhatsApp", "message"]),
    ("Thank you all for coming today", "en", []),
    ("இன்று நாம் ஒரு புதிய திட்டம் தொடங்குகிறோம்", "ta", []),
    ("இந்த meeting நாளை நடைபெறும்", "ta", ["meeting"]),
    ("ok", "und", []),
]


def test_mixed_fixture(tmp_path):
    segs = [{"id": i, "start": i * 3.0, "end": i * 3.0 + 2.5, "src_text": t} for i, (t, *_ ) in enumerate(FIX)]
    j = mkjob(tmp_path, segs={0: segs}, lid="hi")
    sp = L.persist_source_spans(j)
    got = [s["lang"] for s in sp]
    assert got == [f[1] for f in FIX]
    assert sp[1]["mixed"] is False or sp[1]["embedded"] == ["WhatsApp", "message"]
    assert sp[1]["embedded"] == ["WhatsApp", "message"] and sp[4]["embedded"] == ["meeting"]
    assert sp[5]["method"] == "too_short" and sp[5]["confidence"] == 0
    assert [s["start_ms"] for s in sp][:2] == [0, 3000]
    assert L.load_source_spans(j) == sp


def test_override_and_clear(tmp_path):
    segs = [{"id": "a", "start": 0, "end": 2, "src_text": "Thank you all for coming"}]
    j = mkjob(tmp_path, segs={0: segs}, lid="hi")
    assert L.persist_source_spans(j)[0]["lang"] == "en"
    sp = L.set_segment_lang_override(j, 0, "a", "hi", "rev1")
    assert sp[0]["lang"] == "hi" and sp[0]["method"] == "override" and sp[0]["confidence"] == 1.0
    assert L.clear_segment_lang_override(j, 0, "a")[0]["lang"] == "en"
    with pytest.raises(L.LangSchedError):
        L.set_segment_lang_override(j, 0, "a", "Hindi!")


def test_routing_and_masking():
    en = {"lang": "en", "confidence": .95, "mixed": False}
    assert L.route_segment(en, "en")["route"] == "passthrough"
    assert L.route_segment(en, "ta")["route"] == "indic-translate"
    assert L.route_segment({"lang": "hi", "confidence": .9, "mixed": False}, "en")["route"] == "gemini-en"
    assert L.route_segment(en, "mni")["route"] == "it2"
    assert L.route_segment({"lang": "en", "confidence": .95, "mixed": True}, "en")["route"] != "passthrough"
    assert L.route_segment({"lang": "und", "confidence": 0}, "ta")["route"] == "indic-translate"
    m, mp = L.mask_foreign_terms("हमने WhatsApp पर message भेजा।")
    assert "WhatsApp" not in m and set(mp.values()) == {"WhatsApp", "message"}
    out, miss = L.unmask_terms(m, mp)
    assert out == "हमने WhatsApp पर message भेजा।" and not miss
    _, miss = L.unmask_terms("हमने पर भेजा", mp)
    assert sorted(miss) == sorted(mp)


def test_hindustani_script_flip_is_one_group():
    a = L.classify_text("تو یہ طے ہوا کہ بھائی کریں کیسے", "hi")
    b = L.classify_text("तो हुआ कि भाई इसका बेस्ट तरीका", "hi")
    assert a["lang"] == "ur" and b["lang"] == "hi" and L.lang_group("ur") == L.lang_group("hi")


# ---- regression tests for review findings ----
def snap(j):
    return (j / "job.json").read_bytes(), sorted(p.name for p in j.iterdir())


def test_global_inflight_cap_across_jobs_429(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    j1 = mkjob(tmp_path / "a"); j2 = mkjob(tmp_path / "b")
    L.request_track(j1, "ta", 0, 1, hooks=H())
    (j1 / "status.json").write_text('{"state":"running"}')       # what the real ensure_job_running does
    before = snap(j2)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j2, "bn", 0, 1, hooks=H())
    assert e.value.status == 429 and e.value.code == "busy"
    assert snap(j2) == before                                  # nothing mutated
    L.request_track(j1, "ta", 5000, 2, hooks=H())              # re-request of the in-flight track still allowed


def test_inflight_frees_when_track_ready(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    j1 = mkjob(tmp_path / "a"); j2 = mkjob(tmp_path / "b")
    L.request_track(j1, "ta", 0, 1, hooks=H())
    for i in range(5):
        d = j1 / "ta" / f"c{i:04d}"; d.mkdir(parents=True); (d / "status.json").write_text('{"state":"done"}'); (d / "dub_ta.mp4").write_bytes(b"x")
    assert L.request_track(j2, "bn", 0, 1, hooks=H())["created"]


def test_busy_hook_429_no_mutation(tmp_path):
    j = mkjob(tmp_path); b = snap(j)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "ta", 0, 1, hooks=H(busy=lambda: True))
    assert e.value.status == 429 and snap(j) == b and not (j / "tracks.json").exists()


def test_validation_before_mutation(tmp_path):
    vp = {"langs": ["hi", "ta"], "voices": {"hi": {}, "ta": {}}, "unavailable_langs": {}}
    j = mkjob(tmp_path, voice_plan=vp); b = snap(j)
    for lang in ("bn", "zz", "T"):
        with pytest.raises(L.LangSchedError):
            L.request_track(j, lang, 0, 1, hooks=H())
    assert snap(j) == b and not (j / "tracks.json").exists() and not L._INFLIGHT


def test_ensure_running_failure_rolls_back(tmp_path):
    j = mkjob(tmp_path); b = snap(j)
    def boom(_): raise RuntimeError("queue full")
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "ta", 0, 1, hooks=H(ensure_running=boom))
    assert e.value.status == 502 and e.value.code == "schedule_failed" and "queue full" in e.value.message
    assert snap(j) == b                                         # job.json bytes and dirs restored
    assert not (j / "tracks.json").exists() and not L._INFLIGHT
    assert L.request_track(j, "ta", 0, 1, hooks=H())["created"]  # and a retry works


def test_merge_failure_rolls_back(tmp_path):
    j = mkjob(tmp_path); b = snap(j)
    def bad_merge(jd, lg):
        (jd / "job.json").write_text("{partial"); raise OSError("disk")
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "ta", 0, 1, hooks=H(merge_lang=bad_merge))
    assert e.value.code == "schedule_failed" and snap(j) == b


def test_lipsync_hash_honesty(tmp_path):
    import hashlib
    j = mkjob(tmp_path, langs=("hi", "ta"), done=[("ta", 0), ("ta", 1)])
    time.sleep(0.02)
    for i in (0, 1):
        (j / "ta" / f"c{i:04d}" / "dub_ta_lipsync.mp4").write_bytes(b"y")
    r = L.track_readiness(j, "ta")
    assert r["lipsync_state"] == "partial" and not r["lipsync_verified"]       # mtime only: say so
    assert {c["lipsync_basis"] for c in r["chunks"] if c["lipsync"]} == {"mtime"}
    # recorded hash: chunk 0 matches dub.wav, chunk 1 does not
    d0, d1 = j / "ta" / "c0000", j / "ta" / "c0001"
    (d0 / "dub.wav").write_bytes(b"AUD"); (d1 / "dub.wav").write_bytes(b"NEW")
    (d0 / "lipsync.json").write_text(json.dumps({"chunk_audio_hash": hashlib.sha256(b"AUD").hexdigest()}))
    (d1 / "lipsync.json").write_text(json.dumps({"chunk_audio_hash": hashlib.sha256(b"OLD").hexdigest()}))
    r = L.track_readiness(j, "ta")
    assert r["lipsync_intervals"] == [[0, 18050]] and r["lipsync_verified"]
    # expression hook (lipsync_complete) has the last word
    h = L.Hooks(lipsync_hash_ok=lambda jd, lg, i: i == 1)
    assert L.track_readiness(j, "ta", h)["lipsync_intervals"] == [[18050, 42250]]


# ---- Sol r2: tracks.json inside the commit; contract enum ----
def _fail_tracks(monkeypatch, nth=1):
    real = L._atomic_json
    n = {"c": 0}
    def f(p, obj):
        if Path(p).name == "tracks.json":
            n["c"] += 1
            if n["c"] >= nth:
                raise OSError("ENOSPC")
        return real(p, obj)
    monkeypatch.setattr(L, "_atomic_json", f)


def test_tracks_write_failure_after_schedule_records_reconcile(tmp_path, monkeypatch):
    j = mkjob(tmp_path); ran = []
    real = L._atomic_json
    _fail_tracks(monkeypatch, nth=2)               # intent write ok, COMMIT write fails
    r = L.request_track(j, "ta", 0, 1, hooks=H(ensure_running=lambda x: ran.append(x)))
    assert ran and r["reconcile"] and "ENOSPC" in r["reconcile"]["reason"]
    assert json.loads((j / "job.json").read_text())["langs"] == ["hi", "ta"]      # scheduled work is kept, not orphaned
    assert json.loads((j / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduling"   # durable intent
    rd = L.track_readiness(j, "ta")
    assert rd["reconcile"] and rd["audio_state"] in ("preparing", "partial", "ready", "error", "absent")
    monkeypatch.setattr(L, "_atomic_json", real)                                 # disk recovers; retry reconciles
    r2 = L.request_track(j, "ta", 0, 1, hooks=H(ensure_running=lambda x: ran.append(x)))
    assert r2["reconcile"] is None and L.track_readiness(j, "ta")["reconcile"] is None
    assert json.loads((j / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduled"


def test_intent_write_failure_mutates_nothing(tmp_path, monkeypatch):
    j = mkjob(tmp_path); b = snap(j)
    _fail_tracks(monkeypatch, nth=1)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "ta", 0, 1, hooks=H(ensure_running=lambda x: (_ for _ in ()).throw(AssertionError("must not schedule"))))
    assert e.value.code == "persist_failed" and snap(j) == b


def test_tracks_write_failure_on_existing_lang_is_structured(tmp_path, monkeypatch):
    j = mkjob(tmp_path, langs=("hi", "ta"))
    _fail_tracks(monkeypatch)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(j, "ta", 0, 1, hooks=H())
    assert e.value.status == 500 and e.value.code == "persist_failed"


def test_audio_state_within_contract_enum(tmp_path):
    ENUM = {"absent", "preparing", "partial", "ready", "error"}
    j = mkjob(tmp_path, langs=("hi", "ta"))
    assert L.track_readiness(j, "ta")["audio_state"] == "preparing"             # was 'queued'
    d = j / "ta" / "c0000"; d.mkdir(parents=True); (d / "status.json").write_text('{"state":"queued"}')
    assert L.track_readiness(j, "ta")["audio_state"] in ENUM
    assert L.track_readiness(j, "zz")["audio_state"] in ENUM
