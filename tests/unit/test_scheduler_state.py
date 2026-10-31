"""Pure scheduling state and decisions of the add-language scheduler (CPU only, no server)."""

import json
import time

import pytest

from sunwai.jobs.scheduler import requests, store
from sunwai.jobs.scheduler.core import LanguageScheduler
from sunwai.jobs.scheduler.errors import Hooks, LangSchedError
from sunwai.jobs.scheduler.ordering import chunk_for_ms, merge_intervals, plan_order, runway
from sunwai.jobs.scheduler.readiness import audio_state, lipsync_state
from sunwai.jobs.scheduler.script_lid import classify_text, lang_group, mask_foreign_terms, route_segment, unmask_terms
from sunwai.jobs.scheduler.transaction import JobSnapshot

CHUNKS = [
    {"idx": 0, "start": 0.0, "dur": 10.0},
    {"idx": 1, "start": 10.0, "dur": 10.0},
    {"idx": 2, "start": 20.0, "dur": 10.0},
    {"idx": 3, "start": 30.0, "dur": 10.0},
]


def make_job(root, name="aaaaaaaaaaaa", langs=("hi",)):
    job = root / name
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"job_id": name, "langs": list(langs)}))
    plan = {"chunks": [{"idx": c["idx"], "start": c["start"], "dur": c["dur"]} for c in CHUNKS]}
    (job / "plan.json").write_text(json.dumps(plan))
    (job / "speakers_global.json").write_text("{}")
    return job


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("DUB_ADD_LANG", "1")
    monkeypatch.setenv("DUB_MAX_ADDLANG_INFLIGHT", "1")


# ---- ordering


def test_chunk_for_ms_clamps_to_ends():
    assert chunk_for_ms(CHUNKS, 15_000) == 1
    assert chunk_for_ms(CHUNKS, 999_000) == 3
    assert chunk_for_ms([], 5) == 0


def test_runway_and_plan_order_put_playhead_first_then_ahead_then_behind():
    assert runway(CHUNKS, 12_000, lead=10_000) == [1, 2]
    assert plan_order(CHUNKS, 22_000, lead=0) == [2, 3, 1, 0]


def test_merge_intervals_joins_touching_ranges():
    assert merge_intervals([[0, 10], [10, 20], [30, 40]]) == [[0, 20], [30, 40]]


# ---- request validation


def test_parse_request_normalises_and_rejects():
    req = requests.parse_request(" TA ", "1500.9", "3")
    assert (req.lang, req.from_ms, req.generation) == ("ta", 1500, 3)
    for args, code in [(("tamil", 0, 0), "bad_lang"), (("ta", -1, 0), "bad_args"), (("ta", "x", 0), "bad_args")]:
        with pytest.raises(LangSchedError) as err:
            requests.parse_request(*args)
        assert err.value.code == code


@pytest.mark.parametrize(
    ("speakers_global", "expected"),
    [
        ({}, (True, "legacy")),
        ({"voice_plan": 3}, (False, "voice_plan_invalid")),
        ({"voice_plan": {"langs": ["ta"], "voices": {"ta": {}}}}, (True, "planned")),
        ({"voice_plan": {"langs": ["ta"], "voices": {"ta": {}}, "unavailable_langs": {"ta": "no"}}}, (False, None)),
        ({"voice_plan": {"langs": ["hi"], "voices": {"hi": {}}}}, (False, None)),
    ],
)
def test_voice_plan_verdict(speakers_global, expected):
    ok, why = requests.voice_plan_verdict(speakers_global, "ta")
    assert ok is expected[0]
    if expected[1]:
        assert why == expected[1]


def test_releases_cap_rules(monkeypatch):
    monkeypatch.setenv("DUB_ADDLANG_ORPHAN_S", "100")
    now = 10_000.0
    assert requests.releases_cap({"phase": "abandoned"}, job_alive=False)
    assert requests.releases_cap({"phase": "abandoned"}, job_alive=None)
    assert not requests.releases_cap({"phase": "abandoned"}, job_alive=True)
    assert not requests.releases_cap({"phase": "scheduling", "created": 0}, job_alive=False, now=now)
    assert not requests.releases_cap({"phase": "scheduled", "created": now - 50}, job_alive=False, now=now)
    assert requests.releases_cap({"phase": "scheduled", "created": now - 500}, job_alive=False, now=now)
    assert not requests.releases_cap({"phase": "scheduled", "created": 0}, job_alive=None, now=now)


def test_touched_record_counts_requests_and_keeps_phase():
    req = requests.TrackRequest("ta", 5, 2)
    prev = {"created": 1.0, "requests": 2, "phase": "scheduled"}
    rec = requests.touched_record(prev, req, "hi", now=9.0)
    assert (rec["created"], rec["requests"], rec["phase"]) == (1.0, 3, "scheduled")


# ---- readiness summaries


def test_audio_and_lipsync_state_enums():
    assert audio_state(3, 3, 0, 0) == "ready"
    assert audio_state(3, 1, 0, 0) == "partial"
    assert audio_state(3, 0, 0, 1) == "error"
    assert audio_state(3, 0, 1, 1) == "preparing"
    assert audio_state(3, 0, 0, 0) == "preparing"
    assert [lipsync_state(2, n) for n in (0, 1, 2)] == ["none", "partial", "ready"]


# ---- scheduler state


def test_cap_counts_a_scheduled_track_and_frees_it_on_release(tmp_path):
    scheduler = LanguageScheduler()
    a = make_job(tmp_path, "aaaaaaaaaaaa")
    b = make_job(tmp_path, "bbbbbbbbbbbb")
    hooks = Hooks(ensure_running=lambda _jid: None, target_codes=lambda: {"hi", "ta", "bn"})
    first = scheduler.request_track(a, "ta", 0, 1, hooks=hooks)
    assert first["created"] is True
    assert scheduler.inflight_on_disk(tmp_path) == {(str(a), "ta")}
    with pytest.raises(LangSchedError) as err:
        scheduler.request_track(b, "bn", 0, 1, hooks=hooks)
    assert err.value.code == "busy"
    assert scheduler.release_track(a, "ta", "test") is True
    assert scheduler.release_track(a, "ta", "again") is False
    assert scheduler.request_track(b, "bn", 0, 1, hooks=hooks)["created"] is True


def test_repeat_request_is_idempotent_and_obsolete_generation_rejected(tmp_path):
    scheduler = LanguageScheduler()
    job = make_job(tmp_path)
    hooks = Hooks(ensure_running=lambda _jid: None, target_codes=lambda: {"hi", "ta"})
    scheduler.request_track(job, "ta", 0, 5, hooks=hooks)
    again = scheduler.request_track(job, "ta", 1000, 6, hooks=hooks)
    assert again["created"] is False
    assert store.read_json(job / "tracks.json")["tracks"]["ta"]["requests"] == 2
    with pytest.raises(LangSchedError) as err:
        scheduler.request_track(job, "ta", 0, 1, hooks=hooks)
    assert err.value.code == "obsolete_generation"


def test_failed_ensure_running_rolls_back_the_job(tmp_path):
    scheduler = LanguageScheduler()
    job = make_job(tmp_path)
    before = (job / "job.json").read_bytes()

    def boom(_jid):
        raise RuntimeError("no worker")

    hooks = Hooks(ensure_running=boom, target_codes=lambda: {"hi", "ta"})
    with pytest.raises(LangSchedError) as err:
        scheduler.request_track(job, "ta", 0, 1, hooks=hooks)
    assert err.value.code == "schedule_failed"
    assert (job / "job.json").read_bytes() == before
    assert not (job / "tracks.json").exists() and not (job / "ta").exists() and not scheduler.inflight


def test_recover_job_rolls_forward_merged_language_and_drops_unmerged(tmp_path):
    scheduler = LanguageScheduler()
    job = make_job(tmp_path, langs=("hi", "ta"))
    intent = {"phase": "scheduling", "created": time.time()}
    store.atomic_json(job / "tracks.json", {"gen": 1, "tracks": {"ta": dict(intent), "bn": dict(intent)}})
    started = []
    actions = scheduler.recover_job(job, Hooks(ensure_running=started.append))
    assert {(a["lang"], a["action"]) for a in actions} == {("ta", "rolled_forward"), ("bn", "intent_dropped")}
    assert started == ["aaaaaaaaaaaa"]
    assert list(store.read_json(job / "tracks.json")["tracks"]) == ["ta"]


def test_job_snapshot_restores_files_and_new_language_dir(tmp_path):
    job = make_job(tmp_path)
    snapshot = JobSnapshot(job, "ta")
    (job / "job.json").write_text("{}")
    (job / "ta").mkdir()
    (job / "tracks.json").write_text("{}")
    snapshot.restore(restore_job=True)
    assert json.loads((job / "job.json").read_text())["langs"] == ["hi"]
    assert not (job / "ta").exists() and not (job / "tracks.json").exists()


# ---- source language identification


def test_classify_text_uses_script_and_job_lid():
    assert classify_text("வணக்கம் நண்பரே", None)["lang"] == "ta"
    marathi = classify_text("नमस्कार मित्रांनो कसे आहात", "mr")
    assert (marathi["lang"], marathi["method"]) == ("mr", "script+job_lid")
    assert classify_text("ab", "hi")["lang"] == "und"
    assert lang_group("ur") == lang_group("hi")


def test_route_segment_and_term_masking_round_trip():
    assert route_segment({"lang": "ta", "confidence": 0.95}, "ta")["route"] == "passthrough"
    assert route_segment({"lang": "hi"}, "en")["route"] == "gemini-en"
    assert route_segment({"lang": "hi"}, "mni")["route"] == "it2"
    masked, mapping = mask_foreign_terms("नमस्ते Google दोस्तों")
    assert "Google" not in masked and mapping
    restored, missing = unmask_terms(masked, mapping)
    assert restored == "नमस्ते Google दोस्तों" and missing == []
