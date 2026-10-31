"""Crash-window tests for add-language: real process death (``os._exit``) inside ``request_track``, then a fresh process.

``LS_UNDER_TEST`` can point at another copy of the scheduler to check that a test discriminates (it must fail there).
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import sunwai.jobs.language_scheduler as _ls

LS = os.environ.get("LS_UNDER_TEST") or str(Path(_ls.__file__))
CH = [(0, 0.0, 18.05), (1, 18.05, 24.2), (2, 42.25, 37.5)]


def mkjob(root, name, langs=("hi",), status="done"):
    j = root / name
    j.mkdir(parents=True)
    (j / "job.json").write_text(json.dumps({"job_id": name, "langs": list(langs)}))
    (j / "plan.json").write_text(json.dumps({"chunks": [{"idx": i, "start": s, "dur": d} for i, s, d in CH]}))
    (j / "speakers_global.json").write_text("{}")
    (j / "status.json").write_text(json.dumps({"state": status}))
    return j


CHILD = textwrap.dedent('''
    import importlib.util, os, sys
    spec = importlib.util.spec_from_file_location("lang_sched", sys.argv[1]); L = importlib.util.module_from_spec(spec); sys.modules["lang_sched"] = L; spec.loader.exec_module(L)
    from pathlib import Path
    jd, point = Path(sys.argv[2]), sys.argv[3]
    called = {"ensure": False}
    def ensure(jid):
        called["ensure"] = True
        if point == "in_ensure":
            os._exit(9)
    from sunwai.jobs.scheduler import store
    real = store.atomic_json
    def aj(p, obj):
        if point == "before_commit" and Path(p).name == "tracks.json" and called["ensure"]:
            os._exit(9)
        return real(p, obj)
    store.atomic_json = aj
    h = L.Hooks(ensure_running=ensure, target_codes=lambda: {"hi", "ta", "bn"})
    L.request_track(jd, "ta", 0, 1, hooks=h)
''')


def crash(root, jd, point):
    env = dict(os.environ, DUB_ADD_LANG="1")
    p = subprocess.run([sys.executable, "-I", "-c", CHILD, LS, str(jd), point], env=env, capture_output=True, text=True)
    assert p.returncode == 9, p.stderr[-800:]


def load():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ls_under_test", LS)
    m = importlib.util.module_from_spec(spec); sys.modules["ls_under_test"] = m; spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def on(monkeypatch):
    monkeypatch.setenv("DUB_ADD_LANG", "1")


@pytest.mark.parametrize("point", ["in_ensure", "before_commit"])
def test_cap_survives_a_crash(tmp_path, point):
    """After a crash mid-transaction the stranded track still counts toward the global cap (old code: in-memory only -> bypass)."""
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    crash(tmp_path, A, point)
    L = load()
    assert "ta" in json.loads((A / "job.json").read_text())["langs"]          # the merge really happened before the crash
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(B, "bn", 0, 1, hooks=L.Hooks(target_codes=lambda: {"hi", "ta", "bn"}))
    assert e.value.status == 429 and e.value.code == "busy"


@pytest.mark.parametrize("point", ["in_ensure", "before_commit"])
def test_stranded_track_is_rolled_forward_on_recovery(tmp_path, point):
    """A finished job whose new language was merged but never scheduled is NOT resumed by server2's startup reconcile
    (status done). recover_all must call ensure_running and commit the track."""
    A = mkjob(tmp_path, "aaaaaaaaaaaa")
    crash(tmp_path, A, point)
    L = load()
    rd = L.track_readiness(A, "ta")
    assert rd["reconcile"], "stranded track must be reported, not silent"
    ran = []
    acts = L.recover_all(tmp_path, L.Hooks(ensure_running=lambda j: ran.append(j)))
    assert ran == ["aaaaaaaaaaaa"] and acts[0]["action"] == "rolled_forward"
    assert json.loads((A / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduled"
    assert L.track_readiness(A, "ta")["reconcile"] is None
    assert L.recover_all(tmp_path, L.Hooks(ensure_running=lambda j: ran.append(j))) == []     # idempotent


def test_crash_before_merge_drops_intent(tmp_path):
    A = mkjob(tmp_path, "aaaaaaaaaaaa")
    L = load()
    (A / "tracks.json").write_text(json.dumps({"gen": 1, "tracks": {"ta": {"phase": "scheduling", "created": __import__("time").time(), "from_ms": 0, "generation": 1}}}))
    B = mkjob(tmp_path, "bbbbbbbbbbbb")
    # intent exists but lang was never merged: not in-flight (no work), and recovery drops it
    assert L.track_readiness(A, "ta")["audio_state"] == "absent"
    L.request_track(B, "bn", 0, 1, hooks=L.Hooks(target_codes=lambda: {"hi", "ta", "bn"}))   # not blocked
    assert L.recover_all(tmp_path, L.Hooks())[0]["action"] == "intent_dropped"
    assert "ta" not in json.loads((A / "tracks.json").read_text())["tracks"]


def test_retry_after_crash_recovers_then_is_idempotent(tmp_path):
    A = mkjob(tmp_path, "aaaaaaaaaaaa")
    crash(tmp_path, A, "before_commit")
    L = load(); ran = []
    r = L.request_track(A, "ta", 0, 2, hooks=L.Hooks(ensure_running=lambda j: ran.append(j), target_codes=lambda: {"hi", "ta"}))
    assert ran == ["aaaaaaaaaaaa"] and r["reconcile"] is None
    assert json.loads((A / "job.json").read_text())["langs"] == ["hi", "ta"]


def test_missing_chunk_state_does_not_release_cap(tmp_path):
    """review r3: job 'done' + no chunk status (purged) is NOT abandonment; only an explicit marker releases."""
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    L = load()
    L.request_track(A, "ta", 0, 1, hooks=L.Hooks(ensure_running=lambda j: None, target_codes=lambda: {"hi", "ta", "bn"}))
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(B, "bn", 0, 1, hooks=L.Hooks(target_codes=lambda: {"hi", "ta", "bn"}))
    assert e.value.status == 429


def test_release_marker_frees_cap_and_retry_reschedules(tmp_path):
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    L = load(); ran = []
    h = L.Hooks(ensure_running=lambda j: ran.append(j), target_codes=lambda: {"hi", "ta", "bn"})
    L.request_track(A, "ta", 0, 1, hooks=h)
    assert L.release_track(A, "ta", "operation failed") and not L.release_track(A, "ta")
    assert json.loads((A / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "abandoned"
    assert L.request_track(B, "bn", 0, 1, hooks=h)["created"]                  # cap freed by the marker
    L.release_track(B, "bn", "x")
    ran.clear()
    r = L.request_track(A, "ta", 5000, 2, hooks=h)                              # retry of an abandoned track: re-ensure, no second merge
    assert r["created"] and ran == ["aaaaaaaaaaaa"]
    assert json.loads((A / "job.json").read_text())["langs"] == ["hi", "ta"]
    assert json.loads((A / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduled"


def test_retry_blocked_by_cap_429(tmp_path):
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    L = load(); h = L.Hooks(ensure_running=lambda j: None, target_codes=lambda: {"hi", "ta", "bn"})
    L.request_track(A, "ta", 0, 1, hooks=h); L.release_track(A, "ta", "x")
    L.request_track(B, "bn", 0, 1, hooks=h)
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(A, "ta", 0, 2, hooks=h)
    assert e.value.status == 429

def test_kill_switch_stops_recovery(tmp_path, monkeypatch):
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); crash(tmp_path, A, "in_ensure")
    L = load(); ran = []
    monkeypatch.setenv("DUB_ADD_LANG", "0")
    assert L.recover_all(tmp_path, L.Hooks(ensure_running=lambda j: ran.append(j))) == [] and not ran
    assert json.loads((A / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduling"   # intent intact
    monkeypatch.setenv("DUB_ADD_LANG", "1")
    assert L.recover_all(tmp_path, L.Hooks(ensure_running=lambda j: ran.append(j)))[0]["action"] == "rolled_forward"


def test_recovery_without_hook_stays_scheduling(tmp_path):
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); crash(tmp_path, A, "in_ensure")
    L = load()
    acts = L.recover_all(tmp_path, L.Hooks())
    assert acts[0]["action"] == "roll_forward_failed"
    assert json.loads((A / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduling"


def test_live_track_never_expires_by_time(tmp_path, monkeypatch):
    """Job running, chunk running with an ANCIENT mtime and created 10 h ago: still counts."""
    A = mkjob(tmp_path, "aaaaaaaaaaaa", status="running"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    L = load()
    L.request_track(A, "ta", 0, 1, hooks=L.Hooks(ensure_running=lambda j: None, target_codes=lambda: {"hi", "ta", "bn"}))
    t = json.loads((A / "tracks.json").read_text()); t["tracks"]["ta"]["created"] -= 36000
    (A / "tracks.json").write_text(json.dumps(t))
    d = A / "ta" / "c0000"; d.mkdir(parents=True); (d / "status.json").write_text('{"state":"running"}')
    os.utime(d / "status.json", (1, 1))
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(B, "bn", 0, 1, hooks=L.Hooks(target_codes=lambda: {"hi", "ta", "bn"}))
    assert e.value.status == 429


def test_request_on_unrecovered_track_is_not_success(tmp_path):
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); crash(tmp_path, A, "in_ensure")
    L = load()
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(A, "ta", 0, 2, hooks=L.Hooks(target_codes=lambda: {"hi", "ta"}))      # no ensure hook: recovery cannot run
    assert e.value.code == "recovery_pending" and e.value.status == 503


def test_service_release_hook_wiring(tmp_path):
    """ctl_service._release_lang calls ctl_hooks.release_language; failures in the hook never propagate."""
    pytest.importorskip("fastapi")
    from sunwai.control import hooks as ctl_hooks
    from sunwai.control import service as ctl_service
    got = []
    ctl_hooks.release_language = lambda j, l, r: got.append((j, l, r))
    ctl_service.ActionService.release_language("jid", "ta", "boom")
    assert got == [("jid", "ta", "boom")]
    ctl_hooks.release_language = lambda *a: (_ for _ in ()).throw(RuntimeError("x"))
    ctl_service.ActionService.release_language("jid", "ta", "boom")         # swallowed


def test_marker_frees_only_when_job_task_dead(tmp_path):
    """review r4: a release marker written while the job task may still schedule chunks must NOT free the cap (TOCTOU)."""
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    L = load(); alive = {"v": True}
    h = L.Hooks(ensure_running=lambda j: None, target_codes=lambda: {"hi", "ta", "bn"}, job_active=lambda j: alive["v"] if j == "aaaaaaaaaaaa" else False)
    L.request_track(A, "ta", 0, 1, hooks=h); L.release_track(A, "ta", "watcher failed")
    with pytest.raises(L.LangSchedError) as e:
        L.request_track(B, "bn", 0, 1, hooks=h)
    assert e.value.status == 429                                  # job task still alive: work may still start
    alive["v"] = False                                            # no retry needed: frees by itself
    assert L.request_track(B, "bn", 0, 1, hooks=h)["created"]


def test_orphaned_committed_track_frees_after_grace_when_job_dead(tmp_path, monkeypatch):
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); B = mkjob(tmp_path, "bbbbbbbbbbbb")
    L = load()
    h = L.Hooks(ensure_running=lambda j: None, target_codes=lambda: {"hi", "ta", "bn"}, job_active=lambda j: False)
    L.request_track(A, "ta", 0, 1, hooks=h)
    with pytest.raises(L.LangSchedError):
        L.request_track(B, "bn", 0, 1, hooks=h)                   # young: still counts
    t = json.loads((A / "tracks.json").read_text()); t["tracks"]["ta"]["created"] -= 7200
    (A / "tracks.json").write_text(json.dumps(t))
    assert L.request_track(B, "bn", 0, 1, hooks=h)["created"]
    # unknown liveness (no hook) never frees a scheduled track
    C = mkjob(tmp_path, "cccccccccccc")
    h2 = L.Hooks(target_codes=lambda: {"hi", "ta", "bn"})
    L.SCHEDULER.inflight.clear()
    with pytest.raises(L.LangSchedError):
        L.request_track(C, "ta", 0, 1, hooks=h2)


def test_retry_commit_failure_is_reconcile_not_error(tmp_path, monkeypatch):
    A = mkjob(tmp_path, "aaaaaaaaaaaa")
    L = load(); ran = []
    h = L.Hooks(ensure_running=lambda j: ran.append(j), target_codes=lambda: {"hi", "ta"})
    L.request_track(A, "ta", 0, 1, hooks=h); L.release_track(A, "ta", "x"); ran.clear()
    from sunwai.jobs.scheduler import store
    real = store.atomic_json; n = {"c": 0}
    def aj(p, o):
        if Path(p).name == "tracks.json":
            n["c"] += 1
            if n["c"] == 2:
                raise OSError("ENOSPC")
        return real(p, o)
    monkeypatch.setattr(store, "atomic_json", aj)
    r = L.request_track(A, "ta", 0, 2, hooks=h)
    assert ran and r["reconcile"] and "ENOSPC" in r["reconcile"]["reason"]


def test_startup_recovery_hook_registers_and_runs_on_real_fastapi(tmp_path, monkeypatch):
    """The first live deploy found app.add_event_handler missing on the live FastAPI: use app.on_event('startup') (what server2 itself uses)."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    sys.modules.pop("lang_sched", None)
    import importlib.util
    spec = importlib.util.spec_from_file_location("lang_sched", LS); L = importlib.util.module_from_spec(spec); sys.modules["lang_sched"] = L; spec.loader.exec_module(L)
    from sunwai.control import adapters as ctl_adapters
    A = mkjob(tmp_path, "aaaaaaaaaaaa"); crash(tmp_path, A, "in_ensure")
    ran = []
    app = FastAPI()
    ns = {"app": app, "DUB_ROOT": tmp_path, "ensure_job_running": lambda j: ran.append(j) or "running", "get_target_codes": lambda: {"hi", "ta"},
          "resolve_job": lambda j: tmp_path / j, "get_lock": None, "resolve_chunk_dir": lambda *a: None}
    monkeypatch.setenv("DUB_CTL_ADDLANG", "1")
    got = ctl_adapters.install(ns, log=lambda *a: None)
    assert "language_scheduler" in got
    with TestClient(app):                      # triggers startup handlers
        pass
    assert ran == ["aaaaaaaaaaaa"]
    assert json.loads((A / "tracks.json").read_text())["tracks"]["ta"]["phase"] == "scheduled"
