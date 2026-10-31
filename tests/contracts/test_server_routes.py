"""Route contracts of the HTTP API: status codes, response shapes and headers for every route."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from harness import CHUNKS, JOB_ID, Harness, seed_job

pytestmark = pytest.mark.filterwarnings("ignore")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- health / languages / web
def test_health_reports_admission_and_closed_dependencies(server: Harness) -> None:
    r = server.client.get("/health")
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {
        "ok",
        "ready",
        "degraded",
        "root",
        "pipeline",
        "max_concurrent",
        "admission",
        "deps",
        "lip",
        "pid",
        "time",
    }
    assert body["ok"] is True and body["ready"] is False and body["pipeline"] is True
    assert body["root"] == str(server.root.resolve())
    assert body["max_concurrent"] == 3
    assert body["admission"] == {
        "on": True,
        "max_jobs": 1,
        "running": [],
        "queued": [],
        "targets_max": 3,
        "max_queued": 4,
        "max_dur_s": 1800.0,
        "max_work_s": 5400.0,
        "fanout": 12,
        "token_required": False,
    }
    assert set(body["deps"]) == {"asr", "mt", "it2", "tts", "sep"}
    assert body["deps"]["asr"] == {"up": 0, "total": 1, "down": ["127.0.0.1:1"]}
    assert body["lip"] == {"wA": None, "wB": None}
    assert "asr:0/1 down=['127.0.0.1:1']" in body["degraded"]


def test_languages_serves_the_catalogue(server: Harness) -> None:
    r = server.client.get("/api/languages")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"targets", "sources"} and body["targets"]
    assert {"code", "name"} <= set(body["targets"][0])


def test_index_and_static_mount(server: Harness) -> None:
    r = server.client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert server.client.get("/static/styles.css").status_code == 200
    assert server.client.get("/static/nope.css").status_code == 404


def test_unknown_route_is_404(server: Harness) -> None:
    assert server.client.get("/api/nope").status_code == 404


# ---------------------------------------------------------------------------- token + CORS middleware
def test_token_guard_protects_mutating_api_calls(make_server: Callable[..., Harness]) -> None:
    h = make_server(DUB_TOKEN="s3cret")
    denied = h.client.post("/api/jobs", json={})
    assert denied.status_code == 401 and denied.json() == {"detail": "missing or wrong token"}
    assert h.client.get("/api/languages").status_code == 200  # reads are open
    assert h.client.post("/api/jobs", json={}, headers={"x-dub-token": "s3cret"}).status_code == 422
    assert h.client.post("/api/jobs?token=s3cret", json={}).status_code == 422
    assert h.client.get("/health").json()["admission"]["token_required"] is True


def test_cors_headers_on_preflight(server: Harness) -> None:
    r = server.client.options("/api/jobs", headers={"Origin": "http://x.test", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"


# ---------------------------------------------------------------------------- POST /api/jobs
def _upload(h: Harness, name: str = "clip.mp4") -> str:
    path = h.root / "uploads" / name
    path.write_bytes(b"video")
    return name


def test_create_job_writes_job_json_and_reports_mode(server: Harness) -> None:
    server.disable_job_runs()
    name = _upload(server)
    r = server.client.post("/api/jobs", json={"url": name, "langs": [" HI ", "hi", "ta"], "start": 2, "dur": 5})
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {"job_id", "langs", "mode", "state", "queue_position", "queue_len"}
    assert body["langs"] == ["hi", "ta"] and body["mode"] == "running" and body["state"] == "queued"
    assert body["queue_position"] is None and body["queue_len"] == 0
    meta = _json(server.root / body["job_id"] / "job.json")
    assert meta["langs"] == ["hi", "ta"] and meta["lipsync"] is True and meta["url"].endswith("clip.mp4")
    assert (server.root / body["job_id"] / "hi").is_dir()
    assert _json(server.root / body["job_id"] / "voice_rollout.json")["voice_plan"] == 1


def test_create_job_merges_languages_and_keeps_lipsync_setting(server: Harness) -> None:
    server.disable_job_runs()
    name = _upload(server)
    first = server.client.post("/api/jobs", json={"url": name, "langs": ["hi"], "lipsync": False}).json()
    second = server.client.post("/api/jobs", json={"url": name, "langs": ["ta"]}).json()
    assert second["job_id"] == first["job_id"] and second["langs"] == ["hi", "ta"]
    assert _json(server.root / first["job_id"] / "job.json")["lipsync"] is False


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"url": "https://example.com/v", "langs": ["hi"]}, "only youtube.com / youtu.be links are supported"),
        ({"url": "", "langs": ["hi"]}, "url is required"),
        ({"url": "missing.mp4", "langs": ["hi"]}, "file not found: missing.mp4"),
        ({"url": "/etc/passwd", "langs": ["hi"]}, "file path must live under uploads/"),
        ({"url": "https://youtu.be/x", "langs": ["EN-US!"]}, "no valid languages"),
        ({"url": "https://youtu.be/x", "langs": ["zz"]}, "Unsupported languages: ['zz']"),
        ({"url": "https://youtu.be/x", "langs": ["hi", "ta", "bn", "mr"]}, "at most 3 target languages per request"),
        ({"url": "https://youtu.be/x", "langs": ["hi"], "start": -1}, "start must be >= 0"),
        ({"url": "https://youtu.be/x", "langs": ["hi"], "dur": -1}, "dur must be >= 0"),
        ({"url": "https://youtu.be/x", "langs": ["hi"], "start": 1801}, "start must be at most 1800 s"),
        ({"url": "https://youtu.be/x", "langs": ["hi"], "dur": 1801}, "dur must be at most 1800 s"),
    ],
)
def test_create_job_validation_errors(server: Harness, payload: dict[str, Any], detail: str) -> None:
    r = server.client.post("/api/jobs", json=payload)
    assert r.status_code == 422 and r.json() == {"detail": detail}


def test_create_job_rejects_malformed_body_with_pydantic_422(server: Harness) -> None:
    r = server.client.post("/api/jobs", json={"url": "https://youtu.be/x", "langs": []})
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)


def test_create_job_rejects_too_much_work(make_server: Callable[..., Harness]) -> None:
    h = make_server(DUB_MAX_WORK_S="100")
    r = h.client.post("/api/jobs", json={"url": "https://youtu.be/x", "langs": ["hi", "ta"], "dur": 60})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("too much work: 60 s x 2 languages; this demo allows at most 100 s")


def test_create_job_busy_returns_429_with_retry_after(make_server: Callable[..., Harness]) -> None:
    h = make_server(DUB_MAX_QUEUED_JOBS="0", DUB_MAX_UNCACHED_JOBS="1")
    h.disable_job_runs()
    h.occupy_admission("aaaaaaaaaaaa")
    r = h.client.post("/api/jobs", json={"url": "https://youtu.be/x", "langs": ["hi"]})
    assert r.status_code == 429 and r.headers["retry-after"] == "60"
    assert r.json()["detail"].startswith("The demo is busy: 1 dubbing jobs")


# ---------------------------------------------------------------------------- GET /api/jobs/{id} (+ metrics)
def test_get_job_unknown_ids_are_404(server: Harness) -> None:
    for job_id in ("zzz", "0123456789ab", "NOTHEX"):
        r = server.client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 404 and r.json() == {"detail": "unknown job"}


def test_get_job_shape(server: Harness) -> None:
    seed_job(server.root)
    r = server.client.get(f"/api/jobs/{JOB_ID}")
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {
        "job_id",
        "url",
        "state",
        "queue_position",
        "queue_len",
        "error_tail",
        "services_down",
        "duration",
        "lipsync",
        "chunks",
        "speakers",
        "langs",
        "now",
    }
    assert body["state"] == "done" and body["queue_position"] is None and body["error_tail"] is None
    assert body["services_down"] == "" and body["duration"] == 15.0 and body["lipsync"] is True
    assert body["speakers"] == [{"id": "S1"}]
    assert [c["idx"] for c in body["chunks"]] == [0, 1]
    assert body["chunks"][0] == {**CHUNKS[0], "has_source": True}
    hi = body["langs"]["hi"]
    assert set(hi) == {"done", "total", "chunks", "sync"} and hi["done"] == 2 and hi["total"] == 2
    assert set(hi["chunks"][0]) == {
        "idx",
        "state",
        "stage",
        "version",
        "has_media",
        "sync",
        "started",
        "finished",
        "editing",
    }
    assert hi["chunks"][0]["version"] == 3 and hi["chunks"][0]["has_media"] is True
    assert hi["chunks"][0]["sync"] == {
        "segments": 2,
        "overflowed": 1,
        "cut_s": 0.5,
        "refit": 0,
        "tempo_mean": 1.1,
        "tempo_max": 1.1,
        "tempo_n": 1,
        "lipsync": False,
    }
    assert set(hi["sync"]) == {
        "chunks_measured",
        "segments",
        "overflowed",
        "cut_s",
        "refit",
        "tempo_mean",
        "tempo_max",
        "lipsync_chunks",
        "chunks_done",
        "lipsync_coverage",
    }
    assert hi["sync"]["chunks_done"] == 2 and hi["sync"]["lipsync_coverage"] == 0.0


def test_get_job_lists_missing_chunk_dirs_as_queued(server: Harness) -> None:
    job = seed_job(server.root, state="running")
    for path in (job / "hi" / "c0001").iterdir():
        path.unlink()
    (job / "hi" / "c0001").rmdir()
    body = server.client.get(f"/api/jobs/{JOB_ID}").json()
    assert body["langs"]["hi"]["chunks"][1] == {
        "idx": 1,
        "state": "queued",
        "stage": "",
        "version": 0,
        "has_media": False,
    }
    assert body["langs"]["hi"]["done"] == 1 and body["state"] == "running"


def test_get_job_metrics_shape(server: Harness) -> None:
    seed_job(server.root)
    r = server.client.get(f"/api/jobs/{JOB_ID}/metrics")
    body = r.json()
    assert r.status_code == 200 and set(body) == {"job_id", "metrics", "null"}
    assert set(body["metrics"]["hi"]) == {"total", "chunks"} and set(body["metrics"]["hi"]["chunks"]) == {"0", "1"}
    assert server.client.get("/api/jobs/ffffffffffff/metrics").status_code == 404


# ---------------------------------------------------------------------------- POST /api/jobs/{id}/focus
def test_focus_persists_and_increments_generation(server: Harness) -> None:
    seed_job(server.root)
    first = server.client.post(f"/api/jobs/{JOB_ID}/focus", json={"lang": "HI", "chunk": 1})
    second = server.client.post(f"/api/jobs/{JOB_ID}/focus", json={"lang": "hi", "chunk": 0})
    assert first.status_code == 200 and first.json() == {"ok": True, "gen": 1}
    assert second.json() == {"ok": True, "gen": 2}
    saved = _json(server.root / "focus.json")
    assert saved["job"] == JOB_ID and saved["lang"] == "hi" and saved["chunk"] == 0 and saved["gen"] == 2


@pytest.mark.parametrize(
    ("job_id", "payload", "status", "detail"),
    [
        ("ffffffffffff", {"lang": "hi", "chunk": 0}, 404, "job not found"),
        (JOB_ID, {"lang": "ta", "chunk": 0}, 422, "language is not part of this job"),
        (JOB_ID, {"lang": "hi", "chunk": 9}, 422, "unknown chunk"),
    ],
)
def test_focus_errors(server: Harness, job_id: str, payload: dict[str, Any], status: int, detail: str) -> None:
    seed_job(server.root)
    r = server.client.post(f"/api/jobs/{job_id}/focus", json=payload)
    assert r.status_code == status and r.json() == {"detail": detail}


# ---------------------------------------------------------------------------- segments: list / edit / undo
def test_list_segments_returns_global_times_sorted(server: Harness) -> None:
    seed_job(server.root)
    r = server.client.get(f"/api/jobs/{JOB_ID}/hi/segments")
    segs = r.json()["segments"]
    assert r.status_code == 200 and len(segs) == 4
    assert [s["start"] for s in segs] == [1.0, 3.0, 11.0, 13.0]
    assert set(segs[1]) == {
        "chunk",
        "id",
        "start",
        "end",
        "speaker",
        "gender",
        "src_text",
        "mt_text",
        "tgt_text",
        "edited",
        "fallback",
        "fallback_reason",
        "history_len",
    }
    assert segs[1]["history_len"] == 1 and segs[1]["edited"] is True and segs[0]["fallback"] is False


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        ("/api/jobs/NOTAJOB/hi/segments", "unknown job"),
        (f"/api/jobs/{JOB_ID}/HI/segments", "unknown lang"),
        (f"/api/jobs/{JOB_ID}/ta/segments", "unknown job/lang"),
    ],
)
def test_list_segments_404s(server: Harness, path: str, detail: str) -> None:
    seed_job(server.root)
    r = server.client.get(path)
    assert r.status_code == 404 and r.json() == {"detail": detail}


def _edit_url(seg: str = "s1", chunk: int = 0, lang: str = "hi") -> str:
    return f"/api/jobs/{JOB_ID}/{lang}/chunks/{chunk}/segments/{seg}"


def test_edit_segment_rewrites_text_and_bumps_version(server: Harness) -> None:
    job = seed_job(server.root)
    calls: list[tuple[str, list[str]]] = []

    async def fake(
        module: str,
        args: list[str],
        cwd: Path,
        log_path: Path,
        extra_env: Any = None,
        timeout: float = 0,
        kind: str = "job",
    ) -> tuple[int, str]:
        calls.append((kind, args))
        return 0, ""

    server.fake_process(fake)
    r = server.client.post(_edit_url(), json={"text": "new words"})
    body = r.json()
    assert r.status_code == 200 and set(body) == {"ok", "version", "segment"}
    assert body["ok"] is True and body["version"] == 4
    assert (
        body["segment"]["tgt_text"] == "new words"
        and body["segment"]["history"] == ["b"]
        and body["segment"]["edited"] is True
    )
    assert calls[0][0] == "resynth" and calls[0][1][-4:] == ["--resynth", "s1", "--text", "new words"]
    out = job / "hi" / "c0000"
    assert _json(out / "status.json")["editing"] is False and not (out / "segments.json.pre_edit").exists()


def test_edit_failure_restores_segments(server: Harness) -> None:
    job = seed_job(server.root)
    before = (job / "hi" / "c0000" / "segments.json").read_bytes()

    async def failing(*_a: Any, **_k: Any) -> tuple[int, str]:
        return 3, "boom"

    server.fake_process(failing)
    r = server.client.post(_edit_url(), json={"text": "x"})
    assert r.status_code == 500 and r.json()["detail"] == "resynth failed (exit 3): boom"
    assert (job / "hi" / "c0000" / "segments.json").read_bytes() == before
    assert _json(job / "hi" / "c0000" / "status.json")["editing"] is False


def test_undo_restores_previous_text(server: Harness) -> None:
    seed_job(server.root)

    async def ok(*_a: Any, **_k: Any) -> tuple[int, str]:
        return 0, ""

    server.fake_process(ok)
    r = server.client.post(_edit_url("s2") + "/undo")
    body = r.json()
    assert r.status_code == 200 and body["segment"]["tgt_text"] == "old"
    assert body["segment"]["history"] == [] and body["segment"]["edited"] is False and body["version"] == 4
    nothing = server.client.post(_edit_url("s2") + "/undo")
    assert nothing.status_code == 409 and nothing.json() == {"detail": "nothing to undo"}


@pytest.mark.parametrize(
    ("kwargs", "body", "status", "detail"),
    [
        ({}, {"text": "   "}, 422, "text must not be empty"),
        ({}, {"text": "x" * 601}, 422, "text too long (601 chars; the limit is 600)"),
        ({"seg": "bad id!"}, {"text": "x"}, 400, "invalid seg id"),
        ({"seg": "zz"}, {"text": "x"}, 404, "segment not found"),
        ({"chunk": 7}, {"text": "x"}, 404, "chunk not found in plan"),
        ({"lang": "ta"}, {"text": "x"}, 404, "unknown job/lang"),
    ],
)
def test_edit_errors(server: Harness, kwargs: dict[str, Any], body: dict[str, str], status: int, detail: str) -> None:
    seed_job(server.root)
    r = server.client.post(_edit_url(**kwargs), json=body)
    assert r.status_code == status and r.json() == {"detail": detail}


def test_edit_requires_done_chunk(server: Harness) -> None:
    job = seed_job(server.root)
    (job / "hi" / "c0000" / "status.json").write_text(json.dumps({"state": "running"}))
    r = server.client.post(_edit_url(), json={"text": "x"})
    assert r.status_code == 409 and r.json() == {"detail": "chunk not done (state=running)"}


# ---------------------------------------------------------------------------- media
def test_chunk_media_streams_dub(server: Harness) -> None:
    seed_job(server.root)
    r = server.client.get(f"/media/{JOB_ID}/hi/chunks/0.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and r.content == b"dub-bytes"
    assert r.headers["accept-ranges"] == "bytes"
    part = server.client.get(f"/media/{JOB_ID}/hi/chunks/0.mp4", headers={"Range": "bytes=0-2"})
    assert part.status_code == 206 and part.content == b"dub" and part.headers["content-range"] == "bytes 0-2/9"


def test_chunk_media_prefers_newer_lipsync_unless_disabled(server: Harness) -> None:
    job = seed_job(server.root)
    out = job / "hi" / "c0000"
    os.utime(out / "dub_hi.mp4", (1_000_000, 1_000_000))
    (out / "dub_hi_lipsync.mp4").write_bytes(b"lip")
    assert server.client.get(f"/media/{JOB_ID}/hi/chunks/0.mp4").content == b"lip"
    assert server.client.get(f"/media/{JOB_ID}/hi/chunks/0.mp4?lip=0").content == b"dub-bytes"
    meta = _json(job / "job.json")
    (job / "job.json").write_text(json.dumps({**meta, "lipsync": False}))
    assert server.client.get(f"/media/{JOB_ID}/hi/chunks/0.mp4").content == b"dub-bytes"


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        ("/media/NOTAJOB/hi/chunks/0.mp4", "unknown job"),
        (f"/media/{JOB_ID}/HI/chunks/0.mp4", "unknown lang"),
        (f"/media/{JOB_ID}/hi/chunks/-1.mp4", "chunk dir invalid"),
        (f"/media/{JOB_ID}/hi/chunks/9.mp4", "video not ready"),
        (f"/media/{JOB_ID}/original/9.mp4", "chunk source not found"),
        ("/media/NOTAJOB/original/0.mp4", "unknown job"),
    ],
)
def test_media_404s(server: Harness, path: str, detail: str) -> None:
    seed_job(server.root)
    r = server.client.get(path)
    assert r.status_code == 404 and r.json() == {"detail": detail}


def test_chunk_media_not_served_while_running(server: Harness) -> None:
    job = seed_job(server.root)
    (job / "hi" / "c0000" / "status.json").write_text(json.dumps({"state": "running"}))
    r = server.client.get(f"/media/{JOB_ID}/hi/chunks/0.mp4")
    assert r.status_code == 404 and r.json() == {"detail": "video not ready"}


def test_original_chunk_media_uses_stable_src(server: Harness) -> None:
    job = seed_job(server.root)
    r = server.client.get(f"/media/{JOB_ID}/original/0.mp4")
    assert r.status_code == 200 and r.content == b"orig-bytes" and r.headers["content-type"] == "video/mp4"
    os.utime(job / "hi" / "c0000" / "src.mp4", None)  # touched just now: not yet stable
    assert server.client.get(f"/media/{JOB_ID}/original/0.mp4").status_code == 404
