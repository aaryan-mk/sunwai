"""Torch-free parts of the model mirrors: body limit middleware, admission control and path sandbox."""

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from sunwai.mirrors.admission import Admission, Limits
from sunwai.mirrors.body_limit import BodyLimitMiddleware
from sunwai.mirrors.separator_config import SeparatorConfig


def build_client(limit):
    api = FastAPI()
    api.add_middleware(BodyLimitMiddleware, limit_fn=lambda _path: limit)

    @api.post("/echo")
    async def echo(request: Request):
        return {"size": len(await request.body())}

    return TestClient(api)


def test_body_limit_allows_small_and_rejects_large_bodies():
    client = build_client(10)
    assert client.post("/echo", content=b"x" * 10).json() == {"size": 10}
    response = client.post("/echo", content=b"x" * 11)
    assert response.status_code == 413 and "too large" in response.json()["detail"]


def test_body_limit_bounds_chunked_uploads():
    client = build_client(10)
    response = client.post("/echo", content=(b"x" for _ in range(20)))
    assert response.status_code == 413


def test_admission_rejects_long_text_and_full_queue():
    gate = Admission(Limits(max_pending=1, max_chars=5))
    gate.check(["short"])
    with pytest.raises(HTTPException) as long_text:
        gate.check(["too long!"])
    assert long_text.value.status_code == 413
    gate.pending += 1
    with pytest.raises(HTTPException) as busy:
        gate.check(["ok"])
    assert busy.value.status_code == 503
    gate.release()
    gate.check(["ok"])


def test_separator_root_is_component_wise(tmp_path):
    root = tmp_path / "amk"
    root.mkdir()
    (tmp_path / "amk-evil").mkdir()
    config = SeparatorConfig(root=str(root))
    assert config.contains(str(root)) and config.contains(str(root / "job" / "a.wav"))
    assert not config.contains(str(tmp_path / "amk-evil" / "a.wav"))
    assert not config.contains(str(root / ".." / "elsewhere"))


def test_limits_from_env(monkeypatch):
    monkeypatch.setenv("IT2_MAX_PENDING", "3")
    monkeypatch.setenv("IT2_MAX_CHARS", "7")
    limits = Limits.from_env()
    assert (limits.max_pending, limits.max_chars) == (3, 7)
