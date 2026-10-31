"""Fixtures for the HTTP route-contract tests."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from harness import Harness, build_harness, write_services_file


@pytest.fixture
def make_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Harness]]:
    """Factory fixture: ``make_server(DUB_TOKEN="x")`` builds an app with extra environment overrides."""
    root = tmp_path / "jobs"
    (root / "uploads").mkdir(parents=True)
    base = {"DUB_ROOT": str(root), "DUB_SERVICES": str(write_services_file(tmp_path)), "DUB_ACTIONS": "0"}
    for key in list(os.environ):
        if key.startswith(("DUB_", "LIP_")) and key not in base:
            monkeypatch.delenv(key, raising=False)
    for key, value in base.items():
        monkeypatch.setenv(key, value)

    def factory(**overrides: str) -> Harness:
        for name in [m for m in sys.modules if m.startswith("sunwai.server")]:
            if name.endswith("legacy_app"):
                sys.modules.pop(name)
        importlib.invalidate_caches()
        env = {**base, **overrides}
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return build_harness(tmp_path, env)

    yield factory


@pytest.fixture
def server(make_server: Callable[..., Harness]) -> Harness:
    """Default app: admission on, no token."""
    return make_server()
