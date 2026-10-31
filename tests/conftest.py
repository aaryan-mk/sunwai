"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep tests independent of the developer's environment: no stray SUNWAI_* or ``.env`` influence."""
    import os

    for name in [n for n in os.environ if n.startswith("SUNWAI_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
