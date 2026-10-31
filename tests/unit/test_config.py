"""Settings groups and the legacy-variable bridge."""

from __future__ import annotations

import pytest

from sunwai.core.config import Settings
from sunwai.core.legacy_env import BY_NAME, LEGACY_VARS, LegacyVar, export_to_environ


def test_every_legacy_variable_maps_to_a_real_field() -> None:
    settings = Settings()
    for var in LEGACY_VARS:
        group = getattr(settings, var.group)
        assert hasattr(group, var.field), var.name
    assert len(BY_NAME) == len(LEGACY_VARS), "duplicate legacy variable"


def test_defaults_are_unset_so_engine_defaults_apply() -> None:
    settings = Settings()
    assert settings.tts.runaway_ladder is None and settings.fit.meaning_guard is None


def test_legacy_variable_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPE_RUNAWAY_LADDER", "off")
    monkeypatch.setenv("TTS_WORKERS", "7")
    settings = Settings()
    assert settings.tts.runaway_ladder is False and settings.tts.workers == 7


def test_canonical_wins_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_WORKERS", "7")
    monkeypatch.setenv("SUNWAI_TTS__WORKERS", "3")
    assert Settings().tts.workers == 3


def test_unparseable_value_is_ignored_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_WORKERS", "many")
    assert Settings().tts.workers is None


def test_export_only_overrides_with_canonical_names() -> None:
    env = {"TTS_WORKERS": "7", "SUNWAI_TTS__WORKERS": "3", "ASR_WORKERS": "5"}
    settings = Settings(tts={"workers": 3}, asr={"workers": 9}, fit={"trim": False})  # type: ignore[arg-type]
    written = export_to_environ(settings, env)
    assert env["TTS_WORKERS"] == "3"  # canonical present -> overrides
    assert env["ASR_WORKERS"] == "5"  # only a flat variable present -> untouched
    assert env["FIT_TRIM"] == "0"  # absent -> written, booleans as 0/1
    assert "TTS_WORKERS" in written and "ASR_WORKERS" not in written


def test_secrets_are_masked_but_exportable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUNWAI_LLM__OPENROUTER_API_KEY", "sk-test-value")
    settings = Settings()
    assert "sk-test-value" not in repr(settings) and "sk-test-value" not in settings.model_dump_json()
    env: dict[str, str] = {}
    export_to_environ(settings, env)
    assert env["OPENROUTER_API_KEY"] == "sk-test-value"


def test_bool_parsing() -> None:
    var = LegacyVar("X", "tts", "slow", "bool")
    assert [var.parse(v) for v in ("1", "TRUE", "on", "0", "No", "garbage")] == [True, True, True, False, False, None]
