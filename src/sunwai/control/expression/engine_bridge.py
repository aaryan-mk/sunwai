"""Lazy bridge to the dubbing engine: the only place the expression subsystem reaches into ``sunwai.pipeline.engine``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sunwai.control.expression.fsutil import read_json


def _engine() -> Any:
    from sunwai.pipeline import engine

    return engine


def default_tts(text: str, lang: str, voice: str, style: str | None) -> bytes:
    """Synthesise ``text`` with the live TTS service and return the WAV bytes."""
    engine = _engine()
    return engine.bodhan_tts(engine.thread_session(), engine.load_keys()["tts"], text, lang, voice, style=style)  # type: ignore[no-any-return,no-untyped-call]


def default_trim(raw: Path, dst: Path) -> tuple[Path, float]:
    """Trim leading/trailing silence of ``raw`` into ``dst``; returns ``(path, duration_s)``."""
    return _engine()._pretrim_wav(raw, dst)  # type: ignore[no-any-return,no-untyped-call]


def pick_voice(lang: str, gender: str, speaker: str) -> str:
    """Pick the legacy voice for a speaker."""
    return _engine().pick_voice(lang, gender, speaker)  # type: ignore[no-any-return,no-untyped-call]


def default_remix(out_dir: Path, lang: str) -> Any:
    """Re-mix the chunk's dub from its current ``segments.json``."""
    engine = _engine()
    args = SimpleNamespace(lang=lang)
    return engine.stage_mix(args, Path(out_dir), read_json(Path(out_dir) / "segments.json"))
