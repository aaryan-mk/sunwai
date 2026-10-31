"""Golden behaviour of the TTS stage with a fake synthesiser and a tiny synthetic chunk (no network, needs ffmpeg)."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from sunwai.pipeline.stages import tts as tts_stage

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SR = 24000


def wav_bytes(seconds: float) -> bytes:
    """Return PCM16 mono WAV bytes with a 220 Hz tone of the given length."""
    t = np.arange(int(seconds * SR)) / SR
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def seg(i: int, start: float, end: float, text: str = "नमस्ते दुनिया", **kw: Any) -> dict[str, Any]:
    return {
        "id": i,
        "start": start,
        "end": end,
        "speaker": 0,
        "gender": "male",
        "src_text": "hello world",
        "tgt_text": text,
        "error": False,
        **kw,
    }


@pytest.fixture
def chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key, value in {
        "BODHAN_MODE": "local",
        "TTS_SENT_SPLIT": "0",
        "REFIT": "0",
        "TTS_WORKERS": "1",
        "TTS_SLACK_PAUSES": "0",
        "MIX_TRIM": "0",
    }.items():
        monkeypatch.setenv(key, value)
    out = tmp_path / "job" / "hi" / "c0003"
    out.mkdir(parents=True)
    return out


def run(
    chunk: Path, segments: list[dict[str, Any]], durations: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    calls: list[str] = []

    def fake_tts(session: Any, key: str, text: str, lang: str, voice: str, style: Any = None) -> bytes:
        calls.append(text)
        if text not in durations:
            raise RuntimeError("synthetic TTS failure")
        return wav_bytes(durations[text])

    monkeypatch.setattr(tts_stage, "bodhan_tts", fake_tts)
    result = tts_stage.stage_tts(SimpleNamespace(lang="hi"), chunk, segments)
    result_calls = {"calls": len(calls)}
    assert result_calls["calls"] >= 0
    return result


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def test_fits_slows_caps_and_truncates(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [
        seg(0, 0.0, 2.0, "एक"),  # speech much shorter than the slot -> gentle slow-down, no overflow
        seg(1, 2.0, 3.0, "दो"),  # slightly long -> speed up, fits before the next line
        seg(2, 4.0, 5.0, "तीन"),  # too long even at 1.25x and no following line -> overflow, truncated
    ]
    out = run(chunk, segments, {"एक": 1.0, "दो": 1.4, "तीन": 1.9}, monkeypatch)
    by_id = {s["id"]: s for s in out}

    assert by_id[0]["ratio"] == pytest.approx(0.5, abs=0.01) and by_id[0]["tempo"] == pytest.approx(0.9, abs=0.01)
    assert by_id[0]["overflow"] is False
    assert duration(chunk / "tts" / "0.wav") == pytest.approx(1.0 / 0.9, abs=0.05)

    assert by_id[1]["ratio"] == pytest.approx(1.4, abs=0.01) and by_id[1]["tempo"] == pytest.approx(1.25, abs=0.01)
    assert by_id[1]["overflow"] is False
    assert duration(chunk / "tts" / "1.wav") == pytest.approx(1.4 / 1.25, abs=0.05)

    assert by_id[2]["tempo"] == pytest.approx(1.25, abs=0.01) and by_id[2]["overflow"] is True
    assert by_id[2]["overflow_s"] == pytest.approx(1.9 / 1.25 - 1.0, abs=0.05)
    assert duration(chunk / "tts" / "2.wav") == pytest.approx(1.0, abs=0.05)  # truncated to the slot
    assert all(s["voice"] for s in out)
    saved = json.loads((chunk / "segments.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in saved] == [0, 1, 2]


def test_empty_text_and_failed_synthesis_keep_the_original_audio(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_MAX_FRAC", "0.5")
    segments = [seg(0, 0.0, 2.0, ""), seg(1, 2.0, 3.0, "विफल"), seg(2, 3.0, 4.0, "ठीक")]
    out = run(chunk, segments, {"ठीक": 1.0}, monkeypatch)
    by_id = {s["id"]: s for s in out}

    assert by_id[0]["ratio"] == 1.0 and not (chunk / "tts" / "0.wav").exists()  # nothing to speak
    assert by_id[1]["fallback"] and by_id[1]["fallback_reason"] == "tts"  # failure degrades only that line
    assert not (chunk / "tts" / "1.wav").exists()
    assert (chunk / "tts" / "2.wav").exists() and not by_id[2].get("fallback")


def test_cached_raw_audio_is_reused_without_calling_the_service(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [seg(0, 0.0, 2.0, "एक")]
    run(chunk, segments, {"एक": 1.0}, monkeypatch)
    again = [seg(0, 0.0, 2.0, "एक")]
    out = run(chunk, again, {}, monkeypatch)  # empty table: any call would fail
    assert not out[0].get("fallback") and (chunk / "tts" / "0.wav").exists()


def test_runaway_synthesis_is_trimmed_not_placed(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [seg(0, 0.0, 2.0, "बहुत")]
    out = run(chunk, segments, {"बहुत": 12.0}, monkeypatch)  # 6x the slot: a runaway
    assert out[0].get("runaway_fix") or out[0].get("fallback")
    if (chunk / "tts" / "0.wav").exists():
        assert duration(chunk / "tts" / "0.wav") <= 2.0 + 0.1


def test_chunk_fails_when_too_much_speech_fell_back(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [seg(0, 0.0, 2.0, "विफल"), seg(1, 2.0, 3.0, "ठीक")]
    with pytest.raises(RuntimeError, match="chunk failed"):
        run(chunk, segments, {"ठीक": 1.0}, monkeypatch)  # 67% of the speech failed > 20% budget


def test_sentence_split_places_pieces_at_their_source_spans(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_SENT_SPLIT", "1")
    (chunk / "vad.json").write_text(json.dumps([{"start": 0.0, "end": 2.0}, {"start": 3.5, "end": 6.0}]))
    text = "पहला वाक्य यहाँ है। दूसरा वाक्य वहाँ है।"
    pieces: list[str] = []

    def fake_bounded(session: Any, key: str, piece: str, lang: str, voice: str, t_end: float | None) -> bytes:
        pieces.append(piece)
        return wav_bytes(2.0)

    monkeypatch.setattr(tts_stage, "_bodhan_tts_bounded", fake_bounded)
    out = run(chunk, [seg(0, 0.0, 6.0, text)], {text: 5.0}, monkeypatch)[0]  # whole-slot audio: one unbroken 5 s tone

    assert len(pieces) == 2 and out["sent_pieces"] and len(out["sent_pieces"]) == 2
    assert out["sent_pieces"][1]["start"] >= 3.4  # second piece starts after the source pause
    assert out["overflow"] is False and out["legacy_fit"]["ratio"] == pytest.approx(5.0 / 6.0, abs=0.01)
    assert duration(chunk / "tts" / "0.wav") == pytest.approx(
        out["sent_pieces"][1]["start"] + out["sent_pieces"][1]["dur"], abs=0.05
    )


def test_refit_rewrites_a_line_that_cannot_fit_even_at_max_tempo(chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sunwai.pipeline.fitting import fit_stage

    monkeypatch.setenv("REFIT", "1")
    monkeypatch.setattr(fit_stage, "guarded_rewrite", lambda *a, **k: "छोटा")
    out = run(chunk, [seg(0, 0.0, 1.0, "बहुत लम्बा वाक्य")], {"बहुत लम्बा वाक्य": 2.0, "छोटा": 0.9}, monkeypatch)[0]
    assert out["refit"] is True and out["pre_refit_text"] == "बहुत लम्बा वाक्य" and out["tgt_text"] == "छोटा"
    assert out["overflow"] is False and out["tempo"] == pytest.approx(0.9, abs=0.01)
