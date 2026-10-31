"""Pure-logic tests for the planning package: no ffmpeg, GPU or model downloads."""

from __future__ import annotations

import numpy as np
import pytest

from sunwai.planning.chunks import gap_points, lowest_rms_cut, make_cuts, rms_windows
from sunwai.planning.diarization import normalise_turns, windows_to_turns
from sunwai.planning.envvars import env_flag, env_float
from sunwai.planning.errors import MediaError
from sunwai.planning.fetch import CutRange, cut_range_mismatch, encode_args, plan_cut
from sunwai.planning.gender import assign_gender, merge_small_speakers, normalise_gender_keys
from sunwai.planning.speaker_turns import first_appearance_order, main_speakers, nearest_speaker, speaker_totals
from sunwai.planning.speakers import build_speaker_map
from sunwai.planning.subprocesses import parse_probe_json
from sunwai.planning.vad import blocked_vad, merge_intervals
from sunwai.planning.vetting_logic import (
    SpeakerEvidence,
    VetConstants,
    decide_mapping,
    overlap_regions,
    overlap_spans,
    subtract_intervals,
)


def turn(start: float, end: float, speaker: int) -> dict[str, float]:
    return {"start": start, "end": end, "speaker": speaker}


# ---- env parsing -------------------------------------------------------------------------------------------------


def test_env_float_falls_back_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_TEST", "abc")
    assert env_float("X_TEST", 3.0) == 3.0
    monkeypatch.setenv("X_TEST", "nan")
    assert env_float("X_TEST", 3.0) == 3.0
    monkeypatch.setenv("X_TEST", "5")
    assert env_float("X_TEST", 3.0, hi=4.0) == 3.0
    assert env_float("X_TEST", 3.0, hi=9.0) == 5.0


def test_env_flag_kill_switch_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_FLAG", raising=False)
    assert env_flag("X_FLAG") is True
    for off in ("0", "false", "No", "OFF", "n", "f"):
        monkeypatch.setenv("X_FLAG", off)
        assert env_flag("X_FLAG") is False
    monkeypatch.setenv("X_FLAG", "yes")
    assert env_flag("X_FLAG") is True


# ---- ffprobe summary ---------------------------------------------------------------------------------------------


def test_parse_probe_json_ignores_cover_art_and_uses_first_audio_track() -> None:
    doc = {
        "streams": [
            {"codec_type": "video", "duration": "10.0"},
            {"codec_type": "video", "duration": "1.0", "disposition": {"attached_pic": 1}},
            {"codec_type": "audio", "duration": "9.5"},
            {"codec_type": "audio", "duration": "10.0"},
        ],
        "format": {},
    }
    info = parse_probe_json(doc)
    assert (info["n_video"], info["n_audio"]) == (1, 2)
    assert info["a0_dur"] == 9.5 and info["duration"] == 10.0


# ---- cut planning and ffmpeg arguments ---------------------------------------------------------------------------


def test_plan_cut_whole_video() -> None:
    cut = plan_cut(0.0, 0.0, raw_dur=30.0, meta_dur=0.0)
    assert cut == CutRange(False, 0.0, None, 30.0)


def test_plan_cut_range_is_clamped_to_the_video() -> None:
    cut = plan_cut(10.0, 100.0, raw_dur=30.0, meta_dur=30.0)
    assert (cut.start, cut.dur, cut.expect) == (10.0, 100.0, 20.0)


def test_plan_cut_to_end_uses_metadata_duration() -> None:
    cut = plan_cut(5.0, 0.0, raw_dur=0.0, meta_dur=20.0)
    assert (cut.dur, cut.expect) == (15.0, None)


def test_plan_cut_rejects_start_past_end() -> None:
    with pytest.raises(MediaError):
        plan_cut(30.0, 5.0, raw_dur=30.0, meta_dur=30.0)


def test_cut_range_mismatch() -> None:
    cut = CutRange(True, 2.0, 5.0, 5.0)
    assert cut_range_mismatch({"start": 2.0, "dur": 5.0}, cut) is None
    assert cut_range_mismatch({"start": 1.0, "dur": 5.0}, cut) is not None
    whole = CutRange(False, 0.0, None, 30.0)
    assert cut_range_mismatch({}, whole) is None
    assert cut_range_mismatch({"start": 2.0}, whole) is not None


def test_encode_args_seek_position() -> None:
    cut = CutRange(True, 2.0, 5.0, 5.0)
    fast = encode_args("raw.mp4", "out.mp4", cut, input_seek=True)
    slow = encode_args("raw.mp4", "out.mp4", cut, input_seek=False)
    assert fast[:5] == ["-ss", "2.000", "-i", "raw.mp4", "-t"]
    assert slow[:3] == ["-i", "raw.mp4", "-ss"]
    assert fast[-1] == "out.mp4" and "-map" in fast
    whole = encode_args("raw.mp4", "out.mp4", CutRange(False, 0.0, None, 1.0), input_seek=True)
    assert whole[:2] == ["-i", "raw.mp4"] and "-map" not in whole


# ---- VAD and chunking --------------------------------------------------------------------------------------------


def test_merge_intervals() -> None:
    assert merge_intervals([(5, 6), (0, 1), (0.9995, 2), (3, 3)]) == [[0.0, 2.0], [5.0, 6.0]]
    assert merge_intervals([]) == []


def test_blocked_vad_adds_block_offsets() -> None:
    sr = 16000
    samples = np.zeros(sr * 1250, dtype=np.float32)
    for a, b in [(5, 9), (598, 602), (700, 703)]:
        samples[a * sr : b * sr] = 1

    def detect(segment: np.ndarray) -> list[dict[str, float]]:
        edges = np.diff(np.concatenate([[0], (segment > 0.5).astype(np.int8), [0]]))
        return [
            {"start": s / sr, "end": e / sr}
            for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True)
        ]

    got = merge_intervals(blocked_vad(samples, 1250.0, detect))
    assert [(round(a), round(b)) for a, b in got] == [(5, 9), (598, 602), (700, 703)]


def test_gap_points_returns_silence_only() -> None:
    assert gap_points([[10, 20], [30, 40]], 50) == [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]
    assert gap_points([], 50) == []


def test_lowest_rms_cut_picks_quietest_frame_and_skips_nan() -> None:
    rms = [0.5] * 100
    rms[40] = float("nan")
    rms[60] = 0.01
    assert lowest_rms_cut(rms, 0.5, 1.5) == pytest.approx(60.5 * 0.02)
    assert lowest_rms_cut([float("nan")] * 10, 0.0, 0.1) is None


def test_rms_windows_block_boundaries() -> None:
    samples = np.ones(16000 + 100, dtype=np.float32)
    rms = rms_windows(samples)
    assert len(rms) == 50 + 1 and rms[0] == pytest.approx(1.0)


def test_make_cuts_never_cuts_inside_speech() -> None:
    vad = [[i * 9.0, i * 9.0 + 8.0] for i in range(40)]  # 8 s speech, 1 s gaps
    samples = np.full(16000 * 360, 0.01, dtype=np.float32)
    chunks = make_cuts(vad, 360.0, 60.0, samples)
    assert chunks[0]["start"] == 0.0
    assert sum(c["dur"] for c in chunks) == pytest.approx(360.0, abs=0.01)
    assert [c["idx"] for c in chunks] == list(range(len(chunks)))
    for c in chunks[1:]:
        assert not any(a < c["start"] < b for a, b in vad)


def test_make_cuts_ramp_starts_with_short_chunks() -> None:
    samples = np.full(16000 * 600, 0.01, dtype=np.float32)
    chunks = make_cuts([], 600.0, 60.0, samples)
    assert chunks[0]["dur"] < chunks[-2]["dur"] <= 75.0


# ---- diarization post-processing ---------------------------------------------------------------------------------


def test_normalise_turns_drops_short_and_relabels_by_first_appearance() -> None:
    raw = [(5.0, 8.0, "B"), (0.0, 3.0, "A"), (3.0, 3.1, "C"), (8.0, 9.0, "A")]
    turns, relabel = normalise_turns(raw)
    assert relabel == {"A": 0, "B": 1}
    assert [(t["start"], t["speaker"]) for t in turns] == [(0.0, 0), (5.0, 1), (8.0, 0)]


def test_windows_to_turns_merges_equal_labels() -> None:
    turns = windows_to_turns(np.array([0, 0, 1, 1, 1]), [0.0, 0.75, 1.5, 2.25, 3.0], 1.5)
    assert turns == [(0.0, 2.25, 0), (1.5, 4.5, 1)]
    assert windows_to_turns(np.array([], dtype=int), [], 1.5) == []


# ---- speaker turns / gender / speaker map ------------------------------------------------------------------------


def test_speaker_turn_helpers() -> None:
    turns = [turn(0, 10, 0), turn(10, 12, 1), turn(12, 30, 2)]
    totals = speaker_totals(turns)
    assert totals == {0: 10, 1: 2, 2: 18}
    assert main_speakers(totals, 5.0) == {0, 2}
    assert main_speakers({0: 1.0, 1: 2.0}, 5.0) == {1}
    assert nearest_speaker([turn(0, 10, 0), turn(12, 30, 2)], 11.5) == 2
    assert first_appearance_order([turn(5, 6, 9), turn(0, 1, 4)]) == [4, 9]


def test_merge_small_speakers_folds_unless_vetted() -> None:
    def make() -> list[dict[str, float]]:
        return [turn(0, 20, 0), turn(20, 22, 1), turn(22, 40, 2)]

    folded = merge_small_speakers(make(), vetted=False)
    assert sorted({t["speaker"] for t in folded}) == [0, 1]
    kept = merge_small_speakers(make(), vetted=True)
    assert sorted({t["speaker"] for t in kept}) == [0, 1, 2]


def test_assign_gender_band(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GENDER_HI", "GENDER_LO"):
        monkeypatch.delenv(name, raising=False)
    assert assign_gender(0.9, 30, 3) == "female"
    assert assign_gender(0.1, 30, 3) == "male"
    assert assign_gender(0.5, 30, 3) == "unknown"
    assert assign_gender(0.9, 5, 3) == "unknown"  # too little speech
    assert assign_gender(0.9, 10, 1) == "unknown"  # one short turn
    assert assign_gender(float("nan"), 30, 3) == "unknown"
    monkeypatch.setenv("GENDER_HI", "0.2")
    monkeypatch.setenv("GENDER_LO", "0.8")  # inverted pair -> defaults
    assert assign_gender(0.9, 30, 3) == "female"


def test_normalise_gender_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    assert normalise_gender_keys({"0": 1, "x": 2}) == {0: 1, "x": 2}
    monkeypatch.setenv("GENDER_KEYFIX", "0")
    assert normalise_gender_keys({"0": 1}) == {"0": 1}


def test_build_speaker_map_tolerates_str_keys_and_infers_degraded() -> None:
    turns = [turn(0, 20, 0), turn(20, 25, 1)]
    data = build_speaker_map("ecapa-agglomerative", turns, {"0": {"mean_p": 0.9}})
    assert data["degraded"] is True
    first, second = data["speakers"]
    assert first["gender"] == "female" and first["p_female"] == 0.9
    assert second["gender"] == "unknown" and second["p_female"] == 0.5
    assert build_speaker_map("pyannote-community-1", turns, {})["degraded"] is False


# ---- speaker vetting rules ---------------------------------------------------------------------------------------


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def evidence(vec: np.ndarray, clean_s: float = 10.0, n: int = 3) -> SpeakerEvidence:
    return SpeakerEvidence(pieces=[(0.0, clean_s, vec)], clean_s=clean_s, n_clean_turns=n, logf0=5.0)


def test_subtract_intervals() -> None:
    assert subtract_intervals([(0, 10)], [(2, 3), (8, 12)]) == [(0, 2), (3, 8)]


def test_overlap_regions_and_pair_seconds() -> None:
    full = [(0.0, 10.0, "A"), (5.0, 8.0, "B")]
    regions, pairs = overlap_regions(full)
    assert regions == [(5.0, 8.0, frozenset({"A", "B"}))]
    assert pairs == {("A", "B"): 3.0}


def test_short_speaker_merges_into_similar_main_speaker() -> None:
    turns = [turn(0, 30, 0), turn(30, 60, 1), turn(60, 63, 2)]
    ev = {0: evidence(unit(1, 0)), 1: evidence(unit(0, 1)), 2: evidence(unit(0.05, 1))}
    mapping, log = decide_mapping(turns, ev, {}, VetConstants())
    assert mapping == {0: 0, 1: 1, 2: 1}
    assert log[0]["action"] == "merge_identity" and log[0]["best"] == 1


def test_short_speaker_stays_when_voices_differ_or_overlap() -> None:
    turns = [turn(0, 30, 0), turn(30, 60, 1), turn(60, 63, 2)]
    ev = {0: evidence(unit(1, 0)), 1: evidence(unit(0, 1)), 2: evidence(unit(-1, -1))}
    assert decide_mapping(turns, ev, {}, VetConstants())[0][2] == 2
    similar = {**ev, 2: evidence(unit(0.05, 1))}
    assert decide_mapping(turns, similar, {(1, 2): 3.0}, VetConstants())[0][2] == 2  # talks over speaker 1


def test_short_speaker_without_evidence_folds_to_nearest_in_time() -> None:
    turns = [turn(0, 30, 0), turn(30, 31, 2), turn(31, 60, 1)]
    mapping, log = decide_mapping(turns, {}, {}, VetConstants())
    assert mapping[2] in (0, 1) and log[0]["action"] == "fold_no_evidence"


def test_oversplit_speakers_with_one_voice_are_repaired() -> None:
    turns = [turn(0, 30, 0), turn(30, 70, 1)]
    ev = {0: evidence(unit(1, 0.02)), 1: evidence(unit(1, 0.0))}
    mapping, log = decide_mapping(turns, ev, {}, VetConstants())
    assert mapping == {0: 1, 1: 1}  # the speaker with more speech survives
    assert log[-1]["rule"] == "oversplit"


def test_decide_mapping_empty() -> None:
    assert decide_mapping([], {}, {}, VetConstants()) == ({}, [])


def test_overlap_spans_merge_close_regions_and_mark_short_turns() -> None:
    full = [(0.0, 10.0, "A"), (2.0, 3.0, "B"), (3.05, 4.0, "B")]
    spans = overlap_spans([(5.0, 5.2, "A")], full, {"A": 0, "B": 1}, VetConstants())
    assert [(s["kind"], s["start"], s["end"], s["speakers"]) for s in spans] == [
        ("overlap", 2.0, 4.0, [0, 1]),
        ("short_turn", 5.0, 5.2, [0]),
    ]
