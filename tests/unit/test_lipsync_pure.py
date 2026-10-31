"""CPU-only tests for the pure lip-sync and worker logic (no models, no GPU)."""

import json
import os
import time

import numpy as np

from sunwai.lipsync.denoise import plan_window_groups
from sunwai.lipsync.faces import is_usable_face
from sunwai.lipsync.pasteback import crossfade, roi_bounds
from sunwai.lipsync.prepare import fill_missing_landmarks, pad_audio_chunks
from sunwai.lipsync.report import AsdReport, coverage_warning
from sunwai.lipsync.runs import (
    FaceRun,
    cluster_runs,
    gate_runs,
    plan_clusters,
    split_into_runs,
    split_long_run,
)
from sunwai.lipsync.syncnet import average_window_scores, window_starts
from sunwai.lipsync.tracking import slice_padded
from sunwai.lipsync.windows import create_segments
from sunwai.workers.candidates import (
    Candidate,
    Focus,
    next_failure_count,
    parse_chunk_dir,
    rank,
    retry_blocked,
)
from sunwai.workers.claims import ChunkLock
from sunwai.workers.services import load_services


def test_split_into_runs_bridges_and_filters():
    mask = [True] * 30 + [False] * 2 + [True] * 30 + [False] * 10 + [True] * 5
    assert split_into_runs(mask, 25, 1.0, 3) == [(0, 62)]
    assert split_into_runs([], 25) == []


def test_gate_runs_ramps_and_bridges():
    gated, alpha = gate_runs([(0, 100)], lambda a, b: [i >= 20 for i in range(100)])
    assert gated == [(15, 100)]
    assert alpha[15] < alpha[19] < 1.0 and 14 not in alpha


def test_gate_runs_drops_short_speech():
    gated, alpha = gate_runs([(0, 100)], lambda a, b: [10 <= i < 20 for i in range(100)])
    assert gated == [] and alpha == {}


def test_split_long_run_and_clusters():
    assert split_long_run(0, 600, 500) == [(0, 600)]
    assert split_long_run(0, 1000, 500) == [(0, 500), (500, 1000)]
    assert split_long_run(0, 1000, 0) == [(0, 1000)]
    clusters = plan_clusters([FaceRun((0, 0), 0, 50), FaceRun((0, 1), 40, 90), FaceRun(None, 100, 120)], 0)
    assert [(c.start, c.end, len(c.runs)) for c in clusters] == [(0, 90, 2), (100, 120, 1)]
    assert cluster_runs([]) == []


def test_create_segments_merges_and_pads():
    segs = [{"start": 1.0, "end": 2.0}, {"start": 2.5, "end": 3.0}, {"start": 10.0, "end": 11.0}]
    assert create_segments(segs, 0.3, 10.5) == [[0.7, 3.3], [9.7, 10.5]]
    assert create_segments([], 0.3, 5) == []


def test_window_starts_cover_end_and_average_fills_tail():
    assert window_starts(60, 50, 5) == [0, 5, 10]
    assert window_starts(52, 50, 5) == [0, 2]
    conf, off = average_window_scores([(0, 4.0, 1), (5, 2.0, 3)], 70, 50)
    assert conf[0] == 4.0 and conf[10] == 3.0 and off[10] == 2.0
    assert not np.isnan(conf[-1]) and conf[-1] == conf[58]


def test_slice_padded_pads_outside_frame():
    frame = np.full((10, 10, 3), 7, np.uint8)
    crop = slice_padded(frame, (-2, 4, -1, 3))
    assert crop.shape == (6, 4, 3) and crop[0, 0, 0] == 110 and crop[3, 3, 0] == 7


def test_denoise_window_groups():
    windows, groups = plan_window_groups(40, 2)
    assert windows == [(0, 16), (16, 32), (32, 40)]
    assert groups == [[0, 1], [2]]


def test_face_filter_and_prepare_helpers():
    assert is_usable_face([0, 0, 100, 120], 0.9) and not is_usable_face([0, 0, 40, 120], 0.9)
    assert not is_usable_face([0, 0, 100, 120], 0.3)
    assert fill_missing_landmarks([0, 1, 2], {0: "a", 2: "c"}) == ["a", "a", "c"]
    assert fill_missing_landmarks([0, 1], {}) is None
    assert pad_audio_chunks([1, 2], 4) == [1, 2, 2, 2]
    try:
        pad_audio_chunks([1], 6)
    except RuntimeError as ex:
        assert "whisper chunks" in str(ex)
    else:
        raise AssertionError("expected RuntimeError")


def test_pasteback_math():
    corners = np.array([[10.2, 5.0], [50.0, 5.0], [50.0, 40.7], [10.2, 40.7]])
    assert roi_bounds(corners, 100, 100) == (7, 2, 54, 45)
    assert roi_bounds(corners + 500, 100, 100) is None
    base = np.zeros((2, 2, 3), np.uint8)
    assert crossfade(np.full((2, 2, 3), 100, np.uint8), base, 0.5).max() == 50


def test_coverage_warning():
    asd = AsdReport(True, True, 2.7, speech_s=10.0, face_s=8.0, synced_s=2.0)
    report = asd.finalize()
    assert report["coverage"] == 0.2 and report["gated_skip_s"] == 6.0
    assert "COVERAGE_WARN" in coverage_warning("job", report)
    assert coverage_warning("job", AsdReport(True, True, speech_s=10, face_s=8, synced_s=9).finalize()) is None


def test_candidate_ranking_and_retry_policy():
    def cand(job, idx, mtime=0.0):
        return Candidate("d", "ta", job, idx, "dub", mtime)

    focus = Focus("j1", "ta", 5)
    order = sorted(
        [cand("j2", 0, 9), cand("j1", 3), cand("j1", 7), cand("j1", 5), cand("j3", 0, 99)], key=lambda c: rank(c, focus)
    )
    assert [(c.job, c.idx) for c in order] == [("j1", 5), ("j1", 7), ("j1", 3), ("j3", 0), ("j2", 0)]
    assert parse_chunk_dir("/r/job/ta/c0007") == ("job", 7) and parse_chunk_dir("/r/job/ta") == ("job", 0)
    assert retry_blocked(3, 1e9) and retry_blocked(2, 30) and not retry_blocked(1, 61)


def test_failure_marker_count(tmp_path):
    dub = tmp_path / "dub.mp4"
    dub.write_bytes(b"x")
    marker = tmp_path / "lipsync.failed"
    assert next_failure_count(str(marker), str(dub)) == 1
    marker.write_text("2|boom")
    os.utime(marker, (time.time() + 5, time.time() + 5))
    assert next_failure_count(str(marker), str(dub)) == 3


def test_chunk_lock_is_exclusive(tmp_path):
    first = ChunkLock.acquire(str(tmp_path))
    assert first is not None and ChunkLock.acquire(str(tmp_path)) is None
    first.release()
    assert ChunkLock.acquire(str(tmp_path)) is not None
    assert ChunkLock.acquire(str(tmp_path / "missing")) is None


def test_load_services_defaults_and_file(tmp_path):
    base = load_services({"DEP_HOST": "h"})
    assert base.deps["asr"] == ["h:18931"] and base.lip_workers == ["wA", "wB"]
    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"asr": ["x:1"], "mt": [], "lip_mode": "none", "lip_workers": ["w1"]}))
    mapped = load_services({"SUP_SERVICES": str(cfg)})
    assert mapped.deps["asr"] == ["x:1"] and mapped.deps["mt"][0].endswith(":18933")
    assert mapped.lip_mode == "none" and mapped.lip_workers == ["w1"]
