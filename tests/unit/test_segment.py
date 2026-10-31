"""Segment record round trip."""

from __future__ import annotations

from sunwai.pipeline.models import Segment, from_records, to_records


def test_roundtrip_is_lossless_for_full_and_partial_records() -> None:
    full = {
        "id": 3,
        "start": 1.0,
        "end": 2.5,
        "speaker": 1,
        "gender": "female",
        "src_text": "a",
        "tgt_text": "b",
        "voice": "Kavya",
        "tempo": 1.05,
        "sent_pieces": [1, 2],
        "edited": False,
    }
    partial = {"id": 4, "start": 0.0, "end": 1.0, "src_text": "x"}  # no speaker/gender/tgt_text keys yet
    for record in (full, partial):
        assert Segment.from_dict(record).to_dict() == record


def test_unknown_keys_are_preserved_and_core_keys_typed() -> None:
    seg = Segment.from_dict({"id": 1, "start": 0.0, "end": 2.0, "fallback_reason": "asr", "error": "boom"})
    assert seg.duration_s == 2.0 and seg.failed and not seg.edited
    assert seg.extra["fallback_reason"] == "asr"


def test_record_helpers() -> None:
    records = [{"id": 0, "start": 0.0, "end": 1.0}, {"id": 1, "start": 1.0, "end": 2.0, "tgt_text": "t"}]
    assert to_records(from_records(records)) == records
