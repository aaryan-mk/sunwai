"""Typed records exchanged between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_CORE = ("id", "start", "end", "speaker", "gender", "src_text", "tgt_text")


@dataclass
class Segment:
    """One spoken line of a chunk.

    The first seven fields are the identity of the line; every other key of the on-disk ``segments.json`` record
    (voice, tempo, fit and TTS bookkeeping, error and fallback markers, expression state, ...) is kept unchanged in
    :attr:`extra`, so converting to a dict and back never loses or invents a key.

    Attributes:
        id: Index of the line within the chunk.
        start: Start on the chunk timeline, seconds.
        end: End on the chunk timeline, seconds.
        speaker: Speaker index within the job.
        gender: ``"male"``, ``"female"`` or ``"unknown"``.
        src_text: Source-language transcript.
        tgt_text: Target-language translation (display text).
        extra: All remaining record keys.
    """

    id: int
    start: float
    end: float
    speaker: int = 0
    gender: str = "unknown"
    src_text: str = ""
    tgt_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    _core_keys: frozenset[str] = field(default=frozenset(_CORE), repr=False, compare=False)

    @property
    def duration_s(self) -> float:
        """Length of the line in seconds."""
        return self.end - self.start

    @property
    def edited(self) -> bool:
        """True when a person edited the line (it is then never re-translated or re-fitted)."""
        return bool(self.extra.get("edited"))

    @property
    def failed(self) -> bool:
        """True when a stage recorded an error or a fallback for this line."""
        return bool(self.extra.get("error") or self.extra.get("fallback"))

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Segment:
        """Build a segment from a ``segments.json`` record, keeping unknown keys in :attr:`extra`."""
        core = {k: record[k] for k in _CORE if k in record}
        extra = {k: v for k, v in record.items() if k not in _CORE}
        return cls(**core, extra=extra, _core_keys=frozenset(core))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Return the ``segments.json`` record (same keys that :meth:`from_dict` received)."""
        record = {k: getattr(self, k) for k in _CORE if k in self._core_keys}
        record.update(self.extra)
        return record


def to_records(segments: list[Segment]) -> list[dict[str, Any]]:
    """Convert typed segments to the dict records the stage implementations operate on."""
    return [s.to_dict() for s in segments]


def from_records(records: list[dict[str, Any]]) -> list[Segment]:
    """Convert dict records produced by a stage implementation back to typed segments."""
    return [Segment.from_dict(r) for r in records]
