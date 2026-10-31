"""Parsing and validation of an action request body into an immutable :class:`ActionRequest`. Pure: no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sunwai.control.codec import canon, sha
from sunwai.control.errors import CtlError
from sunwai.control.guard import LANG_RE
from sunwai.control.tools import BadArgs, validate_args

SERVER_TYPES = {
    "edit_line",
    "undo_line",
    "redo_line",
    "change_expression",
    "retime",
    "regenerate_chunk",
    "change_voice",
    "add_language",
    "report_issue",
}
SEG_TYPES = {"edit_line", "undo_line", "redo_line", "change_expression", "retime"}
COMMIT_TYPES = {"edit_line", "undo_line", "redo_line", "add_language", "regenerate_chunk", "change_voice"}
PREVIEW_TYPES = {"change_expression", "retime"}
ENVELOPE_FIELDS = {
    "action_id",
    "expected_revision",
    "idempotency_key",
    "type",
    "args",
    "preview_only",
    "confirmation_token",
}
MAX_CHUNK = 9999
MAX_CANDIDATE = 10**6


@dataclass(frozen=True)
class ActionRequest:
    """A validated action request.

    Attributes:
        typ: Action type.
        core: Validated tool arguments (for ``add_language`` including ``playback_generation``).
        lang: Language the action targets, if any.
        chunk: Chunk the action targets, if any.
        apply_cand: Candidate revision to apply (``change_expression`` confirm), if any.
        idem: Idempotency key.
        expected: Expected head revision (segment actions).
        preview_only: Server-derived: a confirmed apply always applies, whatever the caller sent.
        payload_hash: Hash of the semantically significant payload, for idempotency conflict detection.
        token: Confirmation token supplied by the caller, if any.
    """

    typ: str
    core: dict[str, Any]
    lang: str | None
    chunk: int | None
    apply_cand: int | None
    idem: str
    expected: Any
    preview_only: bool
    payload_hash: str
    token: str | None = None

    @property
    def is_segment_action(self) -> bool:
        """Whether the action targets one line."""
        return self.typ in SEG_TYPES

    @property
    def bind_args(self) -> dict[str, Any]:
        """The payload a confirmation token binds to, including which candidate is applied."""
        return {**self.core, "lang": self.lang, "chunk": self.chunk, "apply_candidate": self.apply_cand}

    @property
    def needs_confirmation(self) -> bool:
        """Whether the server demands a valid token: commits, GPU spend and forks (not previews or ``report_issue``)."""
        return self.typ in COMMIT_TYPES or (self.typ in PREVIEW_TYPES and self.apply_cand is not None)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_envelope(body: dict[str, Any]) -> None:
    extra = set(body) - ENVELOPE_FIELDS
    if extra:
        raise CtlError(422, "invalid_args", f"unexpected fields: {sorted(extra)}")
    if not isinstance(body.get("args", {}), dict):
        raise CtlError(422, "invalid_args", "args must be an object")


def _numeric_string_to_int(value: Any) -> Any:
    return int(value) if isinstance(value, str) and value.isdigit() else value


def _validate_extras(
    typ: str, args: dict[str, Any], apply_cand: Any, pgen: Any, lang: Any, chunk: Any
) -> dict[str, Any]:
    """Validate the tool args and the envelope-level extras; returns the cleaned tool args."""
    core = validate_args(typ, args)
    if apply_cand is not None and (not _integer(apply_cand) or not (0 <= apply_cand <= MAX_CANDIDATE)):
        raise BadArgs("apply_candidate must be an integer revision (0..1000000)")
    if pgen is not None and (not _integer(pgen) or pgen < 0):
        raise BadArgs("playback_generation must be a non-negative integer")
    if typ in SEG_TYPES or typ == "regenerate_chunk":
        if not isinstance(lang, str) or not _integer(chunk) or not (0 <= chunk <= MAX_CHUNK):
            raise BadArgs("args.lang (string) and args.chunk (int 0..9999) are required")
        if not LANG_RE.match(lang):
            raise BadArgs("invalid lang")
    return core


def parse_request(body: dict[str, Any], addlang_on: bool) -> ActionRequest:
    """Validate an action body.

    Args:
        body: The JSON body posted by the client.
        addlang_on: Whether the add-language feature is enabled.

    Returns:
        The validated request.

    Raises:
        CtlError: 422 ``invalid_args`` for any malformed field, 403 when add-language is disabled.
    """
    typ = body.get("type")
    if typ not in SERVER_TYPES:
        raise CtlError(422, "invalid_args", f"unknown action type {typ!r}")
    if typ == "add_language" and not addlang_on:
        raise CtlError(403, "feature_disabled", "Adding a language while watching is not available yet.")
    _check_envelope(body)
    args = dict(body.get("args") or {})
    apply_cand = _numeric_string_to_int(
        args.pop("apply_candidate", None)
    )  # tolerate numeric strings from older clients
    if apply_cand is not None and typ != "change_expression":
        raise CtlError(422, "invalid_args", "apply_candidate is only valid for change_expression")
    pgen = args.pop("playback_generation", None) if typ == "add_language" else None
    lang: Any = None
    chunk: Any = None
    if typ in SEG_TYPES or typ == "regenerate_chunk":
        lang = args.pop("lang", None)
        chunk = _numeric_string_to_int(args.pop("chunk", None))
        if typ == "regenerate_chunk":
            args = {"chunk_id": chunk}
    try:
        core = _validate_extras(typ, args, apply_cand, pgen, lang, chunk)
    except BadArgs as exc:
        raise CtlError(422, "invalid_args", str(exc)) from exc
    if typ == "add_language":
        lang = core["lang"]
        core = {**core, "playback_generation": pgen or 0}
    idem = body.get("idempotency_key")
    if not isinstance(idem, str) or not (8 <= len(idem) <= 80):
        raise CtlError(422, "invalid_args", "idempotency_key (8..80 chars) is required")
    expected = body.get("expected_revision")
    preview_only = typ in PREVIEW_TYPES and apply_cand is None
    significant = core if typ != "add_language" else {"lang": core["lang"]}
    payload_hash = sha(
        canon({"t": typ, "a": significant, "l": lang, "c": chunk, "x": expected, "p": preview_only, "ac": apply_cand})
    )
    return ActionRequest(
        typ, core, lang, chunk, apply_cand, idem, expected, preview_only, payload_hash, body.get("confirmation_token")
    )


def summarise(typ: str, args: dict[str, Any]) -> str:
    """Describe a proposed action in one sentence for the confirmation prompt."""
    if typ == "regenerate_chunk":
        return f"Regenerate chunk {args.get('chunk')} in {args.get('lang')}: re-dubs the whole chunk (takes minutes) and replaces its audio (undoable)"
    if typ == "change_voice":
        return f"Create a NEW job with speaker {args['speaker_id']} voiced as {args['voice']}; this job is unchanged"
    if typ == "add_language":
        return f"Start dubbing {args['lang']} from {args['from_ms']} ms (uses GPU time; this job already has two or more languages)"
    if typ == "edit_line":
        return f"Replace line {args['segment_id']} (chunk {args.get('chunk')}) with: {args['text'][:120]}"
    if typ == "undo_line":
        return f"Undo the last edit of line {args['segment_id']} (chunk {args.get('chunk')})"
    if typ == "redo_line":
        return f"Re-apply the edit of line {args['segment_id']} (chunk {args.get('chunk')}) that was just undone"
    return typ
