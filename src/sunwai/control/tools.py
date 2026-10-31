"""Strict tool schemas shared by the Gemini declarations, server-side validation and the action API."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, NamedTuple

from sunwai.control.guard import LANG_RE, SEG_RE

INTENTS = ["angrier", "sadder", "calmer", "excited"]
SUBMODES = ["off", "original", "dub", "dual"]
MAX_EDIT = 600
TEMPO_RANGE = (0.92, 1.08)
ACTION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
VOICE_RE = re.compile(r"^[A-Za-z0-9_.\- ]+$")

_STR: dict[str, Any] = {"type": "string"}


class ToolSpec(NamedTuple):
    """Schema of one tool: description, JSON-schema properties and the required property names."""

    description: str
    props: dict[str, dict[str, Any]]
    required: list[str]


TOOLS: dict[str, ToolSpec] = {
    "inspect": ToolSpec("Read the job state (segments, languages, metrics). No side effects.", {"lang": {"type": "string"}}, []),
    "edit_line": ToolSpec(
        "Replace the dubbed text of one line and re-synthesise it. Reversible.",
        {"segment_id": _STR, "text": {"type": "string"}},
        ["segment_id", "text"],
    ),
    "undo_line": ToolSpec(
        "Undo the last edit of one line (creates a new revision restoring the previous text).", {"segment_id": _STR}, ["segment_id"]
    ),
    "redo_line": ToolSpec(
        "Re-apply the edit that the last undo_line removed (only right after an undo).", {"segment_id": _STR}, ["segment_id"]
    ),
    "change_expression": ToolSpec(
        "Preview a different delivery for one line.",
        {"segment_id": _STR, "intent": {"type": "string", "enum": INTENTS}},
        ["segment_id", "intent"],
    ),
    "retime": ToolSpec(
        "Preview a speaking-tempo change for one line inside its fixed slot (0.92-1.08).",
        {"segment_id": _STR, "tempo_factor": {"type": "number"}},
        ["segment_id", "tempo_factor"],
    ),
    "switch_language": ToolSpec("Switch the player to another language of this job.", {"lang": _STR}, ["lang"]),
    "add_language": ToolSpec(
        "Prepare another target language, starting from a position in ms.",
        {"lang": _STR, "from_ms": {"type": "integer"}},
        ["lang", "from_ms"],
    ),
    "regenerate_chunk": ToolSpec(
        "Regenerate a whole chunk (long, needs confirmation).", {"chunk_id": {"type": "integer"}}, ["chunk_id"]
    ),
    "change_voice": ToolSpec(
        "Propose a different voice for a speaker. Never applied in place: creates a fork proposal for a NEW job.",
        {"speaker_id": _STR, "voice": _STR},
        ["speaker_id", "voice"],
    ),
    "subtitles": ToolSpec("Set the subtitle display mode.", {"mode": {"type": "string", "enum": SUBMODES}}, ["mode"]),
    "explain": ToolSpec("Explain what a past action did (or the latest one).", {"action_id": _STR}, []),
    "report_issue": ToolSpec("Record a problem report for the team.", {"details": {"type": "string"}}, ["details"]),
}
CLIENT_TOOLS = {"inspect", "switch_language", "subtitles", "explain"}
AUTO_TOOLS = {"edit_line", "undo_line", "redo_line", "change_expression", "retime", "report_issue"}  # reversible / preview: no token
CONFIRM_TOOLS = {"regenerate_chunk", "change_voice"}  # long / fork: token required


def declarations() -> list[dict[str, Any]]:
    """Return the function declarations handed to the planner model."""
    out = []
    for name, spec in TOOLS.items():
        parameters: dict[str, Any] = {"type": "object", "properties": spec.props}
        if spec.required:
            parameters["required"] = spec.required
        out.append({"name": name, "description": spec.description, "parameters": parameters})
    return out


class BadArgs(ValueError):
    """Tool arguments failed validation."""


def _coerce(key: str, spec: dict[str, Any], value: Any) -> Any:
    """Check one value against its property spec and return it in canonical form."""
    kind = spec["type"]
    if kind == "string":
        if not isinstance(value, str):
            raise BadArgs(f"{key} must be a string")
        if "enum" in spec and value not in spec["enum"]:
            raise BadArgs(f"{key} must be one of {spec['enum']}")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise BadArgs(f"{key} must be an integer")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadArgs(f"{key} must be a number")
        return float(value)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BadArgs(message)


def _check_edit_line(args: dict[str, Any]) -> None:
    text = args["text"]
    _require(bool(text.strip()) and len(text) <= MAX_EDIT, f"text must be 1..{MAX_EDIT} chars")


def _check_retime(args: dict[str, Any]) -> None:
    _require(TEMPO_RANGE[0] <= args["tempo_factor"] <= TEMPO_RANGE[1], "tempo_factor must be within 0.92..1.08")


def _check_add_language(args: dict[str, Any]) -> None:
    _require(args["from_ms"] >= 0, "from_ms must be >= 0")


def _check_regenerate(args: dict[str, Any]) -> None:
    _require(args["chunk_id"] >= 0, "chunk_id must be >= 0")


def _check_explain(args: dict[str, Any]) -> None:
    _require("action_id" not in args or bool(ACTION_ID_RE.match(args["action_id"])), "invalid action_id")


def _check_report(args: dict[str, Any]) -> None:
    _require(1 <= len(args["details"]) <= 1000, "details must be 1..1000 chars")


def _check_voice(args: dict[str, Any]) -> None:
    _require(1 <= len(args["voice"]) <= 64, "voice must be 1..64 chars")
    _require(bool(VOICE_RE.match(args["voice"])), "voice has invalid characters")


_SEMANTIC_CHECKS: dict[str, Callable[[dict[str, Any]], None]] = {
    "edit_line": _check_edit_line,
    "retime": _check_retime,
    "add_language": _check_add_language,
    "regenerate_chunk": _check_regenerate,
    "explain": _check_explain,
    "report_issue": _check_report,
    "change_voice": _check_voice,
}


def validate_args(name: str, args: Any) -> dict[str, Any]:
    """Strictly validate a tool call.

    Unknown tools, extra or missing fields, wrong types and out-of-range values all raise.

    Args:
        name: Tool name.
        args: Raw arguments from the model or client.

    Returns:
        A cleaned copy of the arguments.

    Raises:
        BadArgs: When anything is invalid.
    """
    if name not in TOOLS:
        raise BadArgs(f"unknown tool {name!r}")
    if not isinstance(args, dict):
        raise BadArgs("args must be an object")
    spec = TOOLS[name]
    extra = set(args) - set(spec.props)
    if extra:
        raise BadArgs(f"unexpected fields: {sorted(extra)}")
    for required in spec.required:
        if required not in args:
            raise BadArgs(f"missing field {required}")
    out = {key: _coerce(key, spec.props[key], value) for key, value in args.items()}
    if "segment_id" in out and not SEG_RE.match(out["segment_id"]):
        raise BadArgs("invalid segment_id")
    if "lang" in out and not LANG_RE.match(out["lang"]):
        raise BadArgs("invalid lang")
    check = _SEMANTIC_CHECKS.get(name)
    if check:
        check(out)
    return out
