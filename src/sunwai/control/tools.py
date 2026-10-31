"""strict tool schemas (shared by Gemini declarations, server validation and the action API)."""
import re

LANG_RE = re.compile(r"^[a-z]{2,3}$")
SEG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
INTENTS = ["angrier", "sadder", "calmer", "excited"]
SUBMODES = ["off", "original", "dub", "dual"]
MAX_EDIT = 600

# name -> (description, {prop: spec}, required)
_S = {"type": "string"}
TOOLS = {
    "inspect": ("Read the job state (segments, languages, metrics). No side effects.", {"lang": {"type": "string"}}, []),
    "edit_line": ("Replace the dubbed text of one line and re-synthesise it. Reversible.", {"segment_id": _S, "text": {"type": "string"}}, ["segment_id", "text"]),
    "undo_line": ("Undo the last edit of one line (creates a new revision restoring the previous text).", {"segment_id": _S}, ["segment_id"]),
    "redo_line": ("Re-apply the edit that the last undo_line removed (only right after an undo).", {"segment_id": _S}, ["segment_id"]),
    "change_expression": ("Preview a different delivery for one line.", {"segment_id": _S, "intent": {"type": "string", "enum": INTENTS}}, ["segment_id", "intent"]),
    "retime": ("Preview a speaking-tempo change for one line inside its fixed slot (0.92-1.08).", {"segment_id": _S, "tempo_factor": {"type": "number"}}, ["segment_id", "tempo_factor"]),
    "switch_language": ("Switch the player to another language of this job.", {"lang": _S}, ["lang"]),
    "add_language": ("Prepare another target language, starting from a position in ms.", {"lang": _S, "from_ms": {"type": "integer"}}, ["lang", "from_ms"]),
    "regenerate_chunk": ("Regenerate a whole chunk (long, needs confirmation).", {"chunk_id": {"type": "integer"}}, ["chunk_id"]),
    "change_voice": ("Propose a different voice for a speaker. Never applied in place: creates a fork proposal for a NEW job.", {"speaker_id": _S, "voice": _S}, ["speaker_id", "voice"]),
    "subtitles": ("Set the subtitle display mode.", {"mode": {"type": "string", "enum": SUBMODES}}, ["mode"]),
    "explain": ("Explain what a past action did (or the latest one).", {"action_id": _S}, []),
    "report_issue": ("Record a problem report for the team.", {"details": {"type": "string"}}, ["details"]),
}
CLIENT_TOOLS = {"inspect", "switch_language", "subtitles", "explain"}
AUTO_TOOLS = {"edit_line", "undo_line", "redo_line", "change_expression", "retime", "report_issue"}   # reversible / preview: no token
CONFIRM_TOOLS = {"regenerate_chunk", "change_voice"}                                        # long / fork: token required


def declarations():
    out = []
    for n, (d, props, req) in TOOLS.items():
        p = {"type": "object", "properties": props}
        if req:
            p["required"] = req
        out.append({"name": n, "description": d, "parameters": p})
    return out


class BadArgs(ValueError):
    pass


def validate_args(name, args):
    """Strict: unknown tool / extra fields / wrong types / out-of-range all raise BadArgs. Returns a cleaned copy."""
    if name not in TOOLS:
        raise BadArgs(f"unknown tool {name!r}")
    if not isinstance(args, dict):
        raise BadArgs("args must be an object")
    _, props, req = TOOLS[name]
    extra = set(args) - set(props)
    if extra:
        raise BadArgs(f"unexpected fields: {sorted(extra)}")
    for r in req:
        if r not in args:
            raise BadArgs(f"missing field {r}")
    out = {}
    for k, v in args.items():
        spec = props[k]
        t = spec["type"]
        if t == "string":
            if not isinstance(v, str):
                raise BadArgs(f"{k} must be a string")
            if "enum" in spec and v not in spec["enum"]:
                raise BadArgs(f"{k} must be one of {spec['enum']}")
        elif t == "integer":
            if isinstance(v, bool) or not isinstance(v, int):
                if isinstance(v, float) and v.is_integer():
                    v = int(v)
                else:
                    raise BadArgs(f"{k} must be an integer")
        elif t == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise BadArgs(f"{k} must be a number")
            v = float(v)
        out[k] = v
    if "segment_id" in out and not SEG_RE.match(out["segment_id"]):
        raise BadArgs("invalid segment_id")
    for lk in ("lang",):
        if lk in out and not LANG_RE.match(out[lk]):
            raise BadArgs("invalid lang")
    if name == "edit_line":
        t = out["text"]
        if not t.strip() or len(t) > MAX_EDIT:
            raise BadArgs(f"text must be 1..{MAX_EDIT} chars")
    if name == "retime" and not (0.92 <= out["tempo_factor"] <= 1.08):
        raise BadArgs("tempo_factor must be within 0.92..1.08")
    if name == "add_language" and out["from_ms"] < 0:
        raise BadArgs("from_ms must be >= 0")
    if name == "regenerate_chunk" and out["chunk_id"] < 0:
        raise BadArgs("chunk_id must be >= 0")
    if name == "explain" and "action_id" in out and not re.match(r"^[A-Za-z0-9_\-]{1,40}$", out["action_id"]):
        raise BadArgs("invalid action_id")
    if name == "report_issue" and not (1 <= len(out["details"]) <= 1000):
        raise BadArgs("details must be 1..1000 chars")
    if name == "change_voice" and not (1 <= len(out["voice"]) <= 64):
        raise BadArgs("voice must be 1..64 chars")
    if name == "change_voice" and not re.match(r"^[A-Za-z0-9_.\- ]+$", out["voice"]):
        raise BadArgs("voice has invalid characters")
    return out
