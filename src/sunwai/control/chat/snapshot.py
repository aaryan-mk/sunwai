"""Server-built snapshot of a job's visible lines: the only vocabulary the planner may use to name a line."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sunwai.control.errors import CtlError
from sunwai.control.store import JobStore

MAX_SNAP = 60
LINE_ACTIONS = ("edit_line", "undo_line", "redo_line", "change_expression", "retime")


def line_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Build the planner-visible entry of one segment row."""
    return {
        "ref": f"c{row['chunk']}s{row['id']}",
        "chunk": row["chunk"],
        "id": row["id"],
        "t": round(row["start"], 1),
        "speaker": row.get("speaker"),
        "dub": (row.get("tgt_text") or "")[:160],
        "src": (row.get("src_text") or "")[:120],
    }


def select_rows(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose at most :data:`MAX_SNAP` rows nearest the playhead, always including the highlighted line.

    Sorts ``rows`` in place by distance to the playhead when one is given.
    """
    playhead = context.get("playhead_ms")
    if isinstance(playhead, (int, float)):
        rows.sort(key=lambda r: abs((r["start"] + r["end"]) * 500 - playhead))
    has_current = context.get("chunk") is not None and context.get("segment_id") is not None
    current = next((r for r in rows if has_current and r["chunk"] == context["chunk"] and str(r["id"]) == str(context["segment_id"])), None)
    keep = rows[:MAX_SNAP]
    if current is not None and current not in keep:
        keep = keep[:-1] + [current]  # the highlighted line is ALWAYS visible to the planner and resolvable, whatever the playhead
    return keep


def last_line_action(store: JobStore) -> dict[str, Any] | None:
    """Describe the most recent line action of the job (what "undo", "redo" or "that" means), or ``None``."""
    try:
        with store.lock:
            actions = sorted(
                (a for a in store.actions.values() if (a.get("args") or {}).get("segment_id") is not None and a.get("type") in LINE_ACTIONS),
                key=lambda a: a.get("ts", ""),
            )
        if not actions:
            return None
        action = actions[-1]
        args = action["args"]
        return {"type": action["type"], "ref": f"c{args.get('chunk')}s{args.get('segment_id')}", "lang": args.get("lang"), "at": action.get("ts")}
    except Exception:
        return None


def build_snapshot(rows_of: Callable[[str], list[dict[str, Any]]], store: JobStore, context: dict[str, Any], langs: list[str]) -> dict[str, Any]:
    """Build the snapshot handed to the planner.

    Args:
        rows_of: Callable ``lang -> list of segment rows`` (the host's ``all_segments`` bound to the job).
        store: The job's store, for the last line action.
        context: Client context (lang, playhead, highlighted line).
        langs: Languages of the job.

    Returns:
        ``{"langs", "lang", "lines", "last_action"}``.
    """
    lang = context.get("lang") if context.get("lang") in langs else (langs[0] if langs else None)
    lines: list[dict[str, Any]] = []
    if lang:
        lines = sorted((line_entry(r) for r in select_rows(rows_of(lang), context)), key=lambda e: (e["chunk"], e["t"]))
    return {"langs": langs, "lang": lang, "lines": lines, "last_action": last_line_action(store)}


def resolve_segment(snap: dict[str, Any], ref: str) -> tuple[int, str]:
    """Resolve a snapshot ref ``c<chunk>s<id>`` to ``(chunk, segment id)``.

    Raises:
        CtlError: 422 when the ref is not in the server-built snapshot.
    """
    for line in snap["lines"]:
        if line["ref"] == ref:
            return line["chunk"], str(line["id"])
    raise CtlError(422, "unknown_segment", "the model referred to a line that is not in this job's visible lines")
