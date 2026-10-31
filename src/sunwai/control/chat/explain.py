"""Plain-language explanation of a past action, built only from the job's own action and operation log."""

from __future__ import annotations

from typing import Any

ACTION_PHRASES = {
    "edit_line": "replaced the text of a line",
    "undo_line": "undid the last edit of a line",
    "redo_line": "re-applied an undone edit",
    "change_expression": "previewed or applied a different delivery for a line",
    "retime": "tried a tempo change on a line",
    "report_issue": "recorded an issue report",
    "add_language": "requested another language",
}


def _operation_state(op: dict[str, Any] | None) -> str:
    if not op:
        return "no background operation was needed"
    state = f"its operation is {op['state']} at stage {op['stage']}"
    if op.get("error"):
        state += f" with error: {op['error']}"
    return state + ("; lip-sync is ready" if op.get("lipsync_ready") else "; lip-sync is not ready")


def explain_action(actions: dict[str, dict[str, Any]], ops: list[dict[str, Any]], action_id: str | None) -> str:
    """Explain an action of this job.

    The model never supplies content, only an optional id that must belong to the job; an unknown id is not explained,
    and the latest action is described instead.

    Args:
        actions: The job's actions by id.
        ops: The job's operation records.
        action_id: Id to explain, or ``None`` for the latest action.
    """
    ordered = sorted(actions.values(), key=lambda a: a.get("ts", ""))
    if action_id is not None:
        action = actions.get(action_id)
        if action is None:
            return "I can't find that action in this job's log. " + (explain_action(actions, ops, None) if ordered else "")
    else:
        if not ordered:
            return "Nothing has been done on this job yet, so there is nothing to explain."
        action = ordered[-1]
    mine = [o for o in ops if o.get("action_id") == action["action_id"]]
    op = max(mine, key=lambda o: o.get("ts", "")) if mine else None
    args = {k: v for k, v in (action.get("args") or {}).items() if v is not None}
    what = ACTION_PHRASES.get(action["type"], action["type"])
    where = f" (language {args['lang']}, chunk {args['chunk']}, line {args['segment_id']})" if "segment_id" in args and "chunk" in args else ""
    when = action.get("ts", "an unknown date")[:19].replace("T", " ")
    return f"On {when} UTC the studio {what}{where}; {_operation_state(op)}."
