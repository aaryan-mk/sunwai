"""Pure rule planner: turns a delivery intent into a style tag and punctuation, and vets wording patches."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from sunwai.control.expression.errors import ExprError

INTENT_STYLE: dict[str, str | None] = {"angrier": "anger", "sadder": "sad", "excited": "happy", "calmer": None}
TERMINATORS = "।.!?…۔"
_TERM_RE = re.compile(r"[" + re.escape(TERMINATORS) + r"\s]+$")
_ELLIPSIS_RE = re.compile(r"\.{2,}|…")
_PADDED_ELLIPSIS_RE = re.compile(r"\s*(\.{2,}|…)\s*")
DEFAULT_STOP = {"hi": "।", "mr": "।", "ur": "۔", "bn": "।", "ta": ".", "te": ".", "ml": ".", "kn": ".", "gu": "।"}
NEGATIONS = {
    "hi": ["नहीं", "मत", "न "],
    "mr": ["नाही", "नको"],
    "ur": ["نہیں", "مت"],
    "ta": ["இல்லை", "வேண்டாம்", "மாட்டேன்", "முடியாது", "அல்ல"],
    "te": ["లేదు", "వద్దు"],
    "bn": ["না", "নেই"],
}
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.-]+")


def split_terminator(text: str | None) -> tuple[str, str]:
    """Split ``text`` into its body and the trailing terminator run (punctuation and spaces, stripped)."""
    stripped = (text or "").strip()
    match = _TERM_RE.search(stripped)
    body = stripped[: match.start()].rstrip() if match else stripped
    return body, (match.group(0).strip() if match else "")


def _terminal_for(intent: str, question: bool, stop: str) -> tuple[str, str | None, str]:
    """Return ``(terminator, style, rationale)`` of the normal (non-compact) plan for ``intent``."""
    if intent == "angrier":
        return ("?!" if question else "!"), "anger", "terminal '!' + style anger"
    if intent == "excited":
        return ("?!" if question else "!"), "happy", "terminal '!' + style happy (auditioned)"
    if intent == "sadder":
        return ("?" if question else "..."), "sad", "trailing ellipsis + style sad"
    return ("?" if question else stop), None, "style omitted, restrained punctuation"  # calmer: no style tag


def plan_expression(
    text: str, lang: str, intent: str | None, *, compact: bool = False, wording_patch: str | None = None
) -> dict[str, Any]:
    """Plan the spoken text and style for a delivery intent.

    Args:
        text: Current display text of the line.
        lang: Language code.
        intent: One of :data:`INTENT_STYLE`.
        compact: Use the restrained "repair" plan (used once when the first take overruns its slot).
        wording_patch: Optional replacement wording; must pass :func:`check_wording`.

    Returns:
        ``{style, punctuation_patch, spoken_text, wording_patch, rationale, intent}``.

    Raises:
        ExprError: 422 for an unknown intent or a rejected wording patch.
    """
    intent = (intent or "").strip().lower()
    if intent not in INTENT_STYLE:
        raise ExprError(422, f"unknown intent {intent!r}; one of {sorted(INTENT_STYLE)}")
    base = text
    if wording_patch:
        ok, why = check_wording(text, wording_patch, lang)
        if not ok:
            raise ExprError(422, "wording patch rejected: " + "; ".join(why))
        base = wording_patch
    body, term = split_terminator(base)
    if intent != "sadder":
        body = _ELLIPSIS_RE.sub(",", body)
    question = "?" in term
    if compact:
        new_term = "?" if question else DEFAULT_STOP.get(lang, ".")
        style, why = INTENT_STYLE[intent], "compact repair: restrained punctuation"
        body = _PADDED_ELLIPSIS_RE.sub(" ", body)
    else:
        new_term, style, why = _terminal_for(intent, question, DEFAULT_STOP.get(lang, "."))
    patch = [{"op": "terminal", "from": term, "to": new_term}] if term != new_term else []
    return {
        "style": style,
        "punctuation_patch": patch,
        "spoken_text": body + new_term,
        "wording_patch": wording_patch,
        "rationale": why,
        "intent": intent,
    }


def _digit_groups(text: str) -> list[str]:
    """Return the digit runs of ``text`` with every Unicode decimal digit mapped to ASCII."""
    chars = []
    for ch in text:
        try:
            chars.append(str(unicodedata.digit(ch)))
        except ValueError:
            chars.append(" " if not ch.isalnum() else ch)
    return re.findall(r"\d+", "".join(chars))


def check_wording(old: str, new: str, lang: str) -> tuple[bool, list[str]]:
    """Check that a wording patch keeps numbers, Latin/entity tokens and negation count.

    This cannot guarantee semantic fidelity.

    Returns:
        ``(ok, reasons)`` where ``reasons`` lists every difference found.
    """
    why = []
    if sorted(_digit_groups(old)) != sorted(_digit_groups(new)):
        why.append("numbers differ")
    if sorted(_LATIN_TOKEN_RE.findall(old)) != sorted(_LATIN_TOKEN_RE.findall(new)):
        why.append("Latin/entity tokens differ")
    negations = NEGATIONS.get(lang, [])
    if sum(old.count(n) for n in negations) != sum(new.count(n) for n in negations):
        why.append("negation count differs")
    return (not why), why
