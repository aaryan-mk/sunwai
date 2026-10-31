"""Pure text analysis of the user's chat message: quoted spans, verbatim restoration and tempo-request limits."""

from __future__ import annotations

import re
import unicodedata

TEMPO_MIN, TEMPO_MAX = 0.92, 1.08
QUOTE_PATTERNS = [
    re.compile(r"(?<!\w)'(.+?)'(?!\w)", re.S),
    re.compile(r'"(.+?)"', re.S),
    re.compile("‘(.+?)’", re.S),
    re.compile("“(.+?)”", re.S),
]
_TEMPO_WORDS = re.compile(
    r"(?<![\w-])(?:twice|thrice|(?:double|doubled|triple|tripled|quadruple|quadrupled))"
    r"(?:\s+as\s+\w+|\s+the\s+(?:speed|tempo|rate|pace)|\s+(?:speed|tempo|rate|pace))\b"
    r"|(?<![\w-])half(?:\s+(?:the\s+)?(?:speed|tempo|rate|pace|time)|\s+as\s+(?:fast|quick|slow|long|short))\b"
    r"|\b(?:by|to|at)\s+(?:half|a\s+half|a\s+quarter|quarter)\b(?!\s*(?:of\s+)?(?:the\s+)?(?:line|sentence|part|chunk))"
    r"|(?<![\w-])quarter\s+(?:the\s+)?(?:speed|tempo|rate|pace)\b"
    r"|one and a half times"
    r"|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|fifty|hundred|a\s+hundred|dozen|\d+)\s+times\b"
    r"|(?<![\w-])(?:halve|halved|double|triple|quadruple)\s+(?:it|this|that|the\s+(?:speed|tempo|rate|pace))\b",
    re.I,
)
_NUM_X = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|×|times)\b", re.I)
_NUM_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", re.I)
_ABSOLUTE_BEFORE = re.compile(r"(?:\bto|\bat|\bset\w*|=|\bof)\s*$", re.I)
_ABSOLUTE_AFTER = re.compile(r"\s*(?:speed|tempo|rate)\b", re.I)
_REPLACEMENT_MARK = re.compile(
    r"(?:\b(?:to|as|with|say|says|saying|said|read|reads|reading|write|put|use|be|becomes?|become|instead)"
    r"|=|\b(?:this|following|that|it)\s*:)\s*$",
    re.I,
)
_IDENTIFYING_AFTER = re.compile(r"^\s*(?:to|with|into|should|so|and)\b", re.I)  # '"X" to end with a period': X identifies the existing line
_ZERO_WIDTH = re.compile("[​‌‍﻿]")
_EDGE_PUNCTUATION = " .,;:!?।॥…-"


def quoted_spans(message: str) -> list[str]:
    """Return the exact quoted spans of ``message`` in order of appearance (not stripped: surrounding spaces are the user's)."""
    found = []
    for pattern in QUOTE_PATTERNS:
        for match in pattern.finditer(message):
            if match.group(1).strip():
                found.append((match.start(), match.group(1)))
    return [span for _, span in sorted(found)]


def canonical_text(text: str) -> str:
    """Reduce ``text`` to the differences a model commonly introduces.

    Unicode form, chandrabindu vs anusvara, nukta, zero-width characters, spacing, edge punctuation and case are erased.
    A real spelling difference such as ``colour`` vs ``color`` is NOT.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text).replace("ँ", "ं").replace("़", "")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text.strip(_EDGE_PUNCTUATION)


def replacement_spans(message: str) -> list[str]:
    """Return the quoted spans that are plausibly the NEW text.

    Only spans immediately preceded by a replacement marker (to / as / with / say / put / use / ``:`` / ``=`` ...) count. A quote
    that merely identifies the existing line (``Change "Hello!" to end with a period``) is never used.
    """
    found = []
    for pattern in QUOTE_PATTERNS:
        for match in pattern.finditer(message):
            before = message[max(0, match.start() - 18) : match.start()]
            after = message[match.end() : match.end() + 12]
            if match.group(1).strip() and _REPLACEMENT_MARK.search(before) and not _IDENTIFYING_AFTER.match(after):
                found.append((match.start(), match.group(1)))
    return [span for _, span in sorted(found)]


def restore_verbatim(message: str, text: str) -> str:
    """Prefer the user's own quoted words over the model's rendering of them.

    If ``text`` equals a replacement-position quoted span after canonicalisation but is not byte-identical, the user's span is
    returned verbatim. There is no fuzzy matching, and identifying quotes or ambiguous collisions are never substituted.
    """
    if text in quoted_spans(message):
        return text  # byte-identical to one of the user's quotes: it IS the user's text
    canonical = canonical_text(text)
    hits = [span for span in replacement_spans(message) if canonical_text(span) == canonical]
    return hits[0] if len(set(hits)) == 1 else text  # ambiguous => fail safe, keep the model's text


def _percent_out_of_range(msg: str, match: re.Match[str]) -> bool:
    value = float(match.group(1))
    before, after = msg[max(0, match.start() - 16) : match.start()], msg[match.end() : match.end() + 14]
    absolute = bool(_ABSOLUTE_BEFORE.search(before) or _ABSOLUTE_AFTER.match(after))
    if absolute and 92 <= value <= 108:
        return False  # "set speed to 105%" / "92% speed": an absolute factor inside the range
    return absolute or value > 8  # an absolute factor outside the range, or any relative change > 8 %


def retime_out_of_scope(message: str) -> bool:
    """Tell whether the wording asks for a tempo change beyond 0.92-1.08.

    Called ONLY when the model chose the retime tool, so the request is a tempo request by definition. Quoted text is
    ignored; explicit numbers are compared with the supported factor (105% / 1.05x are fine, 30% faster / 2x are not);
    "twice / double / half ..." are out.
    """
    msg = message or ""
    for pattern in QUOTE_PATTERNS:
        msg = pattern.sub(" ", msg)
    if _TEMPO_WORDS.search(msg):
        return True
    if any(not (TEMPO_MIN <= float(m.group(1)) <= TEMPO_MAX) for m in _NUM_X.finditer(msg)):
        return True
    return any(_percent_out_of_range(msg, m) for m in _NUM_PCT.finditer(msg))
