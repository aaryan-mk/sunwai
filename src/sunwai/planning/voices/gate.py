"""Intelligibility (CER) gate: which voices may speak which language.

A language's NATIVE voices are always eligible, because CER is measured relative to them (they are the baseline);
every other voice must pass ``CER <= native + 5`` (capped at 15) and have no failed synthesis.
"""

from __future__ import annotations

from sunwai.planning.voices.types import Bank, Gate

CER_DELTA = 5.0  # intelligibility gate: CER <= native + 5 points ...
CER_CEIL = 15.0  # ... and <= 15 % (raised to native + 5 if the native voice itself is above 10 %)


def cer_gate(bank: Bank) -> Gate:
    """Return, per language, the native CER (mean over its native voices) and the ceiling actually applied."""
    gate: Gate = {}
    for lang, voices in bank.get("langs", {}).items():
        native = [voices[v]["cer"] for v in bank.get("native", {}).get(lang, []) if v in voices]
        mean = sum(native) / len(native) if native else 0.0
        ceiling = CER_CEIL if mean <= 10 else mean + CER_DELTA
        gate[lang] = {"native_cer": round(mean, 2), "limit": round(min(mean + CER_DELTA, ceiling), 2)}
    return gate


def eligible(bank: Bank, gate: Gate, lang: str, voice: str) -> bool:
    """Return True when ``voice`` passes the hard CER gate in ``lang`` (native voices always do)."""
    if voice in bank.get("native", {}).get(lang, []):
        return True
    entry = bank["langs"].get(lang, {}).get(voice)
    return bool(entry) and entry["cer"] <= gate[lang]["limit"] and entry.get("fails", 0) == 0


def eligible_in_all(bank: Bank, gate: Gate, voice: str) -> bool:
    """Return True when ``voice`` passes the gate in every measured language."""
    return all(eligible(bank, gate, lang, voice) for lang in bank["langs"])
