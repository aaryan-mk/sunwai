"""Cheap per-segment source language identification from Unicode script, plus translation routing and term masking."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Unicode script -> candidate source languages (first = default when no job LID agrees).
SCRIPT_LANGS: dict[str, list[str]] = {
    "DEVANAGARI": ["hi", "mr", "ne", "sa", "kok", "mai", "doi", "brx"],
    "TAMIL": ["ta"],
    "BENGALI": ["bn", "as"],
    "GUJARATI": ["gu"],
    "GURMUKHI": ["pa"],
    "ORIYA": ["or"],
    "TELUGU": ["te"],
    "KANNADA": ["kn"],
    "MALAYALAM": ["ml"],
    "ARABIC": ["ur", "sd", "ks"],
    "LATIN": ["en"],
}
_TOKEN = re.compile(r"\S+")
_SCRIPT_CACHE: dict[str, str] = {}
_HINDUSTANI = {"hi": "hi-ur", "ur": "hi-ur"}  # one spoken language; ASR flips between Devanagari and Arabic script
_TOKEN_EDGE_PUNCT = ".,;:!?\"'()[]{}।|-"


def _script_of(ch: str) -> str | None:
    if not (ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc")):  # Indic vowel signs are marks, not letters
        return None
    script = _SCRIPT_CACHE.get(ch)
    if script is None:
        try:
            script = unicodedata.name(ch).split()[0]
        except ValueError:
            script = "OTHER"
        _SCRIPT_CACHE[ch] = script
    return script


def script_profile(text: str) -> dict[str, int]:
    """Count letters per Unicode script."""
    profile: dict[str, int] = {}
    for ch in text or "":
        script = _script_of(ch)
        if script:
            profile[script] = profile.get(script, 0) + 1
    return profile


def _token_script(token: str) -> str | None:
    profile = script_profile(token)
    return max(profile, key=lambda k: profile[k]) if profile else None


def token_profile(text: str) -> dict[str, int]:
    """Count words per script (a word counts for its majority script).

    Token-level, so a few long Latin words do not outvote the Indic-script words around them.
    """
    profile: dict[str, int] = {}
    for m in _TOKEN.finditer(text or ""):
        script = _token_script(m.group(0))
        if script:
            profile[script] = profile.get(script, 0) + 1
    return profile


def _dominant(profile: dict[str, int]) -> str:
    return max(profile, key=lambda k: (profile[k], k != "LATIN"))  # tie -> the non-Latin script


def _unresolved(method: str) -> dict[str, Any]:
    return {"lang": "und", "confidence": 0.0, "method": method, "mixed": False, "embedded": []}


def _resolve_language(dominant: str, job_lid: str | None) -> tuple[str, float, str]:
    candidates = SCRIPT_LANGS[dominant]
    if dominant == "ARABIC" and job_lid in ("hi", "ur"):
        return "ur", 0.7, "script+job_lid_hindustani"  # script says Urdu; same spoken language as hi
    if dominant == "DEVANAGARI" and job_lid == "ur":
        return "hi", 0.7, "script+job_lid_hindustani"
    if len(candidates) == 1:
        return candidates[0], 0.95, "script"
    if job_lid in candidates:
        return str(job_lid), 0.7, "script+job_lid"
    return candidates[0], 0.4, "script_ambiguous"


def classify_text(text: str, job_lid: str | None, min_letters: int = 3) -> dict[str, Any]:
    """Classify one segment's source language from its script.

    Same-script ambiguity (e.g. Devanagari hi/mr/ne) is resolved by the video-level LID only when it lies in that
    script's candidate set (lower confidence). Latin text in a job whose LID is a non-Latin language is English
    (ASR emits native script for native speech) but flagged lower. Too few letters or an unknown script gives
    ``und``, never a guess.

    Returns:
        ``{"lang", "confidence", "method", "mixed", "embedded"}``.
    """
    profile = token_profile(text)
    if sum(script_profile(text).values()) < min_letters or not profile:
        return _unresolved("too_short")
    dominant = _dominant(profile)
    if dominant not in SCRIPT_LANGS:
        return _unresolved("script_unknown")
    fraction = profile[dominant] / sum(profile.values())
    lang, confidence, method = _resolve_language(dominant, job_lid)
    if dominant == "LATIN" and job_lid and job_lid != "en":
        confidence = min(confidence, 0.6)  # could be romanised native speech; not resolvable from script
    mixed = fraction < 0.9
    confidence = round(confidence * (fraction if mixed else 1.0), 3)
    embedded = [m.group(0) for m in _TOKEN.finditer(text or "") if _token_script(m.group(0)) not in (None, dominant)]
    return {"lang": lang, "confidence": confidence, "method": method, "mixed": mixed, "embedded": embedded}


def lang_group(lang: str) -> str:
    """Return the spoken-language group for scoring/routing (``hi`` and ``ur`` share Hindustani)."""
    return _HINDUSTANI.get(lang, lang)


def route_segment(
    span: dict[str, Any], target_lang: str, it2_targets: tuple[str, ...] = ("mni", "sd")
) -> dict[str, str]:
    """Choose the translation route for one source span (mirrors ``dub_pipeline.routed_mt``).

    Text already in the target language is passed through; unresolved (``und``) spans use the job path unchanged;
    homogeneous spans go to the local indic-translate.
    """
    src = span.get("lang") or "und"
    if src == target_lang and src != "und" and span.get("confidence", 0) >= 0.5 and not span.get("mixed"):
        return {"route": "passthrough", "src": src}
    if target_lang == "en":
        return {"route": "gemini-en", "src": src}
    if target_lang in it2_targets:
        return {"route": "it2", "src": src}
    return {"route": "indic-translate", "src": src}


def mask_foreign_terms(text: str, dominant_script: str | None = None) -> tuple[str, dict[str, str]]:
    """Replace tokens in a different script from the segment's (brand names, English terms) with placeholders.

    Keeps MT from mangling them; :func:`unmask_terms` reports missing placeholders so the caller can fall back.

    Returns:
        ``(masked_text, placeholder -> original term)``.
    """
    profile = token_profile(text)
    dominant = dominant_script or (_dominant(profile) if profile else None)
    mapping: dict[str, str] = {}

    def substitute(m: re.Match[str]) -> str:
        token = m.group(0)
        core = token.strip(_TOKEN_EDGE_PUNCT)
        script = _token_script(core)
        if core and script and dominant and script != dominant:
            key = f"QX{len(mapping)}Z"
            mapping[key] = core
            return token.replace(core, key)
        return token

    return _TOKEN.sub(substitute, text or ""), mapping


def unmask_terms(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Restore masked terms; return the text and the placeholders MT dropped."""
    missing = [k for k in mapping if k not in text]
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text, missing
