"""Pure cost and distinctness model for matching speakers to bank voices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sunwai.planning.voices.gate import CER_DELTA
from sunwai.planning.voices.types import Bank, Features, Gate, Speaker

SC_COS, SC_F0, W_GENDER = 0.25, 0.20, 3.0  # fixed scales that standardise cosine distance, |d ln F0| and gender terms


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product of two unit-norm vectors."""
    return sum(x * y for x, y in zip(a, b, strict=False))


def gender_prior(speaker: Speaker) -> tuple[bool, float, float]:
    """Return ``(confident, p_female, weight)`` for a speaker.

    ``confident`` is True when the gender is known; ``weight`` grows from 0.5 to 1 with up to 20 s of speech.
    """
    confident = speaker.get("gender") in ("male", "female")
    p_female = 0.5 if speaker.get("p_female") is None else float(speaker["p_female"])
    weight = 0.5 + 0.5 * min(1.0, float(speaker.get("total_s") or 0) / 20.0)
    return confident, p_female, weight


class VoiceCostModel:
    """Scores speaker-voice pairs and voice-voice distinctness over the optimisation languages.

    Args:
        bank: The voice bank.
        gate: The CER gate of the bank.
        languages: Measured languages the voices are compared in.
    """

    def __init__(self, bank: Bank, gate: Gate, languages: list[str]) -> None:
        """Create the model."""
        self.bank = bank
        self.gate = gate
        self.languages = languages
        self._separation_cache: dict[tuple[str, str, tuple[str, ...]], float] = {}

    def embedding(self, lang: str, voice: str) -> list[float]:
        """Return the voice's embedding in ``lang``."""
        emb: list[float] = self.bank["langs"][lang][voice]["emb"]
        return emb

    def _mean_logf0(self, voice: str) -> float | None:
        values = [
            self.bank["langs"][lang][voice]["logf0"]
            for lang in self.languages
            if self.bank["langs"][lang].get(voice, {}).get("logf0") is not None
        ]
        return sum(values) / len(values) if values else None

    def cost(self, speaker: Speaker, features: Features | None, voice: str) -> float:
        """Return the assignment cost of ``voice`` for ``speaker`` (lower is better).

        Sums timbre distance, pitch distance, a gender penalty for speakers of unknown gender, and the voice's CER
        excess over the language's native voices.
        """
        confident, p_female, weight = gender_prior(speaker)
        total = 0.0
        if features is not None:
            distances = [1 - cosine(features["emb"], self.embedding(lang, voice)) for lang in self.languages]
            total += sum(distances) / len(self.languages) / SC_COS
            voice_f0 = self._mean_logf0(voice)
            if features.get("logf0") is not None and voice_f0 is not None:
                total += abs(features["logf0"] - voice_f0) / SC_F0
        if not confident:
            male = self.bank["voices"][voice]["gender"] == "male"
            total += W_GENDER * weight * (p_female if male else 1 - p_female)
        excess = [
            max(0.0, self.bank["langs"][lang][voice]["cer"] - self.gate[lang]["native_cer"]) / CER_DELTA
            for lang in self.languages
        ]
        total += sum(excess) / len(self.languages)
        return round(float(total), 4)

    def separation(self, a: str, b: str, languages: list[str] | None = None) -> float:
        """Return the smallest cosine distance between two voices over ``languages`` (default: the model's)."""
        langs = tuple(languages or self.languages)
        key = (a, b, langs) if a < b else (b, a, langs)
        if key not in self._separation_cache:
            self._separation_cache[key] = round(
                min(1 - cosine(self.embedding(lang, a), self.embedding(lang, b)) for lang in langs), 4
            )
        return self._separation_cache[key]

    def too_close(
        self, voices: Sequence[str], min_separation: float, languages: list[str] | None = None
    ) -> list[tuple[int, int]]:
        """Return the index pairs of ``voices`` closer than ``min_separation``."""
        return [
            (i, j)
            for i in range(len(voices))
            for j in range(i + 1, len(voices))
            if self.separation(voices[i], voices[j], languages) < min_separation
        ]


def describe_speakers(
    speakers: list[Speaker],
    feats: dict[Any, Features | None],
    per_language: dict[str, dict[str, str]],
    languages: list[str],
    bank: Bank,
) -> dict[str, Any]:
    """Return the per-speaker record stored in the plan (gender prior, evidence and similarity per language)."""
    detail: dict[str, Any] = {}
    for speaker in speakers:
        sid = int(speaker["id"])
        f = feats.get(sid)
        similarity: dict[str, float | None] = {}
        for lang in languages:
            voice = per_language.get(lang, {}).get(str(sid))
            if f and voice is not None and voice in bank["langs"].get(lang, {}):
                similarity[lang] = round(float(cosine(f["emb"], bank["langs"][lang][voice]["emb"])), 4)
            else:
                similarity[lang] = None
        detail[str(sid)] = {
            "gender": speaker.get("gender"),
            "p_female": speaker.get("p_female"),
            "total_s": speaker.get("total_s"),
            "logf0": None if not f else f.get("logf0"),
            "evidence_s": None if not f else f.get("evidence_s"),
            "emb": None if not f else [round(float(x), 4) for x in f["emb"]],
            "sim": similarity,
        }
    return detail
