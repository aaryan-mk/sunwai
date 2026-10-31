"""VoicePlanner: assigns one distinct, CER-gated bank voice to every speaker in every language.

HARD guarantees: every (voice, measured language) pair in the result passes the CER gate, and no voice is used twice
within a measured language. Voices that are too similar to tell apart are avoided where possible and flagged
(``degraded``) where not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sunwai.planning.voices.costs import W_GENDER, VoiceCostModel, cosine, describe_speakers, gender_prior
from sunwai.planning.voices.gate import cer_gate, eligible, eligible_in_all
from sunwai.planning.voices.search import (
    BIG,
    NODE_LIMIT,
    TIME_LIMIT,
    BoundedSearch,
    Candidates,
    min_cost_matching,
    separated_matching,
)
from sunwai.planning.voices.types import Bank, Features, PlanError, Speaker, VoiceAssignment

TOPK = 8
_COST_SCALE = 10000


class VoicePlanner:
    """Plans voices for a set of speakers across languages.

    Args:
        bank: The voice bank (``voices``, per-language stats, ``native`` baselines and the distinctness ``tau``).
        languages: All languages to cover.
        opt_langs: Languages the optimisation compares voices in (defaults to ``languages``); only measured ones count.
    """

    def __init__(self, bank: Bank, languages: list[str], opt_langs: list[str] | None = None) -> None:
        """Create a planner."""
        self.bank = bank
        self.languages = languages
        self.requested = list(opt_langs or languages)
        self.gate = cer_gate(bank)
        self.opt_languages = [lang for lang in self.requested if lang in bank["langs"]] or ["hi"]
        self.measured = [lang for lang in languages if lang in bank["langs"]]
        self.tau: float = bank["tau"]["all"]
        self.min_separation = round(1 - self.tau, 4)
        self.voice_names = sorted(
            v for v in bank["voices"] if all(v in bank["langs"][lang] for lang in self.opt_languages)
        )
        self.costs = VoiceCostModel(bank, self.gate, self.opt_languages)
        self._warnings: list[str] = []

    # ------------------------------------------------------------------ public entry
    def assign(self, speakers: list[Speaker], feats: dict[Any, Features | None]) -> VoiceAssignment:
        """Assign voices to ``speakers`` (``feats`` holds their acoustic features, None when unavailable).

        Raises:
            PlanError: Not enough gated voices to give every speaker a distinct one.
        """
        self._warnings = []
        if not speakers:
            return VoiceAssignment({}, {lang: {} for lang in self.languages}, False)
        ordered = sorted(speakers, key=lambda s: int(s["id"]))
        ids = [int(s["id"]) for s in ordered]
        self._check_capacity(ordered)
        chosen = self._choose_canonical(ordered, ids, feats)
        degraded = self._is_degraded(chosen)
        canonical = {str(i): v for i, v in zip(ids, chosen, strict=True)}
        per_language, unavailable, language_degraded = self._assign_languages(ordered, ids, canonical)
        return VoiceAssignment(
            canonical=canonical,
            per_language=per_language,
            degraded=degraded or language_degraded,
            warnings=self._warnings,
            detail=describe_speakers(ordered, feats, per_language, self.languages, self.bank),
            unavailable=unavailable,
        )

    # ------------------------------------------------------------------ helpers
    def _pairs_too_close(self, voices: list[str], languages: list[str] | None = None) -> bool:
        return bool(self.costs.too_close(voices, self.min_separation, languages))

    def _eligible_by_gender(self, lang: str) -> dict[str, list[str]]:
        genders = self.bank["voices"]
        return {
            g: [v for v in self.voice_names if genders[v]["gender"] == g and eligible(self.bank, self.gate, lang, v)]
            for g in ("male", "female")
        }

    def _check_capacity(self, speakers: list[Speaker]) -> None:
        """Confident speakers cannot cross gender, unknown ones may spill: require enough unique eligible voices."""
        for lang in [lang for lang in self.measured if lang in self.requested]:
            pool = self._eligible_by_gender(lang)
            male = sum(1 for s in speakers if s.get("gender") == "male")
            female = sum(1 for s in speakers if s.get("gender") == "female")
            if (
                male > len(pool["male"])
                or female > len(pool["female"])
                or len(speakers) > len(pool["male"]) + len(pool["female"])
            ):
                raise PlanError(
                    f"capacity: {male} male + {female} female confident of {len(speakers)} speakers but only "
                    f"{len(pool['male'])} male / {len(pool['female'])} female eligible voices in {lang}"
                )

    @staticmethod
    def _features(feats: dict[Any, Features | None], sid: int) -> Features | None:
        return feats.get(sid) or feats.get(str(sid))  # type: ignore[call-overload]

    # ------------------------------------------------------------------ canonical voices
    def _candidates(self, speakers: list[Speaker], ids: list[int], feats: dict[Any, Features | None]) -> Candidates:
        genders = self.bank["voices"]
        eligible_everywhere = {
            v for v in self.voice_names if all(eligible(self.bank, self.gate, lang, v) for lang in self.opt_languages)
        }
        candidates: Candidates = {}
        for speaker in speakers:
            sid = int(speaker["id"])
            f = self._features(feats, sid)
            confident, _, _ = gender_prior(speaker)
            pool = [v for v in self.voice_names if not confident or genders[v]["gender"] == speaker["gender"]]
            usable = [v for v in pool if v in eligible_everywhere]
            if len(usable) < len(ids):  # not enough all-language-eligible: widen to per-language gating
                usable = [
                    v for v in pool if any(eligible(self.bank, self.gate, lang, v) for lang in self.opt_languages)
                ] or pool
            rows = sorted((self.costs.cost(speaker, f, v), v) for v in usable)
            candidates[sid] = rows[: max(TOPK, len(ids) + 4)]
            if f is None:
                self._warnings.append(f"speaker {sid}: no clean speech features; cost uses gender/quality only")
        return candidates

    def _choose_canonical(
        self, speakers: list[Speaker], ids: list[int], feats: dict[Any, Features | None]
    ) -> list[str]:
        """Bounded exact search, then a deterministic penalised matching when it fails or hits its limits."""
        candidates = self._candidates(speakers, ids, feats)
        search = BoundedSearch(ids, candidates, self.costs.separation, self.min_separation)
        chosen = search.run()
        if chosen is not None and not search.limit_hit:
            return chosen
        if search.limit_hit:
            self._warnings.append(
                f"optimizer bound hit (nodes>{NODE_LIMIT} or >{TIME_LIMIT}s): deterministic greedy assignment used"
            )
        else:
            self._warnings.append(
                f"INFEASIBLE under tau={self.tau:.3f}: taking the maximum-separation assignment (distinctness_degraded)"
            )
        return self._matching_fallback(speakers, ids, feats)

    def _matching_fallback(
        self, speakers: list[Speaker], ids: list[int], feats: dict[Any, Features | None]
    ) -> list[str]:
        """Min-cost matching over ALL gated voices (feasibility-preserving)."""
        genders = self.bank["voices"]
        pool = [
            v for v in self.voice_names if any(eligible(self.bank, self.gate, lang, v) for lang in self.opt_languages)
        ]
        if len(pool) < len(ids):
            raise PlanError(f"capacity: only {len(pool)} gated voices for {len(ids)} speakers")
        base: dict[tuple[int, str], int] = {}
        by_id = {int(s["id"]): s for s in speakers}
        for sid in ids:
            speaker = by_id[sid]
            f = self._features(feats, sid)
            confident, _, _ = gender_prior(speaker)
            for v in pool:
                crosses = confident and genders[v]["gender"] != speaker["gender"]
                base[(sid, v)] = BIG if crosses else int(round(self.costs.cost(speaker, f, v) * _COST_SCALE))
        return separated_matching(ids, pool, base, self.costs.separation, self.min_separation)

    def _is_degraded(self, chosen: list[str]) -> bool:
        if len(set(chosen)) < len(chosen):
            raise PlanError("capacity: optimizer produced a repeated voice")
        if not self._pairs_too_close(chosen):
            return False
        if not any("INFEASIBLE" in w for w in self._warnings):
            self._warnings.append(f"distinctness below tau={self.tau:.3f} for some pair (distinctness_degraded)")
        return True

    # ------------------------------------------------------------------ per-language voices
    def _assign_languages(
        self, speakers: list[Speaker], ids: list[int], canonical: dict[str, str]
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], bool]:
        per_language: dict[str, dict[str, str]] = {}
        unavailable: dict[str, str] = {}
        degraded = False
        for lang in self.languages:
            measured = lang in self.bank["langs"]
            ok = self._language_gate(lang, measured)
            matched = self._match_language(speakers, ids, canonical, ok, lang if measured else self.opt_languages[0])
            if matched is None:
                message = f"capacity: no full matching of {len(ids)} speakers to distinct gated voices in {lang}"
                if lang in self.requested:
                    raise PlanError(message)
                unavailable[lang] = message
                self._warnings.append(f"{lang}: unavailable ({message})")
                continue
            per_language[lang] = matched
            for sid in ids:
                if matched[str(sid)] != canonical[str(sid)]:
                    self._warnings.append(
                        f"speaker {sid} {lang}: canonical {canonical[str(sid)]} -> {matched[str(sid)]} (gate/matching)"
                    )
            if measured and self._pairs_too_close([matched[str(i)] for i in ids], [lang]):
                degraded = True
                self._warnings.append(f"{lang}: distinctness below tau (distinctness_degraded)")
        return per_language, unavailable, degraded

    def _language_gate(self, lang: str, measured: bool) -> Callable[[str], bool]:
        """Return the voice filter for ``lang``: its own gate, or the gate of every measured language if unmeasured."""
        if measured:
            return lambda voice: eligible(self.bank, self.gate, lang, voice)
        return lambda voice: eligible_in_all(self.bank, self.gate, voice)

    def _match_language(
        self,
        speakers: list[Speaker],
        ids: list[int],
        canonical: dict[str, str],
        ok: Callable[[str], bool],
        ref_lang: str,
    ) -> dict[str, str] | None:
        """Deterministic min-cost matching of speakers to gated voices in one language (canonical preferred)."""
        genders = self.bank["voices"]
        voices = [v for v in self.voice_names if ok(v) and v in self.bank["langs"].get(ref_lang, {})]
        if len(voices) < len(ids):
            return None
        by_id = {int(s["id"]): s for s in speakers}
        matrix: list[list[int]] = []
        for sid in ids:
            speaker = by_id[sid]
            confident, p_female, _ = gender_prior(speaker)
            preferred = canonical[str(sid)]
            row: list[int] = []
            for v in voices:
                if confident and genders[v]["gender"] != speaker["gender"]:
                    row.append(BIG)
                    continue
                cost = (
                    0.0
                    if v == preferred
                    else 1.0
                    + (1 - cosine(self.costs.embedding(ref_lang, preferred), self.costs.embedding(ref_lang, v)))
                )
                if not confident:
                    cost += W_GENDER * (p_female if genders[v]["gender"] == "male" else 1 - p_female)
                row.append(int(round(cost * _COST_SCALE)))
            matrix.append(row)
        pairs = min_cost_matching(matrix)
        if pairs is None:
            return None
        return {str(ids[i]): voices[j] for i, j in pairs}
