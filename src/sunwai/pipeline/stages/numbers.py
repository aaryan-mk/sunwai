"""Numbers stage: spoken forms of clear quantities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List


log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import Segments, StageBase

from sunwai.language.spoken import set_stage_language
from sunwai.language.numbers import _TN_VER, _tn_flag, normalize_numbers
from sunwai.pipeline.support.expression_flags import _expr_on
from sunwai.pipeline.support.storage import atomic_write_json




def stage_numbers(args, out_dir: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """After the fit, deterministic, no external call, tgt_text (display, subtitles) is never modified. A line gets seg["num_text"]
    (+ num_src / num_ver / num_lang) only when (a) it contains a clear quantity, (b) fit.json holds a budget for exactly this fitted
    text and (c) the spoken form still fits that budget. Anything else is spoken exactly as today. If segments.json cannot be
    written, every spoken form is dropped (memory and disk must agree)."""
    from sunwai.pipeline.fitting.fit_stage import graphemes as _gr
    set_stage_language(args.lang)
    on = _tn_flag("TTS_NORMALIZE", True) and _tn_flag("TTS_NORMALIZE_NUMBERS", True)
    budgets = {}
    try:
        budgets = json.load(open(out_dir / "fit.json", encoding="utf-8"))
    except Exception:
        pass
    n = 0
    for s in segments:
        t = s.get("tgt_text") or ""
        for k in ("num_text", "num_src", "num_ver", "num_lang", "pre_numfit_text"):
            s.pop(k, None)
        if not on or not t.strip() or s.get("edited") or s.get("error") or s.get("fallback") or _expr_on(s):
            continue
        try:
            sp = normalize_numbers(t, args.lang)
            if sp == t:
                continue
            ent = budgets.get(str(s.get("id"))) or {}
            b = ent.get("budget")
            if not b or ent.get("fit_text") != t or _gr(sp) > b:
                continue                                       # no matching fit record, or the spoken form would exceed the fit budget
            s["num_text"], s["num_src"], s["num_ver"], s["num_lang"] = sp, t, _TN_VER, args.lang
            n += 1
        except Exception as e:
            log.warning("Stage numbers seg %s failed open (%s)", s.get("id"), str(e)[:100])
            for k in ("num_text", "num_src", "num_ver", "num_lang"):
                s.pop(k, None)
    try:
        atomic_write_json(out_dir / "segments.json", segments, indent=2)
    except Exception as e:
        log.warning("Stage numbers: segments.json write failed (%s); dropping spoken forms", e)
        for s in segments:
            for k in ("num_text", "num_src", "num_ver", "num_lang"):
                s.pop(k, None)
        n = 0
    log.info("Stage numbers: %d/%d lines have a spoken number form", n, len(segments))
    return segments


class NumbersStage(StageBase[Segments, Segments]):
    """Produce normalised spoken text (numbers spelled out)."""

    name = "numbers"
    policy = StagePolicy(timing_key="numbers")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Produce normalised spoken text (numbers spelled out)."""
        return stage_numbers(ctx.legacy_args(), ctx.work_dir, value)  # type: ignore[no-any-return]
