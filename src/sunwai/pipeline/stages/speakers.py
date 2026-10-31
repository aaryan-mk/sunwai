"""Speakers stage: speaker identity per slot and source-language identification."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import StageBase

from sunwai.media.audio import load_wav_mono
from sunwai.pipeline.support.storage import atomic_write_json


def _source_lang(out_dir):
    for d in (out_dir, os.path.dirname(os.environ.get("SPEAKER_MAP", "")) or None):
        try:
            lab = json.load(open(os.path.join(str(d), "lid.json"))).get("label") or ""
            return lab.split(":")[0].strip() or "en"
        except Exception:
            continue
    return "en"


# ---------------------------------------------------------------------------
# Stage 5: Speakers (ECAPA + clustering + gender)
# ---------------------------------------------------------------------------
def stage_speakers(out_dir: Path, slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assign a speaker index and gender to every slot using the job's speaker map or local clustering.

    Args:
        out_dir: Chunk directory.
        slots: Slot records from the slots stage.

    Returns:
        The speaker table, also written to ``speakers.json``.
    """
    sp_path = out_dir / "speakers.json"
    if sp_path.exists() and slots:
        log.info("Stage speakers: cached")
        with open(sp_path, encoding="utf-8") as f:
            return json.load(f)

    smap = os.environ.get("SPEAKER_MAP")
    if smap and os.path.exists(smap):
        g = json.load(open(smap, encoding="utf-8"))
        off = float(os.environ.get("CHUNK_START", "0"))
        turns = [(t["start"] - off, t["end"] - off, int(t["speaker"])) for t in g.get("turns", [])]
        ginfo = {int(x["id"]): x for x in g.get("speakers", [])}
        for sl in slots:
            ov = {}
            for a, b, spk in turns:
                o = min(b, sl["end"]) - max(a, sl["start"])
                if o > 0:
                    ov[spk] = ov.get(spk, 0.0) + o
            if ov:
                sl["speaker"] = max(ov, key=ov.get)
            elif turns:
                c = (sl["start"] + sl["end"]) / 2
                sl["speaker"] = min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - c))[2]
            else:
                sl["speaker"] = 0
        info = {}
        for spk in sorted({sl["speaker"] for sl in slots}):
            mine = [sl for sl in slots if sl["speaker"] == spk]
            gs = ginfo.get(spk, {})
            info[str(spk)] = {"gender": gs.get("gender", "unknown"), "mean_p_female": gs.get("p_female", 0.5),
                              "total_speech": sum(sl["end"] - sl["start"] for sl in mine), "n_slots": len(mine), "global": True}
            for sl in mine:
                sl["gender"] = info[str(spk)]["gender"]
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        log.info("Stage speakers: %d global speakers mapped from %s", len(info), smap)
        return info

    w16 = out_dir / "src16k.wav"
    y = load_wav_mono(w16, sr=16000)
    sr = 16000

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(out_dir / ".ecapa"),
    )

    embs: Dict[int, Optional[np.ndarray]] = {}
    for sl in slots:
        dur = sl["end"] - sl["start"]
        if dur >= 0.8:
            a = int(sl["start"] * sr)
            b = int(sl["end"] * sr)
            seg = y[a:b]
            import torch
            emb = ecapa.encode_batch(torch.from_numpy(seg.reshape(1, -1)))
            embs[sl["id"]] = emb.squeeze().cpu().numpy()
        else:
            embs[sl["id"]] = None

    # Cluster only slots with embeddings
    ids_with = [sl["id"] for sl in slots if embs[sl["id"]] is not None]
    if ids_with:
        X = np.stack([embs[i] for i in ids_with])
        from sklearn.cluster import AgglomerativeClustering
        if len(ids_with) < 2:                   # sklearn cannot cluster a single embedding -> one speaker
            labels = np.zeros(len(ids_with), dtype=int)
        else:
            clustering = AgglomerativeClustering(
                metric="cosine", linkage="average",
                distance_threshold=0.6, n_clusters=None,
            )
            labels = clustering.fit_predict(X)
        label_map = dict(zip(ids_with, labels))
        max_label = max(labels) if len(labels) else 0
    else:
        label_map = {}
        max_label = -1

    # Assign short slots nearest slot speaker
    for sl in slots:
        if sl["id"] in label_map:
            sl["speaker"] = int(label_map[sl["id"]])
        else:
            # nearest by slot center
            c = (sl["start"] + sl["end"]) / 2
            best = None
            best_d = float("inf")
            for sl2 in slots:
                if sl2["id"] in label_map:
                    d = abs((sl2["start"] + sl2["end"]) / 2 - c)
                    if d < best_d:
                        best_d = d
                        best = sl2
            sl["speaker"] = int(label_map[best["id"]]) if best else 0     # best["speaker"] may not be assigned yet

    # Gender per speaker
    n_speakers = max_label + 1 if max_label >= 0 else (max((sl["speaker"] for sl in slots)) + 1 if slots else 0)
    gender_model = None
    if n_speakers > 0:                          # zero-speech clip -> no gender model load
        from model import ECAPA_gender
        gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
        gender_model.eval()

    speakers_info: Dict[int, Dict[str, Any]] = {}
    for spk in range(n_speakers):
        spk_slots = [sl for sl in slots if sl["speaker"] == spk]
        # evidence: each >=1s, up to 10 slots
        eval_slots = [sl for sl in spk_slots if (sl["end"] - sl["start"]) >= 1.0][:10]
        total_speech = sum((sl["end"] - sl["start"]) for sl in spk_slots)
        p_female_scores = []
        for sl in eval_slots:
            a = int(sl["start"] * sr)
            b = int(sl["end"] * sr)
            seg = y[a:b]
            import torch
            inp = torch.from_numpy(seg.reshape(1, -1))
            with torch.no_grad():
                logits = gender_model(inp)
                probs = torch.softmax(logits, dim=-1).squeeze()
                pf = float(probs[1])
            p_female_scores.append(pf)
        mean_pf = float(np.mean(p_female_scores)) if p_female_scores else 0.5
        n_slots_ge1 = len(eval_slots)

        gender = "unknown"
        if mean_pf >= 0.9 and total_speech >= 8 and n_slots_ge1 >= 2:
            gender = "female"
        elif mean_pf <= 0.1 and total_speech >= 8 and n_slots_ge1 >= 2:
            gender = "male"
        speakers_info[spk] = {
            "gender": gender,
            "mean_p_female": mean_pf,
            "total_speech": float(total_speech),
            "n_slots": len(spk_slots),
        }

    with open(sp_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in speakers_info.items()}, f, indent=2)

    # attach speaker + gender back to slots (we'll store per-slot in segments later)
    for sl in slots:
        spk = sl["speaker"]
        sl["gender"] = speakers_info.get(spk, {}).get("gender", "unknown")

    with open(out_dir / "slots.json", "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=2)
    log.info("Stage speakers: %d speakers", n_speakers)
    return speakers_info


# ---------------------------------------------------------------------------
# Stage 6: ASR
# ---------------------------------------------------------------------------
ASR_LANGS = {"en", "hi", "ta", "te", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur", "as", "ne", "sa"}


def _read_lid_record(path):
    """A published LID record, or None if absent / empty / torn / malformed."""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d if isinstance(d, dict) and "label" in d else None
    except Exception:
        return None


def _lid_compute(w16: Path, slots: List[Dict[str, Any]], out_dir: Path) -> Optional[Dict[str, Any]]:
    """The VoxLingua107 ECAPA LID on up to 60 s of VAD speech. Record {label, prob, seconds, code} or None (< 3 s speech)."""
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier as _LID
    y = load_wav_mono(w16, 16000)
    parts, total = [], 0.0
    for sl in slots:
        a, b = int(sl["start"] * 16000), int(sl["end"] * 16000)
        parts.append(y[a:b]); total += (b - a) / 16000
        if total >= 60:
            break
    if total < 3:
        return None
    lid = _LID.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(out_dir / ".lid"), run_opts={"device": "cpu"})
    out = lid.classify_batch(torch.from_numpy(np.concatenate(parts)).unsqueeze(0))
    prob = float(out[1].exp().item())
    label = out[3][0]
    code = label.split(":")[0].strip()
    log.info("Source LID: %s p=%.3f (%.1f s)", label, prob, total)
    return {"label": label, "prob": prob, "seconds": total, "code": code if (prob >= 0.7 and code in ASR_LANGS) else None}


def _lid_is_leader(out_dir) -> bool:
    """True when this chunk has the lowest cNNNN index present next to it (chunk 0 normally)."""
    import re
    try:
        m = re.fullmatch(r"c(\d+)", Path(out_dir).name)
        if not m:
            return True
        mine = int(m.group(1))
        idx = [int(x.group(1)) for x in (re.fullmatch(r"c(\d+)", d.name) for d in Path(out_dir).resolve().parent.iterdir()) if x]
        return mine <= min(idx + [mine])
    except Exception:
        return True


LID_LOCK = os.environ.get("LID_LOCK", "1").strip().lower() not in ("0", "false", "off", "no")


def _lid_first_writer(job_cache: str, compute, leader: bool = True, wait_s: float = 15.0) -> Optional[Dict[str, Any]]:
    """ONE language ID per job without delaying chunk 0. Only the leader chunk (lowest index present) computes and
    publishes (flock, tmp + fsync + os.replace, first writer wins). Every other chunk waits at most `wait_s` for the published
    record, then computes locally WITHOUT publishing. LID_LOCK=0 restores the old unlocked behaviour (compute + overwrite)."""
    rec = _read_lid_record(job_cache)
    if rec is not None:
        return rec
    if not LID_LOCK:
        rec = compute()
        if rec is not None:
            with open(job_cache, "w") as f:
                json.dump(rec, f)
        return rec
    lf = None
    t0 = time.time()
    deadline = t0 + wait_s          # ONE shared deadline for the lock attempt and the record wait
    if leader:
        lf = open(job_cache + ".lock", "a+")
        while True:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    lf.close(); lf = None
                    break
                time.sleep(0.2)
    if lf is None:
        while time.time() < deadline:
            rec = _read_lid_record(job_cache)
            if rec is not None:
                return rec
            time.sleep(0.25)
        rec = _read_lid_record(job_cache)
        return rec if rec is not None else compute()        # local, NOT published
    try:
        rec = _read_lid_record(job_cache)
        if rec is not None:
            return rec
        rec = compute()
        if rec is not None:
            tmp = f"{job_cache}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, job_cache)
        return rec
    finally:
        try:
            fcntl.flock(lf, fcntl.LOCK_UN)
        finally:
            lf.close()


def detect_source_language(w16: Path, slots: List[Dict[str, Any]], out_dir: Path) -> Optional[str]:
    """Video-level spoken-language ID (VoxLingua107 ECAPA) on up to 60 s of VAD speech; None unless confident."""
    # one LID per job: chunks of the same video share JOB_DIR/lid.json (next to the speaker map), first writer wins
    smap = os.environ.get("SPEAKER_MAP")
    job_cache = os.path.join(os.path.dirname(smap), "lid.json") if smap else None
    if job_cache:
        cached = _read_lid_record(job_cache) is not None
        rec = _lid_first_writer(job_cache, lambda: _lid_compute(w16, slots, out_dir), leader=_lid_is_leader(out_dir))
        if rec is None:
            return None
        if cached:
            log.info("Source LID (job cache): %s p=%.3f", rec.get("label"), rec.get("prob", 0))
        # the chunk copy mirrors the job record (so _source_lang agrees across chunks)
        atomic_write_json(out_dir / "lid.json", {"label": rec["label"], "prob": rec.get("prob", 0), "seconds": rec.get("seconds", 0)})
        return rec.get("code")
    rec = _lid_compute(w16, slots, out_dir)
    if rec is None:
        return None
    atomic_write_json(out_dir / "lid.json", {"label": rec["label"], "prob": rec["prob"], "seconds": rec["seconds"]})
    return rec["code"]


class SpeakersStage(StageBase[list[dict[str, Any]], dict[str, Any]]):
    """Attach speaker identity and voice assignments to slots."""

    name = "speakers"
    policy = StagePolicy(timing_key="speakers")

    def run(self, value: list[dict[str, Any]], ctx: ChunkContext) -> dict[str, Any]:
        """Attach speaker identity and voice assignments to slots."""
        return stage_speakers(ctx.work_dir, value)  # type: ignore[no-any-return]
