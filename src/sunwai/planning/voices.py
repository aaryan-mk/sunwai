"""voice assignment.
Gate ruling (lead): a language's NATIVE voices are always eligible, because CER is measured RELATIVE to them (they are the
baseline); every other voice must pass CER <= native + 5 (cap 15) and have no failed synthesis.

voice assignment. Planner computes a voice_plan ONCE per NEW job and embeds it in speakers_global.json; workers
only validate and read it. The assignment function (assign_voices) is pluggable: plan_job() takes it as a parameter."""
import os, json, hashlib, math
from pathlib import Path

from sunwai.paths import DATA_DIR  # noqa: E402

APP = DATA_DIR
PLAN_VERSION = 2
CER_DELTA = 5.0              # intelligibility gate: CER <= native + 5 points ...
CER_CEIL = 15.0              # ... and <= 15 % (raised to native + 5 if the native voice itself is above 10 %)
SC_COS, SC_F0, W_GENDER = 0.25, 0.20, 3.0      # fixed scales that standardise the cost terms (cosine dist, |d ln F0|, gender)
TOPK = 8


class PlanError(RuntimeError):
    pass


def env_on(name, default=True):
    v = os.environ.get(name, "").strip().lower()
    return default if v == "" else v not in ("0", "false", "no", "off", "n", "f")


def _bytes(p):
    try:
        return Path(p).read_bytes()
    except OSError:
        return b""


ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"


def catalogue_rev(app=APP):
    """hash of the voice catalogue (languages.json + voice bank), the ECAPA model id and THIS module's source (assignment +
    gate + optimiser implementation)."""
    h = hashlib.sha256()
    for n in ("languages.json", "voice_bank.json"):
        h.update(_bytes(Path(app) / n))
    h.update(ECAPA_ID.encode())
    h.update(_bytes(Path(__file__)))
    return h.hexdigest()[:16]


def smap_hash(g):
    core = {"speakers": g.get("speakers", []), "turns": g.get("turns", [])}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_bank(app=APP):
    return json.loads(_bytes(Path(app) / "voice_bank.json").decode("utf-8") or "{}")


# ------------------------------------------------------------------ acoustic features (planner / bank builder only)
_ECAPA = None


def ecapa(savedir):
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _ECAPA = EncoderClassifier.from_hparams(source=ECAPA_ID, savedir=str(savedir), run_opts={"device": "cpu"})
    return _ECAPA


def embed(y16, savedir):
    import torch
    import numpy as np
    with torch.no_grad():
        e = ecapa(savedir).encode_batch(torch.from_numpy(np.ascontiguousarray(y16, dtype="float32"))[None]).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def median_logf0(y16):
    import numpy as np
    import librosa
    f0, vf, _ = librosa.pyin(np.asarray(y16, dtype="float32"), fmin=60, fmax=450, sr=16000, frame_length=1024)
    f0 = f0[vf & ~np.isnan(f0)]
    return float(np.median(np.log(f0))) if len(f0) >= 20 else None


def speaker_features(y, turns, savedir, max_total=40.0, max_piece=8.0):
    """per speaker: ECAPA centroid + median ln F0 from CLEAN (non-overlapping) speech; falls back to any >=1 s turn."""
    import numpy as np
    spk = sorted({t["speaker"] for t in turns})
    out = {}
    for s in spk:
        others = sorted((t["start"], t["end"]) for t in turns if t["speaker"] != s)
        pieces = []
        for t in turns:
            if t["speaker"] != s:
                continue
            segs = [(t["start"], t["end"])]
            for a, b in others:
                nxt = []
                for x, z in segs:
                    if b <= x or a >= z:
                        nxt.append((x, z))
                    else:
                        if a > x: nxt.append((x, a))
                        if b < z: nxt.append((b, z))
                segs = nxt
            pieces += [(x, z) for x, z in segs if z - x >= 1.0]
        clean = bool(pieces)
        if not pieces:
            pieces = [(t["start"], t["end"]) for t in turns if t["speaker"] == s and t["end"] - t["start"] >= 1.0]
        pieces.sort(key=lambda p: (-(p[1] - p[0]), p[0]))
        embs, wts, tot, chunks = [], [], 0.0, []
        for x, z in pieces:
            if tot >= max_total: break
            z = min(z, x + max_piece)
            seg = y[int(x * 16000):int(z * 16000)]
            embs.append(embed(seg, savedir)); wts.append(z - x); tot += z - x; chunks.append(seg)
        if not embs:
            out[s] = None; continue
        c = np.average(np.stack(embs), axis=0, weights=wts); c = c / (np.linalg.norm(c) + 1e-9)
        f0 = median_logf0(np.concatenate(chunks)[:16000 * 20])
        out[s] = {"emb": c, "logf0": f0, "evidence_s": round(tot, 2), "clean": clean}
    return out


# ------------------------------------------------------------------ assignment (pluggable)
def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def cer_gate(bank):
    """per language: native CER (mean of that language's native voices) and the ceiling actually applied."""
    g = {}
    for lang, vs in bank.get("langs", {}).items():
        nat = [vs[v]["cer"] for v in bank.get("native", {}).get(lang, []) if v in vs]
        n = sum(nat) / len(nat) if nat else 0.0
        g[lang] = {"native_cer": round(n, 2), "limit": round(min(n + CER_DELTA, CER_CEIL if n <= 10 else n + CER_DELTA), 2)}
    return g


def eligible(bank, gate, lang, voice):
    """hard CER gate. The language's native voices are the baseline and always eligible."""
    if voice in bank.get("native", {}).get(lang, []):
        return True
    e = bank["langs"].get(lang, {}).get(voice)
    return bool(e) and e["cer"] <= gate[lang]["limit"] and e.get("fails", 0) == 0


NODE_LIMIT, TIME_LIMIT = 20000, 2.0


def ok_all_fn(bank, gate):
    return lambda v: all(eligible(bank, gate, m, v) for m in bank["langs"])


def _native_for(lang, gender, bank, cfg_pools):
    return (cfg_pools.get(lang, {}).get(gender) or [None])[0]


def ids_of(speakers):
    return [int(s["id"]) for s in speakers]


def assign_voices(speakers, feats, langs, bank, opt_langs=None):
    """speakers: [{id,gender,p_female,total_s}] ; feats: {id:{emb,logf0,evidence_s}|None}; langs: all languages to cover.
    Returns (canonical{id:voice}, per_lang{lang:{id:voice}}, degraded, warnings, detail). Raises PlanError on a capacity
    failure. HARD guarantees: every (voice, measured language) pair in the result passes the CER gate; no voice is used
    twice within a measured language."""
    import time
    warns = []
    vinfo = bank["voices"]
    gate = cer_gate(bank)
    Lm = [l for l in (opt_langs or langs) if l in bank["langs"]] or ["hi"]
    measured = [l for l in langs if l in bank["langs"]]
    unmeasured = [l for l in langs if l not in bank["langs"]]
    tau = bank["tau"]["all"]
    min_sep_ok = round(1 - tau, 4)
    names = sorted(v for v in vinfo if all(v in bank["langs"][l] for l in Lm))      # only voices with bank entries in every optimisation language
    if not ids_of(speakers):
        return {}, {l: {} for l in langs}, False, [], {}
    def vemb(l, v): return bank["langs"][l][v]["emb"]
    def vlf0(v):
        xs = [bank["langs"][l][v]["logf0"] for l in Lm if bank["langs"][l].get(v, {}).get("logf0") is not None]
        return sum(xs) / len(xs) if xs else None
    elig_all = {v for v in names if all(eligible(bank, gate, l, v) for l in Lm)}
    spk = sorted(speakers, key=lambda s: int(s["id"]))
    ids = [int(s["id"]) for s in spk]
    sgen = {int(s["id"]): s.get("gender") for s in spk}
    # ---- capacity (unique eligible voices): confident speakers cannot cross gender, unknown ones may spill
    requested = [l for l in (opt_langs or langs)]
    for l in [x for x in measured if x in requested]:
        el = {g: [v for v in names if vinfo[v]["gender"] == g and eligible(bank, gate, l, v)] for g in ("male", "female")}
        nm_, nf_ = sum(1 for i in ids if sgen[i] == "male"), sum(1 for i in ids if sgen[i] == "female")
        if nm_ > len(el["male"]) or nf_ > len(el["female"]) or len(ids) > len(el["male"]) + len(el["female"]):
            raise PlanError(f"capacity: {nm_} male + {nf_} female confident of {len(ids)} speakers but only "
                            f"{len(el['male'])} male / {len(el['female'])} female eligible voices in {l}")
    def cost_sv(s, f, v):
        conf = s.get("gender") in ("male", "female")
        pf = 0.5 if s.get("p_female") is None else float(s["p_female"])
        w = 0.5 + 0.5 * min(1.0, float(s.get("total_s") or 0) / 20.0)
        c = 0.0
        if f is not None:
            c += sum(1 - _cos(f["emb"], vemb(l, v)) for l in Lm) / len(Lm) / SC_COS
            vf = vlf0(v)
            if f.get("logf0") is not None and vf is not None:
                c += abs(f["logf0"] - vf) / SC_F0
        if not conf:
            c += W_GENDER * w * (pf if vinfo[v]["gender"] == "male" else 1 - pf)
        c += sum(max(0.0, bank["langs"][l][v]["cer"] - gate[l]["native_cer"]) / CER_DELTA for l in Lm) / len(Lm)
        return round(c, 4)
    cands = {}
    for s in spk:
        sid = int(s["id"]); f = feats.get(sid) or feats.get(str(sid))
        conf = s.get("gender") in ("male", "female")
        pf = 0.5 if s.get("p_female") is None else float(s["p_female"])
        w = 0.5 + 0.5 * min(1.0, float(s.get("total_s") or 0) / 20.0)
        pool = [v for v in names if (not conf or vinfo[v]["gender"] == s["gender"])]
        pe = [v for v in pool if v in elig_all]
        if len(pe) < len(ids):                                        # not enough all-language-eligible: widen to per-language gating
            pe = [v for v in pool if any(eligible(bank, gate, l, v) for l in Lm)] or pool       # pool is already limited to voices with entries in every Lm language
        rows = [(cost_sv(s, f, v), v) for v in pe]
        rows.sort()
        cands[sid] = rows[:max(TOPK, len(ids) + 4)]
        if f is None: warns.append(f"speaker {sid}: no clean speech features; cost uses gender/quality only")
    sepcache = {}
    def sep(a, b, langs_=None):
        k = (a, b, tuple(langs_ or Lm)) if a < b else (b, a, tuple(langs_ or Lm))
        if k not in sepcache:
            sepcache[k] = round(min(1 - _cos(vemb(l, a), vemb(l, b)) for l in (langs_ or Lm)), 4)
        return sepcache[k]
    # ---- bounded exact search (node + time limit), then deterministic max-separation greedy
    best = [None, None]; nodes = [0]; t0 = time.monotonic(); hit = [False]
    def dfs(i, chosen, cost):
        if hit[0]: return
        nodes[0] += 1
        if nodes[0] > NODE_LIMIT or (nodes[0] % 512 == 0 and time.monotonic() - t0 > TIME_LIMIT):
            hit[0] = True; return
        if best[0] is not None and cost >= best[0] - 1e-9: return
        if i == len(ids):
            best[0], best[1] = cost, list(chosen); return
        for c, v in cands[ids[i]]:
            if v in chosen: continue
            if any(sep(v, u) < min_sep_ok for u in chosen): continue
            chosen.append(v); dfs(i + 1, chosen, cost + c); chosen.pop()
    dfs(0, [], 0.0)
    degraded = False
    chosen = best[1]
    if chosen is None or hit[0]:
        if hit[0]: warns.append(f"optimizer bound hit (nodes>{NODE_LIMIT} or >{TIME_LIMIT}s): deterministic greedy assignment used")
        if chosen is None and not hit[0]:
            warns.append(f"INFEASIBLE under tau={tau:.3f}: taking the maximum-separation assignment (distinctness_degraded)")
        # min-cost bipartite matching over ALL gated voices (feasibility-preserving); separation preference = iterative
        # penalties on the later speaker of every pair closer than tau, re-solved deterministically.
        from scipy.optimize import linear_sum_assignment
        BIG = 10 ** 9
        pool_v = [x for x in names if any(eligible(bank, gate, l, x) for l in Lm)]
        if len(pool_v) < len(ids):
            raise PlanError(f"capacity: only {len(pool_v)} gated voices for {len(ids)} speakers")
        base = {}
        for sid in ids:
            s = next(x for x in spk if int(x["id"]) == sid); f = feats.get(sid) or feats.get(str(sid)); conf = s.get("gender") in ("male", "female")
            for x in pool_v:
                base[(sid, x)] = BIG if (conf and vinfo[x]["gender"] != s["gender"]) else int(round(cost_sv(s, f, x) * 10000))
        pen = {}; bestm = None
        for _ in range(8):
            M = [[base[(sid, x)] + pen.get((sid, x), 0) for x in pool_v] for sid in ids]
            r, c = linear_sum_assignment(M)
            if any(M[i][j] >= BIG for i, j in zip(r, c)):
                raise PlanError(f"capacity: no full matching of {len(ids)} speakers to distinct gated voices")
            sol = [pool_v[j] for _, j in sorted(zip(r, c))]
            bad = [(i, j) for i in range(len(sol)) for j in range(i + 1, len(sol)) if sep(sol[i], sol[j]) < min_sep_ok]
            key = (len(bad), sum(base[(ids[i], sol[i])] for i in range(len(sol))))
            if bestm is None or key < bestm[0]: bestm = (key, sol)
            if not bad: break
            for i, j in bad: pen[(ids[j], sol[j])] = pen.get((ids[j], sol[j]), 0) + 30000
        chosen = bestm[1]
    if len(set(chosen)) < len(chosen):
        raise PlanError("capacity: optimizer produced a repeated voice")
    if any(sep(chosen[i], chosen[j]) < min_sep_ok for i in range(len(chosen)) for j in range(i + 1, len(chosen))):
        degraded = True
        if not any("INFEASIBLE" in w for w in warns): warns.append(f"distinctness below tau={tau:.3f} for some pair (distinctness_degraded)")
    canonical = {str(i): v for i, v in zip(ids, chosen)}
    # ---- per language: deterministic min-cost bipartite matching of speakers to gated voices (canonical preferred)
    from scipy.optimize import linear_sum_assignment
    BIG = 10 ** 9
    unavailable = {}
    def match(l, ok, refl):
        voices = [v for v in names if ok(v) and v in bank["langs"].get(refl, {})]
        if len(voices) < len(ids): return None
        M = []
        for sid in ids:
            s = next(x for x in spk if int(x["id"]) == sid); conf = s.get("gender") in ("male", "female")
            pf = 0.5 if s.get("p_female") is None else float(s["p_female"]); c0 = canonical[str(sid)]
            row = []
            for v in voices:
                if conf and vinfo[v]["gender"] != s["gender"]: row.append(BIG); continue
                c = 0.0 if v == c0 else 1.0 + (1 - _cos(vemb(refl, c0), vemb(refl, v)))
                if not conf: c += W_GENDER * (pf if vinfo[v]["gender"] == "male" else 1 - pf)
                row.append(int(round(c * 10000)))
            M.append(row)
        r, c = linear_sum_assignment(M)                                  # deterministic: integer costs, lexical column order
        if any(M[i][j] >= BIG for i, j in zip(r, c)): return None       # no full matching exists
        return {str(ids[i]): voices[j] for i, j in zip(r, c)}
    per = {}
    for l in langs:
        meas = l in bank["langs"]
        ok = (lambda v, l=l: eligible(bank, gate, l, v)) if meas else ok_all_fn(bank, gate)
        m = match(l, ok, l if meas else Lm[0])
        if m is None:
            msg = f"capacity: no full matching of {len(ids)} speakers to distinct gated voices in {l}"
            if l in requested: raise PlanError(msg)
            unavailable[l] = msg; warns.append(f"{l}: unavailable ({msg})"); continue
        per[l] = m
        for sid in ids:
            if m[str(sid)] != canonical[str(sid)]:
                warns.append(f"speaker {sid} {l}: canonical {canonical[str(sid)]} -> {m[str(sid)]} (gate/matching)")
        vs = [m[str(i)] for i in ids]
        if meas and any(sep(vs[i], vs[j], [l]) < min_sep_ok for i in range(len(vs)) for j in range(i + 1, len(vs))):
            degraded = True; warns.append(f"{l}: distinctness below tau (distinctness_degraded)")
    detail = {}
    for s in spk:
        sid = int(s["id"]); f = feats.get(sid)
        detail[str(sid)] = {"gender": s.get("gender"), "p_female": s.get("p_female"), "total_s": s.get("total_s"),
                            "logf0": None if not f else f.get("logf0"), "evidence_s": None if not f else f.get("evidence_s"),
                            "emb": None if not f else [round(float(x), 4) for x in f["emb"]],
                            "sim": {l: (round(float(_cos(f["emb"], vemb(l, per[l][str(sid)]))), 4) if (f and l in per and l in bank["langs"] and per[l][str(sid)] in bank["langs"][l]) else None) for l in langs}}
    assign_voices.unavailable = unavailable
    return canonical, per, degraded, warns, detail


def plan_job(g, langs, feats, bank=None, assign=assign_voices, app=APP, opt_langs=None):
    """Build the immutable voice_plan for speakers_global.json content g (without voice_plan)."""
    bank = bank or load_bank(app)
    if not bank.get("voices"):
        raise PlanError("voice bank missing")
    spk = [{"id": x["id"], "gender": x.get("gender"), "p_female": x.get("p_female"), "total_s": x.get("total_s")} for x in g.get("speakers", [])]
    canonical, per, degraded, warns, detail = assign(spk, feats, list(langs), bank, opt_langs=opt_langs)
    return {"version": PLAN_VERSION, "catalogue_rev": catalogue_rev(app), "smap_hash": smap_hash(g), "langs": list(langs),
            "canonical": canonical, "voices": per, "distinctness_degraded": bool(degraded), "tau": bank["tau"],
            "unmeasured_langs": [l for l in langs if l not in bank["langs"]],
            "unavailable_langs": dict(getattr(assign, "unavailable", {})),
            "cer_gate": cer_gate(bank), "warnings": warns, "speakers": detail}


def validate_plan(g, lang, app=APP):
    """Chunk preflight (worker side, read-only). Raises PlanError for ANY problem of a flagged job; returns the plan."""
    p = g.get("voice_plan")
    if not isinstance(p, dict): raise PlanError("voice_plan is not an object")
    if p.get("version") != PLAN_VERSION: raise PlanError(f"unsupported voice_plan version {p.get('version')!r}")
    if p.get("smap_hash") != smap_hash(g): raise PlanError("voice_plan smap_hash does not match the speaker map")
    rev = catalogue_rev(app)
    if p.get("catalogue_rev") != rev: raise PlanError(f"voice_plan catalogue_rev {p.get('catalogue_rev')} != {rev}")
    if lang not in (p.get("langs") or []): raise PlanError(f"language {lang} not in voice_plan.langs")
    if lang in (p.get("unavailable_langs") or {}): raise PlanError(f"language {lang} is unavailable for this job: {p['unavailable_langs'][lang]}")
    vm = (p.get("voices") or {}).get(lang)
    if not isinstance(vm, dict): raise PlanError(f"voice_plan has no mapping for {lang}")
    bank = load_bank(app); known = set(bank.get("voices", {})); gate = cer_gate(bank)
    um = set(p.get("unmeasured_langs") or [])
    if (lang in bank.get("langs", {})) == (lang in um): raise PlanError(f"{lang}: measured/unmeasured flag inconsistent with the bank")
    cfgp = {t["code"]: t.get("voices", {}) for t in json.loads(_bytes(Path(app) / "languages.json").decode()).get("targets", [])}
    used = []
    for x in g.get("speakers", []):
        v = vm.get(str(x["id"]))
        if not isinstance(v, str) or v not in known: raise PlanError(f"speaker {x['id']} has no valid voice for {lang}: {v!r}")
        if lang in um:
            if not all(eligible(bank, gate, m, v) for m in bank["langs"]):
                raise PlanError(f"unmeasured language {lang}: {v} fails the CER gate in a measured language")
        elif not eligible(bank, gate, lang, v):
            raise PlanError(f"voice {v} fails the CER gate in {lang}")
        used.append(v)
    if len(set(used)) != len(used): raise PlanError(f"voice collision in {lang}: {used}")
    return p
