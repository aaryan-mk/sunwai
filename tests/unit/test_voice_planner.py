"""Voice-plan logic: the CER gate, VoicePlanner distinctness, and the plan_job / validate_plan round trip."""

import json
import math

import pytest

from sunwai.planning import voices
from sunwai.planning.voices.costs import VoiceCostModel
from sunwai.planning.voices.features import subtract_intervals

LANGS = ["hi", "ta"]


def unit_vec(angle):
    return [math.cos(angle), math.sin(angle)]


def make_bank(angles, genders, tau=0.9, cer=None):
    """Voice ``i`` points at ``angles[i]`` in both languages; ``V0`` is the native baseline."""
    names = [f"V{i}" for i in range(len(angles))]
    cer = cer or {}
    langs = {
        lang: {
            n: {"cer": cer.get(n, 5.0), "fails": 0, "emb": unit_vec(a), "logf0": 5.0}
            for n, a in zip(names, angles, strict=True)
        }
        for lang in LANGS
    }
    return {
        "voices": {n: {"gender": g} for n, g in zip(names, genders, strict=True)},
        "langs": langs,
        "native": {lang: ["V0"] for lang in LANGS},
        "tau": {"all": tau},
    }


def speakers(*genders):
    return [{"id": i, "gender": g, "p_female": None, "total_s": 30.0} for i, g in enumerate(genders)]


def feats_for(angles):
    return {i: {"emb": unit_vec(a), "logf0": 5.0, "evidence_s": 10.0, "clean": True} for i, a in enumerate(angles)}


# ---- gate


def test_cer_gate_limits_and_native_always_eligible():
    bank = make_bank([0, 1, 2], ["male"] * 3, cer={"V0": 12.0, "V1": 16.0, "V2": 17.5})
    gate = voices.cer_gate(bank)
    assert gate["hi"] == {"native_cer": 12.0, "limit": 17.0}  # native above 10: native + 5, no 15 cap
    assert voices.eligible(bank, gate, "hi", "V0")
    assert voices.eligible(bank, gate, "hi", "V1")
    assert not voices.eligible(bank, gate, "hi", "V2")
    bank["langs"]["hi"]["V1"]["fails"] = 1
    assert not voices.eligible(bank, gate, "hi", "V1")


# ---- distinctness


def test_assignment_prefers_voices_matching_each_speaker_and_keeps_them_distinct():
    bank = make_bank([0.0, 0.1, 1.5, 1.6], ["male"] * 4, tau=0.9)
    result = voices.VoicePlanner(bank, LANGS).assign(speakers("male", "male"), feats_for([0.0, 1.5]))
    chosen = [result.canonical["0"], result.canonical["1"]]
    assert chosen[0] in ("V0", "V1") and chosen[1] in ("V2", "V3")
    assert len(set(chosen)) == 2 and not result.degraded
    assert set(result.per_language) == set(LANGS)


def test_confident_gender_is_respected():
    bank = make_bank([0.0, 0.2, 1.5, 1.7], ["male", "male", "female", "female"])
    result = voices.VoicePlanner(bank, LANGS).assign(speakers("female", "male"), feats_for([0.0, 0.0]))
    assert bank["voices"][result.canonical["0"]]["gender"] == "female"
    assert bank["voices"][result.canonical["1"]]["gender"] == "male"


def test_indistinguishable_voices_are_flagged_degraded_but_never_repeated():
    bank = make_bank([0.0, 0.01, 0.02], ["male"] * 3, tau=0.9)  # every pair is far below the separation threshold
    result = voices.VoicePlanner(bank, LANGS).assign(speakers("male", "male"), feats_for([0.0, 0.0]))
    assert result.degraded
    assert any("INFEASIBLE" in w for w in result.warnings)
    assert len(set(result.canonical.values())) == 2


def test_capacity_failure_raises():
    bank = make_bank([0.0, 1.0], ["male", "male"])
    with pytest.raises(voices.PlanError, match="capacity"):
        voices.VoicePlanner(bank, LANGS).assign(speakers("male", "male", "male"), feats_for([0, 1, 2]))


def test_unmeasured_language_reuses_voices_gated_in_every_measured_language():
    bank = make_bank([0.0, 1.5, 3.0], ["male"] * 3)
    result = voices.VoicePlanner(bank, [*LANGS, "mni"]).assign(speakers("male"), feats_for([0.0]))
    assert result.per_language["mni"]["0"] in bank["voices"]


def test_cost_model_separation_is_symmetric_and_cached():
    bank = make_bank([0.0, 1.0], ["male", "male"])
    model = VoiceCostModel(bank, voices.cer_gate(bank), LANGS)
    assert model.separation("V0", "V1") == model.separation("V1", "V0") == round(1 - math.cos(1.0), 4)
    assert model.too_close(["V0", "V1"], min_separation=1.0) == [(0, 1)]


def test_subtract_intervals_removes_overlap():
    assert subtract_intervals((0.0, 10.0), [(2.0, 4.0), (8.0, 12.0)]) == [(0.0, 2.0), (4.0, 8.0)]
    assert subtract_intervals((0.0, 1.0), [(0.0, 1.0)]) == []


# ---- plan round trip


def write_catalogue(app, bank):
    app.mkdir()
    (app / "voice_bank.json").write_text(json.dumps(bank))
    (app / "languages.json").write_text(json.dumps({"targets": [{"code": lang} for lang in LANGS]}))


def test_plan_job_then_validate_plan_round_trip(tmp_path):
    bank = make_bank([0.0, 1.5, 3.0], ["male"] * 3)
    write_catalogue(tmp_path / "app", bank)
    app = tmp_path / "app"
    smap = {"speakers": speakers("male", "male"), "turns": [{"speaker": 0, "start": 0, "end": 3}]}
    plan = voices.plan_job(smap, LANGS, feats_for([0.0, 1.5]), app=app)
    assert plan["version"] == voices.PLAN_VERSION and plan["smap_hash"] == voices.smap_hash(smap)
    smap["voice_plan"] = plan
    assert voices.validate_plan(smap, "ta", app=app) is plan
    with pytest.raises(voices.PlanError, match="not in voice_plan.langs"):
        voices.validate_plan(smap, "bn", app=app)
    smap["speakers"] = speakers("male")
    with pytest.raises(voices.PlanError, match="smap_hash"):
        voices.validate_plan(smap, "ta", app=app)


def test_validate_plan_detects_catalogue_change_and_voice_collision(tmp_path):
    bank = make_bank([0.0, 1.5], ["male"] * 2)
    app = tmp_path / "app"
    write_catalogue(app, bank)
    smap = {"speakers": speakers("male", "male"), "turns": []}
    smap["voice_plan"] = voices.plan_job(smap, LANGS, feats_for([0.0, 1.5]), app=app)
    colliding = json.loads(json.dumps(smap))
    colliding["voice_plan"]["voices"]["hi"] = {"0": "V0", "1": "V0"}
    with pytest.raises(voices.PlanError, match="collision"):
        voices.validate_plan(colliding, "hi", app=app)
    (app / "voice_bank.json").write_text(json.dumps(make_bank([0.0, 1.5], ["male"] * 2, tau=0.5)))
    with pytest.raises(voices.PlanError, match="catalogue_rev"):
        voices.validate_plan(smap, "hi", app=app)


def test_plan_job_requires_a_bank(tmp_path):
    with pytest.raises(voices.PlanError, match="voice bank missing"):
        voices.plan_job({"speakers": []}, LANGS, {}, app=tmp_path)
