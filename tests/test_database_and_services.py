import json
import re

import pytest

from src.database.db import Database, MIGRATIONS
from src.livestock.schema import LivestockCase
from src.ml.inference import PlantDiseasePredictor
from src.services.assessment_service import AssessmentService
from src.services.assistant_service import assessment_context, ask_about_assessment
from src.services.knowledge import condition_guide
from src.services.recommendation import livestock_recommendations, plant_recommendations
from src.utils.image_quality import InvalidImageError
from src.utils.storage import resolve_image, save_image
from tests.helpers import blurry_image, leaf_image, make_tiny_model_dir, to_bytes

DOSE_LIKE = re.compile(r"\b\d+(\.\d+)?\s*(mg|ml|g|kg|l|litres?)\b", re.I)


@pytest.fixture
def svc(tmp_path, settings):
    pred = PlantDiseasePredictor(settings, model_dir=make_tiny_model_dir(tmp_path / "m"))
    return AssessmentService(db=Database(":memory:"), predictor=pred, settings=settings)


def test_migrations_apply_once_and_are_versioned(tmp_path):
    db = Database(tmp_path / "x.db")
    assert db.schema_version() == MIGRATIONS[-1][0]
    Database(tmp_path / "x.db")  # idempotent reopen
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) n FROM schema_migrations").fetchone()["n"] == len(MIGRATIONS)
        tables = {r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"assessments", "crops", "animals", "assessment_symptoms", "predictions", "recommendations", "follow_ups"} <= tables


def test_sql_injection_is_inert():
    db = Database(":memory:")
    evil = "x'); DROP TABLE assessments;--"
    aid = db.save_plant_assessment(crop=evil, image_path=None, status="ok", summary=evil, confidence=0.9, model_version="v",
                                   result={}, predictions=[{"key": "k", "name": evil, "probability": 0.9}],
                                   recommendations={"symptoms": [evil]}, notes=evil)
    got = db.get_assessment(aid)
    assert got["summary"] == evil and got["crop"] == evil and db.count_assessments() == 1


def test_cascade_delete_and_follow_ups():
    db = Database(":memory:")
    aid = db.save_plant_assessment(crop="Tomato", image_path=None, status="ok", summary="s", confidence=0.5, model_version="v", result={},
                                   predictions=[{"key": "a", "name": "A", "probability": 1.0}], recommendations={"management": ["m"]})
    db.add_follow_up(aid, "q?", "a.", [{"title": "t"}])
    assert db.get_assessment(aid)["follow_ups"][0]["sources"][0]["title"] == "t"
    assert db.delete_assessment(aid) and db.get_assessment(aid) is None
    with db.connect() as c:
        for t in ("predictions", "recommendations", "follow_ups"):
            assert c.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"] == 0
    assert not db.delete_assessment(999)


def test_images_stored_on_disk_not_in_db(tmp_path):
    rel = save_image(leaf_image(), tmp_path / "up", prefix="plant")
    assert (tmp_path / "up" / rel).is_file() and rel.endswith(".jpg")
    assert resolve_image(tmp_path / "up", rel) is not None
    assert resolve_image(tmp_path / "up", "../../etc/passwd") is None
    assert resolve_image(tmp_path / "up", None) is None


def test_plant_flow_saves_history(svc):
    out = svc.analyze_plant(to_bytes(leaf_image()), "Tomato", notes="spots after rain")
    assert out.result.status == "ok" and out.assessment_id
    a = svc.db.get_assessment(out.assessment_id)
    assert a["assessment_type"] == "plant" and a["crop"] == "Tomato" and a["notes"] == "spots after rain"
    assert len(a["predictions"]) == 3 and a["recommendations"]["management"]
    assert svc.db.list_assessments()[0]["id"] == out.assessment_id
    assert (svc.settings.uploads_dir / a["image_path"]).is_file()


def test_rejected_and_invalid_uploads_not_saved(svc):
    out = svc.analyze_plant(to_bytes(blurry_image()))
    assert out.result.status == "rejected_quality" and out.assessment_id is None and svc.db.count_assessments() == 0
    with pytest.raises(InvalidImageError):
        svc.analyze_plant(b"garbage")
    with pytest.raises(ValueError, match="not supported"):
        svc.analyze_plant(to_bytes(leaf_image()), crop="Maize")


def test_model_unavailable_not_saved(tmp_path, settings):
    s = AssessmentService(db=Database(":memory:"), predictor=PlantDiseasePredictor(settings, model_dir=tmp_path / "none"), settings=settings)
    out = s.analyze_plant(to_bytes(leaf_image()))
    assert out.result.status == "model_unavailable" and out.assessment_id is None and s.db.count_assessments() == 0


def test_livestock_flow_and_history(svc):
    case = LivestockCase(animal_id="Daisy", breed="Friesian", temperature_c=40.2, symptoms=["mouth_sores", "drooling"])
    aid, a, recs = svc.assess_livestock(case, to_bytes(leaf_image()))
    got = svc.db.get_assessment(aid)
    assert got["assessment_type"] == "livestock" and got["animal"]["animal_tag"] == "Daisy"
    assert {s["symptom_key"] for s in got["symptoms"]} == {"mouth_sores", "drooling"}
    assert got["status"] == "emergency" and got["recommendations"]["referral"] and got["image_path"]
    with pytest.raises(ValueError):
        svc.assess_livestock(LivestockCase(temperature_c=80))


def test_assessment_specific_followup_roundtrip(svc, retriever_settings):
    from src.assistant.chat import AgriAssistant
    from src.assistant.retriever import KnowledgeRetriever
    out = svc.analyze_plant(to_bytes(leaf_image()))
    assistant = AgriAssistant(KnowledgeRetriever(retriever_settings))
    ctx = assessment_context(svc.db.get_assessment(out.assessment_id))
    assert ctx["species"] == "tomato" and "not a confirmed diagnosis" in ctx["text"]
    reply = ask_about_assessment(assistant, svc.db, out.assessment_id, "What does this result mean?")
    assert reply.answer and len(svc.db.get_assessment(out.assessment_id)["follow_ups"]) == 1
    with pytest.raises(KeyError):
        ask_about_assessment(assistant, svc.db, 9999, "x")


@pytest.fixture
def retriever_settings(tmp_path, settings):
    import dataclasses
    return dataclasses.replace(settings, chroma_dir=tmp_path / "c2")


def test_recommendations_for_every_class_have_no_doses_and_correct_sections():
    from src.ml.inference import Candidate, PredictionResult
    for key, cond in condition_guide()["conditions"].items():
        r = PredictionResult(status="ok", predicted_key=key, predicted_name=cond["name"], confidence=0.95,
                             top_k=[Candidate(key, cond["name"], 0.95)])
        recs = plant_recommendations(r)
        assert set(recs) == {"symptoms", "possible_causes", "prevention", "management", "referral"}
        assert recs["referral"] and recs["symptoms"]
        assert not DOSE_LIKE.search(json.dumps(recs)), key


def test_uncertain_recommendations_discourage_chemicals_and_refer():
    from src.ml.inference import Candidate, PredictionResult
    r = PredictionResult(status="uncertain", top_k=[Candidate("a", "A", 0.4)], reasons=["x"])
    recs = plant_recommendations(r)
    assert any("Do not apply chemicals" in m for m in recs["management"]) and any("extension officer" in m for m in recs["referral"])


def test_livestock_recommendations_structure():
    from src.livestock.triage import assess
    recs = livestock_recommendations(assess(LivestockCase(symptoms=["bloated_left_side"], breathing="Laboured breathing")))
    assert recs["referral"] and recs["management"] and recs["possible_causes"]
    assert not DOSE_LIKE.search(json.dumps(recs))
