import json

import pytest

from src.assistant.chat import AgriAssistant
from src.assistant.safety import _DOSE
from src.database.db import Database
from src.livestock.schema import LivestockCase
from src.ml.inference import PlantDiseasePredictor, PredictionResult, Candidate
from src.services.assessment_service import AssessmentService
from src.services.assistant_service import ask_about_assessment, assessment_context
from src.services.knowledge import condition_guide
from src.services.recommendation import SECTIONS, livestock_recommendations, plant_recommendations
from src.utils.image_quality import InvalidImageError
from tests.helpers import blurry_image, leaf_image, make_tiny_model_dir, to_bytes


def ok_result(key):
    return PredictionResult(status="ok", predicted_key=key, predicted_name=key, confidence=0.95,
                            top_k=[Candidate(key, key, 0.95)])


def test_every_condition_has_all_sections_and_no_doses():
    for key in condition_guide()["conditions"]:
        recs = plant_recommendations(ok_result(key))
        assert set(recs) == set(SECTIONS)
        assert recs["referral"] and recs["management"]
        assert not _DOSE.search(json.dumps(recs)), key


def test_moderate_confidence_adds_caution():
    r = ok_result("tomato_early_blight")
    r.confidence = 0.7
    assert "not a fact" in plant_recommendations(r)["management"][0]


def test_uncertain_result_gives_no_chemical_advice_and_refers():
    r = PredictionResult(status="uncertain", top_k=[Candidate("a", "A", 0.4), Candidate("b", "B", 0.3)], reasons=["x"])
    recs = plant_recommendations(r)
    assert any("Do not apply chemicals" in m for m in recs["management"]) and recs["referral"]


def test_livestock_recommendations_structure():
    from src.livestock.triage import assess
    recs = livestock_recommendations(assess(LivestockCase(symptoms=["mouth_sores", "drooling"], temperature_c=40.3)))
    assert set(recs) == set(SECTIONS) and any("notifiable" in r for r in recs["referral"])


@pytest.fixture
def svc(tmp_path, settings):
    pred = PlantDiseasePredictor(settings, model_dir=make_tiny_model_dir(tmp_path / "m"))
    return AssessmentService(db=Database(":memory:"), predictor=pred, settings=settings)


def test_plant_analysis_saved_and_retrievable(svc):
    out = svc.analyze_plant(to_bytes(leaf_image()), "Tomato", notes="spots")
    assert out.assessment_id and out.result.status == "ok"
    a = svc.db.get_assessment(out.assessment_id)
    assert a["image_path"] and (svc.settings.uploads_dir / a["image_path"]).is_file()
    assert a["notes"] == "spots" and len(a["predictions"]) == 3 and a["recommendations"]["management"]


def test_invalid_upload_and_unsupported_crop(svc):
    with pytest.raises(InvalidImageError):
        svc.analyze_plant(b"junk")
    with pytest.raises(ValueError, match="not supported"):
        svc.analyze_plant(to_bytes(leaf_image()), "Maize")
    assert svc.db.count_assessments() == 0


def test_rejected_photo_not_saved(svc):
    out = svc.analyze_plant(to_bytes(blurry_image()))
    assert out.result.status == "rejected_quality" and out.assessment_id is None and svc.db.count_assessments() == 0


def test_model_unavailable_not_saved(tmp_path, settings):
    s = AssessmentService(db=Database(":memory:"), predictor=PlantDiseasePredictor(settings, model_dir=tmp_path / "none"), settings=settings)
    out = s.analyze_plant(to_bytes(leaf_image()))
    assert out.result.status == "model_unavailable" and out.assessment_id is None and s.db.count_assessments() == 0


def test_livestock_flow_saves_history(svc):
    case = LivestockCase(animal_id="C1", symptoms=["coughing"], temperature_c=40.0, breathing="Laboured breathing")
    aid, a, recs = svc.assess_livestock(case, to_bytes(leaf_image()))
    assert a.urgency == "urgent" and svc.db.get_assessment(aid)["symptoms"][0]["symptom_key"] == "coughing"
    with pytest.raises(ValueError):
        svc.assess_livestock(LivestockCase(temperature_c=80))


def test_followup_about_assessment_uses_context_and_is_stored(svc, tmp_path, settings):
    from src.assistant.ingest import index_knowledge_base
    from src.assistant.retriever import KnowledgeRetriever
    import dataclasses
    s2 = dataclasses.replace(settings, chroma_dir=tmp_path / "c")
    index_knowledge_base(s2)
    assistant = AgriAssistant(KnowledgeRetriever(s2, auto_index=False))
    aid = svc.db.save_plant_assessment(
        crop="Tomato", image_path=None, status="ok", summary="Early blight", confidence=0.9, model_version="t",
        result={"status": "ok", "reasons": []},
        predictions=[{"key": "tomato_early_blight", "name": "Early blight", "probability": 0.9}], recommendations={})
    ctx = assessment_context(svc.db.get_assessment(aid))
    assert "not a confirmed diagnosis" in ctx["text"] and ctx["species"] == "tomato"
    reply = ask_about_assessment(assistant, svc.db, aid, "How can I prevent it?")
    assert reply.sources and any("blight" in (s["section"] or "").lower() or "tomato" in s["title"].lower() for s in reply.sources)
    assert svc.db.get_assessment(aid)["follow_ups"][0]["question"] == "How can I prevent it?"
    with pytest.raises(KeyError):
        ask_about_assessment(assistant, svc.db, 999, "x")
