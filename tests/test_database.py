import sqlite3

import pytest

from src.database.db import Database, MIGRATIONS
from src.utils.storage import resolve_image, save_image
from tests.helpers import leaf_image

RECS = {"symptoms": ["a"], "possible_causes": ["b"], "prevention": ["c"], "management": ["d", "e"], "referral": ["f"]}


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.db")


def save_plant(db, summary="Early blight", notes="", n=1):
    return db.save_plant_assessment(
        crop="Tomato", image_path="x.jpg", status="ok", summary=summary, confidence=0.93, model_version="v1",
        result={"status": "ok"}, predictions=[{"key": "k1", "name": "Early blight", "probability": 0.93},
                                              {"key": "k2", "name": "Target spot", "probability": 0.05}],
        recommendations=RECS, notes=notes)


def test_migrations_idempotent(tmp_path):
    p = tmp_path / "a.db"
    assert Database(p).schema_version() == MIGRATIONS[-1][0]
    assert Database(p).schema_version() == MIGRATIONS[-1][0]


def test_save_and_get_plant_assessment(db):
    aid = save_plant(db, notes="spots after rain")
    a = db.get_assessment(aid)
    assert a["crop"] == "Tomato" and a["confidence"] == 0.93 and a["notes"] == "spots after rain"
    assert [p["rank"] for p in a["predictions"]] == [1, 2]
    assert a["recommendations"]["management"] == ["d", "e"]
    assert db.get_assessment(9999) is None


def test_livestock_assessment_and_symptoms(db):
    aid = db.save_livestock_assessment(case={"species": "cattle", "animal_id": "T-1", "age_range": "Adult (2-8 years)", "breed": "Friesian"},
                                       image_path=None, status="urgent", summary="Contact a vet today: fever",
                                       result={"rules_version": "0.1"}, symptoms={"coughing": "Coughing"}, recommendations=RECS)
    a = db.get_assessment(aid)
    assert a["animal"]["breed"] == "Friesian" and a["symptoms"][0]["symptom_key"] == "coughing"


def test_history_listing_order_filter_and_counts(db):
    ids = [save_plant(db, f"r{i}") for i in range(3)]
    db.save_livestock_assessment(case={}, image_path=None, status="soon", summary="L", result={}, symptoms={}, recommendations={})
    rows = db.list_assessments()
    assert rows[0]["assessment_type"] == "livestock" and rows[1]["id"] == ids[2]
    assert len(db.list_assessments(assessment_type="plant")) == 3
    assert db.count_assessments() == 4 and db.count_assessments("plant") == 3
    assert len(db.list_assessments(limit=2, offset=1)) == 2


def test_follow_ups_and_cascade_delete(db):
    aid = save_plant(db)
    db.add_follow_up(aid, "q1", "a1", [{"title": "t"}])
    db.add_follow_up(aid, "q2", "a2")
    a = db.get_assessment(aid)
    assert [f["question"] for f in a["follow_ups"]] == ["q1", "q2"] and a["follow_ups"][0]["sources"][0]["title"] == "t"
    assert db.delete_assessment(aid) and db.get_assessment(aid) is None
    with db.connect() as c:
        for t in ("predictions", "recommendations", "follow_ups"):
            assert c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0


def test_sql_injection_is_inert(db):
    evil = "x'); DROP TABLE assessments; --"
    aid = save_plant(db, summary=evil, notes=evil)
    assert db.get_assessment(aid)["summary"] == evil
    assert db.count_assessments() == 1
    assert db.list_assessments(assessment_type="plant' OR '1'='1") == []


def test_check_constraint_rejects_bad_type(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as c:
            c.execute("INSERT INTO assessments(created_at, assessment_type, status, summary, result_json) VALUES ('t','bad','s','s','{}')")


def test_images_stored_on_disk_not_in_db(tmp_path):
    rel = save_image(leaf_image(), tmp_path / "up", "plant")
    assert (tmp_path / "up" / rel).is_file() and rel.endswith(".jpg")
    assert resolve_image(tmp_path / "up", rel) is not None


def test_path_traversal_rejected(tmp_path):
    (tmp_path / "secret.txt").write_text("s")
    (tmp_path / "up").mkdir()
    assert resolve_image(tmp_path / "up", "../secret.txt") is None
    assert resolve_image(tmp_path / "up", "/etc/passwd") is None
    assert resolve_image(tmp_path / "up", None) is None
