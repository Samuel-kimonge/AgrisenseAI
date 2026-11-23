"""SQLite persistence: migration-friendly schema + parameterized repository."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Append new migrations to the END of this list; never edit an applied one.
MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE crops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE animals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        species TEXT NOT NULL,
        animal_tag TEXT,
        age_range TEXT,
        breed TEXT
    );
    CREATE TABLE assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        assessment_type TEXT NOT NULL CHECK (assessment_type IN ('plant','livestock')),
        crop_id INTEGER REFERENCES crops(id),
        animal_id INTEGER REFERENCES animals(id),
        image_path TEXT,
        status TEXT NOT NULL,
        summary TEXT NOT NULL,
        confidence REAL,
        model_version TEXT,
        notes TEXT,
        result_json TEXT NOT NULL
    );
    CREATE INDEX idx_assessments_created ON assessments(created_at DESC);
    CREATE TABLE assessment_symptoms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
        symptom_key TEXT NOT NULL,
        symptom_label TEXT NOT NULL
    );
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
        rank INTEGER NOT NULL,
        label_key TEXT NOT NULL,
        label_name TEXT NOT NULL,
        probability REAL NOT NULL
    );
    CREATE TABLE recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
        section TEXT NOT NULL,
        position INTEGER NOT NULL,
        text TEXT NOT NULL
    );
    CREATE TABLE follow_ups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        sources_json TEXT
    );
    """),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem_conn: sqlite3.Connection | None = None
        self.migrate()

    # -- connection handling ---------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:")
                self._mem_conn.row_factory = sqlite3.Row
                self._mem_conn.execute("PRAGMA foreign_keys = ON")
            conn, owned = self._mem_conn, False
        else:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            owned = True
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if owned:
                conn.close()

    def migrate(self) -> None:
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            done = {r["version"] for r in c.execute("SELECT version FROM schema_migrations")}
            for version, sql in MIGRATIONS:
                if version in done:
                    continue
                for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                    c.execute(stmt)
                c.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, _now()))

    def schema_version(self) -> int:
        with self.connect() as c:
            return c.execute("SELECT COALESCE(MAX(version),0) AS v FROM schema_migrations").fetchone()["v"]

    # -- writes -------------------------------------------------------------------
    def _get_or_create_crop(self, c: sqlite3.Connection, name: str) -> int:
        c.execute("INSERT OR IGNORE INTO crops(name) VALUES (?)", (name,))
        return c.execute("SELECT id FROM crops WHERE name = ?", (name,)).fetchone()["id"]

    def save_plant_assessment(self, *, crop: str, image_path: str | None, status: str, summary: str,
                              confidence: float | None, model_version: str | None, result: dict,
                              predictions: list[dict], recommendations: dict[str, list[str]],
                              notes: str = "") -> int:
        with self.connect() as c:
            crop_id = self._get_or_create_crop(c, crop)
            cur = c.execute(
                "INSERT INTO assessments(created_at, assessment_type, crop_id, image_path, status, summary, confidence, model_version, notes, result_json) "
                "VALUES (?, 'plant', ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), crop_id, image_path, status, summary, confidence, model_version, notes, json.dumps(result, default=str)))
            aid = cur.lastrowid
            for rank, p in enumerate(predictions, start=1):
                c.execute("INSERT INTO predictions(assessment_id, rank, label_key, label_name, probability) VALUES (?,?,?,?,?)",
                          (aid, rank, p["key"], p["name"], float(p["probability"])))
            self._save_recs(c, aid, recommendations)
            return aid

    def save_livestock_assessment(self, *, case: dict, image_path: str | None, status: str, summary: str,
                                  result: dict, symptoms: dict[str, str], recommendations: dict[str, list[str]]) -> int:
        with self.connect() as c:
            cur = c.execute("INSERT INTO animals(species, animal_tag, age_range, breed) VALUES (?,?,?,?)",
                            (case.get("species", "cattle"), case.get("animal_id") or None, case.get("age_range"), case.get("breed") or None))
            animal_id = cur.lastrowid
            cur = c.execute(
                "INSERT INTO assessments(created_at, assessment_type, animal_id, image_path, status, summary, confidence, model_version, notes, result_json) "
                "VALUES (?, 'livestock', ?, ?, ?, ?, NULL, ?, ?, ?)",
                (_now(), animal_id, image_path, status, summary, result.get("rules_version"), case.get("notes", ""),
                 json.dumps({"case": case, "assessment": result}, default=str)))
            aid = cur.lastrowid
            for k, label in symptoms.items():
                c.execute("INSERT INTO assessment_symptoms(assessment_id, symptom_key, symptom_label) VALUES (?,?,?)", (aid, k, label))
            self._save_recs(c, aid, recommendations)
            return aid

    @staticmethod
    def _save_recs(c: sqlite3.Connection, aid: int, recs: dict[str, list[str]]) -> None:
        for section, items in recs.items():
            for pos, text in enumerate(items):
                c.execute("INSERT INTO recommendations(assessment_id, section, position, text) VALUES (?,?,?,?)", (aid, section, pos, text))

    def add_follow_up(self, assessment_id: int, question: str, answer: str, sources: list[dict] | None = None) -> int:
        with self.connect() as c:
            cur = c.execute("INSERT INTO follow_ups(assessment_id, created_at, question, answer, sources_json) VALUES (?,?,?,?,?)",
                            (assessment_id, _now(), question, answer, json.dumps(sources or [])))
            return cur.lastrowid

    def delete_assessment(self, assessment_id: int) -> bool:
        with self.connect() as c:
            return c.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,)).rowcount > 0

    # -- reads --------------------------------------------------------------------
    def count_assessments(self, assessment_type: str | None = None) -> int:
        with self.connect() as c:
            if assessment_type:
                return c.execute("SELECT COUNT(*) AS n FROM assessments WHERE assessment_type = ?", (assessment_type,)).fetchone()["n"]
            return c.execute("SELECT COUNT(*) AS n FROM assessments").fetchone()["n"]

    def list_assessments(self, limit: int = 50, offset: int = 0, assessment_type: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT a.id, a.created_at, a.assessment_type, a.status, a.summary, a.confidence, a.image_path, a.model_version, "
               "c.name AS crop FROM assessments a LEFT JOIN crops c ON c.id = a.crop_id")
        params: list[Any] = []
        if assessment_type:
            sql += " WHERE a.assessment_type = ?"
            params.append(assessment_type)
        sql += " ORDER BY a.created_at DESC, a.id DESC LIMIT ? OFFSET ?"
        params += [int(limit), int(offset)]
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, params)]

    def get_assessment(self, assessment_id: int) -> dict[str, Any] | None:
        with self.connect() as c:
            row = c.execute("SELECT a.*, c.name AS crop FROM assessments a LEFT JOIN crops c ON c.id = a.crop_id WHERE a.id = ?",
                            (assessment_id,)).fetchone()
            if not row:
                return None
            out = dict(row)
            out["result"] = json.loads(out.pop("result_json"))
            out["predictions"] = [dict(r) for r in c.execute("SELECT rank, label_key, label_name, probability FROM predictions WHERE assessment_id = ? ORDER BY rank", (assessment_id,))]
            out["symptoms"] = [dict(r) for r in c.execute("SELECT symptom_key, symptom_label FROM assessment_symptoms WHERE assessment_id = ? ORDER BY id", (assessment_id,))]
            recs: dict[str, list[str]] = {}
            for r in c.execute("SELECT section, text FROM recommendations WHERE assessment_id = ? ORDER BY section, position", (assessment_id,)):
                recs.setdefault(r["section"], []).append(r["text"])
            out["recommendations"] = recs
            out["follow_ups"] = [dict(r) | {"sources": json.loads(r["sources_json"] or "[]")} for r in
                                 c.execute("SELECT id, created_at, question, answer, sources_json FROM follow_ups WHERE assessment_id = ? ORDER BY id", (assessment_id,))]
            if out.get("animal_id"):
                a = c.execute("SELECT species, animal_tag, age_range, breed FROM animals WHERE id = ?", (out["animal_id"],)).fetchone()
                out["animal"] = dict(a) if a else None
            return out


_db: Database | None = None


def get_database() -> Database:
    global _db
    from config import get_settings
    path = str(get_settings().database_path)
    if _db is None or _db.path != path:
        _db = Database(path)
    return _db
