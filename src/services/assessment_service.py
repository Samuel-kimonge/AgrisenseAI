"""Application service: validates input, runs analysis, saves history."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from config import Settings, get_settings
from src.database.db import Database, get_database
from src.livestock.schema import LivestockCase, SYMPTOMS
from src.livestock.triage import LivestockAssessment, assess as triage
from src.ml.inference import PlantDiseasePredictor, PredictionResult, get_predictor
from src.utils.image_quality import InvalidImageError, load_image_from_bytes
from src.utils.storage import save_image

from .recommendation import livestock_recommendations, plant_recommendations


@dataclass
class PlantOutcome:
    result: PredictionResult
    recommendations: dict[str, list[str]]
    assessment_id: int | None   # None when nothing was saved (rejected photo / model unavailable)


class AssessmentService:
    def __init__(self, db: Database | None = None, predictor: PlantDiseasePredictor | None = None,
                 settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db = db or get_database()
        self.predictor = predictor or get_predictor()

    # ---------------------------------------------------------------- plants
    def analyze_plant(self, image_bytes: bytes, crop: str = "Tomato", notes: str = "") -> PlantOutcome:
        if crop.lower() != "tomato":
            raise ValueError(f"Crop '{crop}' is not supported yet. Supported crops: Tomato.")
        img = load_image_from_bytes(image_bytes, self.settings.max_upload_mb)  # raises InvalidImageError
        result = self.predictor.predict(img)
        recs = plant_recommendations(result)
        if result.status not in ("ok", "uncertain"):
            return PlantOutcome(result, recs, None)

        rel = save_image(img, self.settings.uploads_dir, prefix="plant")
        if result.status == "ok":
            summary = f"{result.predicted_name} (preliminary AI assessment)"
        else:
            summary = "Uncertain - the model could not name a condition"
        aid = self.db.save_plant_assessment(
            crop=crop, image_path=rel, status=result.status, summary=summary, confidence=result.confidence,
            model_version=result.model_version, result=result.to_dict(),
            predictions=[{"key": c.key, "name": c.name, "probability": c.probability} for c in result.top_k],
            recommendations=recs, notes=notes[:1000])
        return PlantOutcome(result, recs, aid)

    # ------------------------------------------------------------- livestock
    def assess_livestock(self, case: LivestockCase, photo_bytes: bytes | None = None) -> tuple[int, LivestockAssessment, dict]:
        errs = case.validate()
        if errs:
            raise ValueError(" ".join(errs))
        rel = None
        if photo_bytes:
            img = load_image_from_bytes(photo_bytes, self.settings.max_upload_mb)
            rel = save_image(img, self.settings.uploads_dir, prefix="cattle")
            case.photo_path = rel
        a = triage(case)
        recs = livestock_recommendations(a)
        top = a.concerns[0].name
        aid = self.db.save_livestock_assessment(
            case=case.to_dict(), image_path=rel, status=a.urgency,
            summary=f"{a.urgency_label}: {top}", result=a.to_dict(),
            symptoms={k: SYMPTOMS[k] for k in case.symptoms}, recommendations=recs)
        return aid, a, recs
