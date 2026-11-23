"""Real-model inference with quality gating and abstention."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from config import Settings, get_settings
from src.utils.image_quality import QualityReport, assess_quality

from .classes import display_name

MODEL_FILE = "model.pt"
META_FILE = "metadata.json"


@dataclass
class Candidate:
    key: str
    name: str
    probability: float


@dataclass
class PredictionResult:
    status: str  # ok | uncertain | rejected_quality | model_unavailable
    message: str = ""
    predicted_key: str | None = None
    predicted_name: str | None = None
    confidence: float | None = None          # calibrated (temperature-scaled) probability
    raw_confidence: float | None = None      # plain softmax, for transparency
    margin: float | None = None
    entropy: float | None = None
    top_k: list[Candidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)      # why the system abstained
    quality: QualityReport | None = None
    model_version: str | None = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("quality", "top_k")}
        d["top_k"] = [c.__dict__ for c in self.top_k]
        d["quality"] = self.quality.__dict__ if self.quality else None
        return d


class PlantDiseasePredictor:
    """Loads the production model lazily. If it is missing, `available` is False and
    `predict` returns status='model_unavailable' - it never fabricates predictions."""

    def __init__(self, settings: Settings | None = None, model_dir: Path | None = None):
        self.settings = settings or get_settings()
        self.model_dir = Path(model_dir) if model_dir else self.settings.production_model_dir
        self._model = None
        self.meta: dict | None = None
        self.error: str | None = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        mp, jp = self.model_dir / MODEL_FILE, self.model_dir / META_FILE
        if not mp.exists() or not jp.exists():
            self.error = ("No trained model found. Train one with `python scripts/train.py` "
                          f"(expected {mp.name} and {jp.name} in {self.model_dir}).")
            return
        try:
            import torch
            from .model import build_model
            torch.set_num_threads(max(1, self.settings.torch_threads))
            self.meta = json.loads(jp.read_text())
            model = build_model(len(self.meta["class_keys"]), pretrained_weights=None)
            state = torch.load(mp, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._model = model
        except Exception as exc:  # corrupted file, arch mismatch, etc.
            self._model, self.meta = None, None
            self.error = f"The trained model could not be loaded: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def class_keys(self) -> list[str]:
        return list(self.meta["class_keys"]) if self.meta else []

    @property
    def version(self) -> str | None:
        return self.meta.get("model_version") if self.meta else None

    # ------------------------------------------------------------------
    def preprocess(self, img: Image.Image) -> "torch.Tensor":  # noqa: F821
        import torch
        import torchvision.transforms as T
        size = int(self.meta["image_size"]) if self.meta else self.settings.image_size
        tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return tf(img.convert("RGB")).unsqueeze(0)

    def predict(self, img: Image.Image, top_k: int = 3) -> PredictionResult:
        if not self.available:
            return PredictionResult(status="model_unavailable", message=self.error or "Model unavailable.")
        import torch
        from scipy.special import softmax

        quality = assess_quality(img, self.settings.min_image_side)
        if not quality.ok:
            return PredictionResult(status="rejected_quality", message=" ".join(quality.issues),
                                    quality=quality, model_version=self.version)

        with torch.inference_mode():
            logits = self._model(self.preprocess(img)).float().numpy()[0]
        T = float(self.meta["temperature"])
        probs = softmax(logits / T)
        raw = softmax(logits)
        order = np.argsort(-probs)
        cands = [Candidate(self.meta["class_keys"][i], display_name(self.meta["class_keys"][i]), float(probs[i]))
                 for i in order[:top_k]]
        margin = float(probs[order[0]] - probs[order[1]])
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))
        thr = self.meta["thresholds"]

        reasons = []
        if probs[order[0]] < thr["min_confidence"]:
            reasons.append(f"calibrated confidence {probs[order[0]]:.0%} is below the {thr['min_confidence']:.0%} reliability threshold")
        if margin < thr["min_margin"]:
            reasons.append("the top two possibilities are too close to separate")
        if float(logits.max()) < thr["min_max_logit"]:
            reasons.append("the leaf does not closely match any condition the model was trained on")

        common = dict(top_k=cands, confidence=float(probs[order[0]]), raw_confidence=float(raw[order[0]]),
                      margin=margin, entropy=entropy, quality=quality, model_version=self.version)
        if reasons:
            return PredictionResult(
                status="uncertain", reasons=reasons,
                message="The model is not certain enough to name a condition for this photo. "
                        "Try a clearer, closer photo of a single tomato leaf, or consult an extension officer.",
                **common)
        top = cands[0]
        return PredictionResult(status="ok", predicted_key=top.key, predicted_name=top.name,
                                message="Preliminary AI assessment (not a confirmed diagnosis).", **common)


_cached: PlantDiseasePredictor | None = None


def get_predictor(force_reload: bool = False) -> PlantDiseasePredictor:
    global _cached
    if _cached is None or force_reload:
        _cached = PlantDiseasePredictor()
    return _cached
