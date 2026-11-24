import json
from pathlib import Path

import pytest
import torch

from config import get_settings
from src.ml.inference import PlantDiseasePredictor
from tests.helpers import blurry_image, leaf_image, make_tiny_model_dir, non_leaf_image


def test_model_unavailable_does_not_fabricate(tmp_path, settings):
    p = PlantDiseasePredictor(settings, model_dir=tmp_path / "missing")
    assert not p.available
    r = p.predict(leaf_image())
    assert r.status == "model_unavailable"
    assert r.predicted_key is None and r.confidence is None and not r.top_k
    assert "train" in r.message.lower()


def test_corrupt_model_file_handled(tmp_path, settings):
    d = make_tiny_model_dir(tmp_path / "m")
    (d / "model.pt").write_bytes(b"corrupted")
    p = PlantDiseasePredictor(settings, model_dir=d)
    assert not p.available and "could not be loaded" in p.error


def test_loading_and_preprocessing(tmp_path, settings):
    p = PlantDiseasePredictor(settings, model_dir=make_tiny_model_dir(tmp_path / "m"))
    assert p.available and p.class_keys[0] == "tomato_early_blight"
    t = p.preprocess(leaf_image(300))
    assert tuple(t.shape) == (1, 3, 96, 96) and t.dtype == torch.float32
    assert -3 < float(t.min()) and float(t.max()) < 3.5


def test_prediction_response_structure(tmp_path, settings):
    p = PlantDiseasePredictor(settings, model_dir=make_tiny_model_dir(tmp_path / "m"))
    r = p.predict(leaf_image())
    assert r.status == "ok"
    assert r.predicted_key in p.class_keys
    assert len(r.top_k) == 3 and abs(sum(c.probability for c in r.top_k) - 1) < 1e-4
    assert r.top_k[0].probability >= r.top_k[1].probability >= r.top_k[2].probability
    d = r.to_dict()
    json.dumps(d)  # serialisable for storage
    assert 0 <= r.confidence <= 1 and 0 <= r.entropy <= 1


def test_uncertain_image_is_not_forced_into_a_class(tmp_path, settings):
    strict = make_tiny_model_dir(tmp_path / "m", min_conf=0.999, min_margin=0.5)
    r = PlantDiseasePredictor(settings, model_dir=strict).predict(leaf_image())
    assert r.status == "uncertain"
    assert r.predicted_key is None and r.predicted_name is None
    assert r.reasons and r.top_k  # shows candidates, but names nothing


def test_low_max_logit_triggers_abstention(tmp_path, settings):
    d = make_tiny_model_dir(tmp_path / "m", min_logit=1e6)
    r = PlantDiseasePredictor(settings, model_dir=d).predict(leaf_image())
    assert r.status == "uncertain" and any("does not closely match" in x for x in r.reasons)


@pytest.mark.parametrize("img_fn", [blurry_image, non_leaf_image])
def test_bad_quality_images_rejected_before_model(tmp_path, settings, img_fn):
    p = PlantDiseasePredictor(settings, model_dir=make_tiny_model_dir(tmp_path / "m"))
    r = p.predict(img_fn())
    assert r.status == "rejected_quality" and r.predicted_key is None and r.quality and r.message


# ----- integration tests against the REAL trained model (skipped if not trained) ----------
PROD = get_settings().production_model_dir
REAL = (PROD / "model.pt").exists() and (get_settings().processed_dir / "manifest.csv").exists()


@pytest.mark.skipif(not REAL, reason="trained model/manifest not present")
def test_real_model_on_held_out_test_images():
    import pandas as pd
    from PIL import Image
    s = get_settings()
    p = PlantDiseasePredictor(s)
    assert p.available
    man = pd.read_csv(s.processed_dir / "manifest.csv")
    test = man[man["split"] == "test"].groupby("label", group_keys=False).sample(n=8, random_state=7)
    correct = answered = 0
    for row in test.itertuples():
        img = Image.open(s.raw_data_dir / row.path).convert("RGB")
        r = p.predict(img)
        if r.status == "ok":
            answered += 1
            correct += r.predicted_key == row.label
    assert answered >= 0.5 * len(test)
    assert correct / answered >= 0.80, f"held-out accuracy among answered = {correct}/{answered}"


@pytest.mark.skipif(not REAL, reason="trained model/manifest not present")
def test_real_model_abstains_on_non_leaf_scene():
    r = PlantDiseasePredictor(get_settings()).predict(non_leaf_image())
    assert r.status in ("rejected_quality", "uncertain") and r.predicted_key is None


@pytest.mark.skipif(not REAL, reason="trained model/manifest not present")
def test_quality_gate_rarely_rejects_genuine_leaves():
    """Regression: an early leaf-tissue heuristic wrongly rejected/abstained on ~30% of valid leaves."""
    import pandas as pd
    from PIL import Image
    from src.utils.image_quality import assess_quality
    s = get_settings()
    man = pd.read_csv(s.processed_dir / "manifest.csv")
    sample = man[man["split"] == "test"].groupby("label", group_keys=False).sample(n=12, random_state=11)
    rejected = sum(not assess_quality(Image.open(s.raw_data_dir / r.path).convert("RGB")).ok for r in sample.itertuples())
    assert rejected / len(sample) <= 0.05, f"{rejected}/{len(sample)} genuine leaves rejected by quality gate"


# ----- shipped-artifact check: needs only models/production (works inside the Docker image) -----
HAS_MODEL = (get_settings().production_model_dir / "model.pt").exists()


@pytest.mark.skipif(not HAS_MODEL, reason="no trained model shipped")
def test_shipped_model_loads_and_predicts_without_dataset():
    p = PlantDiseasePredictor(get_settings())
    assert p.available, p.error
    assert len(p.class_keys) == 10 and p.meta["temperature"] > 0
    r = p.predict(leaf_image())
    assert r.status in ("ok", "uncertain") and len(r.top_k) == 3
    assert abs(sum(c.probability for c in r.top_k) - 1) < 0.2  # top-3 of a 10-class softmax
