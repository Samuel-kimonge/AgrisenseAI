import pandas as pd
import pytest

from src.ml.dataset import cap_split, discover_dataset, stratified_split, validate_dataset
from tests.helpers import leaf_image


@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / "raw"
    for ci, folder in enumerate(["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]):
        d = root / folder
        d.mkdir(parents=True)
        for i in range(20):
            leaf_image(128, seed=ci * 100 + i).save(d / f"{i}.jpg")
    (root / "Tomato___healthy" / "corrupt.jpg").write_bytes(b"garbage")
    leaf_image(128, seed=0).save(root / "Tomato___Early_blight" / "dup.jpg")  # not byte-identical to healthy/0 on purpose
    (root / "Tomato___Late_blight" / "20.jpg").write_bytes((root / "Tomato___Late_blight" / "0.jpg").read_bytes())  # exact duplicate
    (root / "empty_folder").mkdir()
    return root


def test_discovery_finds_classes_and_ignores_empty(fake_dataset):
    info = discover_dataset(fake_dataset)
    assert info.class_keys == ["tomato_early_blight", "tomato_healthy", "tomato_late_blight"]
    assert info.counts["tomato_healthy"] == 21


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_dataset(tmp_path / "nope")


def test_integrity_validation_flags_corrupt_and_duplicates(fake_dataset):
    df, rep = validate_dataset(discover_dataset(fake_dataset))
    assert rep["problems"]["corrupt"] == 1
    assert rep["problems"]["exact_duplicate"] >= 1
    assert len(df) == rep["total_kept"]


def test_split_is_stratified_and_disjoint(fake_dataset):
    df, _ = validate_dataset(discover_dataset(fake_dataset))
    m = stratified_split(df, 0.2, 0.2, seed=1)
    assert set(m["split"]) == {"train", "val", "test"}
    assert m["path"].is_unique
    counts = m.groupby(["label", "split"]).size().unstack()
    assert (counts > 0).all().all()
    m2 = stratified_split(df, 0.2, 0.2, seed=1)
    pd.testing.assert_series_equal(m["split"], m2["split"])  # reproducible


def test_cap_split_limits_per_class(fake_dataset):
    df, _ = validate_dataset(discover_dataset(fake_dataset))
    m = stratified_split(df, 0.2, 0.2)
    capped = cap_split(m, "train", 5)
    assert capped.groupby("label").size().max() <= 5 and set(capped["split"]) == {"train"}
