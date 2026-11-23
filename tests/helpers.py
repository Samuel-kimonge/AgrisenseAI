"""Shared helpers for tests: synthetic images, tiny untrained model package."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def leaf_image(size=256, seed=0, spots=True) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[..., 0], arr[..., 1], arr[..., 2] = 70, 140, 50
    arr += rng.normal(0, 14, arr.shape)
    if spots:
        for _ in range(12):
            x, y, r = rng.integers(20, size - 20, 2).tolist() + [int(rng.integers(4, 12))]
            yy, xx = np.ogrid[:size, :size]
            arr[(yy - y) ** 2 + (xx - x) ** 2 < r * r] = (110, 70, 30)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def blurry_image() -> Image.Image:
    return leaf_image(seed=3).filter(ImageFilter.GaussianBlur(25))


def dark_image() -> Image.Image:
    a = np.asarray(leaf_image(seed=4)).astype(np.float32) * 0.08
    return Image.fromarray(a.astype(np.uint8))


def non_leaf_image() -> Image.Image:
    rng = np.random.default_rng(5)
    arr = np.zeros((256, 256, 3), dtype=np.float32)
    arr[..., 0], arr[..., 1], arr[..., 2] = 40, 60, 200
    arr += rng.normal(0, 14, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_tiny_model_dir(path: Path, n_classes=3, min_conf=0.0, min_margin=0.0, min_logit=-1e9) -> Path:
    """Randomly initialised MobileNetV3 + metadata. For UNIT TESTS ONLY (never user-facing)."""
    import torch
    from src.ml.model import build_model
    path.mkdir(parents=True, exist_ok=True)
    keys = ["tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_mold"][:n_classes]
    torch.manual_seed(0)
    model = build_model(n_classes, None)
    torch.save(model.state_dict(), path / "model.pt")
    meta = {"model_version": "unit-test", "architecture": "test", "class_keys": keys,
            "class_names": {k: k for k in keys}, "image_size": 96, "temperature": 1.0,
            "thresholds": {"min_confidence": min_conf, "min_margin": min_margin, "min_max_logit": min_logit},
            "trained_at": "2026-01-01T00:00:00+00:00", "metrics": {}}
    (path / "metadata.json").write_text(json.dumps(meta))
    return path
