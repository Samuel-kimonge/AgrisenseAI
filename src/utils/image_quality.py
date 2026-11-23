"""Upload validation and image-quality checks.

These checks run *before* the classifier so that unusable photos (corrupt files,
tiny thumbnails, blur, darkness, non-leaf scenes) are rejected with a friendly
message instead of being forced into a disease class.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}


class InvalidImageError(ValueError):
    """Raised for uploads that are not usable images."""


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float          # variance of Laplacian (higher = sharper)
    brightness: float          # mean V channel, 0-255
    vegetation_ratio: float    # fraction of pixels that look like plant tissue
    issues: list[str] = field(default_factory=list)   # blocking problems
    warnings: list[str] = field(default_factory=list)  # non-blocking

    @property
    def ok(self) -> bool:
        return not self.issues


def load_image_from_bytes(data: bytes, max_mb: int = 8) -> Image.Image:
    """Decode upload bytes safely; enforce size/format; fix EXIF orientation."""
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise InvalidImageError(f"The image is larger than {max_mb} MB. Please upload a smaller photo.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImageError("This file could not be read as an image. Please upload a JPG or PNG photo.")
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"Unsupported image type ({fmt}). Please upload a JPG, JPEG or PNG photo.")
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def assess_quality(img: Image.Image, min_side: int = 96) -> QualityReport:
    w, h = img.size
    arr = np.asarray(img)
    # Analyse at a bounded size so results do not depend on camera resolution.
    scale = 512 / max(w, h)
    if scale < 1:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    brightness = float(hsv[..., 2].mean())
    # plant-like hues: yellow-green-brown range with some saturation
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Plant-like hues (olive/brown through green). Ranges were set from measurements on held-out
    # PlantVillage leaves (median ratio ~0.34 because leaves sit on plain backgrounds; min ~0.10)
    # while blue sky, grey/white walls and red objects score 0.
    veg = ((hue >= 5) & (hue <= 100) & (sat >= 25) & (val >= 30)).mean()
    rep = QualityReport(width=w, height=h, blur_score=blur, brightness=brightness, vegetation_ratio=float(veg))

    if min(w, h) < min_side:
        rep.issues.append(f"The photo is too small ({w}x{h}). Please take a closer, higher-resolution photo.")
    if brightness < 35:
        rep.issues.append("The photo is too dark. Please retake it in daylight.")
    elif brightness > 245:
        rep.warnings.append("The photo may be over-exposed.")
    if blur < 15:
        rep.issues.append("The photo looks blurry. Hold the camera steady and focus on the leaf.")
    elif blur < 40:
        rep.warnings.append("The photo is slightly soft; a sharper photo gives more reliable results.")
    if veg < 0.08:
        rep.issues.append("This does not look like a plant leaf photo. Please photograph a single tomato leaf filling most of the frame.")
    elif veg < 0.15:
        rep.warnings.append("Little leaf area is visible. Move closer so the leaf fills the frame.")
    return rep
