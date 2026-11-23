import pytest
from PIL import Image

from src.utils.image_quality import InvalidImageError, assess_quality, load_image_from_bytes
from tests.helpers import blurry_image, dark_image, leaf_image, non_leaf_image, to_bytes


def test_valid_jpeg_and_png_load():
    for fmt in ("JPEG", "PNG"):
        img = load_image_from_bytes(to_bytes(leaf_image(), fmt))
        assert img.mode == "RGB" and img.size == (256, 256)


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_bytes_rejected(data):
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(data)


def test_unsupported_format_rejected():
    with pytest.raises(InvalidImageError, match="Unsupported"):
        load_image_from_bytes(to_bytes(leaf_image(), "GIF"))


def test_oversize_rejected():
    with pytest.raises(InvalidImageError, match="larger than"):
        load_image_from_bytes(b"x" * (2 * 1024 * 1024 + 1), max_mb=2)


def test_good_leaf_passes_quality():
    q = assess_quality(leaf_image())
    assert q.ok, q.issues
    assert q.vegetation_ratio > 0.5


def test_tiny_image_rejected():
    q = assess_quality(Image.new("RGB", (40, 40), (60, 140, 50)), min_side=96)
    assert not q.ok and any("too small" in i for i in q.issues)


def test_blurry_dark_and_nonleaf_flagged():
    assert any("blurry" in i for i in assess_quality(blurry_image()).issues)
    assert any("dark" in i for i in assess_quality(dark_image()).issues)
    assert any("not look like a plant" in i for i in assess_quality(non_leaf_image()).issues)


def test_non_leaf_scenes_score_zero_vegetation():
    for colour in [(110, 170, 230), (128, 128, 128), (240, 240, 240), (200, 40, 40)]:
        assert assess_quality(Image.new("RGB", (256, 256), colour)).vegetation_ratio < 0.08
