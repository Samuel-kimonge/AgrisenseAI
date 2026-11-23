"""Safe image storage: files on disk, only a relative path in the database."""
from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image


def save_image(img: Image.Image, uploads_dir: Path, prefix: str = "img", max_side: int = 1280) -> str:
    """Re-encode as JPEG (strips metadata), cap size, and save under a random name.
    Returns the path relative to `uploads_dir`."""
    uploads_dir = Path(uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in "-_")[:20] or "img"
    name = f"{safe_prefix}_{uuid.uuid4().hex}.jpg"
    im = img.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    im.save(uploads_dir / name, format="JPEG", quality=88)
    return name


def resolve_image(uploads_dir: Path, rel: str | None) -> Path | None:
    """Resolve a stored relative path, refusing anything that escapes uploads_dir."""
    if not rel:
        return None
    base = Path(uploads_dir).resolve()
    p = (base / rel).resolve()
    if base not in p.parents or not p.is_file():
        return None
    return p
