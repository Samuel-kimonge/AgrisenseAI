"""Access to the structured tomato condition guide."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from config import get_settings


@lru_cache(maxsize=2)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def condition_guide(path: Path | None = None) -> dict:
    p = path or get_settings().knowledge_base_dir / "tomato_diseases.json"
    return _load(str(p))


def get_condition(key: str, path: Path | None = None) -> dict | None:
    return condition_guide(path)["conditions"].get(key)


def guide_meta(path: Path | None = None) -> dict:
    return condition_guide(path)["_meta"]
