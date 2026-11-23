import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    s = get_settings()
    return dataclasses.replace(
        s, uploads_dir=tmp_path / "uploads", chroma_dir=tmp_path / "chroma", database_path=tmp_path / "t.db",
        production_model_dir=tmp_path / "prod", processed_dir=tmp_path / "processed", reports_dir=tmp_path / "reports")


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point the app at temp storage (for Streamlit AppTest and singletons)."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("PRODUCTION_MODEL_DIR", str(tmp_path / "no_model"))
    return tmp_path
