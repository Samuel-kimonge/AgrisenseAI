"""Central configuration for AgriSense AI.

All settings come from environment variables (optionally loaded from a .env
file). No credentials are hard-coded anywhere in the code base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _path(name: str, default: Path) -> Path:
    raw = _env(name)
    p = Path(raw) if raw else default
    return p if p.is_absolute() else ROOT_DIR / p


@dataclass(frozen=True)
class Settings:
    # ---- paths -----------------------------------------------------------
    root_dir: Path = ROOT_DIR
    raw_data_dir: Path = field(default_factory=lambda: _path("RAW_DATA_DIR", ROOT_DIR / "data" / "raw" / "plantvillage_tomato"))
    processed_dir: Path = field(default_factory=lambda: _path("PROCESSED_DIR", ROOT_DIR / "data" / "processed"))
    knowledge_base_dir: Path = field(default_factory=lambda: _path("KNOWLEDGE_BASE_DIR", ROOT_DIR / "data" / "knowledge_base"))
    checkpoints_dir: Path = field(default_factory=lambda: _path("CHECKPOINTS_DIR", ROOT_DIR / "models" / "checkpoints"))
    production_model_dir: Path = field(default_factory=lambda: _path("PRODUCTION_MODEL_DIR", ROOT_DIR / "models" / "production"))
    pretrained_weights: Path = field(default_factory=lambda: _path("PRETRAINED_WEIGHTS", ROOT_DIR / "models" / "pretrained" / "mobilenetv3_large_100_ra-f55367f5.pth"))
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", ROOT_DIR / "chroma_db"))
    uploads_dir: Path = field(default_factory=lambda: _path("UPLOADS_DIR", ROOT_DIR / "uploads"))
    reports_dir: Path = field(default_factory=lambda: _path("REPORTS_DIR", ROOT_DIR / "reports"))
    database_path: Path = field(default_factory=lambda: _path("DATABASE_PATH", ROOT_DIR / "data" / "agrisense.db"))

    # ---- ML --------------------------------------------------------------
    image_size: int = int(_env("IMAGE_SIZE", "192"))
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "8"))
    min_image_side: int = int(_env("MIN_IMAGE_SIDE", "96"))
    torch_threads: int = int(_env("TORCH_THREADS", "2"))

    # ---- assistant / RAG -------------------------------------------------
    # LLM_PROVIDER: none | anthropic | openai | ollama
    llm_provider: str = _env("LLM_PROVIDER", "none").lower()
    llm_model: str = _env("LLM_MODEL", "")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.2"))
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY")
    openai_api_key: str = _env("OPENAI_API_KEY")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    # EMBEDDING_PROVIDER: hashing | sentence-transformers
    embedding_provider: str = _env("EMBEDDING_PROVIDER", "hashing").lower()
    embedding_model: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(_env("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "100"))
    retrieval_k: int = int(_env("RETRIEVAL_K", "4"))
    min_relevance: float = float(_env("MIN_RELEVANCE", "0.30"))

    def ensure_dirs(self) -> None:
        for p in (self.processed_dir, self.checkpoints_dir, self.production_model_dir,
                  self.chroma_dir, self.uploads_dir, self.reports_dir / "evaluation",
                  self.reports_dir / "model_cards", self.database_path.parent):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a fresh Settings object (re-reads the environment)."""
    s = Settings()
    s.ensure_dirs()
    return s
