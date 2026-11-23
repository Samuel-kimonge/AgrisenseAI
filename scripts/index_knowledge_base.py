"""Rebuild the ChromaDB index from data/knowledge_base (also happens automatically when files change)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_settings
from src.assistant.ingest import index_knowledge_base

if __name__ == "__main__":
    print(index_knowledge_base(get_settings()))
