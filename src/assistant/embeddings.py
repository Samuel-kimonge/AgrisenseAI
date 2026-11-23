"""Configurable embeddings.

* hashing (default): deterministic lexical embeddings that need no download or
  network. Good enough for a small curated corpus; weaker than neural embeddings
  on paraphrases.
* sentence-transformers: neural embeddings (needs the model to be downloadable
  from Hugging Face or already cached).
"""
from __future__ import annotations

import re

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(ENGLISH_STOP_WORDS)


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def analyzer(text: str) -> list[str]:
    toks = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 4096):
        self._vec = HashingVectorizer(n_features=dim, analyzer=analyzer, alternate_sign=False, norm="l2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vec.transform(texts).toarray().astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load sentence-transformers model '{model_name}': {exc}. "
                "Set EMBEDDING_PROVIDER=hashing to use the offline embeddings.") from exc

    def embed_documents(self, texts):
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def get_embeddings(settings) -> Embeddings:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddings(settings.embedding_model)
    if settings.embedding_provider == "hashing":
        return HashingEmbeddings()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
