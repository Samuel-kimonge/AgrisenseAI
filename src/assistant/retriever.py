"""Hybrid semantic + lexical retrieval with metadata filtering and abstention."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .embeddings import analyzer
from .ingest import ensure_index, get_store

# Very common words that carry no topical signal for this domain.
GENERIC = {"make", "use", "get", "give", "need", "help", "good", "best", "way", "work", "thing", "know", "tell", "want",
           "like", "caus", "common", "sign", "how", "much", "many", "mean", "doe", "wa"}


def _terms(text: str) -> set[str]:
    return {t for t in analyzer(text) if "_" not in t and t not in GENERIC}


@dataclass
class RetrievedChunk:
    text: str
    score: float           # hybrid relevance in [0, 1]
    metadata: dict
    cosine: float = 0.0
    coverage: float = 0.0

    @property
    def source(self) -> dict:
        m = self.metadata
        return {"title": m.get("title"), "section": m.get("section"), "publisher": m.get("publisher"),
                "publication_date": m.get("publication_date"), "jurisdiction": m.get("jurisdiction"),
                "species": m.get("species"), "version": m.get("version"), "review_status": m.get("review_status"),
                "score": round(self.score, 3)}


class KnowledgeRetriever:
    MIN_COVERAGE = 0.34   # share of the question's distinctive terms that must appear in the chunk

    def __init__(self, settings, embeddings=None, auto_index: bool = True):
        self.settings = settings
        self.embeddings = embeddings
        if auto_index:
            ensure_index(settings, embeddings)
        self.store = get_store(settings, embeddings)
        self._build_idf()

    def _build_idf(self) -> None:
        data = self.store.get(include=["documents"])
        docs = data.get("documents") or []
        self._n = max(1, len(docs))
        df: Counter = Counter()
        for d in docs:
            df.update(_terms(d))
        self._df = df

    def _idf(self, term: str) -> float:
        return math.log((self._n + 1) / (self._df.get(term, 0) + 0.5)) + 1.0

    def _coverage(self, q_terms: set[str], chunk_terms: set[str]) -> float:
        if not q_terms:
            return 0.0
        total = sum(self._idf(t) for t in q_terms)
        hit = sum(self._idf(t) for t in q_terms if t in chunk_terms)
        return hit / total

    def search(self, query: str, species: str | None = None, k: int | None = None,
               min_relevance: float | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.retrieval_k
        thr = self.settings.min_relevance if min_relevance is None else min_relevance
        flt = {"species": {"$in": [species, "general"]}} if species else None
        q_terms = _terms(query)
        cands = self.store.similarity_search_with_score(query, k=max(k * 4, 16), filter=flt)
        out = []
        for doc, dist in cands:
            cos = max(0.0, 1.0 - float(dist))
            cov = self._coverage(q_terms, _terms(doc.page_content))
            score = 0.5 * min(1.0, cos / 0.5) + 0.5 * cov
            if thr > 0 and cov < self.MIN_COVERAGE:
                continue
            if score >= thr:
                out.append(RetrievedChunk(doc.page_content, score, doc.metadata, cos, cov))
        out.sort(key=lambda c: -c.score)
        return out[:k]
