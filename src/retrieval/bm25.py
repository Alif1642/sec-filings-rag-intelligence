from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9$%._-]+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class _FallbackBM25Okapi:
    """Small dependency-free BM25 fallback used only when rank-bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in corpus:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.tf.append(counts)
            for token in counts:
                self.df[token] = self.df.get(token, 0) + 1

    def get_scores(self, query: list[str]) -> list[float]:
        n = len(self.corpus)
        scores = [0.0] * n
        if n == 0:
            return scores
        for term in query:
            df = self.df.get(term, 0)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self.tf):
                freq = counts.get(term, 0)
                if not freq:
                    continue
                dl = self.doc_len[i]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (freq * (self.k1 + 1) / denom)
        return scores


try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:  # pragma: no cover - exercised in minimal environments
    _BM25Okapi = _FallbackBM25Okapi


@dataclass(slots=True)
class RetrievalHit:
    chunk: dict
    score: float
    rank: int
    source: str


class BM25Retriever:
    """In-memory BM25 retriever over parsed filing chunks."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [tokenize(c.get('text', '')) for c in chunks]
        self.index = _BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 20) -> list[RetrievalHit]:
        if not self.index or not self.chunks:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [RetrievalHit(self.chunks[i], float(scores[i]), rank + 1, 'bm25') for rank, i in enumerate(ranked)]
