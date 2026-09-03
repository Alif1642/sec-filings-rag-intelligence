from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, get_settings
from src.retrieval.fusion import FusedHit


@dataclass(slots=True)
class RerankedHit:
    chunk: dict
    score: float
    rank: int
    fusion_score: float


class CrossEncoderReranker:
    """Lazy cross-encoder reranker for hybrid retrieval candidates."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = None

    def _load(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise RuntimeError('sentence-transformers is required for reranking') from exc
            self.model = CrossEncoder(self.settings.reranker_model)

    def rerank(self, query: str, candidates: list[FusedHit], top_k: int | None = None) -> list[RerankedHit]:
        top_k = top_k or self.settings.rerank_top_k
        if not candidates:
            return []
        if not self.settings.reranker_enabled:
            return [RerankedHit(c.chunk, c.score, i + 1, c.score) for i, c in enumerate(candidates[:top_k])]
        self._load()
        pairs = [(query, c.chunk.get('text', '')) for c in candidates]
        scores = self.model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [RerankedHit(candidates[i].chunk, float(scores[i]), rank + 1, candidates[i].score) for rank, i in enumerate(order)]
