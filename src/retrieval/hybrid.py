from __future__ import annotations

from src.config import Settings, get_settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import FusedHit, reciprocal_rank_fusion
from src.retrieval.vector_store import FAISSVectorStore


class HybridRetriever:
    """BM25 + dense retrieval merged with Reciprocal Rank Fusion."""

    def __init__(self, chunks: list[dict], settings: Settings | None = None, vector_store: FAISSVectorStore | None = None):
        self.settings = settings or get_settings()
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.vector = vector_store or FAISSVectorStore(self.settings)
        if self.vector.index is None:
            self.vector.build(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[FusedHit]:
        top_k = top_k or self.settings.retrieval_top_k
        sparse = self.bm25.search(query, top_k=top_k)
        dense = self.vector.search(query, top_k=top_k)
        return reciprocal_rank_fusion([sparse, dense], top_k=top_k)
