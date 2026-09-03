from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.bm25 import RetrievalHit


@dataclass(slots=True)
class FusedHit:
    chunk: dict
    score: float
    rank: int
    sources: list[str]


def reciprocal_rank_fusion(rankings: list[list[RetrievalHit]], k: int = 60, top_k: int = 20) -> list[FusedHit]:
    """Fuse ranked lists by stable chunk_id using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}
    for ranking in rankings:
        for hit in ranking:
            cid = str(hit.chunk.get('chunk_id') or id(hit.chunk))
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + hit.rank)
            chunks[cid] = hit.chunk
            sources.setdefault(cid, set()).add(hit.source)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [FusedHit(chunks[cid], scores[cid], rank + 1, sorted(sources[cid])) for rank, cid in enumerate(ordered)]
