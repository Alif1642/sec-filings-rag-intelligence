from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, item in enumerate(retrieved_ids[:k]) if item in relevant_ids)
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    return {
        'recall@1': recall_at_k(retrieved_ids, relevant_ids, 1),
        'recall@3': recall_at_k(retrieved_ids, relevant_ids, 3),
        'recall@5': recall_at_k(retrieved_ids, relevant_ids, 5),
        'recall@10': recall_at_k(retrieved_ids, relevant_ids, 10),
        'mrr': reciprocal_rank(retrieved_ids, relevant_ids),
        'ndcg@10': ndcg_at_k(retrieved_ids, relevant_ids, 10),
    }
