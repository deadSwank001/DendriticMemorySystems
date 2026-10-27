"""
Evaluation metrics for comparing baseline RAG and Dendritic Memory retrieval.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Set

from dendritic_memory.retrieval.retriever import RetrievedMemory
from dendritic_memory.storage.memory_record import ConflictStatus


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r_id in top_k if r_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r_id in top_k if r_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute Reciprocal Rank for a single query."""
    for idx, r_id in enumerate(retrieved_ids, start=1):
        if r_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute binary nDCG@K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 / math.log2(idx + 1))
        for idx, r_id in enumerate(top_k, start=1)
        if r_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def staleness_error_rate(results: Iterable[RetrievedMemory]) -> float:
    """Compute fraction of retrieved records that are superseded or retracted."""
    items = list(results)
    if not items:
        return 0.0
    stale = sum(
        1
        for r in items
        if r.record.conflict_status in (ConflictStatus.SUPERSEDED, ConflictStatus.RETRACTED)
        or not r.record.is_valid()
    )
    return stale / len(items)
