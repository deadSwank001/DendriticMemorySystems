"""Seven-branch scoring system — the core of dendritic retrieval evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import numpy as np

from dendritic_memory.storage.memory_record import ConflictStatus, MemoryRecord, VerificationStatus


# ---------------------------------------------------------------------------
# Branch score container
# ---------------------------------------------------------------------------


@dataclass
class BranchScores:
    """Scores from all seven evaluation branches for a single candidate record."""

    semantic: float = 0.0
    temporal: float = 0.0
    contextual: float = 0.0
    relational: float = 0.0
    provenance: float = 0.0
    conflict_penalty: float = 0.0
    importance: float = 0.0

    def activation(
        self,
        w_semantic: float = 0.40,
        w_temporal: float = 0.20,
        w_contextual: float = 0.20,
        w_provenance: float = 0.20,
        w_conflict: float = 0.40,
        w_relational: float = 0.10,
        w_importance: float = 0.10,
    ) -> float:
        """
        Weighted integration across branches (soma rule).

        Conflict is subtracted as a penalty. Relational and importance add bonus
        weight on top of the primary four branches.
        """
        return (
            w_semantic * self.semantic
            + w_temporal * self.temporal
            + w_contextual * self.contextual
            + w_provenance * self.provenance
            + w_relational * self.relational
            + w_importance * self.importance
            - w_conflict * self.conflict_penalty
        )


# ---------------------------------------------------------------------------
# Individual branch scorers
# ---------------------------------------------------------------------------


def semantic_score(
    record: MemoryRecord,
    query_embedding: list[float],
    keyword_bonus: float = 0.0,
) -> float:
    """
    Cosine similarity between the record's embedding and the query embedding,
    plus an optional keyword-match bonus.
    """
    if record.embedding is None or not query_embedding:
        return 0.0
    a = np.array(record.embedding, dtype=float)
    b = np.array(query_embedding, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return min(1.0, cosine + keyword_bonus)


def temporal_score(
    record: MemoryRecord,
    half_life_days: float = 90.0,
) -> float:
    """
    Exponential decay score based on age of the record.

    Records expire immediately if their valid_until has passed.
    """
    if not record.is_valid():
        return 0.0
    age_days = (datetime.utcnow() - record.created_at).days
    score = math.exp(-age_days * math.log(2) / half_life_days)
    return float(min(1.0, score))


def contextual_score(
    record: MemoryRecord,
    active_task: Optional[str] = None,
    active_project: Optional[str] = None,
) -> float:
    """
    Score based on fit with the current task context and active project.
    """
    score = 0.0
    if active_project and active_project in record.project_tags:
        score += 0.6
    if active_task:
        task_lower = active_task.lower()
        if any(task_lower in tag.lower() for tag in record.project_tags):
            score += 0.4
        if task_lower in record.content.lower():
            score += 0.2
    return min(1.0, score)


def relational_score(
    record: MemoryRecord,
    active_entity_ids: Optional[set[str]] = None,
    graph_hop_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Score based on graph adjacency to currently active entities or memories.

    graph_hop_scores maps memory_id -> score contributed by graph proximity.
    """
    if active_entity_ids:
        overlap = set(record.entities) & active_entity_ids
        if overlap:
            return min(1.0, len(overlap) * 0.3)
    if graph_hop_scores and record.memory_id in graph_hop_scores:
        return float(min(1.0, graph_hop_scores[record.memory_id]))
    return 0.0


def provenance_score(record: MemoryRecord) -> float:
    """
    Score based on source reliability, verification status, and confidence.
    """
    base = record.confidence

    status_bonus = {
        VerificationStatus.USER_CONFIRMED: 0.2,
        VerificationStatus.UNVERIFIED: 0.0,
        VerificationStatus.DISPUTED: -0.2,
        VerificationStatus.RETRACTED: -0.5,
    }.get(record.verification_status, 0.0)

    return float(max(0.0, min(1.0, base + status_bonus)))


def conflict_penalty(record: MemoryRecord) -> float:
    """
    Penalty for records that are contradicted, superseded, or retracted.
    Returns a value in [0, 1] — subtracted from the activation sum.
    """
    return {
        ConflictStatus.NONE: 0.0,
        ConflictStatus.DISPUTED: 0.5,
        ConflictStatus.SUPERSEDED: 0.8,
        ConflictStatus.RETRACTED: 1.0,
    }.get(record.conflict_status, 0.0)


def importance_score(record: MemoryRecord) -> float:
    """
    Importance based on explicit user-set importance, retrieval frequency,
    and memory type.
    """
    freq_bonus = min(0.2, record.retrieval_count * 0.02)
    return float(min(1.0, record.importance + freq_bonus))


# ---------------------------------------------------------------------------
# All-branches evaluator
# ---------------------------------------------------------------------------


def evaluate_all_branches(
    record: MemoryRecord,
    query_embedding: list[float],
    active_task: Optional[str] = None,
    active_project: Optional[str] = None,
    active_entity_ids: Optional[set[str]] = None,
    graph_hop_scores: Optional[dict[str, float]] = None,
    temporal_half_life_days: float = 90.0,
) -> BranchScores:
    """Evaluate all seven branches for a single candidate record."""
    return BranchScores(
        semantic=semantic_score(record, query_embedding),
        temporal=temporal_score(record, temporal_half_life_days),
        contextual=contextual_score(record, active_task, active_project),
        relational=relational_score(record, active_entity_ids, graph_hop_scores),
        provenance=provenance_score(record),
        conflict_penalty=conflict_penalty(record),
        importance=importance_score(record),
    )


def should_activate(
    record: MemoryRecord,
    query_embedding: list[float],
    threshold: float = 0.65,
    active_task: Optional[str] = None,
    active_project: Optional[str] = None,
    active_entity_ids: Optional[set[str]] = None,
    graph_hop_scores: Optional[dict[str, float]] = None,
) -> tuple[bool, BranchScores]:
    """
    Return (activate, branch_scores) for a candidate memory record.

    This is the main gate: a memory is retrieved only if its weighted
    branch integration meets or exceeds the activation threshold.
    """
    scores = evaluate_all_branches(
        record,
        query_embedding,
        active_task,
        active_project,
        active_entity_ids,
        graph_hop_scores,
    )
    return scores.activation() >= threshold, scores
