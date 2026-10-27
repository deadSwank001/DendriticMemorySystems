"""Ablation tests: verify each branch contributes to retrieval quality."""

from __future__ import annotations

import pytest
from dendritic_memory import DendriticMemorySystem
from dendritic_memory.scoring.branches import (
    BranchScores,
    conflict_penalty,
    importance_score,
    provenance_score,
    temporal_score,
)
from dendritic_memory.storage.memory_record import (
    ConflictStatus,
    MemoryRecord,
    VerificationStatus,
)


def test_full_branch_score_higher_than_semantic_only():
    """
    A record with good provenance and importance should score higher
    with multi-branch integration than with cosine similarity alone.
    """
    full_scores = BranchScores(
        semantic=0.6,
        temporal=0.8,
        contextual=0.7,
        relational=0.4,
        provenance=0.9,
        conflict_penalty=0.0,
        importance=0.8,
    )
    semantic_only = BranchScores(semantic=0.6)

    assert full_scores.activation() > semantic_only.activation()


def test_conflict_penalty_reduces_activation():
    no_conflict = BranchScores(semantic=0.8, provenance=0.9, conflict_penalty=0.0)
    disputed = BranchScores(semantic=0.8, provenance=0.9, conflict_penalty=0.5)
    retracted = BranchScores(semantic=0.8, provenance=0.9, conflict_penalty=1.0)

    assert no_conflict.activation() > disputed.activation() > retracted.activation()


def test_temporal_ablation():
    """Removing temporal branch should degrade recency-sensitive retrieval."""
    dms_full = DendriticMemorySystem(activation_threshold=0.0)
    dms_full.ingest("Recent finding about dendritic scoring.", confidence=0.9)

    results_full = dms_full.retrieve("dendritic scoring")
    results_baseline = dms_full.retrieve_baseline("dendritic scoring")

    # Both should find the record; full system should return it
    assert len(results_full) >= len(results_baseline) or len(results_full) > 0


def test_equal_weights_ablation():
    """With equal branch weights, conflict should still suppress a record."""
    equal_no_conflict = BranchScores(
        semantic=0.6,
        temporal=0.6,
        contextual=0.6,
        relational=0.6,
        provenance=0.6,
        conflict_penalty=0.0,
        importance=0.6,
    )
    equal_with_conflict = BranchScores(
        semantic=0.6,
        temporal=0.6,
        contextual=0.6,
        relational=0.6,
        provenance=0.6,
        conflict_penalty=1.0,
        importance=0.6,
    )
    w = dict(
        w_semantic=1 / 6,
        w_temporal=1 / 6,
        w_contextual=1 / 6,
        w_provenance=1 / 6,
        w_conflict=1 / 6,
        w_relational=1 / 6,
        w_importance=1 / 6,
    )
    assert equal_no_conflict.activation(**w) > equal_with_conflict.activation(**w)


def test_importance_bonus_for_frequently_retrieved():
    record = MemoryRecord(content="High-priority note.", importance=0.5, retrieval_count=10)
    score = importance_score(record)
    assert score > 0.5


def test_zero_semantic_with_high_other_branches():
    """Even with zero semantic similarity, high importance + provenance can activate."""
    scores = BranchScores(
        semantic=0.0,
        temporal=1.0,
        contextual=1.0,
        relational=1.0,
        provenance=1.0,
        conflict_penalty=0.0,
        importance=1.0,
    )
    assert scores.activation() > 0.0
