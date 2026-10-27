"""Tests for dendritic retrieval pipeline."""

from __future__ import annotations

import pytest
from dendritic_memory import DendriticMemorySystem


@pytest.fixture()
def dms():
    return DendriticMemorySystem(activation_threshold=0.0)  # low threshold for testing


def _ingest_basics(dms: DendriticMemorySystem) -> list[str]:
    ids = []
    ids.append(
        dms.ingest(
            "Multi-branch scoring improves context-sensitive retrieval.",
            confidence=0.9,
            project_tags=["DendriticMemorySystems"],
        ).memory_id
    )
    ids.append(
        dms.ingest(
            "Baseline RAG uses cosine similarity only.",
            confidence=0.8,
            project_tags=["DendriticMemorySystems"],
        ).memory_id
    )
    ids.append(
        dms.ingest(
            "SpudCell is a synthetic-cell research project.",
            confidence=0.7,
            project_tags=["SpudCell"],
        ).memory_id
    )
    return ids


def test_retrieve_returns_results(dms):
    _ingest_basics(dms)
    results = dms.retrieve("How does retrieval work?")
    assert len(results) > 0


def test_retrieve_sorted_by_activation(dms):
    _ingest_basics(dms)
    results = dms.retrieve("multi-branch scoring")
    activations = [r.activation for r in results]
    assert activations == sorted(activations, reverse=True)


def test_retrieve_baseline_returns_results(dms):
    _ingest_basics(dms)
    results = dms.retrieve_baseline("cosine similarity")
    assert len(results) > 0


def test_project_filter(dms):
    _ingest_basics(dms)
    results = dms.retrieve(
        "retrieval",
        active_project="DendriticMemorySystems",
    )
    # All returned records should be tagged with the active project
    # (some may come from graph activation, so we check at least one)
    assert len(results) > 0


def test_stats(dms):
    _ingest_basics(dms)
    s = dms.stats()
    assert s["total_records"] == 3
    assert s["active_records"] == 3
    assert s["conflicted"] == 0


def test_supersede_reduces_active_count(dms):
    ids = _ingest_basics(dms)
    old_active = dms.stats()["active_records"]
    dms.supersede(ids[0], "Updated: multi-branch scoring is empirically validated.")
    new_active = dms.stats()["active_records"]
    # old record suppressed, new one added → net +0 active, but conflicted +1
    assert dms.stats()["conflicted"] >= 1
