"""Tests for the associative memory graph and spreading activation."""

from __future__ import annotations

import pytest
from dendritic_memory.graph.associative_graph import AssociativeGraph
from dendritic_memory.storage.memory_record import MemoryRecord


@pytest.fixture()
def graph():
    return AssociativeGraph(max_depth=2, decay=0.5)


def make_record(content: str, project_tags=None, entities=None) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        project_tags=project_tags or [],
        entities=entities or [],
    )


def test_add_memory_and_link(graph):
    r = make_record("Dendritic scoring.", project_tags=["DMS"], entities=["David"])
    graph.link_from_record(r)
    assert graph.node_count() >= 3  # record + project + entity


def test_spreading_activation_reaches_neighbor(graph):
    r1 = make_record("Memory A.", entities=["Concept"])
    r2 = make_record("Memory B.", entities=["Concept"])
    graph.link_from_record(r1)
    graph.link_from_record(r2)

    scores = graph.spreading_activation([r1.memory_id])
    # Concept node should be reachable
    assert "Concept" in scores or any("Concept" in k for k in scores)


def test_activation_decays_with_hops(graph):
    a = make_record("A.")
    b = make_record("B.")
    c = make_record("C.")
    graph.add_memory(a)
    graph.add_memory(b)
    graph.add_memory(c)
    graph.link(a.memory_id, b.memory_id, "related", weight=1.0)
    graph.link(b.memory_id, c.memory_id, "related", weight=1.0)

    scores = graph.spreading_activation([a.memory_id])
    score_b = scores.get(b.memory_id, 0.0)
    score_c = scores.get(c.memory_id, 0.0)
    assert score_b > score_c  # decay: 2-hop should be lower than 1-hop


def test_max_depth_respected(graph):
    g = AssociativeGraph(max_depth=1, decay=0.5)
    a = make_record("A.")
    b = make_record("B.")
    c = make_record("C.")
    g.add_memory(a)
    g.add_memory(b)
    g.add_memory(c)
    g.link(a.memory_id, b.memory_id, "related")
    g.link(b.memory_id, c.memory_id, "related")

    scores = g.spreading_activation([a.memory_id])
    assert c.memory_id not in scores  # depth=1 should not reach c


def test_empty_graph_returns_empty(graph):
    scores = graph.spreading_activation(["nonexistent_id"])
    assert scores == {}
