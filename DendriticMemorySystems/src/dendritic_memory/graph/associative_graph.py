"""Associative memory graph using NetworkX."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from dendritic_memory.storage.memory_record import MemoryRecord


class AssociativeGraph:
    """
    Directed graph of relationships between memory records and entities.

    Supports bounded spreading activation: given a set of seed node IDs,
    compute activation scores for reachable nodes within a depth limit.
    """

    def __init__(self, max_depth: int = 2, decay: float = 0.5) -> None:
        self._g = nx.DiGraph()
        self.max_depth = max_depth
        self.decay = decay  # score multiplied per hop

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_memory(self, record: MemoryRecord) -> None:
        """Add a memory record as a node."""
        self._g.add_node(
            record.memory_id,
            node_type="memory",
            content_snippet=record.content[:80],
            memory_type=record.memory_type,
            project_tags=record.project_tags,
        )

    def add_entity(self, entity_id: str, label: str) -> None:
        """Add an entity (person, project, concept) as a node."""
        self._g.add_node(entity_id, node_type="entity", label=label)

    def link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Create a directed edge between two nodes."""
        self._g.add_edge(source_id, target_id, relation=relation_type, weight=weight)

    def link_from_record(self, record: MemoryRecord) -> None:
        """
        Auto-link a record to related entity nodes based on its metadata.
        Creates entity nodes if they don't already exist.
        """
        self.add_memory(record)
        for entity in record.entities:
            if not self._g.has_node(entity):
                self.add_entity(entity, entity)
            self.link(record.memory_id, entity, "mentions", weight=0.8)

        for tag in record.project_tags:
            project_id = f"project:{tag}"
            if not self._g.has_node(project_id):
                self.add_entity(project_id, tag)
            self.link(record.memory_id, project_id, "belongs_to", weight=0.9)

        if record.supersedes:
            self.link(record.memory_id, record.supersedes, "supersedes", weight=1.0)

    # ------------------------------------------------------------------
    # Bounded spreading activation
    # ------------------------------------------------------------------

    def spreading_activation(
        self,
        seed_ids: list[str],
        initial_score: float = 1.0,
    ) -> dict[str, float]:
        """
        Compute activation scores for all nodes reachable from seed_ids
        within self.max_depth hops.

        Returns a dict mapping node_id -> activation_score.
        """
        scores: dict[str, float] = {}
        frontier: list[tuple[str, float, int]] = [
            (sid, initial_score, 0) for sid in seed_ids if self._g.has_node(sid)
        ]

        visited: set[str] = set()

        while frontier:
            node_id, score, depth = frontier.pop(0)
            if node_id in visited or depth > self.max_depth:
                continue
            visited.add(node_id)

            if node_id not in seed_ids:
                scores[node_id] = max(scores.get(node_id, 0.0), score)

            if depth < self.max_depth:
                for neighbor in self._g.successors(node_id):
                    edge_data = self._g[node_id][neighbor]
                    edge_weight = edge_data.get("weight", 1.0)
                    propagated = score * self.decay * edge_weight
                    frontier.append((neighbor, propagated, depth + 1))

        return scores

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str, relation: Optional[str] = None) -> list[str]:
        """Return successor node IDs, optionally filtered by relation type."""
        if not self._g.has_node(node_id):
            return []
        return [
            n
            for n in self._g.successors(node_id)
            if relation is None or self._g[node_id][n].get("relation") == relation
        ]

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"AssociativeGraph(nodes={self.node_count()}, "
            f"edges={self.edge_count()}, max_depth={self.max_depth})"
        )
