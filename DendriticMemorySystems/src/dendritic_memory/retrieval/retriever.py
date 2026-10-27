"""Main retriever — orchestrates candidate fetch, branch evaluation, and context packaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dendritic_memory.embeddings.embedder import EmbeddingLayer
from dendritic_memory.graph.associative_graph import AssociativeGraph
from dendritic_memory.scoring.branches import BranchScores, should_activate
from dendritic_memory.storage.memory_record import MemoryRecord
from dendritic_memory.storage.store import MemoryStore


@dataclass
class RetrievedMemory:
    """A memory record that passed the activation threshold, with its scores."""

    record: MemoryRecord
    scores: BranchScores
    activation: float

    def __repr__(self) -> str:
        return (
            f"RetrievedMemory(id={self.record.memory_id!r}, "
            f"activation={self.activation:.3f}, "
            f"content={self.record.content[:60]!r})"
        )


class DendriticRetriever:
    """
    Full dendritic retrieval pipeline:

    1. Embed the query
    2. Fetch all active candidate records from the store
    3. Compute spreading activation scores from the graph
    4. Evaluate all seven branches for each candidate
    5. Apply the activation threshold
    6. Return ranked RetrievedMemory objects

    This is intentionally transparent: every retrieval decision is fully
    auditable through the BranchScores attached to each result.
    """

    def __init__(
        self,
        store: MemoryStore,
        graph: AssociativeGraph,
        embedder: EmbeddingLayer,
        threshold: float = 0.65,
    ) -> None:
        self.store = store
        self.graph = graph
        self.embedder = embedder
        self.threshold = threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        active_task: Optional[str] = None,
        active_project: Optional[str] = None,
        active_entity_ids: Optional[set[str]] = None,
    ) -> list[RetrievedMemory]:
        """
        Run the full dendritic retrieval pipeline and return the top-k
        activated memories, sorted by composite activation score.
        """
        query_embedding = self.embedder.embed(query)
        candidates = self.store.all_active()

        # Spreading activation from any provided seed entities
        graph_hop_scores: dict[str, float] = {}
        if active_entity_ids:
            graph_hop_scores = self.graph.spreading_activation(list(active_entity_ids))

        results: list[RetrievedMemory] = []
        for record in candidates:
            activate, branch_scores = should_activate(
                record=record,
                query_embedding=query_embedding,
                threshold=self.threshold,
                active_task=active_task,
                active_project=active_project,
                active_entity_ids=active_entity_ids,
                graph_hop_scores=graph_hop_scores,
            )
            if activate:
                record.mark_retrieved()
                results.append(
                    RetrievedMemory(
                        record=record,
                        scores=branch_scores,
                        activation=branch_scores.activation(),
                    )
                )

        results.sort(key=lambda r: r.activation, reverse=True)
        return results[:top_k]

    def retrieve_baseline(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedMemory]:
        """
        Baseline cosine-only retrieval — no branch evaluation.
        Used for ablation and comparison studies.
        """
        from dendritic_memory.scoring.branches import semantic_score, BranchScores

        query_embedding = self.embedder.embed(query)
        candidates = self.store.all_active()

        results: list[RetrievedMemory] = []
        for record in candidates:
            score = semantic_score(record, query_embedding)
            scores = BranchScores(semantic=score)
            results.append(
                RetrievedMemory(record=record, scores=scores, activation=score)
            )

        results.sort(key=lambda r: r.activation, reverse=True)
        return results[:top_k]
