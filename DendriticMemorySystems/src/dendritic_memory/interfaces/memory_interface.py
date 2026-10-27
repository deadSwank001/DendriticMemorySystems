"""Public API — the top-level DendriticMemorySystem facade."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dendritic_memory.consolidation.consolidator import Consolidator
from dendritic_memory.embeddings.embedder import EmbeddingLayer
from dendritic_memory.graph.associative_graph import AssociativeGraph
from dendritic_memory.retrieval.retriever import DendriticRetriever, RetrievedMemory
from dendritic_memory.storage.memory_record import (
    MemoryRecord,
    MemoryType,
    RetentionPolicy,
    SourceType,
)
from dendritic_memory.storage.store import MemoryStore


class DendriticMemorySystem:
    """
    Top-level facade for the Dendritic Memory System.

    Coordinates ingestion, embedding, storage, graph linking, retrieval,
    and consolidation through a single cohesive interface.

    Example
    -------
    >>> from dendritic_memory import DendriticMemorySystem
    >>> dms = DendriticMemorySystem()
    >>> dms.ingest("The project uses multi-branch scoring.", source_type="user_provided")
    >>> results = dms.retrieve("How does retrieval work?", task_context="research")
    >>> for r in results:
    ...     print(r.record.content, r.activation)
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        activation_threshold: float = 0.65,
        graph_max_depth: int = 2,
    ) -> None:
        self.store = MemoryStore(persist_path)
        self.embedder = EmbeddingLayer(embedding_model)
        self.graph = AssociativeGraph(max_depth=graph_max_depth)
        self.retriever = DendriticRetriever(
            store=self.store,
            graph=self.graph,
            embedder=self.embedder,
            threshold=activation_threshold,
        )
        self.consolidator = Consolidator(store=self.store)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        content: str,
        memory_type: str = "episodic",
        source_type: str = "user_provided",
        author: Optional[str] = None,
        confidence: float = 0.8,
        importance: float = 0.5,
        project_tags: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
        retention_policy: str = "persistent",
        source_reference: Optional[str] = None,
    ) -> MemoryRecord:
        """
        Ingest a new memory into the system.

        Embeds the content, stores the record, and links it into the
        associative graph automatically.

        Returns the stored MemoryRecord.
        """
        record = MemoryRecord(
            content=content,
            memory_type=MemoryType(memory_type),
            source_type=SourceType(source_type),
            author=author,
            confidence=confidence,
            importance=importance,
            project_tags=project_tags or [],
            entities=entities or [],
            retention_policy=RetentionPolicy(retention_policy),
            source_reference=source_reference,
        )
        self.embedder.embed_record(record)
        self.store.add(record)
        self.graph.link_from_record(record)
        return record

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        task_context: Optional[str] = None,
        active_project: Optional[str] = None,
        active_entities: Optional[list[str]] = None,
    ) -> list[RetrievedMemory]:
        """
        Run dendritic retrieval for a query.

        All seven branches are evaluated for each candidate. Only records
        that exceed the activation threshold are returned.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            active_task=task_context,
            active_project=active_project,
            active_entity_ids=set(active_entities) if active_entities else None,
        )

    def retrieve_baseline(self, query: str, top_k: int = 10) -> list[RetrievedMemory]:
        """Cosine-only retrieval — no branch evaluation. For comparison/ablation."""
        return self.retriever.retrieve_baseline(query=query, top_k=top_k)

    # ------------------------------------------------------------------
    # Revision
    # ------------------------------------------------------------------

    def supersede(self, old_memory_id: str, new_content: str, **kwargs) -> MemoryRecord:
        """
        Replace an existing memory with a new version.
        The old record is marked as superseded (not deleted).
        """
        new_record = self.ingest(new_content, **kwargs)
        self.store.supersede(old_memory_id, new_record)
        self.graph.link(new_record.memory_id, old_memory_id, "supersedes")
        return new_record

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(
        self,
        episode_ids: list[str],
        summary: str,
        author: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        """
        Consolidate a cluster of episodic memories into a semantic record.
        Provenance links to source episodes are preserved.
        """
        semantic = self.consolidator.consolidate_cluster(episode_ids, summary, author)
        if semantic:
            self.embedder.embed_record(semantic)
            self.store.update(semantic)
            self.graph.link_from_record(semantic)
        return semantic

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the current memory store."""
        all_records = list(self.store._records.values())
        active = self.store.all_active()
        return {
            "total_records": len(all_records),
            "active_records": len(active),
            "graph_nodes": self.graph.node_count(),
            "graph_edges": self.graph.edge_count(),
            "conflicted": len(self.store.conflicted()),
        }
