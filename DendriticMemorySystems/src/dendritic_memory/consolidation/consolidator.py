"""Episodic-to-semantic consolidation policy."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from dendritic_memory.storage.memory_record import (
    MemoryRecord,
    MemoryType,
    RetentionPolicy,
    SourceType,
    VerificationStatus,
)
from dendritic_memory.storage.store import MemoryStore


class Consolidator:
    """
    Consolidates clusters of related episodic records into semantic memories.

    Key invariant: consolidated semantic memories remain traceable to their
    source episodic records through explicit provenance links. Consolidation
    never erases the underlying episodic evidence.
    """

    def __init__(
        self,
        store: MemoryStore,
        min_episodes: int = 3,
        min_confidence: float = 0.7,
    ) -> None:
        self.store = store
        self.min_episodes = min_episodes
        self.min_confidence = min_confidence

    def consolidate_cluster(
        self,
        episode_ids: list[str],
        summary_content: str,
        author: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        """
        Given a list of episodic memory IDs and a human-or-model-generated
        summary, create a new semantic memory record linked to those episodes.

        Returns the new semantic record, or None if the cluster is too small
        or has insufficient confidence.
        """
        episodes = [self.store.get(eid) for eid in episode_ids]
        episodes = [e for e in episodes if e is not None]

        if len(episodes) < self.min_episodes:
            return None

        avg_confidence = sum(e.confidence for e in episodes) / len(episodes)
        if avg_confidence < self.min_confidence:
            return None

        # Collect all project tags and entities from source episodes
        all_tags: set[str] = set()
        all_entities: set[str] = set()
        for ep in episodes:
            all_tags.update(ep.project_tags)
            all_entities.update(ep.entities)

        # Build source reference: comma-separated episode IDs
        source_reference = ",".join(ep.memory_id for ep in episodes)

        semantic = MemoryRecord(
            content=summary_content,
            memory_type=MemoryType.SEMANTIC,
            source_type=SourceType.SYSTEM_INFERRED,
            source_reference=source_reference,
            author=author,
            confidence=avg_confidence,
            verification_status=VerificationStatus.UNVERIFIED,
            project_tags=sorted(all_tags),
            entities=sorted(all_entities),
            relations=[f"consolidated_from:{eid}" for eid in episode_ids],
            retention_policy=RetentionPolicy.PERSISTENT,
        )

        self.store.add(semantic)
        return semantic

    def should_consolidate(self, episode_ids: list[str]) -> bool:
        """Return True if the cluster meets consolidation criteria."""
        episodes = [self.store.get(eid) for eid in episode_ids if self.store.get(eid)]
        if len(episodes) < self.min_episodes:
            return False
        avg_conf = sum(e.confidence for e in episodes) / len(episodes)
        return avg_conf >= self.min_confidence
