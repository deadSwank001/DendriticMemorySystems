"""
Event encoder for ingesting raw text or conversation turns into MemoryRecord candidates.
"""

from __future__ import annotations

from typing import Optional
from dendritic_memory.storage.memory_record import (
    MemoryRecord,
    MemoryType,
    RetentionPolicy,
    SourceType,
    VerificationStatus,
)


class EventEncoder:
    """Normalizes raw text and metadata into structured MemoryRecord instances."""

    def encode(
        self,
        content: str,
        memory_type: str | MemoryType = MemoryType.EPISODIC,
        source_type: str | SourceType = SourceType.USER_PROVIDED,
        verification_status: str | VerificationStatus = VerificationStatus.UNVERIFIED,
        confidence: float = 0.8,
        importance: float = 0.5,
        project_tags: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
        retention_policy: str | RetentionPolicy = RetentionPolicy.PERSISTENT,
    ) -> MemoryRecord:
        """Create a normalized MemoryRecord from raw ingestion parameters."""
        return MemoryRecord(
            content=content.strip(),
            memory_type=MemoryType(memory_type),
            source_type=SourceType(source_type),
            verification_status=VerificationStatus(verification_status),
            confidence=confidence,
            importance=importance,
            project_tags=project_tags or [],
            entities=entities or [],
            retention_policy=RetentionPolicy(retention_policy),
        )
