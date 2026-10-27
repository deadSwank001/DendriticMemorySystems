"""Memory record schema — the atomic unit of the Dendritic Memory System."""

from __future__ import annotations

import uuid
from datetime import datetime, date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    PROJECT = "project"
    PROJECT_FACT = "project_fact"


class SourceType(str, Enum):
    USER_PROVIDED = "user_provided"
    LLM_GENERATED = "llm_generated"
    EXTERNAL_DOCUMENT = "external_document"
    RETRIEVED_SOURCE = "retrieved_source"
    SYSTEM_INFERRED = "system_inferred"


class VerificationStatus(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    RETRACTED = "retracted"


class ConflictStatus(str, Enum):
    NONE = "none"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class RetentionPolicy(str, Enum):
    PERSISTENT = "persistent"
    SESSION = "session"
    EXPIRING = "expiring"


class MemoryRecord(BaseModel):
    """
    Atomic memory trace with full provenance, temporal, and relational metadata.

    This is the core data structure of the Dendritic Memory System. Each record
    carries enough metadata for all seven scoring branches to evaluate it
    independently before a global retrieval decision is made.
    """

    # Identity
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC

    # Vector representation (populated by EmbeddingLayer)
    embedding: Optional[list[float]] = None

    # Temporal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: Optional[date] = None
    valid_until: Optional[date] = None

    # Provenance
    source_type: SourceType = SourceType.USER_PROVIDED
    source_reference: Optional[str] = None
    author: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    # Salience and usage
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None

    # Relational
    project_tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)

    # Conflict and revision
    conflict_status: ConflictStatus = ConflictStatus.NONE
    supersedes: Optional[str] = None   # memory_id this record replaces
    superseded_by: Optional[str] = None  # memory_id that replaced this record

    # Lifecycle
    retention_policy: RetentionPolicy = RetentionPolicy.PERSISTENT
    version: int = 1

    def is_valid(self) -> bool:
        """Return True if the record is within its validity window."""
        today = date.today()
        if self.valid_from and today < self.valid_from:
            return False
        if self.valid_until and today > self.valid_until:
            return False
        return True

    def is_active(self) -> bool:
        """Return True if the record is not superseded or retracted."""
        return self.conflict_status not in (
            ConflictStatus.SUPERSEDED,
            ConflictStatus.RETRACTED,
        )

    def mark_retrieved(self) -> None:
        """Update retrieval statistics in place."""
        self.retrieval_count += 1
        self.last_retrieved = datetime.utcnow()

    def supersede(self, new_id: str) -> None:
        """Mark this record as superseded by a newer record."""
        self.superseded_by = new_id
        self.conflict_status = ConflictStatus.SUPERSEDED
        self.updated_at = datetime.utcnow()
