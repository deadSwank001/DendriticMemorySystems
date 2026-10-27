"""
Provenance tracker for auditing memory origin, verification status, and citation lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from dendritic_memory.storage.memory_record import MemoryRecord, VerificationStatus


@dataclass
class ProvenanceEvent:
    timestamp: datetime
    action: str
    status: VerificationStatus
    confidence: float
    note: str = ""


class ProvenanceTracker:
    """Tracks verification changes and citation lineage for MemoryRecords."""

    def __init__(self) -> None:
        self._history: Dict[str, List[ProvenanceEvent]] = {}

    def record_creation(self, record: MemoryRecord, note: str = "Initial ingestion") -> None:
        self._history.setdefault(record.memory_id, []).append(
            ProvenanceEvent(
                timestamp=datetime.now(timezone.utc),
                action="created",
                status=record.verification_status,
                confidence=record.confidence,
                note=note,
            )
        )

    def update_verification(
        self,
        record: MemoryRecord,
        new_status: VerificationStatus,
        new_confidence: float,
        note: str = "",
    ) -> None:
        record.verification_status = new_status
        record.confidence = max(0.0, min(1.0, new_confidence))
        self._history.setdefault(record.memory_id, []).append(
            ProvenanceEvent(
                timestamp=datetime.now(timezone.utc),
                action="updated",
                status=record.verification_status,
                confidence=record.confidence,
                note=note,
            )
        )

    def get_history(self, memory_id: str) -> List[ProvenanceEvent]:
        return list(self._history.get(memory_id, []))
