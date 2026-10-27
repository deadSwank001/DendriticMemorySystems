"""In-memory and persistent store for MemoryRecord objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dendritic_memory.storage.memory_record import (
    ConflictStatus,
    MemoryRecord,
    MemoryType,
    RetentionPolicy,
)


class MemoryStore:
    """
    Persistent store for MemoryRecord objects.

    Backed by an in-memory dict for fast lookup and an optional JSON file for
    persistence across sessions. A production deployment would replace the JSON
    backend with a vector database (e.g., Qdrant, Chroma, Pinecone).
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load(persist_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, record: MemoryRecord) -> str:
        """Store a new memory record. Returns the memory_id."""
        self._records[record.memory_id] = record
        self._save()
        return record.memory_id

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a single record by ID."""
        return self._records.get(memory_id)

    def update(self, record: MemoryRecord) -> None:
        """Overwrite an existing record."""
        if record.memory_id not in self._records:
            raise KeyError(f"No record with id {record.memory_id!r}")
        self._records[record.memory_id] = record
        self._save()

    def delete(self, memory_id: str) -> None:
        """Hard-delete a record."""
        self._records.pop(memory_id, None)
        self._save()

    def supersede(self, old_id: str, new_record: MemoryRecord) -> str:
        """
        Store new_record and mark old_id as superseded by it.
        Returns the new record's memory_id.
        """
        new_id = self.add(new_record)
        old = self.get(old_id)
        if old:
            old.supersede(new_id)
            self.update(old)
        return new_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_active(self) -> list[MemoryRecord]:
        """Return all records that are not superseded, retracted, or expired."""
        return [r for r in self._records.values() if r.is_active() and r.is_valid()]

    def by_type(self, memory_type: MemoryType) -> list[MemoryRecord]:
        return [r for r in self.all_active() if r.memory_type == memory_type]

    def by_project(self, tag: str) -> list[MemoryRecord]:
        return [r for r in self.all_active() if tag in r.project_tags]

    def by_retention(self, policy: RetentionPolicy) -> list[MemoryRecord]:
        return [r for r in self.all_active() if r.retention_policy == policy]

    def conflicted(self) -> list[MemoryRecord]:
        return [
            r
            for r in self._records.values()
            if r.conflict_status != ConflictStatus.NONE
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._persist_path:
            data = {mid: rec.model_dump(mode="json") for mid, rec in self._records.items()}
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self._records = {mid: MemoryRecord(**rec) for mid, rec in data.items()}

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"MemoryStore(records={len(self._records)}, path={self._persist_path})"
