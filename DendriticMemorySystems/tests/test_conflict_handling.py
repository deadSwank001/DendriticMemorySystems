"""Tests for conflict detection, supersession, and penalty scoring."""

from __future__ import annotations

import pytest
from dendritic_memory import DendriticMemorySystem
from dendritic_memory.scoring.branches import conflict_penalty
from dendritic_memory.storage.memory_record import ConflictStatus, MemoryRecord


def make_record(conflict: ConflictStatus) -> MemoryRecord:
    return MemoryRecord(content="A claim.", conflict_status=conflict)


def test_no_conflict_zero_penalty():
    assert conflict_penalty(make_record(ConflictStatus.NONE)) == 0.0


def test_disputed_partial_penalty():
    assert conflict_penalty(make_record(ConflictStatus.DISPUTED)) == pytest.approx(0.5)


def test_superseded_high_penalty():
    assert conflict_penalty(make_record(ConflictStatus.SUPERSEDED)) == pytest.approx(0.8)


def test_retracted_full_penalty():
    assert conflict_penalty(make_record(ConflictStatus.RETRACTED)) == pytest.approx(1.0)


def test_supersede_marks_old_record():
    dms = DendriticMemorySystem(activation_threshold=0.0)
    old = dms.ingest("The system uses flat vector search.", confidence=0.9)
    dms.supersede(old.memory_id, "The system uses multi-branch dendritic scoring.")

    old_record = dms.store.get(old.memory_id)
    assert old_record.conflict_status == ConflictStatus.SUPERSEDED
    assert old_record.superseded_by is not None


def test_superseded_record_not_in_active():
    dms = DendriticMemorySystem(activation_threshold=0.0)
    old = dms.ingest("Old claim.", confidence=0.9)
    dms.supersede(old.memory_id, "New claim.")

    active_ids = {r.memory_id for r in dms.store.all_active()}
    assert old.memory_id not in active_ids


def test_conflict_suppresses_retrieval():
    """A highly disputed record should score lower than an undisputed one."""
    dms = DendriticMemorySystem(activation_threshold=0.0)
    good = dms.ingest("Multi-branch scoring helps retrieval.", confidence=0.9)
    bad = dms.ingest("Multi-branch scoring helps retrieval.", confidence=0.9)

    # Manually mark bad as disputed
    bad_rec = dms.store.get(bad.memory_id)
    bad_rec.conflict_status = ConflictStatus.DISPUTED
    dms.store.update(bad_rec)

    results = dms.retrieve("multi-branch scoring")
    result_ids = [r.record.memory_id for r in results]
    good_idx = result_ids.index(good.memory_id) if good.memory_id in result_ids else 999
    bad_idx = result_ids.index(bad.memory_id) if bad.memory_id in result_ids else 999
    assert good_idx < bad_idx
