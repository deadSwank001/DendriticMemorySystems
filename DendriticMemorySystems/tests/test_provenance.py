"""Tests for provenance tracking and trust scoring."""

from __future__ import annotations

import pytest
from dendritic_memory.scoring.branches import provenance_score
from dendritic_memory.storage.memory_record import MemoryRecord, VerificationStatus


def make_record(confidence: float, verification: VerificationStatus) -> MemoryRecord:
    return MemoryRecord(
        content="Test claim.",
        confidence=confidence,
        verification_status=verification,
    )


def test_user_confirmed_high_score():
    record = make_record(0.8, VerificationStatus.USER_CONFIRMED)
    score = provenance_score(record)
    assert score >= 0.9


def test_unverified_no_bonus():
    record = make_record(0.8, VerificationStatus.UNVERIFIED)
    score = provenance_score(record)
    assert score == pytest.approx(0.8)


def test_disputed_penalty():
    record = make_record(0.8, VerificationStatus.DISPUTED)
    score = provenance_score(record)
    assert score == pytest.approx(0.6)


def test_retracted_heavily_penalized():
    record = make_record(0.9, VerificationStatus.RETRACTED)
    score = provenance_score(record)
    assert score == pytest.approx(0.4)


def test_low_confidence_user_confirmed():
    record = make_record(0.3, VerificationStatus.USER_CONFIRMED)
    score = provenance_score(record)
    assert score == pytest.approx(0.5)


def test_score_clipped_to_one():
    record = make_record(1.0, VerificationStatus.USER_CONFIRMED)
    score = provenance_score(record)
    assert score <= 1.0


def test_score_never_below_zero():
    record = make_record(0.0, VerificationStatus.RETRACTED)
    score = provenance_score(record)
    assert score >= 0.0
