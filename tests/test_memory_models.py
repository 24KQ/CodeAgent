"""P2 tests: memory data-plane schema contracts (models.py)."""

from __future__ import annotations

import dataclasses

import pytest

from firstcoder.memory.models import (
    MemoryEvidence,
    MemoryNote,
    MemoryQuery,
    PromotionCandidate,
    RetrievalResult,
    RetrievalSelection,
)


def test_evidence_defaults() -> None:
    evidence = MemoryEvidence()
    assert evidence.source_path == ""
    assert evidence.session_id == ""
    assert evidence.anchor_hash == ""
    assert evidence.scope == "workspace"


def test_note_defaults() -> None:
    note = MemoryNote(topic="key-decisions", text="Use pytest")
    assert note.note_id == ""
    assert note.status == "active"
    assert note.supersedes == ""
    assert note.created_at == ""
    assert note.evidence.source_path == ""


def test_note_is_frozen() -> None:
    note = MemoryNote(topic="t", text="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.text = "y"  # type: ignore[misc]


def test_query_defaults() -> None:
    query = MemoryQuery(text="pytest")
    assert query.limit == 5
    assert query.include_quarantined is False


def test_selection_reject_reason() -> None:
    note = MemoryNote(topic="t", text="x")
    selection = RetrievalSelection(note=note, reject_reason="duplicate", score=1.5)
    assert selection.selected is False
    assert selection.reject_reason == "duplicate"
    assert selection.score == 1.5


def test_result_selected_notes_filters() -> None:
    note_a = MemoryNote(topic="t", text="a")
    note_b = MemoryNote(topic="t", text="b")
    result = RetrievalResult(
        query=MemoryQuery(text="q"),
        selections=[
            RetrievalSelection(note=note_a, selected=True, score=1.0),
            RetrievalSelection(note=note_b, reject_reason="below_limit", score=0.5),
        ],
    )
    assert result.selected_notes == [note_a]


def test_promotion_candidate() -> None:
    candidate = PromotionCandidate(topic="key-decisions", text="x", source_path="src/a.py", reason="intent")
    assert candidate.reason == "intent"
