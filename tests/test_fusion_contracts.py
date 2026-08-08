"""Fusion P0 contract tests: ports exist, models behave, packages import clean."""

from __future__ import annotations

from firstcoder.harness.ports import (
    Redactor,
    RunArtifactStore,
    RunReportSink,
    SessionSource,
    TraceSink,
)
from firstcoder.memory.models import MemoryEvidence, MemoryNote, MemoryQuery, RetrievalResult
from firstcoder.memory.ports import (
    BoundedDreamRunner,
    MemoryPromotionPolicy,
    MemoryRetrievalPort,
    MemorySecurityPolicy,
    MemoryStorePort,
    WorkspaceScope,
)

MEMORY_PORTS = (
    MemoryStorePort,
    MemoryRetrievalPort,
    MemoryPromotionPolicy,
    MemorySecurityPolicy,
    WorkspaceScope,
    BoundedDreamRunner,
)

HARNESS_PORTS = (
    SessionSource,
    TraceSink,
    RunReportSink,
    RunArtifactStore,
    Redactor,
)


def test_memory_package_imports_cleanly() -> None:
    import firstcoder.memory  # noqa: F401


def test_harness_package_imports_cleanly() -> None:
    import firstcoder.harness  # noqa: F401


def test_all_fusion_ports_are_protocols() -> None:
    from typing import Protocol

    for port in MEMORY_PORTS + HARNESS_PORTS:
        assert isinstance(port, type)
        assert issubclass(port, Protocol)


def test_memory_note_defaults() -> None:
    note = MemoryNote(topic="t", text="x")
    assert note.status == "active"
    assert note.evidence.scope == "workspace"
    assert note.supersedes == ""


def test_retrieval_result_selected_notes_filters() -> None:
    from firstcoder.memory.models import RetrievalSelection

    note = MemoryNote(topic="t", text="x")
    result = RetrievalResult(
        query=MemoryQuery(text="q"),
        selections=[
            RetrievalSelection(note=note, selected=True),
            RetrievalSelection(note=note, selected=False, reject_reason="duplicate"),
        ],
    )
    assert len(result.selected_notes) == 1
    assert result.selections[1].reject_reason == "duplicate"


def test_evidence_fields() -> None:
    evidence = MemoryEvidence(source_path="src/main.py", session_id="s1", anchor_hash="ab12")
    assert evidence.source_path == "src/main.py"
    assert evidence.anchor_hash == "ab12"
