"""Stable protocol ports for the harness domain (fusion P0).

Run-level evidence contracts: trace events, run reports, session facts
read-only access, and artifact redaction. Contracts only — implementations
land in P1 (RunStore/TaskState/trace/report). See docs/fusion-plan-review.md §3.
"""

from __future__ import annotations

from typing import Any, Protocol


class SessionSource(Protocol):
    """Read-only access to session facts for run projection (fusion §5.4.1)."""

    def list_events(self, session_id: str) -> list[Any]: ...


class TraceSink(Protocol):
    """Emit run-level trace events (redacted before persistence, fusion H3)."""

    def emit(self, event_type: str, payload: dict[str, Any]) -> None: ...


class RunReportSink(Protocol):
    """Write terminal run reports (fusion H4)."""

    def write_report(self, run_id: str, report: dict[str, Any]) -> None: ...


class Redactor(Protocol):
    """Redact secrets from text and artifacts before persistence (fusion H3/M8)."""

    def redact(self, text: str) -> str: ...
    def redact_artifact(self, path: str, content: bytes) -> bytes: ...
