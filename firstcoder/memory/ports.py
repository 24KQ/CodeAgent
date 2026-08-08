"""Stable protocol ports for the memory domain (fusion P0).

All ports are structural Protocols in the FirstCoder idiom
(see firstcoder/agent/ports.py). P0 defines contracts only — no
implementations. Existing behavior must remain unchanged until a
port has a real implementation wired in (P2 memory data plane,
P6 auto-dream).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from firstcoder.memory.models import (
    MemoryEvidence,
    MemoryNote,
    MemoryQuery,
    PromotionCandidate,
    RetrievalResult,
)
from firstcoder.memory.dream.models import DreamProposal, MemoryMaintenanceSnapshot


class MemoryStorePort(Protocol):
    """Durable memory storage: daily logs, topic notes, metadata index.

    签名与 `DurableMemoryStore` 实现对齐（Codex P2 review #8）：daily log
    返回写入路径，evidence 可缺省。
    """

    def append_daily_log(self, text: str, *, source: MemoryEvidence | None = None) -> Path | None: ...
    def upsert_topic(self, note: MemoryNote) -> None: ...
    def promote_maintenance(self, notes: list[MemoryNote]) -> tuple[list[str], list[str]]: ...
    def read_index(self) -> list[MemoryNote]: ...


class MemoryRetrievalPort(Protocol):
    """Memory retrieval: query -> ranked notes with an audit trail."""

    def retrieve(self, query: MemoryQuery) -> RetrievalResult: ...


class MemoryPromotionPolicy(Protocol):
    """Decides which working-memory entries become durable notes."""

    def evaluate(self, entry: dict[str, Any]) -> PromotionCandidate | None: ...


class MemorySecurityPolicy(Protocol):
    """Static security rules: secret patterns, quarantine gate, redaction.

    `redact_artifact` is the recursive redaction used by the harness
    `TraceWriter` before persistence (Codex P1 review fix: the protocol
    must declare what the emitter actually calls).
    """

    def redact(self, text: str) -> str: ...
    def redact_artifact(self, value: Any, key: str | None = None) -> Any: ...
    def passes_quarantine(self, note: MemoryNote) -> bool: ...


class WorkspaceScope(Protocol):
    """Path constraints: memory root location and workspace residency."""

    def memory_root(self) -> str: ...
    def is_within_workspace(self, path: str) -> bool: ...


class BoundedDreamRunner(Protocol):
    """Bounded LLM maintenance runner with a restricted write scope."""

    def run_maintenance(
        self,
        *,
        prompt: str,
        snapshot: MemoryMaintenanceSnapshot,
        write_scope: WorkspaceScope,
    ) -> DreamProposal: ...
