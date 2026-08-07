"""Memory data-plane schema (fusion P0).

Pure dataclasses, zero runtime dependencies. The data model is the
FirstCoder-side contract for the pico-derived memory layer:
durable notes with provenance evidence, retrieval results with an audit
trail, and promotion candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NoteStatus = Literal["active", "superseded", "quarantined"]
MemoryVisibility = Literal["session", "workspace", "global"]
MEMORY_VISIBILITIES = frozenset({"session", "workspace", "global"})
RejectReason = Literal[
    "duplicate",
    "secret_shaped",
    "relative_date",
    "missing_evidence",
    "too_trivial",
    # retrieval 实际产生的拒绝原因（Codex P2 review #8：契约声明
    # 必须覆盖实现产生并写入 audit trail 的全部值）
    "quarantined",
    "superseded",
    "stale_evidence",
    "scope_mismatch",
    "session_mismatch",
    "global_disabled",
    "below_limit",
]


@dataclass(frozen=True)
class MemoryEvidence:
    """Provenance of a durable memory note (fusion M6).

    `source_path` is required in real use (the data plane always records
    provenance); the empty default keeps the contract constructible for
    tests and intermediate values.
    """

    source_path: str = ""
    session_id: str = ""
    anchor_hash: str = ""
    # `scope` 保留 P2 的 workspace fingerprint 契约；visibility 才表示
    # 这条记忆允许在哪个上下文中被读取。显式 capture 在 runtime 层使用
    # session，durable promote 默认使用 workspace，避免两个概念混用。
    scope: str = "workspace"
    visibility: MemoryVisibility = "workspace"


@dataclass(frozen=True)
class MemoryNote:
    """A durable memory topic entry (fusion M3)."""

    topic: str
    text: str
    note_id: str = ""
    status: NoteStatus = "active"
    supersedes: str = ""
    evidence: MemoryEvidence = field(default_factory=MemoryEvidence)
    created_at: str = ""

    # note_id = sha256(topic + text)[:12] 的推导由数据面实现（P2），
    # 契约层只保证字段存在。


@dataclass(frozen=True)
class MemoryQuery:
    """Retrieval request (fusion M4)."""

    text: str
    limit: int = 5
    include_quarantined: bool = False
    # 检索上下文由宿主 session 注入；缺少 session_id 时不会命中 session-only
    # 记忆。global 记忆必须由调用方显式开启，默认拒绝。
    session_id: str = ""
    include_global: bool = False


@dataclass(frozen=True)
class RetrievalSelection:
    """One selected/rejected candidate with its audit trail (fusion M4)."""

    note: MemoryNote
    selected: bool = False
    reject_reason: RejectReason | None = None
    score: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked retrieval output with audit trail (fusion M4)."""

    query: MemoryQuery
    selections: list[RetrievalSelection] = field(default_factory=list)
    query_hash: str = ""

    @property
    def selected_notes(self) -> list[MemoryNote]:
        return [s.note for s in self.selections if s.selected]


@dataclass(frozen=True)
class PromotionCandidate:
    """Working-memory entry that passed the durable promotion heuristics."""

    topic: str
    text: str
    source_path: str
    reason: str = ""
