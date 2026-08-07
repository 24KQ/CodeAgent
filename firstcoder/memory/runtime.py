"""P3 显式记忆 runtime。

本模块把 daily log、durable promotion 和 memory audit 的共用安全边界集中起来。
它只依赖 memory 数据面和一个抽象的 audit callback，不直接 import context/agent，
因此不会反向改变 FirstCoder 的事件事实层职责。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from firstcoder.memory.durable import DurableMemoryStore, note_id_for
from firstcoder.memory.models import (
    MEMORY_VISIBILITIES,
    MemoryEvidence,
    MemoryNote,
    MemoryQuery,
    MemoryVisibility,
    RetrievalResult,
)
from firstcoder.memory.redact import MemoryRedactor
from firstcoder.memory.retrieval import MemoryRetriever

MemoryAuditCallback = Callable[[str, dict[str, Any]], None]

MAX_MEMORY_ENTRY_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class MemoryWriteReceipt:
    """一次显式记忆写入的安全摘要。

    receipt 故意不保存原文和绝对路径。工具、命令和 UI 只需要知道操作是否
    成功、是否被 quarantine 拦截以及一个稳定 note id，避免把敏感输入再次
    放进结果 artifact。
    """

    ok: bool
    operation: str
    chars: int = 0
    redacted: bool = False
    quarantined: bool = False
    promoted: bool = False
    note_id: str = ""
    error: str = ""
    superseded_count: int = 0
    visibility: MemoryVisibility = "session"

    def public_data(self) -> dict[str, object]:
        """返回可直接给工具/UI 的非敏感字段。"""

        return {
            "operation": self.operation,
            "chars": self.chars,
            "redacted": self.redacted,
            "quarantined": self.quarantined,
            "promoted": self.promoted,
            "note_id": self.note_id,
            "superseded_count": self.superseded_count,
            "visibility": self.visibility,
        }


@dataclass(slots=True)
class MemoryRuntime:
    """绑定一个 workspace memory store 的显式操作 facade。"""

    store: DurableMemoryStore
    session_id: str
    security: MemoryRedactor = field(default_factory=MemoryRedactor)
    audit: MemoryAuditCallback | None = None
    max_entry_chars: int = MAX_MEMORY_ENTRY_CHARS
    # Global store 独立于当前 workspace；存在这个引用不等于允许读取，
    # retrieve/projector 仍必须收到 include_global=True 才会加入查询快照。
    global_store: DurableMemoryStore | None = None

    def record(self, text: object, *, source: str) -> MemoryWriteReceipt:
        """脱敏后追加 daily log，并追加不含原文的 ``memory_recorded`` audit。

        daily log 是显式捕获的安全缓冲层：即使内容带 prompt injection，也不会
        直接进入 request projector。quarantine 标记会随 receipt 和 audit 保存，
        让上层知道这条输入不能被提升为可检索 durable note。
        """

        original = self._normalize_text(text)
        sanitized_full = self.security.redact_text(original)
        sanitized = self._bounded(sanitized_full)
        if not sanitized:
            return MemoryWriteReceipt(ok=False, operation="daily_log", error="记忆内容不能为空")

        redacted = sanitized_full != original
        quarantined = redacted or not self.security.passes_quarantine(
            MemoryNote(topic="capture", text=original)
        )
        evidence = self._evidence()
        try:
            self.store.append_daily_log(sanitized, source=evidence)
        except (OSError, ValueError):
            return MemoryWriteReceipt(
                ok=False,
                operation="daily_log",
                chars=len(sanitized),
                redacted=redacted,
                quarantined=quarantined,
                error="记忆写入失败",
            )

        receipt = MemoryWriteReceipt(
            ok=True,
            operation="daily_log",
            chars=len(sanitized),
            redacted=redacted,
            quarantined=quarantined,
            visibility="session",
        )
        self._emit(
            "memory_recorded",
            {
                "operation": receipt.operation,
                "source": str(source),
                "chars": receipt.chars,
                "redacted": receipt.redacted,
                "quarantined": receipt.quarantined,
            },
        )
        return receipt

    def promote(
        self,
        topic: object,
        text: object,
        *,
        source: str,
        visibility: MemoryVisibility = "workspace",
    ) -> MemoryWriteReceipt:
        """安全地提升一条 durable note。

        gate 在脱敏前检查原文，确保 secret 或 prompt injection 不会因为先被
        替换成 ``<redacted>`` 而意外变成 active memory。被隔离的输入只留下
        audit，不写入 durable topic，也就不可能被下一轮 request 注入。
        """

        requested_visibility = self._normalize_visibility(visibility)
        if requested_visibility is None:
            return MemoryWriteReceipt(
                ok=False,
                operation="promote",
                error="记忆作用域必须是 session、workspace 或 global",
                visibility="workspace",
            )

        target_store = self.store
        if requested_visibility == "global":
            if self.global_store is None:
                return MemoryWriteReceipt(
                    ok=False,
                    operation="promote",
                    error="全局记忆存储未启用",
                    visibility="global",
                )
            target_store = self.global_store

        topic_text = str(topic or "").strip()
        original = self._normalize_text(text)
        sanitized_full = self.security.redact_text(original)
        sanitized = self._bounded(sanitized_full)
        if not topic_text:
            return MemoryWriteReceipt(ok=False, operation="promote", error="记忆主题不能为空")
        if not sanitized:
            return MemoryWriteReceipt(ok=False, operation="promote", error="记忆内容不能为空")

        redacted = sanitized_full != original
        quarantined = redacted or not self.security.passes_quarantine(
            MemoryNote(topic=topic_text, text=original)
        )
        if quarantined:
            receipt = MemoryWriteReceipt(
                ok=False,
                operation="promote",
                chars=len(sanitized),
                redacted=redacted,
                quarantined=True,
                error="记忆内容已隔离，未提升为 durable memory",
                visibility=requested_visibility,
            )
            self._emit(
                "memory_recorded",
                {
                    "operation": receipt.operation,
                    "source": str(source),
                    "topic": topic_text,
                    "chars": receipt.chars,
                    "redacted": receipt.redacted,
                    "quarantined": True,
                    "promoted": False,
                },
            )
            return receipt

        # /remember 的 record 已经在当天 sidecar 保存了捕获证据。提升时优先
        # 取回同一 session、同一脱敏文本的最新 sidecar 行，让 durable metadata
        # 保留 capture provenance；直接调用 promote 或跨日调用则回退到当前
        # session 的默认证据，不把其他 session 的 sidecar 误挂到本条 note。
        note = MemoryNote(
            topic=topic_text,
            text=sanitized,
            evidence=self._evidence_for_text(sanitized, visibility=requested_visibility),
        )
        try:
            target_store.upsert_topic(note)
            snapshot = target_store.snapshot(
                None if target_store.global_store else target_store.workspace_root
            )
        except (OSError, ValueError, KeyError):
            return MemoryWriteReceipt(
                ok=False,
                operation="promote",
                chars=len(sanitized),
                redacted=redacted,
                error="durable memory 提升失败",
                visibility=requested_visibility,
            )

        resolved_id = note_id_for(topic_text, sanitized)
        stored = next((item for item in snapshot if item.get("note_id") == resolved_id), {})
        superseded_count = sum(
            1
            for item in snapshot
            if item.get("supersedes") == resolved_id
        )
        receipt = MemoryWriteReceipt(
            ok=True,
            operation="promote",
            chars=len(sanitized),
            redacted=redacted,
            promoted=str(stored.get("status", "active")) == "active",
            note_id=resolved_id,
            superseded_count=superseded_count,
            visibility=requested_visibility,
        )
        self._emit(
            "memory_recorded",
            {
                "operation": receipt.operation,
                "source": str(source),
                "topic": topic_text,
                "chars": receipt.chars,
                "redacted": receipt.redacted,
                "quarantined": False,
                "promoted": receipt.promoted,
                "note_id": receipt.note_id,
                "superseded_count": receipt.superseded_count,
                "visibility": receipt.visibility,
            },
        )
        return receipt

    def retrieve(
        self,
        query: object,
        *,
        limit: int = 5,
        include_global: bool = False,
    ) -> RetrievalResult:
        """用脱敏 query 做一次快照检索，并写入 audit-only retrieval 事件。"""

        sanitized_query = self._bounded(self.security.redact_text(str(query or "")).strip())
        result = MemoryRetriever(
            store=self.store,
            global_store=self.global_store,
            workspace_root=self.store.workspace_root,
            session_id=self.session_id,
        ).retrieve(
            MemoryQuery(
                text=sanitized_query,
                limit=max(0, int(limit)),
                session_id=self.session_id,
                include_global=include_global,
            )
        )
        self._emit(
            "memory_retrieved",
            {
                "query": sanitized_query,
                "query_hash": result.query_hash,
                "selected_note_ids": [selection.note.note_id for selection in result.selections if selection.selected],
                "selected_count": len(result.selected_notes),
                "include_global": include_global,
            },
        )
        return result

    def _evidence(self) -> MemoryEvidence:
        # scope 使用契约默认占位值，DurableMemoryStore 会在 workspace 绑定时
        # 保留其真实 fingerprint；不能在这里用字面量覆盖真实 workspace scope。
        return MemoryEvidence(session_id=self.session_id, scope="workspace", visibility="session")

    def _evidence_for_text(self, text: str, *, visibility: MemoryVisibility) -> MemoryEvidence:
        """从当天 sidecar 恢复本 session 的捕获证据，找不到时使用默认值。"""

        try:
            rows = self.store.load_daily_log_evidence()
        except (OSError, ValueError):
            # sidecar 是 provenance 增强信息；损坏或暂时不可读时不应阻断
            # 已经明确要求的 durable promotion，调用方仍保留 session scope。
            return MemoryEvidence(
                session_id=self.session_id,
                scope="workspace",
                visibility=visibility,
            )
        for row in reversed(rows):
            if not isinstance(row, dict):
                continue
            if str(row.get("session_id") or "") != self.session_id:
                continue
            if str(row.get("text") or "").strip() != text:
                continue
            return MemoryEvidence(
                source_path=str(row.get("source_path") or ""),
                session_id=self.session_id,
                anchor_hash=str(row.get("evidence_anchor_hash") or ""),
                scope=str(row.get("scope") or "workspace"),
                visibility=visibility,
            )
        return MemoryEvidence(
            session_id=self.session_id,
            scope="workspace",
            visibility=visibility,
        )

    @staticmethod
    def _normalize_visibility(value: object) -> MemoryVisibility | None:
        """把外部命令/工具参数收敛到三个公开作用域枚举。"""

        normalized = str(value or "").strip().lower()
        if normalized not in MEMORY_VISIBILITIES:
            return None
        return normalized  # type: ignore[return-value]

    def _bounded(self, value: str) -> str:
        if len(value) <= self.max_entry_chars:
            return value
        return value[: self.max_entry_chars]

    @staticmethod
    def _normalize_text(value: object) -> str:
        """把显式输入折叠为 daily log/topic 可安全承载的单行文本。"""

        # durable store 会以同样的单行语义计算 note_id；runtime 先统一规范化，
        # 才能让 receipt、sidecar 和最终 metadata 共享同一个稳定标识。
        return " ".join(str(value or "").split()).strip()

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.audit is None:
            return
        # audit payload 再走一次递归脱敏，防止未来新增字段绕过写入侧的规则。
        self.audit(event_type, self.security.redact_value(payload))


__all__ = ["MemoryAuditCallback", "MemoryRuntime", "MemoryWriteReceipt"]
