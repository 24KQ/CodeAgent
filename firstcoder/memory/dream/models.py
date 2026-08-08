"""P6 memory maintenance 的结构化输入输出模型。

这些模型把 provider 输出和 durable 写入隔开：runner 可以提出候选，scheduler
负责验证候选并选择 workspace store 提交。模型本身不包含文件写入逻辑，也不保存
原始 prompt 或用户会话 transcript 的副本。
"""

from __future__ import annotations

from dataclasses import dataclass

from firstcoder.memory.models import MemoryNote, MemoryVisibility


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceEntry:
    """一条供 dream prompt 使用的 daily-log 证据摘要。"""

    text: str
    session_id: str = ""
    source_path: str = ""
    anchor_hash: str = ""
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceSnapshot:
    """一次一致性读取得到的维护输入。

    ``notes`` 来自 ``DurableMemoryStore.snapshot``；``entries`` 只包含 gate 选中
    session 的 daily-log 内容。两者在交给 runner 前都会经过字符上限和脱敏处理。
    """

    snapshot_id: str
    index_version: int
    session_ids: tuple[str, ...] = ()
    notes: tuple[MemoryNote, ...] = ()
    entries: tuple[MemoryMaintenanceEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class DreamCandidate:
    """runner 提出的一个 workspace durable 候选。"""

    topic: str
    text: str
    source_path: str = ""
    session_id: str = ""
    reason: str = ""
    visibility: MemoryVisibility = "workspace"


@dataclass(frozen=True, slots=True)
class DreamRejection:
    """runner 或安全校验拒绝的候选，只保留稳定原因和可选 id。"""

    reason: str
    candidate_id: str = ""


@dataclass(frozen=True, slots=True)
class DreamProposal:
    """一次 bounded runner 的结构化结果。"""

    candidates: tuple[DreamCandidate, ...] = ()
    rejections: tuple[DreamRejection, ...] = ()
    relative_dates_absolutized: int = 0


__all__ = [
    "DreamCandidate",
    "DreamProposal",
    "DreamRejection",
    "MemoryMaintenanceEntry",
    "MemoryMaintenanceSnapshot",
]
