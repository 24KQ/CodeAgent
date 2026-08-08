"""P6 maintenance task 的持久状态和恢复原语。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from firstcoder.memory.paths import ensure_no_link_or_junction
from firstcoder.memory.write import (
    atomic_write_text,
    cross_process_lock,
    try_cross_process_lock,
)

MaintenanceStatus = Literal["pending", "running", "succeeded", "failed"]
STATE_SCHEMA_VERSION = 1


class MaintenanceStateError(ValueError):
    """维护状态文件不存在或无法安全解析。"""


@dataclass(frozen=True, slots=True)
class DreamTaskState:
    """一个可跨进程恢复的 dream task 状态快照。"""

    task_id: str
    status: MaintenanceStatus
    workspace_fingerprint: str
    trigger_session_id: str = ""
    session_ids: tuple[str, ...] = ()
    snapshot_id: str = ""
    attempt: int = 0
    recovery_count: int = 0
    created_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    last_success_at: float | None = None
    report_path: str = ""
    error_code: str = ""
    error_message: str = ""
    schema_version: int = STATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """转换为稳定 JSON 形状；tuple 会由 json 编码为数组。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> DreamTaskState:
        """从磁盘数据恢复状态，并在遇到未知 schema/status 时 fail closed。"""

        if not isinstance(raw, dict):
            raise MaintenanceStateError("maintenance state must be an object")
        schema_version = raw.get("schema_version", STATE_SCHEMA_VERSION)
        if schema_version != STATE_SCHEMA_VERSION:
            raise MaintenanceStateError(f"unsupported maintenance state schema: {schema_version!r}")
        status = raw.get("status")
        if status not in {"pending", "running", "succeeded", "failed"}:
            raise MaintenanceStateError(f"unsupported maintenance status: {status!r}")
        task_id = str(raw.get("task_id") or "").strip()
        workspace_fingerprint = str(raw.get("workspace_fingerprint") or "").strip()
        if not task_id or not workspace_fingerprint:
            raise MaintenanceStateError("maintenance state requires task_id and workspace_fingerprint")
        session_ids = raw.get("session_ids", [])
        if not isinstance(session_ids, list) or not all(isinstance(item, str) for item in session_ids):
            raise MaintenanceStateError("maintenance state session_ids must be a string list")
        return cls(
            task_id=task_id,
            status=status,
            workspace_fingerprint=workspace_fingerprint,
            trigger_session_id=str(raw.get("trigger_session_id") or ""),
            session_ids=tuple(session_ids),
            snapshot_id=str(raw.get("snapshot_id") or ""),
            attempt=_non_negative_int(raw.get("attempt", 0), "attempt"),
            recovery_count=_non_negative_int(raw.get("recovery_count", 0), "recovery_count"),
            created_at=float(raw.get("created_at", 0.0)),
            started_at=_optional_float(raw.get("started_at")),
            completed_at=_optional_float(raw.get("completed_at")),
            last_success_at=_optional_float(raw.get("last_success_at")),
            report_path=str(raw.get("report_path") or ""),
            error_code=str(raw.get("error_code") or ""),
            error_message=str(raw.get("error_message") or ""),
            schema_version=STATE_SCHEMA_VERSION,
        )


class MaintenanceStateStore:
    """在 dream 目录中以锁保护、原子写入维护状态。"""

    def __init__(self, root: str | Path, *, clock: Callable[[], float] | None = None) -> None:
        self.root = Path(root)
        self.state_path = self.root / "maintenance_state.json"
        self.lock_path = self.root / ".scheduler.lock"
        self._clock = clock or time.time

    @contextmanager
    def locked(self) -> Iterator[None]:
        """给 scheduler 组合多个状态读写动作时使用的锁边界。"""

        self._ensure_root()
        with cross_process_lock(self.lock_path):
            yield

    @contextmanager
    def try_locked(self) -> Iterator[bool]:
        """非阻塞取得 scheduler 锁，供普通 turn/启动恢复路径使用。"""

        self._ensure_root()
        with try_cross_process_lock(self.lock_path) as acquired:
            yield acquired

    def load(self) -> DreamTaskState | None:
        """持锁读取当前状态；首次运行没有状态文件时返回 ``None``。"""

        self._ensure_root()
        with cross_process_lock(self.lock_path):
            return self.load_unlocked()

    def load_unlocked(self) -> DreamTaskState | None:
        """在调用方已经持有 scheduler lock 时读取状态。"""

        if not self.state_path.exists():
            return None
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MaintenanceStateError(f"failed to read maintenance state: {exc}") from exc
        return DreamTaskState.from_dict(raw)

    def save(self, state: DreamTaskState) -> None:
        """以 temp+replace 原子替换状态文件。"""

        self._ensure_root()
        with cross_process_lock(self.lock_path):
            self.save_unlocked(state)

    def save_unlocked(self, state: DreamTaskState) -> None:
        """在调用方已经持有 scheduler lock 时写入状态。"""

        if state.schema_version != STATE_SCHEMA_VERSION:
            raise MaintenanceStateError("cannot write an unknown maintenance state schema")
        payload = json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        atomic_write_text(self.state_path, payload)

    def now(self) -> float:
        """返回注入的时钟值，便于 scheduler 测试恢复边界。"""

        return float(self._clock())

    def _ensure_root(self) -> None:
        # dream 目录属于 workspace memory root；预置链接必须在创建/替换前被拒绝，
        # 否则状态和 audit 也可能被导向 workspace 外部。
        if self.root.exists():
            ensure_no_link_or_junction(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        ensure_no_link_or_junction(self.root)
        if self.lock_path.exists():
            ensure_no_link_or_junction(self.lock_path)
        if self.state_path.exists():
            ensure_no_link_or_junction(self.state_path)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MaintenanceStateError(f"maintenance state {name} must be a non-negative integer")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MaintenanceStateError("maintenance state timestamp must be numeric") from exc


__all__ = [
    "STATE_SCHEMA_VERSION",
    "DreamTaskState",
    "MaintenanceStateError",
    "MaintenanceStateStore",
    "MaintenanceStatus",
]
