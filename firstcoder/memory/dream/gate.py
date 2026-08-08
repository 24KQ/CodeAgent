"""Auto-dream 的纯门控逻辑。

Pico 的原实现把当前时间、目录扫描和配置读取混在一起，难以在重启恢复和测试中
复用。FirstCoder 把它收敛成两个公开函数：``list_sessions_since`` 只读取 session
文件元数据，``evaluate_auto_dream_gate`` 只根据注入的时间和门槛作决定。这样
scheduler 可以在持锁前先完成纯判断，测试也不需要真实等待时间流逝。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DreamGateResult:
    """一次 auto-dream gate 的稳定结果。

    ``session_ids`` 是门控使用的完整候选集合，包含排序后的稳定 session id；
    不保存 transcript 内容，避免 gate 结果变成新的敏感数据副本。
    """

    should_run: bool
    skip_reason: str
    session_ids: tuple[str, ...]
    session_count: int
    last_success_at: float
    hours_since: float


def list_sessions_since(
    sessions_dir: str | Path | None,
    since: float,
    *,
    current_session_id: str = "",
) -> tuple[str, ...]:
    """返回 ``since`` 之后被修改过的 session JSONL 文件名。

    FirstCoder 的会话文件位于 ``<data-root>/sessions``，而不是 data-root 本身。
    这里只读取普通 ``.jsonl`` 文件的 mtime；文件内容不会被 gate 解析，因此损坏
    的 session 不会被误当成可维护内容。当前 session 必须排除，否则每一轮自身
    写入都会让 session gate 人为满足。
    """

    if sessions_dir is None:
        return ()
    directory = Path(sessions_dir)
    if not directory.is_dir():
        return ()
    threshold = float(since)
    current = str(current_session_id or "")
    candidates: list[tuple[float, str]] = []
    for path in directory.iterdir():
        if path.suffix != ".jsonl" or not path.is_file():
            continue
        session_id = path.stem
        if current and session_id == current:
            continue
        try:
            modified_at = path.stat().st_mtime
            if modified_at > threshold:
                candidates.append((modified_at, session_id))
        except OSError:
            # gate 只是触发条件，不应因为一个刚被删除或暂时不可读的文件阻断
            # 正常会话；下一轮仍会重新扫描该目录。
            continue
    # session id 是随机标识符，字母序与新鲜度没有关系；按 mtime 保留稳定的
    # 时间顺序，scheduler 在超出 max_sessions 时才能真正取到最近的 session。
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple(session_id for _, session_id in candidates)


def evaluate_auto_dream_gate(
    *,
    last_success_at: float,
    sessions_dir: str | Path | None,
    current_session_id: str = "",
    min_interval_hours: float = 24.0,
    min_sessions: int = 3,
    now: float | None = None,
) -> DreamGateResult:
    """评估 interval + session 双门槛。

    两个门槛必须同时满足。首次维护没有成功时间戳时，时间门槛视为满足；但仍然
    必须有足够数量的其他 session。``now`` 是可选注入值，生产调用方传当前
    POSIX 时间，测试调用方传固定值即可。
    """

    if isinstance(min_interval_hours, bool) or float(min_interval_hours) < 0:
        raise ValueError("min_interval_hours must be non-negative")
    if isinstance(min_sessions, bool) or not isinstance(min_sessions, int) or min_sessions < 0:
        raise ValueError("min_sessions must be a non-negative integer")

    current_time = float(now if now is not None else time.time())
    last = float(last_success_at or 0.0)
    hours_since = (current_time - last) / 3600 if last > 0 else float("inf")
    session_ids = list_sessions_since(
        sessions_dir,
        last,
        current_session_id=current_session_id,
    )
    result = DreamGateResult(
        should_run=False,
        skip_reason="",
        session_ids=session_ids,
        session_count=len(session_ids),
        last_success_at=last,
        hours_since=hours_since,
    )
    if hours_since < float(min_interval_hours):
        return replace(result, skip_reason="interval_gate")
    if len(session_ids) < min_sessions:
        return replace(result, skip_reason="session_gate")
    return replace(result, should_run=True)
