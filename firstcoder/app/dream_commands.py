"""受限 memory maintenance 的 ``/dream`` slash command。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from firstcoder.app.commands import CommandResult
from firstcoder.memory.dream.scheduler import ScheduleResult


class DreamSessionLike(Protocol):
    """命令只依赖当前 session id，不把 session transcript 传给 scheduler。"""

    session_id: str


class DreamSchedulerLike(Protocol):
    """scheduler 的手动触发窄接口，便于 TUI/CLI 测试使用 fake。"""

    def request_run(self, current_session_id: str) -> ScheduleResult: ...


@dataclass(slots=True)
class DreamCommandHandler:
    """把 ``/dream`` 映射为非阻塞维护请求。

    维护结果写入 memory store 的 dream 目录；命令输出只是一条 UI 反馈，绝不
    通过 ``AgentSession`` 追加 user/assistant 消息，因此不会污染可 resume 的
    普通会话历史。
    """

    session: DreamSessionLike
    scheduler: DreamSchedulerLike

    def handle(self, text: str) -> CommandResult:
        command = " ".join(str(text or "").strip().split())
        if command != "/dream":
            if command.startswith("/dream "):
                return CommandResult(handled=True, output="Usage: /dream")
            return CommandResult(handled=False)

        try:
            result = self.scheduler.request_run(self.session.session_id)
        except Exception:  # noqa: BLE001 - 命令边界只返回稳定的用户提示
            return CommandResult(handled=True, output="Dream maintenance could not be scheduled.")
        return CommandResult(handled=True, output=_render_schedule_result(result))


def _render_schedule_result(result: ScheduleResult) -> str:
    """将 scheduler 状态压成不含 prompt/note 正文的稳定反馈。"""

    if result.status == "scheduled":
        return f"Dream maintenance scheduled: {result.task_id}"
    if result.status == "running":
        return f"Dream maintenance already running: {result.task_id}"
    if result.status == "busy":
        return "Dream maintenance is already running in another process."
    if result.status == "closed":
        return "Dream maintenance is shutting down."
    if result.status == "disabled":
        # 手动入口理论上绕过 auto_dream 开关；保留这个分支处理注入 fake 或
        # 未来策略拒绝的情形，避免把内部枚举直接暴露给用户。
        return "Dream maintenance is unavailable."
    if result.status == "skipped":
        return f"Dream maintenance skipped: {result.skip_reason or 'gate'}"
    return "Dream maintenance failed to start."


__all__ = ["DreamCommandHandler"]
