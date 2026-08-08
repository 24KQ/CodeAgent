"""P6 ``/dream`` command 的行为边界测试。"""

from __future__ import annotations

from dataclasses import dataclass

from firstcoder.app.dream_commands import DreamCommandHandler
from firstcoder.memory.dream.scheduler import ScheduleResult


@dataclass
class _Session:
    session_id: str = "sess_current"


class _Scheduler:
    def __init__(self, result: ScheduleResult) -> None:
        self.result = result
        self.calls: list[str] = []

    def request_run(self, current_session_id: str) -> ScheduleResult:
        self.calls.append(current_session_id)
        return self.result


def test_dream_command_uses_current_session_and_does_not_accept_prompt_text() -> None:
    """手动 dream 只触发 scheduler，不能把用户输入伪装成维护 prompt。"""

    scheduler = _Scheduler(ScheduleResult(status="scheduled", task_id="dream_task"))
    handler = DreamCommandHandler(session=_Session(), scheduler=scheduler)

    result = handler.handle("/dream")
    rejected = handler.handle("/dream consolidate this")

    assert result.handled is True
    assert result.output == "Dream maintenance scheduled: dream_task"
    assert scheduler.calls == ["sess_current"]
    assert rejected.output == "Usage: /dream"


def test_dream_command_renders_skip_and_failure_without_internal_exception() -> None:
    """gate/失败状态只显示稳定原因，不暴露路径或异常正文。"""

    skipped = DreamCommandHandler(
        session=_Session(),
        scheduler=_Scheduler(ScheduleResult(status="skipped", skip_reason="session_gate")),
    ).handle("/dream")
    failed = DreamCommandHandler(
        session=_Session(),
        scheduler=_Scheduler(ScheduleResult(status="failed", skip_reason="submit_failed")),
    ).handle("/dream")

    assert skipped.output == "Dream maintenance skipped: session_gate"
    assert failed.output == "Dream maintenance failed to start."
