"""FirstCoder loop outcome -> harness STOP_REASON 映射（fusion P1，§7.3）。

评审文档 §7.3 的 stop_reason 映射表落地为纯函数。输入是 AgentLoop 现存
的收敛信号（AgentTurnStatus + ChatResponse.finish_reason + error_type），
输出是 harness TaskState 的 STOP_REASON_* 枚举。映射是投影，不改 loop
行为；RunRecorder 接线（AgentLoop 注入）在 P5（H6 步骤③）。

映射表（§7.3）：
- 正常最终回答 -> final_answer_returned
- tool round / provider call limit -> step_limit_reached（保留原始原因）
- turn timeout -> tool_timeout
- interrupted / cancelled -> interrupted / cancelled（pico 枚举扩展）
- provider 不支持合成响应（finish_reason="error"）-> model_error
- waiting for user input（权限等待）-> approval_denied
- persistence / resume 失败 -> persistence_error / resume_load_error
"""

from __future__ import annotations

from firstcoder.agent.loop_limits import AgentLoopStopReason
from firstcoder.agent.user_input import AgentTurnStatus
from firstcoder.harness.task_state import (
    STOP_REASON_APPROVAL_DENIED,
    STOP_REASON_CANCELLED,
    STOP_REASON_FINAL_ANSWER_RETURNED,
    STOP_REASON_INTERRUPTED,
    STOP_REASON_MODEL_ERROR,
    STOP_REASON_PERSISTENCE_ERROR,
    STOP_REASON_RESUME_LOAD_ERROR,
    STOP_REASON_STEP_LIMIT_REACHED,
    STOP_REASON_TOOL_TIMEOUT,
)

#: loop 现在把 cancel 与 interrupt 都折叠成 "interrupted"；
#: CANCELLED 保留给 P5 接线里能区分两者的场景。
_FINISH_REASON_TO_STOP_REASON = {
    AgentLoopStopReason.TOOL_ROUND_LIMIT.value: STOP_REASON_STEP_LIMIT_REACHED,
    AgentLoopStopReason.PROVIDER_CALL_LIMIT.value: STOP_REASON_STEP_LIMIT_REACHED,
    AgentLoopStopReason.TURN_TIMEOUT.value: STOP_REASON_TOOL_TIMEOUT,
    "interrupted": STOP_REASON_INTERRUPTED,
    "cancelled": STOP_REASON_CANCELLED,
    "error": STOP_REASON_MODEL_ERROR,
}


def map_finish_reason(finish_reason: str | None) -> str:
    """把 ChatResponse.finish_reason 映射为 STOP_REASON_*。

    provider 原生 finish_reason（"stop" / "length" 等）不是停止原因，
    返回空串，由外层按状态判定成功路径。
    """
    if finish_reason is None:
        return ""
    return _FINISH_REASON_TO_STOP_REASON.get(str(finish_reason), "")


def map_turn_outcome(
    *,
    status: str,
    finish_reason: str | None = None,
    error_type: str = "",
) -> str:
    """把一轮 loop 的结果映射为 STOP_REASON_*（§7.3 映射表）。

    `error_type` 是接线层传入的失败分类（"persistence" / "resume"）。
    """
    if error_type == "persistence":
        return STOP_REASON_PERSISTENCE_ERROR
    if error_type == "resume":
        return STOP_REASON_RESUME_LOAD_ERROR
    if status == AgentTurnStatus.WAITING_FOR_USER_INPUT.value:
        return STOP_REASON_APPROVAL_DENIED
    if status == AgentTurnStatus.COMPLETED.value:
        mapped = map_finish_reason(finish_reason)
        if mapped:
            return mapped
        return STOP_REASON_FINAL_ANSWER_RETURNED
    return ""
