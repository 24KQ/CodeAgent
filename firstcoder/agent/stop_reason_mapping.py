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
- provider 不支持合成响应 / provider 异常（finish_reason="error" 或 error_type="provider"）-> model_error
- 用户输入等待：只有权限确认的最终结果是明确拒绝才 -> approval_denied；
  ask_user 与等待决定中都不映射（非终局，Codex P1 review fix）
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
    wait_kind: str = "",
    permission_denied: bool = False,
) -> str:
    """把一轮 loop 的结果映射为 STOP_REASON_*（§7.3 映射表）。

    `error_type` 是接线层传入的失败分类（"persistence" / "resume" /
    "provider"）。`wait_kind` 区分 user_input 的两种等待（"ask_user" /
    "permission_confirmation"）；`permission_denied` 表示权限确认的最终
    结果是拒绝——只有明确拒绝才映射 approval_denied，等待用户决定不算
    终局（Codex P1 review fix）。

    失败路径也走 finish_reason 映射（不限于 completed 状态）：真实
    provider 异常在 loop 里直接抛出，P5 接线时以 error_type="provider"
    或 finish_reason="error" 落到这里。
    """
    if error_type == "persistence":
        return STOP_REASON_PERSISTENCE_ERROR
    if error_type == "resume":
        return STOP_REASON_RESUME_LOAD_ERROR
    if error_type == "provider":
        return STOP_REASON_MODEL_ERROR
    # finish_reason 映射优先于状态分支：任意状态（含 failed / waiting）下
    # limit / timeout / interrupted / cancelled / error 都一致映射
    # （Codex P1 review fix 复验）。
    mapped = map_finish_reason(finish_reason)
    if mapped:
        return mapped
    if status == AgentTurnStatus.WAITING_FOR_USER_INPUT.value:
        if wait_kind == "permission_confirmation" and permission_denied:
            return STOP_REASON_APPROVAL_DENIED
        return ""
    if status == AgentTurnStatus.COMPLETED.value:
        return STOP_REASON_FINAL_ANSWER_RETURNED
    return ""
