"""Reduce trace events into TaskState evidence summaries (fusion P1, H9).

Ported from pico `core/evidence_summaries.py` + `core/turn_transitions.py`.
The reducer is the bridge from append-only trace facts to compact
report-ready state; it consumes events as they are emitted during a run
rather than re-reading trace files.

P1 implements the loop-transition leaf. The context-budget (prompt_built /
context_orchestrator_decision), verification-signal (tool_executed) and
final-readiness (final_readiness_decision) leaves arrive in P4 with the
FirstCoder field-name mapping; unhandled events pass through unchanged.
"""

from __future__ import annotations

from typing import Any

from firstcoder.context.budget_summary import (
    context_budget_summary,
    update_from_orchestrator,
)
from firstcoder.harness.final_readiness import reduce_final_readiness_summary
from firstcoder.harness.governance import reduce_governance_summary
from firstcoder.harness.verification import reduce_verification_signal

CONTINUE_KIND = "continue"
TERMINAL_KIND = "terminal"
TRANSITION_SUMMARY_SCHEMA = "firstcoder.transition_summary.v1"

# prompt_built 描述的是下一次请求的最新预算，但它本身没有携带上一次 L4
# 压缩的事实。报告需要保留本 run 已经发生过的压缩证据，不能因后续预算事件
# 的 reducer 更新而把 compact_call_usage 清成 None。
_COMPACTION_SUMMARY_FIELDS = (
    "reductions",
    "summary_called",
    "summary_mode",
    "summary_delta_event_count",
    "compact_call_usage",
    "compact_net_benefit_tokens",
    "compact_summary_has_next_steps",
    "compact_summary_has_file_references",
    "pre_compact_estimated_tokens",
    "post_compact_estimated_tokens",
    "replacement_cache_hits",
    "replacement_records_created",
    "replacement_ledger_enabled",
    "saved_chars",
)


def build_transition(
    *,
    kind: str,
    reason: str,
    attempt_index: int,
    tool_call_count: int = 0,
    tool_requested_count: int = 0,
    tool_executed_count: int = 0,
    stop_reason: str = "",
) -> dict[str, Any]:
    payload = {
        "kind": str(kind),
        "reason": str(reason),
        "attempt_index": int(attempt_index),
    }
    for key, value in {
        "tool_call_count": tool_call_count,
        "tool_requested_count": tool_requested_count,
        "tool_executed_count": tool_executed_count,
    }.items():
        if value:
            payload[key] = int(value)
    if stop_reason:
        payload["stop_reason"] = str(stop_reason)
    return payload


def reduce_transition_summary(
    summary: dict[str, Any] | None,
    transition: dict[str, Any],
) -> dict[str, Any]:
    """Fold one loop-transition event into the transition summary.

    Invariant (pico): a run has exactly one terminal transition — a second
    one raises ValueError so double-stop wiring is caught at the source.
    """
    summary = dict(summary or {})
    summary.setdefault("schema_version", TRANSITION_SUMMARY_SCHEMA)
    kind = str(transition.get("kind", ""))
    reason = str(transition.get("reason", ""))
    reasons = dict(summary.get("reasons", {}) or {})
    reasons[reason] = reasons.get(reason, 0) + 1
    summary["reasons"] = reasons
    summary["max_attempt_index"] = max(
        int(summary.get("max_attempt_index", 0) or 0),
        int(transition.get("attempt_index", 0) or 0),
    )
    if kind == CONTINUE_KIND:
        summary["continue_count"] = int(summary.get("continue_count", 0) or 0) + 1
        summary.setdefault("terminal_count", 0)
        summary["tool_requested_count"] = (
            int(summary.get("tool_requested_count", 0) or 0)
            + int(transition.get("tool_requested_count", 0) or 0)
        )
        summary["tool_executed_count"] = (
            int(summary.get("tool_executed_count", 0) or 0)
            + int(transition.get("tool_executed_count", 0) or 0)
        )
        return summary
    if kind == TERMINAL_KIND:
        if int(summary.get("terminal_count", 0) or 0) >= 1:
            raise ValueError("run already has a terminal transition")
        summary["terminal_count"] = 1
        summary.setdefault("continue_count", 0)
        summary["terminal_reason"] = str(transition.get("stop_reason") or transition.get("reason") or "")
        return summary
    return summary


def update_evidence_summaries(
    summaries: dict[str, Any] | None,
    event: dict[str, Any],
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Fold one trace event into the evidence summaries dict (H9)."""
    summaries = dict(summaries or {})
    event_name = event.get("event")
    if event_name == "loop_transition":
        summaries["transition_summary"] = reduce_transition_summary(
            summaries.get("transition_summary", {}), event
        )
    elif event_name == "prompt_built":
        current = context_budget_summary(event.get("prompt_metadata", {}))
        _preserve_compaction_summary(
            current,
            summaries.get("context_budget_summary", {}),
        )
        summaries["context_budget_summary"] = current
    elif event_name == "context_orchestrator_decision":
        summaries["context_budget_summary"] = update_from_orchestrator(
            summaries.get("context_budget_summary", {}), event
        )
    elif event_name == "governance_decision":
        summaries["governance_summary"] = reduce_governance_summary(
            summaries.get("governance_summary", {}), event
        )
    elif event_name == "tool_executed":
        paths = list(
            changed_paths
            or event.get("changed_paths")
            or event.get("affected_paths")
            or []
        )
        previous_signal = summaries.get("verification_signal", {})
        verification_signal = reduce_verification_signal(previous_signal, event, paths)
        # 没有诊断工具名和可识别命令的普通 tool event 不制造空 evidence leaf；
        # 这样旧调用方仍保持“未处理事件原样透传”的 P1 契约。
        if verification_signal != previous_signal:
            summaries["verification_signal"] = verification_signal
    elif event_name == "final_readiness_decision":
        summaries["final_readiness_summary"] = reduce_final_readiness_summary(
            summaries.get("final_readiness_summary", {}), event
        )
    return summaries


def _preserve_compaction_summary(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> None:
    """在更新最新预算时保留本 run 的已有压缩证据。"""

    previous = dict(previous or {})
    for key in _COMPACTION_SUMMARY_FIELDS:
        current_value = current.get(key)
        previous_value = previous.get(key)
        if current_value in (None, "", 0, False, []) and previous_value not in (None, "", 0, False, []):
            current[key] = previous_value
