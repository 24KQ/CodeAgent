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

CONTINUE_KIND = "continue"
TERMINAL_KIND = "terminal"
TRANSITION_SUMMARY_SCHEMA = "firstcoder.transition_summary.v1"


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
    if event.get("event") == "loop_transition":
        summaries["transition_summary"] = reduce_transition_summary(
            summaries.get("transition_summary", {}), event
        )
    # P4 leaves: prompt_built / context_orchestrator_decision -> context_budget_summary,
    # tool_executed -> verification_signal, final_readiness_decision -> final_readiness_summary.
    return summaries
