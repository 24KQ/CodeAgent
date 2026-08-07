"""Reducer for governance decisions recorded in the run trace.

治理策略仍由 FirstCoder 权限系统负责；本模块只统计 allow/warn/deny 结果，供
final-readiness 和报告使用，避免 harness 反向控制工具执行。
"""

from __future__ import annotations

from typing import Any

GOVERNANCE_SUMMARY_SCHEMA = "firstcoder.governance_summary.v1"


def reduce_governance_summary(
    summary: dict[str, Any] | None,
    event: dict[str, Any],
) -> dict[str, Any]:
    """把一个 governance_decision 事件折叠为计数和最近原因。"""

    summary = dict(summary or {})
    summary.setdefault("schema_version", GOVERNANCE_SUMMARY_SCHEMA)
    decision = str(event.get("decision", ""))
    reason = str(event.get("reason_code") or event.get("reason") or "")
    decision_type = str(event.get("decision_type", ""))
    key = f"{decision}_count"
    summary[key] = int(summary.get(key, 0) or 0) + 1
    for missing in ("allow_count", "deny_count", "warn_count"):
        summary.setdefault(missing, 0)
    type_counts = dict(summary.get("decision_type_counts", {}) or {})
    type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
    summary["decision_type_counts"] = type_counts
    reasons = dict(summary.get("reasons", {}) or {})
    reasons[reason] = reasons.get(reason, 0) + 1
    summary["reasons"] = reasons
    if decision == "deny":
        summary["last_denied_reason"] = reason
    return summary
