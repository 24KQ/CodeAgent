"""Run report field contract (fusion P1, H4).

Ported from pico `core/runtime.py` build_report (:831-858). Fields are
defined as a data contract, not pulled live off the agent: the report is
one run's terminal summary, assembled from the TaskState plus a few
runtime-owned extras, and written atomically by the RunStore.
"""

from __future__ import annotations

from typing import Any

from firstcoder.harness.task_state import TaskState
from firstcoder.harness.trace import now_iso

REPORT_SCHEMA_VERSION = 1


def build_report(
    task_state: TaskState,
    *,
    prompt_metadata: dict[str, Any] | None = None,
    compactions: list[dict[str, Any]] | None = None,
    redacted_env: dict[str, Any] | None = None,
    verifier_suggestions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the terminal report for a run (H4 field contract)."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": now_iso(),
        "run_id": task_state.run_id,
        "task_id": task_state.task_id,
        "session_id": task_state.session_id,
        "status": task_state.status,
        "stop_reason": task_state.stop_reason,
        "final_answer": task_state.final_answer,
        "tool_steps": task_state.tool_steps,
        "attempts": task_state.attempts,
        "checkpoint_id": task_state.checkpoint_id,
        "changed_paths": list(task_state.changed_paths),
        "artifact_graph": dict(task_state.artifact_graph),
        "evidence_summaries": dict(task_state.evidence_summaries),
        "prompt_metadata": dict(prompt_metadata or {}),
        "compactions": list(compactions or []),
        "redacted_env": dict(redacted_env or {}),
        # P4 将验证建议作为报告数据保存；不在 report builder 内执行命令。
        "verifier_suggestions": list(
            verifier_suggestions
            if verifier_suggestions is not None
            else getattr(task_state, "verifier_suggestions", [])
        ),
        "harness_degraded": bool(getattr(task_state, "harness_degraded", False)),
        "harness_degradation_reason": str(
            getattr(task_state, "harness_degradation_reason", "") or ""
        ),
    }
