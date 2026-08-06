"""P1 slice 2 tests: evidence reducer (evidence.py) and report (report.py)."""

from __future__ import annotations

import pytest

from firstcoder.harness.evidence import (
    CONTINUE_KIND,
    TERMINAL_KIND,
    TRANSITION_SUMMARY_SCHEMA,
    build_transition,
    reduce_transition_summary,
    update_evidence_summaries,
)
from firstcoder.harness.report import REPORT_SCHEMA_VERSION, build_report
from firstcoder.harness.task_state import TaskState


def test_build_transition_counts_only_truthy_fields() -> None:
    t = build_transition(kind=CONTINUE_KIND, reason="tool_batch_executed", attempt_index=1)
    assert "tool_call_count" not in t
    t2 = build_transition(
        kind=CONTINUE_KIND,
        reason="tool_batch_executed",
        attempt_index=1,
        tool_executed_count=3,
        stop_reason="step_limit_reached",
    )
    assert t2["tool_executed_count"] == 3
    assert t2["stop_reason"] == "step_limit_reached"


def test_reduce_transition_summary_continue_then_terminal() -> None:
    summary = reduce_transition_summary(
        None,
        build_transition(kind=CONTINUE_KIND, reason="provider_retry", attempt_index=0, tool_requested_count=1),
    )
    summary = reduce_transition_summary(
        summary,
        build_transition(kind=CONTINUE_KIND, reason="tool_batch_executed", attempt_index=2, tool_executed_count=2),
    )
    assert summary["schema_version"] == TRANSITION_SUMMARY_SCHEMA
    assert summary["continue_count"] == 2
    assert summary["max_attempt_index"] == 2
    assert summary["reasons"] == {"provider_retry": 1, "tool_batch_executed": 1}
    assert summary["tool_requested_count"] == 1
    assert summary["tool_executed_count"] == 2
    assert summary.get("terminal_count") == 0

    summary = reduce_transition_summary(
        summary,
        build_transition(
            kind=TERMINAL_KIND,
            reason="final_answer_returned",
            attempt_index=3,
            stop_reason="final_answer_returned",
        ),
    )
    assert summary["terminal_count"] == 1
    assert summary["terminal_reason"] == "final_answer_returned"
    assert summary["continue_count"] == 2


def test_reduce_transition_summary_rejects_double_terminal() -> None:
    summary = reduce_transition_summary(None, build_transition(kind=TERMINAL_KIND, reason="aborted", attempt_index=0))
    with pytest.raises(ValueError, match="already has a terminal transition"):
        reduce_transition_summary(summary, build_transition(kind=TERMINAL_KIND, reason="aborted", attempt_index=1))


def test_update_evidence_summaries_dispatch() -> None:
    summaries = update_evidence_summaries(
        None,
        {
            "event": "loop_transition",
            "kind": TERMINAL_KIND,
            "reason": "final_answer_returned",
            "attempt_index": 0,
        },
    )
    assert summaries["transition_summary"]["terminal_count"] == 1
    # P4 叶子就位前，未处理事件原样透传。
    assert update_evidence_summaries(summaries, {"event": "tool_executed"}) == summaries


def test_build_report_field_contract() -> None:
    state = TaskState.create(
        "t1", "request", run_id="run_r", session_id="s1", workspace_fingerprint="fp"
    )
    state.record_attempt()
    state.record_tool("write")
    state.changed_paths.append("src/a.py")
    state.finish_success("done")

    report = build_report(
        state,
        prompt_metadata={"context_usage": {"total_estimated_tokens": 123}},
        compactions=[{"kind": "task_boundary"}],
        redacted_env={"secret_env_count": 0},
    )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["run_id"] == "run_r"
    assert report["session_id"] == "s1"
    assert report["status"] == "completed"
    assert report["stop_reason"] == "final_answer_returned"
    assert report["tool_steps"] == 1
    assert report["attempts"] == 1
    assert report["changed_paths"] == ["src/a.py"]
    assert report["prompt_metadata"]["context_usage"]["total_estimated_tokens"] == 123
    assert report["compactions"][0]["kind"] == "task_boundary"
    assert report["redacted_env"]["secret_env_count"] == 0
    assert report["verifier_suggestions"] == []
    assert report["created_at"].endswith("Z")
