"""P1 slice 1 tests: TaskState contract (task_state.py)."""

from __future__ import annotations

from firstcoder.harness.task_state import (
    STOP_REASON_FINAL_ANSWER_RETURNED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    TASK_STATE_SCHEMA_VERSION,
    TaskState,
)


def test_create_defaults() -> None:
    state = TaskState.create(task_id="t1", user_request="do the thing", session_id="s1")
    assert state.status == STATUS_RUNNING
    assert state.session_id == "s1"
    assert state.schema_version == TASK_STATE_SCHEMA_VERSION
    assert state.stop_reason == ""
    assert state.run_id.startswith("run_")
    assert state.changed_paths == []


def test_create_accepts_explicit_run_id() -> None:
    state = TaskState.create(task_id="t1", user_request="x", run_id="run_fixed")
    assert state.run_id == "run_fixed"


def test_record_attempt_and_tool() -> None:
    state = TaskState.create("t1", "x")
    state.record_attempt().record_attempt()
    state.record_tool("bash")
    assert state.attempts == 2
    assert state.tool_steps == 1
    assert state.last_tool == "bash"


def test_finish_success() -> None:
    state = TaskState.create("t1", "x")
    state.finish_success("done")
    assert state.status == STATUS_COMPLETED
    assert state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
    assert state.final_answer == "done"


def test_stop_sets_reason_and_status() -> None:
    state = TaskState.create("t1", "x")
    state.stop("step_limit_reached")
    assert state.status == STATUS_STOPPED
    assert state.stop_reason == "step_limit_reached"
    state.stop("model_error", status="failed", final_answer="boom")
    assert state.status == "failed"
    assert state.final_answer == "boom"


def test_to_dict_from_dict_round_trip() -> None:
    state = TaskState.create(
        "t1",
        "user request",
        run_id="run_1",
        session_id="s9",
        workspace_fingerprint="fp-abc",
        parent_run_id="run_0",
    )
    state.record_attempt()
    state.record_tool("write")
    state.changed_paths.append("src/a.py")
    state.evidence_summaries["transition_summary"] = {"terminal_count": 1}
    state.finish_success("all good")

    restored = TaskState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    assert restored.schema_version == TASK_STATE_SCHEMA_VERSION
    assert restored.workspace_fingerprint == "fp-abc"
    assert restored.parent_run_id == "run_0"
