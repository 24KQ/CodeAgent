"""P1 slice 4 tests: stop_reason mapping (stop_reason_mapping.py)."""

from __future__ import annotations

import pytest

from firstcoder.agent.loop_limits import AgentLoopStopReason
from firstcoder.agent.stop_reason_mapping import map_finish_reason, map_turn_outcome
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


def test_finish_reason_limits() -> None:
    assert map_finish_reason(AgentLoopStopReason.TOOL_ROUND_LIMIT.value) == STOP_REASON_STEP_LIMIT_REACHED
    assert map_finish_reason(AgentLoopStopReason.PROVIDER_CALL_LIMIT.value) == STOP_REASON_STEP_LIMIT_REACHED
    assert map_finish_reason(AgentLoopStopReason.TURN_TIMEOUT.value) == STOP_REASON_TOOL_TIMEOUT


def test_finish_reason_interrupt_and_error() -> None:
    assert map_finish_reason("interrupted") == STOP_REASON_INTERRUPTED
    assert map_finish_reason("cancelled") == STOP_REASON_CANCELLED
    assert map_finish_reason("error") == STOP_REASON_MODEL_ERROR


def test_finish_reason_provider_native_is_empty() -> None:
    assert map_finish_reason("stop") == ""
    assert map_finish_reason("length") == ""
    assert map_finish_reason(None) == ""


def test_completed_with_provider_finish_is_final_answer() -> None:
    assert map_turn_outcome(status="completed", finish_reason="stop") == STOP_REASON_FINAL_ANSWER_RETURNED
    assert map_turn_outcome(status="completed") == STOP_REASON_FINAL_ANSWER_RETURNED


def test_completed_with_limit_finish_maps_reason() -> None:
    assert (
        map_turn_outcome(status="completed", finish_reason="tool_round_limit")
        == STOP_REASON_STEP_LIMIT_REACHED
    )
    assert map_turn_outcome(status="completed", finish_reason="interrupted") == STOP_REASON_INTERRUPTED


def test_waiting_for_user_input_is_approval_denied() -> None:
    assert map_turn_outcome(status="waiting_for_user_input") == STOP_REASON_APPROVAL_DENIED


def test_error_type_overrides_status() -> None:
    assert map_turn_outcome(status="completed", error_type="persistence") == STOP_REASON_PERSISTENCE_ERROR
    assert map_turn_outcome(status="completed", error_type="resume") == STOP_REASON_RESUME_LOAD_ERROR


def test_unknown_status_maps_to_empty() -> None:
    assert map_turn_outcome(status="weird") == ""
