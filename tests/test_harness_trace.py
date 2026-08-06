"""P1 slice 2 tests: trace schema and redacting emitter (trace.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from firstcoder.harness.run_store import RunStore
from firstcoder.harness.task_state import TaskState
from firstcoder.harness.trace import PHASE_BY_EVENT, TraceWriter, build_trace_event
from firstcoder.memory.security import REDACTED_VALUE, StaticSecurityPolicy


def test_phase_mapping() -> None:
    assert PHASE_BY_EVENT["prompt_built"] == "prompt"
    assert PHASE_BY_EVENT["tool_executed"] == "tool"
    assert PHASE_BY_EVENT["run_started"] == "runtime"
    assert PHASE_BY_EVENT["compaction_finished"] == "compact"
    assert build_trace_event("unknown_thing", None, trace_id="r", turn_id="t")["phase"] == "runtime"


def test_build_trace_event_defaults() -> None:
    event = build_trace_event("run_started", None, trace_id="run_1", turn_id="t1", span_seq=3)
    assert event["event"] == "run_started"
    assert event["trace_id"] == "run_1"
    assert event["turn_id"] == "t1"
    assert event["span_id"] == "span_000003"
    assert event["duration_ms"] == 0
    assert event["estimated_input_tokens"] == 0
    assert event["artifact_paths"] == []
    assert event["created_at"].endswith("Z")


def test_build_trace_event_preserves_payload_and_parent() -> None:
    event = build_trace_event(
        "loop_transition",
        {"kind": "continue", "reason": "tool_batch_executed", "affected_paths": ["a.py"]},
        trace_id="run_1",
        turn_id="t1",
        span_seq=1,
        parent_span_id="span_000000",
    )
    assert event["kind"] == "continue"
    assert event["parent_span_id"] == "span_000000"
    assert event["artifact_paths"] == ["a.py"]


def test_emitter_redacts_before_persistence(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = TaskState.create("t1", "request", run_id="run_sec")
    writer = TraceWriter(store, StaticSecurityPolicy())
    writer.emit(
        state,
        "tool_executed",
        {"tool": "bash", "output": "key sk-AbCdEfGhIjKlMnOpQrStUvWxYz0123"},
    )
    line = store.trace_path("run_sec").read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert "sk-AbCdEfGhIjKlMnOpQrStUvWxYz0123" not in line
    assert REDACTED_VALUE in record["output"]


def test_emitter_tracks_changed_paths_and_spans(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = TaskState.create("t1", "request", run_id="run_paths")
    writer = TraceWriter(store, StaticSecurityPolicy())
    writer.emit(state, "tool_executed", {"affected_paths": ["src/a.py", "src/b.py"]})
    writer.emit(state, "tool_executed", {"affected_paths": ["src/a.py"]})
    assert state.changed_paths == ["src/a.py", "src/b.py"]
    lines = store.trace_path("run_paths").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["span_id"] == "span_000001"
    assert json.loads(lines[1])["span_id"] == "span_000002"
    assert json.loads(lines[1])["parent_span_id"] == "span_000001"


def test_emitter_fans_out_to_consumers_and_records_errors(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = TaskState.create("t1", "request", run_id="run_cons")
    received: list[dict] = []

    class OkConsumer:
        def handle(self, task_state, event) -> None:
            received.append(event)

    class BadConsumer:
        def handle(self, task_state, event) -> None:
            raise RuntimeError("consumer boom")

    writer = TraceWriter(store, StaticSecurityPolicy(), consumers=[OkConsumer(), BadConsumer()])
    writer.emit(state, "run_finished")
    assert len(received) == 1
    errors = state.evidence_summaries["runtime_consumer_errors"]
    assert errors[0]["consumer"] == "BadConsumer"
    assert "consumer boom" in errors[0]["message"]
    # 消费者失败不中断 run，task_state 照常落盘。
    assert store.load_task_state("run_cons")["evidence_summaries"]["runtime_consumer_errors"]


def test_critical_consumer_failure_raises_after_persistence(tmp_path: Path) -> None:
    """critical consumer 失败 = 审计硬失败：错误先落盘，再中断 run（Codex P1 review fix）。"""
    store = RunStore(tmp_path / "runs")
    state = TaskState.create("t1", "request", run_id="run_crit")

    class CriticalConsumer:
        critical = True

        def handle(self, task_state, event) -> None:
            raise RuntimeError("audit store boom")

    writer = TraceWriter(store, StaticSecurityPolicy(), consumers=[CriticalConsumer()])
    with pytest.raises(RuntimeError, match="audit store boom"):
        writer.emit(state, "run_finished")
    # 异常抛出前 task_state 已落盘，错误记录可审计。
    persisted = store.load_task_state("run_crit")
    errors = persisted["evidence_summaries"]["runtime_consumer_errors"]
    assert errors[0]["consumer"] == "CriticalConsumer"
    assert errors[0]["critical"] is True
