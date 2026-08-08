"""P4.1 普通 AgentLoop -> run-level harness 的集成测试。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from firstcoder.agent.loop import AgentLoop
from firstcoder.agent.session import AgentSession
from firstcoder.agent.tool_execution import ToolExecutionEvent
from firstcoder.app.runtime import AgentChatRunner, CurrentSessionState
from firstcoder.context.llm_compact import LlmCompactEvent
from firstcoder.context.manager import ContextCompactResult
from firstcoder.context.store import (
    JsonlSessionStore,
    SessionLoadError,
    SessionPersistenceError,
)
from firstcoder.context.token_budget import ContextBudget
from firstcoder.harness.recorder import RunRecorder
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.errors import ProviderError, ProviderErrorKind
from firstcoder.providers.types import (
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from firstcoder.runtime.cancellation import AgentCancelledError
from firstcoder.session.index import SessionIndex
from firstcoder.tools.types import Tool, make_error_result, make_text_result


class FailingArtifactStore:
    """模拟 artifact 存储故障，验证旁路故障不会污染主回答。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def start_run(self, task_state, *, task_state_payload=None):
        raise OSError("fixture artifact store unavailable")

    def write_task_state(self, task_state, *, payload=None):
        raise OSError("fixture artifact store unavailable")

    def append_trace(self, task_state, event):
        raise OSError("fixture artifact store unavailable")

    def write_report(self, task_state, report):
        raise OSError("fixture artifact store unavailable")


@dataclass
class RecorderProvider(ChatProvider):
    """无网络 provider，用来验证真实 AgentLoop 事件边界。"""

    responses: list[ChatResponse]
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fixture"

    @property
    def model(self) -> str:
        return "fixture-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.responses.pop(0)


@dataclass
class StreamingRecorderProvider(RecorderProvider):
    """提供最小 message_completed 事件，验证 AgentChatRunner 流式接线。"""

    async def astream(self, request: ChatRequest):
        self.requests.append(request)
        yield ChatStreamEvent(kind="message_completed", response=self.responses.pop(0))


@dataclass
class FailingRecorderProvider(ChatProvider):
    """无网络失败 provider，用来验证异常路径仍生成失败 run。"""

    error: Exception

    @property
    def name(self) -> str:
        return "failing-fixture"

    @property
    def model(self) -> str:
        return "failing-fixture-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        raise self.error


class SkippingContextManager:
    """返回稳定 no-op 结果，验证 context reducer 的生产事件接线。"""

    def compact_if_needed(self, request):
        return ContextCompactResult(
            status="skipped",
            reason="under_threshold",
            view=request.view,
            before_tokens=request.budget.input_tokens,
            after_tokens=request.budget.input_tokens,
        )


def _artifact_paths(store_root: Path) -> tuple[Path, Path, Path]:
    run_dirs = [path for path in (store_root / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    return run_dir / "task_state.json", run_dir / "trace.jsonl", run_dir / "report.json"


def test_run_recorder_defers_persistence_until_terminal_artifacts_are_needed(tmp_path: Path) -> None:
    """初始化和逻辑 start 不应让取消请求先等待 harness 写盘。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_lazy_harness", agents_md="")
    recorder = RunRecorder(session=session, user_request="提前取消")

    recorder.start()

    assert not (store.root / "runs").exists()

    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="完成",
            finish_reason="stop",
        )
    )

    task_state_path, trace_path, report_path = _artifact_paths(store.root)
    assert task_state_path.exists()
    assert trace_path.exists()
    assert report_path.exists()


def test_run_recorder_degrades_when_artifact_persistence_fails(tmp_path: Path) -> None:
    """harness 写盘失败只标记 degraded，不得把 provider 结果变成异常。"""

    session = AgentSession.create(
        store=JsonlSessionStore(tmp_path),
        session_id="sess_degraded_harness",
        agents_md="",
    )
    provider = RecorderProvider([])
    recorder = RunRecorder(
        session=session,
        user_request="普通回答",
        artifact_store_factory=FailingArtifactStore,
    )
    budget = ContextBudget(
        context_window=8_192,
        input_tokens=10,
        output_reserve=100,
        input_capacity=8_092,
        fixed_tokens=0,
        history_tokens=10,
        high_watermark=7_282,
        low_watermark=5_826,
        source="assumed",
    )
    prepared = SimpleNamespace(
        request_id="request-degraded",
        projection_fingerprint="fingerprint-degraded",
        context_budget=budget,
        request=ChatRequest(messages=[]),
    )

    recorder.record_provider_requested(prepared, provider)
    recorder.record_provider_response(
        prepared,
        provider,
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="主回答仍然可用",
            finish_reason="stop",
        ),
    )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="主回答仍然可用",
            finish_reason="stop",
        )
    )

    assert recorder.task_state.harness_degraded is True
    assert recorder.task_state.harness_degradation_reason == "artifact_persistence_failed"


def test_task_boundary_classifier_calls_are_recorded_without_session_messages(tmp_path: Path) -> None:
    """隐藏 classifier 请求进入统一 trace，但响应不污染可恢复对话。"""

    class BoundaryProvider(RecorderProvider):
        def complete(self, request: ChatRequest) -> ChatResponse:
            self.requests.append(request)
            basis = next(
                message.content.split("basis_message_id=", 1)[1].split("]", 1)[0]
                for message in request.messages
                if message.role == "user" and "basis_message_id=" in message.content
            )
            return ChatResponse(
                provider="fixture",
                model="fixture-model",
                content=json.dumps({"decision": "same", "basis_message_id": basis}),
                finish_reason="stop",
            )

    session = AgentSession.create(
        store=JsonlSessionStore(tmp_path),
        session_id="sess_classifier_harness",
        agents_md="",
    )
    session.runtime_state.active_task_hash = "task_existing"
    basis_message_id = session.append_user_message("继续当前任务")
    provider = BoundaryProvider([])
    loop = AgentLoop(session=session, provider=provider)
    loop._ensure_run_recorder("继续当前任务")
    loop._classify_task_boundary(basis_message_id)

    loop.run_recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="完成",
            finish_reason="stop",
        )
    )

    assert loop.run_recorder.run_store is not None
    trace_path = loop.run_recorder.run_store.trace_path(loop.run_recorder.task_state)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    auxiliary = [
        event
        for event in events
        if event.get("call_kind") == "task_boundary_classifier"
    ]
    assert {event["event"] for event in auxiliary} == {
        "prompt_built",
        "model_requested",
        "model_parsed",
    }
    assert len({event["request_id"] for event in auxiliary}) == 1
    assert len({event["projection_fingerprint"] for event in auxiliary}) == 1
    assert [message.role for message in session.rebuild_view().messages] == ["user"]


def test_run_recorder_strict_readiness_persists_final_gate_blocked(tmp_path: Path) -> None:
    """严格 readiness 阻断时，报告状态必须与 final_gate_blocked 一致。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_strict_harness", agents_md="")
    recorder = RunRecorder(
        session=session,
        user_request="修改 result.json",
        readiness_mode="strict",
    )
    recorder.task_state.changed_paths.append("src/changed.py")

    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="已完成",
            finish_reason="stop",
        )
    )

    task_state_path, trace_path, report_path = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    readiness = next(event for event in events if event["event"] == "final_readiness_decision")

    assert task_state["status"] == "stopped"
    assert task_state["stop_reason"] == "final_gate_blocked"
    assert report["stop_reason"] == "final_gate_blocked"
    assert readiness["decision"] == "block"
    assert report_path.exists()


def test_run_recorder_ignores_dry_run_and_external_tool_paths(tmp_path: Path) -> None:
    """dry-run 和 workspace 外路径不能被算作真实 workspace 变更。"""

    store = JsonlSessionStore(tmp_path / ".firstcoder")
    session = AgentSession.from_project(store=store, session_id="sess_path_harness", project_root=tmp_path)
    recorder = RunRecorder(session=session, user_request="检查补丁")
    outside = tmp_path.parent / "outside.txt"
    result = make_text_result(
        "apply_patch",
        "补丁可应用。",
        dry_run=True,
        changed_files=["README.md", str(outside)],
    )
    recorder.record_tool_event(
        ToolExecutionEvent(
            kind="finished",
            tool_call=ToolCall(id="call_patch", name="apply_patch", arguments={"dry_run": True}),
            result=result,
        )
    )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="检查完成",
            finish_reason="stop",
        )
    )

    _, trace_path, report_path = _artifact_paths(store.root)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_event = next(event for event in events if event["event"] == "tool_executed")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert tool_event["affected_paths"] == []
    assert tool_event["workspace_changed"] is False
    assert report["changed_paths"] == []


def test_run_recorder_transition_counts_are_incremental_for_parallel_tools(tmp_path: Path) -> None:
    """并行工具的每条 transition 只能携带本次新增计数，不能重复累计。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_incremental_counts", agents_md="")
    recorder = RunRecorder(session=session, user_request="并行读取")

    calls = [
        ToolCall(id="call_one", name="view", arguments={}),
        ToolCall(id="call_two", name="view", arguments={}),
    ]
    for call in calls:
        recorder.record_tool_event(ToolExecutionEvent(kind="started", tool_call=call))
    for call in calls:
        recorder.record_tool_event(
            ToolExecutionEvent(
                kind="finished",
                tool_call=call,
                result=make_text_result("view", "内容"),
            )
        )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="完成",
            finish_reason="stop",
        )
    )

    _, trace_path, report_path = _artifact_paths(store.root)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    transitions = [
        event
        for event in events
        if event["event"] == "loop_transition" and event.get("kind") == "continue"
    ]
    summary = json.loads(report_path.read_text(encoding="utf-8"))["evidence_summaries"]["transition_summary"]

    assert [
        (event.get("tool_requested_count", 0), event.get("tool_executed_count", 0))
        for event in transitions
    ] == [(2, 1), (0, 1)]
    assert summary["tool_requested_count"] == 2
    assert summary["tool_executed_count"] == 2


def test_run_recorder_keeps_denied_governance_and_tool_result_distinct(tmp_path: Path) -> None:
    """权限拒绝既是治理事实，也要有失败 tool result，但不能伪增执行步数。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_denied_events", agents_md="")
    recorder = RunRecorder(session=session, user_request="拒绝写入")
    recorder.record_tool_event(
        ToolExecutionEvent(
            kind="denied",
            tool_call=ToolCall(id="call_denied", name="write", arguments={}),
            result=make_error_result("write", "用户拒绝了权限请求"),
        )
    )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="已取消写入",
            finish_reason="stop",
        )
    )

    task_state_path, trace_path, _ = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    governance = [event for event in events if event["event"] == "governance_decision"]
    tool_events = [event for event in events if event["event"] == "tool_executed"]

    assert governance[0]["decision"] == "deny"
    assert tool_events[0]["tool_status"] == "failed"
    assert task_state["tool_steps"] == 0


def test_agent_loop_classifies_session_persistence_failure(tmp_path: Path, monkeypatch) -> None:
    """session writer 的 I/O 失败应进入 persistence_error，而不是 model_error。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_persistence_error", agents_md="")
    provider = RecorderProvider(
        [ChatResponse(provider="fixture", model="fixture-model", content="不会调用", finish_reason="stop")]
    )
    loop = AgentLoop(session=session, provider=provider)

    def fail_index_update(*args, **kwargs):
        raise OSError("session index update failed")

    monkeypatch.setattr(SessionIndex, "update_event", fail_index_update)

    with pytest.raises(SessionPersistenceError):
        asyncio.run(loop.run_user_turn("写入失败"))

    task_state_path, _, _ = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    assert task_state["stop_reason"] == "persistence_error"


def test_agent_loop_classifies_resume_load_failure(tmp_path: Path, monkeypatch) -> None:
    """恢复阶段读取 session 失败应进入 resume_load_error。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_resume_error", agents_md="")
    loop = AgentLoop(
        session=session,
        provider=RecorderProvider([]),
    )

    def fail_rebuild(self):
        raise SessionLoadError("session replay failed")

    monkeypatch.setattr(AgentSession, "rebuild_view", fail_rebuild)

    with pytest.raises(SessionLoadError):
        asyncio.run(loop.resume_with_user_input("missing-request", "继续"))

    task_state_path, _, _ = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    assert task_state["stop_reason"] == "resume_load_error"


def test_run_recorder_redacts_task_state_and_report_artifacts(tmp_path: Path) -> None:
    """TaskState/report 也必须遵守 trace 使用的递归脱敏规则。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_redact_harness", agents_md="")
    recorder = RunRecorder(
        session=session,
        user_request="记录 sk-abcdefghijklmnopqrstuvwxyz1234567890",
    )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="结果 sk-abcdefghijklmnopqrstuvwxyz1234567890",
            finish_reason="stop",
        )
    )

    task_state_path, _, report_path = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in task_state["user_request"]
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in report["final_answer"]


def test_run_recorder_persists_l4_compact_usage_in_context_evidence(tmp_path: Path) -> None:
    """L4 provider usage 应进入 context budget summary，供 cost reducer 读取。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_l4_usage", agents_md="")
    recorder = RunRecorder(session=session, user_request="压缩上下文")
    budget = ContextBudget(
        context_window=1000,
        output_reserve=100,
        input_capacity=850,
        fixed_tokens=100,
        history_tokens=300,
        input_tokens=400,
        high_watermark=765,
        low_watermark=612,
        source="configured",
    )
    recorder.record_context_decision(
        ContextCompactResult(
            status="success",
            reason="under_threshold",
            view=session.rebuild_view(),
            before_tokens=700,
            after_tokens=400,
            l4_event=LlmCompactEvent(
                status="success",
                source_fingerprint="source-1",
                compact_call_usage={
                    "input_tokens": 700,
                    "output_tokens": 80,
                    "total_tokens": 780,
                    "cached_tokens": 0,
                },
            ),
        ),
        budget,
    )
    recorder.record_prompt_built(
        SimpleNamespace(
            context_budget=budget,
            projection_fingerprint="prompt-after-compact",
            request=ChatRequest(messages=[]),
        ),
        RecorderProvider([]),
    )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="完成",
            finish_reason="stop",
        )
    )

    _, _, report_path = _artifact_paths(store.root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    context_summary = report["evidence_summaries"]["context_budget_summary"]

    assert context_summary["compact_call_usage"]["total_tokens"] == 780
    assert context_summary["compact_net_benefit_tokens"] == -480


def test_run_recorder_does_not_treat_readonly_tool_path_as_workspace_change(tmp_path: Path) -> None:
    """view 等只读工具返回的 path 不能触发 strict readiness 的变更门禁。"""

    store = JsonlSessionStore(tmp_path / ".firstcoder")
    session = AgentSession.from_project(store=store, session_id="sess_readonly_path", project_root=tmp_path)
    recorder = RunRecorder(session=session, user_request="读取 README", readiness_mode="strict")
    recorder.record_tool_event(
        ToolExecutionEvent(
            kind="finished",
            tool_call=ToolCall(id="call_view", name="view", arguments={"path": "README.md"}),
            result=make_text_result("view", "内容", path="README.md"),
        )
    )
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="读取完成",
            finish_reason="stop",
        )
    )

    _, trace_path, report_path = _artifact_paths(store.root)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_event = next(event for event in events if event["event"] == "tool_executed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert tool_event["affected_paths"] == []
    assert tool_event["workspace_changed"] is False
    assert report["changed_paths"] == []


def test_agent_chat_runner_persists_run_trace_and_provider_metadata(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_harness", agents_md="")
    provider = RecorderProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="完成",
                finish_reason="stop",
                usage=TokenUsage(input_tokens=12, output_tokens=4, total_tokens=16),
            )
        ]
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        context_manager=SkippingContextManager(),
    )

    response = runner.run_user_turn("请完成任务")

    assert response.content == "完成"
    task_state_path, trace_path, report_path = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_names = [event["event"] for event in events]

    assert task_state["status"] == "completed"
    assert task_state["stop_reason"] == "final_answer_returned"
    assert task_state["attempts"] == 1
    assert report["run_id"] == task_state["run_id"]
    assert report["verifier_suggestions"] == []
    assert {
        "run_started",
        "prompt_built",
        "model_requested",
        "model_parsed",
        "context_orchestrator_decision",
        "final_readiness_decision",
        "loop_transition",
        "run_finished",
    }.issubset(event_names)

    parsed = next(event for event in events if event["event"] == "model_parsed")
    completion = parsed["completion_metadata"]
    assert completion["provider_protocol"] == "custom"
    assert completion["input_tokens"] == 12
    assert completion["output_tokens"] == 4
    assert report["evidence_summaries"]["transition_summary"]["terminal_count"] == 1
    assert report["evidence_summaries"]["context_budget_summary"]["pressure_tier"] == "tier0_observe"


def test_reused_agent_loop_creates_one_run_artifact_per_turn(tmp_path: Path) -> None:
    """直接复用 AgentLoop 时，每个已完成用户回合都必须拥有独立 run。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_reused_loop", agents_md="")
    provider = RecorderProvider(
        [
            ChatResponse(provider="fixture", model="fixture-model", content="第一轮", finish_reason="stop"),
            ChatResponse(provider="fixture", model="fixture-model", content="第二轮", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(session=session, provider=provider)

    first = loop._run_user_turn_sync("第一轮请求")
    # 让第二轮绕过既有的隐藏 task-boundary 分类 provider 调用，测试只聚焦
    # RunRecorder 是否按完成回合重新建立 run，而不依赖分类模型的输出格式。
    session.runtime_state.active_task_hash = None
    second = loop._run_user_turn_sync("第二轮请求")

    assert first.response is not None
    assert second.response is not None
    run_dirs = sorted(path for path in (store.root / "runs").iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    states = [json.loads((path / "task_state.json").read_text(encoding="utf-8")) for path in run_dirs]
    assert [state["status"] for state in states] == ["completed", "completed"]


def test_agent_chat_runner_streaming_persists_run_artifacts(tmp_path: Path) -> None:
    """普通 runner 的 streaming 入口也必须产生同一套 run 证据。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_stream_harness", agents_md="")
    provider = StreamingRecorderProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="流式完成",
                finish_reason="stop",
                usage=TokenUsage(input_tokens=10, output_tokens=3, total_tokens=13),
            )
        ]
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        use_streaming=True,
    )

    response = runner.run_user_turn("流式请求")

    assert response.content == "流式完成"
    _, trace_path, report_path = _artifact_paths(store.root)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events].count("model_parsed") == 1
    assert report_path.exists()


def test_agent_chat_runner_provider_error_persists_failed_run(tmp_path: Path) -> None:
    """provider 异常重新抛出给调用方，同时保留失败 run 的审计事实。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_error_harness", agents_md="")
    provider = FailingRecorderProvider(
        ProviderError(ProviderErrorKind.SERVER_ERROR, "fixture provider failed")
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
    )

    with pytest.raises(ProviderError):
        runner.run_user_turn("失败请求")

    task_state_path, trace_path, report_path = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert task_state["status"] == "failed"
    assert task_state["stop_reason"] == "model_error"
    assert any(event["event"] == "run_error" for event in events)
    assert report_path.exists()


def test_agent_chat_runner_user_abort_persists_cancelled_run(tmp_path: Path) -> None:
    """provider USER_ABORT 应保留为 cancelled，而不是误报 model_error。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_user_abort_harness", agents_md="")
    provider = FailingRecorderProvider(
        ProviderError(ProviderErrorKind.USER_ABORT, "user aborted provider request")
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
    )

    with pytest.raises(ProviderError):
        runner.run_user_turn("取消请求")

    task_state_path, trace_path, _ = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    parsed = next(event for event in events if event["event"] == "model_parsed")
    assert task_state["stop_reason"] == "cancelled"
    assert parsed["finish_reason"] == "cancelled"


def test_agent_loop_local_cancellation_persists_interrupted_run(tmp_path: Path) -> None:
    """本地 cooperative cancellation 与 provider USER_ABORT 保持不同 stop reason。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_local_interrupted", agents_md="")
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=FailingRecorderProvider(AgentCancelledError()),
    )

    response = runner.run_user_turn("本地中断")

    task_state_path, _, _ = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    assert response.finish_reason == "interrupted"
    assert task_state["status"] == "stopped"
    assert task_state["stop_reason"] == "interrupted"


def test_strict_readiness_does_not_replace_guardrail_stop_reason(tmp_path: Path) -> None:
    """strict gate 只阻断自然最终回答，不覆盖取消或 step limit 终局。"""

    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(store=store, session_id="sess_strict_guardrail", agents_md="")
    recorder = RunRecorder(
        session=session,
        user_request="修改 result.json",
        readiness_mode="strict",
    )
    recorder.task_state.changed_paths.append("src/changed.py")
    recorder.finish(
        ChatResponse(
            provider="fixture",
            model="fixture-model",
            content="任务已中断",
            finish_reason="interrupted",
        )
    )

    task_state_path, trace_path, _ = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    readiness = next(event for event in events if event["event"] == "final_readiness_decision")
    assert task_state["stop_reason"] == "interrupted"
    assert readiness["decision"] == "block"


def test_agent_loop_tool_event_reaches_verification_reducer(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path)
    diagnostics = Tool(
        definition=ToolDefinition(
            name="diagnostics",
            description="执行 fixture 验证。",
            parameters={"type": "object"},
        ),
        executor=lambda: make_text_result(
            "diagnostics",
            "passed",
            command="python -m pytest -q",
        ),
    )
    session = AgentSession.create(
        store=store,
        session_id="sess_harness_tool",
        agents_md="",
        tools=[diagnostics],
    )
    provider = RecorderProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="",
                tool_calls=[ToolCall(id="call_diagnostics", name="diagnostics", arguments={})],
                finish_reason="tool_calls",
                usage=TokenUsage(input_tokens=20, output_tokens=3, total_tokens=23),
            ),
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="验证完成",
                finish_reason="stop",
                usage=TokenUsage(input_tokens=25, output_tokens=3, total_tokens=28),
            ),
        ]
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        tools=[diagnostics],
    )

    response = runner.run_user_turn("验证当前改动")

    assert response.content == "验证完成"
    task_state_path, trace_path, report_path = _artifact_paths(store.root)
    task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_events = [event for event in events if event["event"] == "tool_executed"]

    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "diagnostics"
    assert task_state["tool_steps"] == 1
    signal = report["evidence_summaries"]["verification_signal"]
    assert signal["state"] == "passed"
    assert signal["command"] == "python -m pytest -q"
