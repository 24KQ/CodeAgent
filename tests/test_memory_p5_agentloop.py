"""P5.3 AgentLoop memory audit 集成测试。

这些测试从 provider、session event 和 run trace 三个公开边界观察行为，确认
memory 投影不会进入 session history，并且同一轮的多次 provider 请求可以用
稳定的 request/fingerprint 二元键关联。fake provider 只模拟外部模型边界，
不会创建真实 client、runner 或网络请求。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from firstcoder.agent.loop import AgentLoop
from firstcoder.agent.session import AgentSession
from firstcoder.context.store import JsonlSessionStore
from firstcoder.harness.experiments.memory_eval import correlate_memory_audit_events
from firstcoder.harness.recorder import RunRecorder
from firstcoder.memory.durable import DurableMemoryStore
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.types import (
    ChatRequest,
    ChatResponse,
    ToolCall,
    ToolDefinition,
)
from firstcoder.tools.types import Tool, make_text_result


@dataclass
class _FixtureProvider(ChatProvider):
    """按预置顺序返回结果，并保留请求供测试观察真实调用边界。"""

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


def _session(tmp_path: Path, *, session_id: str, tool: Tool | None = None) -> AgentSession:
    """创建完全隔离的 session，并显式替换 global store，避免默认用户目录副作用。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_store = JsonlSessionStore(workspace / ".firstcoder" / "sessions")
    global_store = DurableMemoryStore(tmp_path / "global-memory", global_store=True)
    return AgentSession.create(
        store=session_store,
        session_id=session_id,
        agents_md="",
        tools=[tool] if tool is not None else None,
        workspace_root=workspace,
        global_memory_store=global_store,
    )


def test_memory_projection_keeps_retrieval_metadata_for_audit(tmp_path: Path) -> None:
    """投影应保留 query hash、selected note id 和 global opt-in 元数据。"""

    session = _session(tmp_path, session_id="projection-metadata")
    receipt = session.memory_runtime.promote(
        "project-conventions",
        "pytest uses deterministic fixtures",
        source="test",
    )

    message, projection = session.memory_projector.build_message_with_metadata(
        "pytest",
        include_global=True,
    )

    assert message is not None
    assert projection.selected_note_ids == (receipt.note_id,)
    assert len(projection.query_hash) == 12
    assert projection.include_global is True
    assert projection.note_count == 1
    assert session.memory_projector.build_message("pytest", include_global=True) == message


def test_agent_loop_counts_memory_in_budget_without_changing_stable_prefix(tmp_path: Path) -> None:
    """memory 动态消息应进入 token 预算，但不能写入稳定 system prefix 缓存。"""

    without_memory = _session(tmp_path / "without", session_id="budget-without")
    without_memory.append_user_message("pytest")
    without_loop = AgentLoop(session=without_memory, provider=_FixtureProvider([]))
    without_budget = without_loop.context_budget_for_view(without_memory.rebuild_view())

    with_memory = _session(tmp_path / "with", session_id="budget-with")
    with_memory.memory_runtime.promote(
        "project-conventions",
        "pytest uses deterministic fixtures and this note is deliberately long enough for budgeting",
        source="test",
    )
    with_memory.append_user_message("pytest")
    with_loop = AgentLoop(session=with_memory, provider=_FixtureProvider([]))
    with_budget = with_loop.context_budget_for_view(with_memory.rebuild_view())

    assert with_budget.input_tokens > without_budget.input_tokens
    assert all(
        "pytest uses deterministic fixtures" not in message.content
        for message in with_memory.prompt_cache.entry.messages
    )


def test_context_budget_probe_does_not_write_memory_audit(tmp_path: Path) -> None:
    """预算试算可以读取投影，但不能留下待提交或已提交的 memory audit。"""

    session = _session(tmp_path, session_id="budget-probe")
    session.memory_runtime.promote(
        "project-conventions",
        "pytest uses deterministic fixtures",
        source="test",
    )
    session.append_user_message("pytest")
    provider = _FixtureProvider([])
    loop = AgentLoop(session=session, provider=provider)

    loop.context_budget_for_view(session.rebuild_view())

    assert [event for event in session.store.list_events(session.session_id) if event.type == "memory_retrieved"] == []


def test_empty_memory_query_does_not_write_retrieval_audit(tmp_path: Path) -> None:
    """没有可检索 query 时，真实 provider 请求不能伪造 memory_retrieved。"""

    session = _session(tmp_path, session_id="empty-query")
    provider = _FixtureProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="done",
                finish_reason="stop",
            )
        ]
    )

    result = AgentLoop(session=session, provider=provider)._run_user_turn_sync("   ")

    assert result.response is not None
    assert [
        event
        for event in session.store.list_events(session.session_id)
        if event.type == "memory_retrieved"
    ] == []


def test_empty_memory_projection_records_abstention_audit_for_real_query(tmp_path: Path) -> None:
    """有 query 但无命中时，空 projection 仍要保留 abstention 证据。"""

    session = _session(tmp_path, session_id="empty-projection")
    provider = _FixtureProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="done",
                finish_reason="stop",
            )
        ]
    )

    result = AgentLoop(session=session, provider=provider)._run_user_turn_sync(
        "no durable memory matches this query"
    )

    assert result.response is not None
    events = [
        event
        for event in session.store.list_events(session.session_id)
        if event.type == "memory_retrieved"
    ]
    assert len(events) == 1
    assert events[0].payload["projection_empty"] is True
    assert events[0].payload["query_hash"]
    assert events[0].payload["selected_note_ids"] == []
    payload = events[0].payload
    evaluated = correlate_memory_audit_events(
        [
            {"type": "memory_retrieved", "payload": payload},
            {
                "event": "prompt_built",
                "request_id": payload["request_id"],
                "projection_fingerprint": payload["projection_fingerprint"],
            },
            {
                "event": "model_requested",
                "request_id": payload["request_id"],
                "projection_fingerprint": payload["projection_fingerprint"],
            },
        ]
    )
    assert evaluated["associations"][0]["projection_empty"] is True


def test_evidence_only_projection_exposes_missing_memory_evidence(tmp_path: Path) -> None:
    """严格 evidence-only 请求必须把“无证据”状态明确传给 provider。"""

    session = _session(tmp_path, session_id="evidence-only-empty")
    default_message, _ = session.memory_projector.build_message_with_metadata(
        "global support window",
    )
    assert default_message is None
    session.memory_projector.evidence_only = True

    message, projection = session.memory_projector.build_message_with_metadata(
        "global support window",
    )

    assert message is not None
    assert projection.selected_note_ids == ()
    assert "No valid durable memory was selected" in message.content
    assert "only valid answer is exactly the single word 'unknown'" in message.content


def test_evidence_only_audit_keeps_empty_selection_semantics(tmp_path: Path) -> None:
    """状态提示不能把无命中检索伪装成已选 memory。"""

    session = _session(tmp_path, session_id="evidence-only-audit")
    session.memory_projector.evidence_only = True
    provider = _FixtureProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="unknown",
                finish_reason="stop",
            )
        ]
    )

    result = AgentLoop(session=session, provider=provider)._run_user_turn_sync(
        "global support window",
    )

    assert result.response is not None
    events = [
        event
        for event in session.store.list_events(session.session_id)
        if event.type == "memory_retrieved"
    ]
    assert len(events) == 1
    assert events[0].payload["evidence_only"] is True
    assert events[0].payload["projection_empty"] is True
    assert events[0].payload["selected_note_ids"] == []
    payload = events[0].payload
    evaluated = correlate_memory_audit_events(
        [
            {"type": "memory_retrieved", "payload": payload},
            {
                "event": "prompt_built",
                "request_id": payload["request_id"],
                "projection_fingerprint": payload["projection_fingerprint"],
            },
            {
                "event": "model_requested",
                "request_id": payload["request_id"],
                "projection_fingerprint": payload["projection_fingerprint"],
            },
        ]
    )
    assert evaluated["associations"][0]["evidence_only"] is True


def test_agent_loop_does_not_project_generic_build_note_for_full_evaluation_prompt(
    tmp_path: Path,
) -> None:
    """真实 AgentLoop 查询包含评估说明时，低信号词不能命中无关 durable note。"""

    session = _session(tmp_path, session_id="full-evaluation-prompt")
    session.memory_runtime.promote(
        "project-conventions",
        "build tool for this project is uv",
        source="test",
    )
    provider = _FixtureProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="unknown",
                finish_reason="stop",
            )
        ]
    )
    prompt = (
        "You are evaluating a durable-memory assistant. Use only relevant durable "
        "memory provided in the context. Do not guess. If evidence is missing, say "
        "that you cannot determine the answer. Question: unknown production incident case-04"
    )

    result = AgentLoop(session=session, provider=provider)._run_user_turn_sync(prompt)

    assert result.response is not None
    events = [
        event
        for event in session.store.list_events(session.session_id)
        if event.type == "memory_retrieved"
    ]
    assert len(events) == 1
    assert events[0].payload["selected_note_ids"] == []
    assert events[0].payload["projection_empty"] is True


def test_provider_error_trace_keeps_request_correlation_keys(tmp_path: Path) -> None:
    """provider 失败的 model_parsed 也必须暴露统一的顶层关联键。"""

    session = _session(tmp_path, session_id="provider-error")
    provider = _FixtureProvider([])
    loop = AgentLoop(session=session, provider=provider)
    budget = loop.context_budget_for_view(session.rebuild_view())
    prepared = SimpleNamespace(
        request_id="request-error",
        projection_fingerprint="fingerprint-error",
        context_budget=budget,
    )
    recorder = RunRecorder(session=session, user_request="provider failure")

    recorder.record_provider_requested(prepared, provider)
    recorder.record_provider_error(prepared, provider, RuntimeError("fixture failure"))

    assert recorder.run_store is not None
    trace_path = recorder.run_store.trace_path(recorder.task_state)
    trace_events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    parsed = next(event for event in trace_events if event["event"] == "model_parsed")
    assert parsed["request_id"] == "request-error"
    assert parsed["projection_fingerprint"] == "fingerprint-error"


def test_agent_loop_links_each_memory_audit_to_two_provider_requests(tmp_path: Path) -> None:
    """一次工具轮次中的每次真实请求都必须拥有唯一 memory audit 关联键。"""

    tool = Tool(
        definition=ToolDefinition(
            name="fixture_echo",
            description="Return a deterministic fixture result.",
            parameters={"type": "object", "properties": {}},
        ),
        executor=lambda **_: make_text_result("fixture_echo", "fixture tool completed"),
    )
    session = _session(tmp_path, session_id="multi-request", tool=tool)
    session.memory_runtime.promote(
        "project-conventions",
        "pytest uses deterministic fixtures",
        source="test",
    )
    provider = _FixtureProvider(
        [
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="",
                tool_calls=[ToolCall(id="call_fixture", name="fixture_echo", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                provider="fixture",
                model="fixture-model",
                content="done",
                finish_reason="stop",
            ),
        ]
    )
    loop = AgentLoop(session=session, provider=provider)

    result = loop._run_user_turn_sync("pytest")

    assert result.response is not None
    assert len(provider.requests) == 2
    memory_events = [
        event.payload
        for event in session.store.list_events(session.session_id)
        if event.type == "memory_retrieved"
    ]
    memory_keys = {(payload["request_id"], payload["projection_fingerprint"]) for payload in memory_events}
    assert len(memory_events) == 2
    assert len(memory_keys) == 2
    assert all(payload["query_hash"] for payload in memory_events)
    assert all(payload["selected_note_ids"] for payload in memory_events)
    assert all(payload["include_global"] is False for payload in memory_events)

    assert loop.run_recorder is not None
    assert loop.run_recorder.run_store is not None
    trace_path = loop.run_recorder.run_store.trace_path(loop.run_recorder.task_state)
    trace_events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    prompt_keys = {
        (event["request_id"], event["projection_fingerprint"])
        for event in trace_events
        if event["event"] == "prompt_built"
    }
    provider_keys = {
        (event["request_id"], event["projection_fingerprint"])
        for event in trace_events
        if event["event"] == "model_requested"
    }
    assert memory_keys == prompt_keys == provider_keys

    # memory_retrieved 是旁路审计事件，不能成为可恢复的 user/assistant/tool 消息。
    assert all(
        "pytest uses deterministic fixtures" not in part.content
        for message in session.rebuild_view().messages
        for part in message.parts
    )


def test_memory_audit_evaluator_marks_legacy_events_as_fallback() -> None:
    """有二元关联键的事件才是高可信结果，旧事件必须显式降级。"""

    events = [
        {
            "type": "memory_retrieved",
            "payload": {
                "request_id": "req_high",
                "projection_fingerprint": "fp_high",
                "query_hash": "query_high",
                "selected_note_ids": ["note_high"],
                "include_global": False,
            },
        },
        {
            "event": "prompt_built",
            "request_id": "req_high",
            "projection_fingerprint": "fp_high",
        },
        {
            "event": "model_requested",
            "request_id": "req_high",
            "projection_fingerprint": "fp_high",
        },
        {
            "type": "memory_retrieved",
            "payload": {
                "query_hash": "legacy_query",
                "selected_note_ids": ["legacy_note"],
                "include_global": False,
            },
        },
        {"event": "prompt_built", "projection_fingerprint": "legacy_fp"},
        {"event": "model_requested", "request_id": "legacy_req"},
    ]

    result = correlate_memory_audit_events(events)

    assert result["high_confidence_count"] == 1
    assert result["fallback_count"] == 1
    assert result["claimable"] is False
    assert result["associations"][0]["confidence"] == "high"
    assert result["associations"][1]["confidence"] == "fallback"


def test_memory_audit_evaluator_rejects_duplicate_request_pairs() -> None:
    """同一二元键重复出现时，逐行结果也不能继续声称 high。"""

    events = [
        {
            "type": "memory_retrieved",
            "payload": {
                "request_id": "duplicate-request",
                "projection_fingerprint": "duplicate-fingerprint",
                "selected_note_ids": ["note-a"],
            },
        },
        {
            "type": "memory_retrieved",
            "payload": {
                "request_id": "duplicate-request",
                "projection_fingerprint": "duplicate-fingerprint",
                "selected_note_ids": ["note-b"],
            },
        },
        {
            "event": "prompt_built",
            "request_id": "duplicate-request",
            "projection_fingerprint": "duplicate-fingerprint",
        },
        {
            "event": "model_requested",
            "request_id": "duplicate-request",
            "projection_fingerprint": "duplicate-fingerprint",
        },
        {
            "type": "memory_retrieved",
            "payload": {
                "request_id": "unique-request",
                "projection_fingerprint": "unique-fingerprint",
                "selected_note_ids": ["note-unique"],
            },
        },
        {
            "event": "prompt_built",
            "request_id": "unique-request",
            "projection_fingerprint": "unique-fingerprint",
        },
        {
            "event": "model_requested",
            "request_id": "unique-request",
            "projection_fingerprint": "unique-fingerprint",
        },
    ]

    result = correlate_memory_audit_events(events)

    assert result["duplicate_key_count"] == 1
    assert result["duplicate_count"] == 2
    assert result["high_confidence_count"] == 1
    assert result["claimable"] is False
    assert [row["confidence"] for row in result["associations"]] == [
        "duplicate",
        "duplicate",
        "high",
    ]
