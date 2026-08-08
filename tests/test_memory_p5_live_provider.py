"""P5 真实 provider memory smoke benchmark。

本文件是一个显式 opt-in 的外部验收入口，不属于 CI 的默认测试路径。它使用
项目已有的配置加载、provider factory、AgentSession、AgentLoop 和 memory
artifact writer，验证真实模型是否能看到 workspace memory，以及 session/global
作用域和 request-level audit 是否仍然成立。

真实 provider 的回答质量具有成本、网络和模型版本依赖，因此默认只 skip；只有
调用者明确设置 ``FIRSTCODER_LIVE_MEMORY_TEST=1`` 才会发出请求。所有 session、
durable store、global store 和默认 artifact 都建立在 pytest 临时目录中。
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from firstcoder.agent.loop import AgentLoop
from firstcoder.agent.session import AgentSession
from firstcoder.config import ModelProfile, load_config
from firstcoder.context.store import JsonlSessionStore
from firstcoder.harness.experiments.memory_eval import (
    MemoryEvaluationAdapter,
    MemoryFixtureCase,
    MemoryFixtureNote,
    MemoryObservation,
    build_challenge_cases,
    build_contract_cases,
    correlate_memory_audit_events,
    summarize_memory_observations,
    write_memory_eval_artifacts,
)
from firstcoder.memory.durable import DurableMemoryStore, note_id_for
from firstcoder.memory.models import MemoryEvidence, MemoryNote
from firstcoder.memory.provenance import compute_anchor_hash, workspace_fingerprint
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.factory import ProviderConfigError, create_provider_for_model
from firstcoder.providers.types import MainRequestOptions, ProviderCapabilities

LIVE_TEST_ENV = "FIRSTCODER_LIVE_MEMORY_TEST"
LIVE_PROJECT_ROOT_ENV = "FIRSTCODER_LIVE_PROJECT_ROOT"
LIVE_MODEL_REF_ENV = "FIRSTCODER_LIVE_MODEL_REF"
LIVE_CASES_ENV = "FIRSTCODER_LIVE_MEMORY_CASES"
LIVE_ARTIFACT_DIR_ENV = "FIRSTCODER_LIVE_ARTIFACT_DIR"
# 默认一个正常 recall + 一个 stale/scope 安全拒绝 case；真实 provider 仍只需
# 两个可控请求，既能观察回答质量，也能覆盖 fixture 的 provenance 语义。
DEFAULT_LIVE_CASES = ("direct_recall_001", "temporal_rejection_001")
LIVE_TOPIC_POOL = (
    "project-conventions",
    "key-decisions",
    "dependency-facts",
    "user-preferences",
)
LIVE_MEMORY_SYSTEM_RULES = (
    "You are answering a memory benchmark. Use only relevant durable memory in the "
    "system context. Do not inspect files or call tools. Give one short direct answer. "
    "When stating a remembered fact, include both its subject and value rather than "
    "returning only a bare value. If valid evidence is unavailable, say unknown and "
    "do not guess."
)


@dataclass(frozen=True, slots=True)
class _LiveRun:
    """一次真实请求的安全观察结果和关联摘要。

    ``answer`` 只保存在内存中的 ``MemoryObservation``，不会直接写入 artifact。
    ``audit`` 也只保留 evaluator 输出，避免测试辅助对象把 prompt 或 provider
    原始响应带到结果目录。
    """

    observation: MemoryObservation
    audit: dict[str, object]
    provider_calls: int
    input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class _ToollessLiveProvider(ChatProvider):
    """给真实 provider 加一个只读工具边界，防止 benchmark 误触发写工具。

    请求仍由 factory 创建的真实 provider 发送；这里只把 ``supports_tools``
    关闭，使 AgentLoop 不向模型暴露 task/memory/write 工具。真实 provider
    若仍返回 tool call，会在进入 AgentLoop 工具执行前失败，避免评估副作用。
    """

    inner: ChatProvider

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def protocol(self) -> str:
        return self.inner.protocol

    @property
    def base_url(self) -> str | None:
        return self.inner.base_url

    @property
    def capabilities(self) -> ProviderCapabilities:
        """只禁止工具，不改变真实 provider 的其他能力描述。"""

        capabilities = getattr(self.inner, "capabilities", ProviderCapabilities())
        return ProviderCapabilities(
            supports_tools=False,
            supports_forced_tool_choice=False,
            supports_streaming=capabilities.supports_streaming,
            supports_parallel_tool_calls=False,
            supports_json_mode=capabilities.supports_json_mode,
            supports_vision=capabilities.supports_vision,
            supports_reasoning=capabilities.supports_reasoning,
            token_param=capabilities.token_param,
        )

    def complete(self, request):
        """转发真实请求，同时拒绝 provider 意外返回的工具调用。"""

        if request.tools:
            raise AssertionError("live memory benchmark must not expose tool definitions")
        response = self.inner.complete(request)
        if response.tool_calls:
            raise AssertionError("live memory benchmark provider returned an unexpected tool call")
        return response


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _live_project_root() -> Path:
    """解析真实 provider 使用的项目配置根目录。"""

    configured = os.getenv(LIVE_PROJECT_ROOT_ENV, "").strip()
    root = Path(configured) if configured else Path(__file__).resolve().parents[1]
    root = root.expanduser().resolve()
    if not root.is_dir():
        pytest.fail(f"{LIVE_PROJECT_ROOT_ENV} does not point to a directory: {root}")
    return root


def _live_provider() -> tuple[_ToollessLiveProvider, ModelProfile]:
    """用项目正式配置创建 provider；缺少密钥时给出明确 skip。"""

    config = load_config(project_root=_live_project_root())
    catalog = config.model_catalog()
    model_ref = os.getenv(LIVE_MODEL_REF_ENV, "").strip() or catalog.default_ref
    if not model_ref:
        pytest.skip("live provider requires a configured default model or FIRSTCODER_LIVE_MODEL_REF")
    profile = catalog.require(model_ref)
    try:
        provider = create_provider_for_model(config, profile)
    except ProviderConfigError as error:
        if "缺少环境变量" in str(error):
            pytest.skip(f"live provider credential is not configured: {error}")
        raise
    return _ToollessLiveProvider(provider), profile


def _new_session(
    workspace: Path,
    *,
    session_id: str,
    global_store: DurableMemoryStore,
) -> AgentSession:
    """创建完全隔离的 session，并显式注入临时 global store。"""

    session = AgentSession.create(
        store=JsonlSessionStore(workspace / ".firstcoder" / "sessions"),
        session_id=session_id,
        agents_md="",
        tools=[],
        workspace_root=workspace,
        global_memory_store=global_store,
    )
    # 评估约束属于本次 live run 的 system/base rules，不应混入 user query，
    # 否则 retriever 会把 benchmark 说明词也当成 memory 搜索词。
    session.base_rules = LIVE_MEMORY_SYSTEM_RULES
    return session


def _persist_fixture_case(
    session: AgentSession,
    case: MemoryFixtureCase,
) -> MemoryFixtureCase:
    """把 provider-free fixture 转成真实 durable notes，并重写稳定 note id。

    fixture id 是测试语义上的短标识，而 durable id 由 ``topic + text`` 哈希生成。
    这里在测试边界完成一次显式映射，保证 live 结果使用 durable store 实际返回的
    note id，不把 provider-free 的虚拟 id 错当成持久化事实。
    """

    id_map: dict[str, str] = {}
    persisted_notes: list[MemoryFixtureNote] = []
    stale_anchors: list[Path] = []
    workspace = Path(session.memory_store.workspace_root or session.memory_store.root)
    if len(case.notes) > len(LIVE_TOPIC_POOL):
        raise AssertionError(
            "live fixture has more notes than the durable topic pool; add explicit topic mapping"
        )
    for index, fixture_note in enumerate(case.notes):
        # DurableMemoryStore 的 topic 是受控的四项主题枚举；每个 case 最多使用
        # 两条 note，轮换合法主题即可避免测试辅助层伪造不存在的 topic slug。
        topic = LIVE_TOPIC_POOL[index % len(LIVE_TOPIC_POOL)]
        note_id = note_id_for(topic, fixture_note.text)
        evidence = MemoryEvidence(
            session_id=session.session_id,
            visibility="workspace",
        )
        if fixture_note.stale_evidence:
            # 先写入旧内容并保存旧 hash，写入 durable metadata 后再改文件，
            # 让真实 snapshot/provenance 路径产生 stale_evidence rejection。
            anchor_path = workspace / f".live-anchor-{case.case_id}-{index:02d}.txt"
            anchor_path.write_text("before-live-benchmark\n", encoding="utf-8")
            evidence = replace(
                evidence,
                source_path=anchor_path.relative_to(workspace).as_posix(),
                anchor_hash=compute_anchor_hash(anchor_path) or "",
            )
            stale_anchors.append(anchor_path)
        if fixture_note.scope_mismatch:
            # 用另一个临时路径计算真实 fingerprint，避免使用随意字符串绕过
            # retrieval 的 12-hex workspace scope contract。
            other_workspace = workspace.parent / f"other-{case.case_id}-{index:02d}"
            evidence = replace(evidence, scope=workspace_fingerprint(other_workspace))
        session.memory_store.upsert_topic(
            MemoryNote(
                topic=topic,
                text=fixture_note.text,
                status=fixture_note.status,
                evidence=evidence,
            )
        )
        id_map[fixture_note.note_id] = note_id
        persisted_notes.append(replace(fixture_note, note_id=note_id, source=topic))

    for anchor_path in stale_anchors:
        anchor_path.write_text("after-live-benchmark\n", encoding="utf-8")

    return replace(
        case,
        notes=tuple(persisted_notes),
        required_evidence_ids=tuple(id_map[item] for item in case.required_evidence_ids),
        forbidden_memory_ids=tuple(id_map[item] for item in case.forbidden_memory_ids),
    )


def _trace_events(loop: AgentLoop) -> list[dict[str, object]]:
    """读取本次 run 的 trace，供 audit evaluator 做跨存储边界关联。"""

    recorder = loop.run_recorder
    if recorder is None or recorder.run_store is None:
        raise AssertionError("live provider request did not create a run recorder")
    trace_path = recorder.run_store.trace_path(recorder.task_state)
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_provider_case(
    *,
    session: AgentSession,
    case: MemoryFixtureCase,
    provider: ChatProvider,
    profile: ModelProfile,
) -> _LiveRun:
    """通过真实 AgentLoop 执行一个 case，并只返回可脱敏的观察数据。"""

    request_options = MainRequestOptions(
        # live smoke 只需要短答案；低输出上限同时限制意外成本，且不改变
        # provider factory、AgentLoop 或 memory projector 的真实执行路径。
        temperature=profile.request.temperature if profile.request.temperature is not None else 0.0,
        # DeepSeek 等 reasoning provider 会先消耗一部分 output budget 做内部
        # 推理；256 可能在最终短答案前截断，live smoke 使用 512 仍保持低成本。
        max_tokens=min(profile.request.max_tokens or 512, 512),
        extra_body=profile.request.extra_body,
    )
    loop = AgentLoop(
        session=session,
        provider=provider,
        request_options=request_options,
        context_window=profile.context_window,
        enable_delegate_tool=False,
    )
    # benchmark 指令已经放入 system/base rules；user message 保持原始 case query，
    # 这样 MemoryProjector 的检索排名不会被评估说明中的通用词注入污染。
    turn = asyncio.run(loop.run_user_turn(case.query))
    if turn.response is None:
        raise AssertionError(f"live provider returned no final response for case {case.case_id}")
    if turn.response.finish_reason == "length":
        raise AssertionError(
            f"live provider response was truncated by output budget for case {case.case_id}"
        )

    memory_events = [
        event.payload
        for event in session.store.list_events(session.session_id)
        if event.type == "memory_retrieved"
    ]
    if not memory_events:
        raise AssertionError(f"AgentLoop did not write memory_retrieved for case {case.case_id}")
    trace_events = _trace_events(loop)
    audit_events = [
        {"type": "memory_retrieved", "payload": payload}
        for payload in memory_events
    ]
    audit = correlate_memory_audit_events([*audit_events, *trace_events])
    if not audit.get("claimable"):
        raise AssertionError(
            f"live memory audit is not claimable for {case.case_id}: "
            f"{audit.get('associations', [])}"
        )

    # memory_retrieved 是旁路 audit，不应被 session history 当成自然语言消息恢复。
    assert all(
        "## Relevant Durable Memories" not in part.content
        for message in session.rebuild_view().messages
        for part in message.parts
    )
    provider_pairs = [
        (
            str(event.get("request_id") or ""),
            str(event.get("projection_fingerprint") or ""),
        )
        for event in trace_events
        if event.get("event") == "model_requested"
    ]
    if not provider_pairs or any(not request_id or not fingerprint for request_id, fingerprint in provider_pairs):
        raise AssertionError(f"live provider trace lacks complete request pairs for {case.case_id}")
    final_pair = provider_pairs[-1]
    matching_memory_events = [
        payload
        for payload in memory_events
        if (
            str(payload.get("request_id") or ""),
            str(payload.get("projection_fingerprint") or ""),
        )
        == final_pair
    ]
    if len(matching_memory_events) != 1:
        raise AssertionError(
            f"expected exactly one memory audit for final provider request in {case.case_id}, "
            f"got {len(matching_memory_events)}"
        )
    # 多 provider-call 轮次不能凭事件写入顺序猜答案对应的 memory；先按最后一个
    # 真实 provider request 的二元键闭合，再把同一投影交给语义 adapter。
    payload = matching_memory_events[0]
    response = turn.response
    usage = response.usage
    observation = MemoryEvaluationAdapter().observe_provider_answer(
        case,
        response.content,
        payload.get("selected_note_ids", []),
        variant="memory_on",
        rejected_reasons=payload.get("rejected_reasons", {}),
    )
    return _LiveRun(
        observation=observation,
        audit=audit,
        provider_calls=sum(event.get("event") == "model_requested" for event in trace_events),
        input_tokens=int(usage.input_tokens or 0) if usage is not None else 0,
        output_tokens=int(usage.output_tokens or 0) if usage is not None else 0,
    )


def _selected_live_cases() -> tuple[MemoryFixtureCase, ...]:
    """按环境变量选择 live case；默认使用两个低成本代表性 case。

    ``FIRSTCODER_LIVE_MEMORY_CASES=all`` 会执行当前 contract 加 challenge 的
    全部 62 个 case，再加 3 个作用域 case，预计产生约 65 次真实 provider 请求；
    该选项只适合明确要承担对应时间和费用时使用。
    """

    cases = {
        case.case_id: case
        for case in (*build_contract_cases(), *build_challenge_cases())
    }
    raw = os.getenv(LIVE_CASES_ENV, "").strip()
    requested = tuple(case_id.strip() for case_id in raw.split(",") if case_id.strip()) if raw else DEFAULT_LIVE_CASES
    if requested == ("all",):
        return tuple(cases.values())
    unknown = [case_id for case_id in requested if case_id not in cases]
    if unknown:
        pytest.fail(f"unknown live memory case ids: {', '.join(unknown)}")
    return tuple(cases[case_id] for case_id in requested)


def _assert_artifact_is_separate(
    artifact_dir: Path,
    stores: tuple[DurableMemoryStore, ...],
) -> None:
    """确认 benchmark 输出不会落在 memory store 内部。"""

    resolved_artifact = artifact_dir.resolve()
    for store in stores:
        resolved_store = store.root.resolve()
        if resolved_artifact == resolved_store or resolved_store in resolved_artifact.parents:
            raise AssertionError("benchmark artifact directory must be separate from memory stores")


def test_live_provider_memory_benchmark_is_explicit_and_isolated(tmp_path: Path) -> None:
    """真实模型 smoke：workspace 可跨 session，session/global 默认不可泄漏。

    这个测试不把回答质量硬编码成 CI 通过条件；它将真实回答交给统一 adapter
    评分，并把安全的布尔/ID 结果写成 artifact。这样模型变差时结果会明确显示
    失败 case，而不会因为模型措辞变化导致测试套件伪造一个稳定分数。
    """

    if not _is_truthy_env(LIVE_TEST_ENV):
        pytest.skip(f"set {LIVE_TEST_ENV}=1 to run the opt-in live provider benchmark")

    provider, profile = _live_provider()
    observations: list[MemoryObservation] = []
    live_runs: list[_LiveRun] = []
    stores: list[DurableMemoryStore] = []

    # 每个 fixture 使用独立 workspace；由 session A 写入后交给 session B 查询，
    # 真实验证 workspace memory 的跨 session 可见性，而不是只验证同一 runtime。
    selected_cases = _selected_live_cases()
    for index, fixture_case in enumerate(selected_cases):
        workspace = tmp_path / f"fixture-workspace-{index:02d}"
        global_store = DurableMemoryStore(workspace / "global-memory", global_store=True)
        session_a = _new_session(workspace, session_id=f"fixture-writer-{index:02d}", global_store=global_store)
        durable_case = _persist_fixture_case(session_a, fixture_case)
        session_b = _new_session(workspace, session_id=f"fixture-reader-{index:02d}", global_store=global_store)
        live_run = _run_provider_case(
            session=session_b,
            case=durable_case,
            provider=provider,
            profile=profile,
        )
        live_runs.append(live_run)
        observations.append(live_run.observation)
        stores.extend((session_b.memory_store, global_store))
        if fixture_case.case_id == "temporal_rejection_001":
            assert live_run.observation.selected_note_ids == ()
            assert (
                live_run.observation.rejected_reasons.get(
                    live_run.observation.forbidden_memory_ids[0]
                )
                == "scope_mismatch"
            )

    visibility_workspace = tmp_path / "visibility-workspace"
    visibility_global_store = DurableMemoryStore(
        visibility_workspace / "global-memory",
        global_store=True,
    )
    stores.append(visibility_global_store)

    writer = _new_session(
        visibility_workspace,
        session_id="visibility-writer",
        global_store=visibility_global_store,
    )
    workspace_receipt = writer.memory_runtime.promote(
        "project-conventions",
        "release train is blue",
        source="live-smoke",
        visibility="workspace",
    )
    assert workspace_receipt.ok and workspace_receipt.promoted
    workspace_reader = _new_session(
        visibility_workspace,
        session_id="visibility-workspace-reader",
        global_store=visibility_global_store,
    )
    workspace_case = MemoryFixtureCase(
        case_id="live_workspace_cross_session",
        category="multi_session_reasoning",
        query="release train",
        expected_answer="release train is blue",
        required_evidence_ids=(workspace_receipt.note_id,),
    )
    workspace_run = _run_provider_case(
        session=workspace_reader,
        case=workspace_case,
        provider=provider,
        profile=profile,
    )
    assert workspace_run.observation.selected_note_ids == (workspace_receipt.note_id,)
    live_runs.append(workspace_run)
    observations.append(workspace_run.observation)
    stores.append(workspace_reader.memory_store)

    session_receipt = writer.memory_runtime.promote(
        "user-preferences",
        "private launch codename is aurora",
        source="live-smoke",
        visibility="session",
    )
    assert session_receipt.ok and session_receipt.promoted
    session_reader = _new_session(
        visibility_workspace,
        session_id="visibility-session-reader",
        global_store=visibility_global_store,
    )
    session_case = MemoryFixtureCase(
        case_id="live_session_isolation",
        category="multi_session_reasoning",
        query="private launch codename",
        expected_answer="unknown",
        forbidden_memory_ids=(session_receipt.note_id,),
        no_evidence=True,
    )
    session_run = _run_provider_case(
        session=session_reader,
        case=session_case,
        provider=provider,
        profile=profile,
    )
    assert session_run.observation.selected_note_ids == ()
    assert session_run.observation.rejected_reasons.get(session_receipt.note_id) == "session_mismatch"
    live_runs.append(session_run)
    observations.append(session_run.observation)
    stores.append(session_reader.memory_store)

    global_note = MemoryNote(
        topic="key-decisions",
        text="global support window is Friday",
        evidence=MemoryEvidence(scope="global", visibility="global"),
    )
    visibility_global_store.upsert_topic(global_note)
    global_reader = _new_session(
        visibility_workspace,
        session_id="visibility-global-reader",
        global_store=visibility_global_store,
    )
    global_case = MemoryFixtureCase(
        case_id="live_global_default_disabled",
        category="abstention",
        query="global support window",
        expected_answer="unknown",
        no_evidence=True,
    )
    global_run = _run_provider_case(
        session=global_reader,
        case=global_case,
        provider=provider,
        profile=profile,
    )
    assert global_run.observation.selected_note_ids == ()
    assert global_run.audit["associations"][0]["include_global"] is False
    # MemoryProjector.retrieve 的 public seam 接收 query 文本，global 开关由显式
    # keyword 控制；把 MemoryQuery 对象传进去会被 projector 当成字符串而丢掉字段。
    opted_in = global_reader.memory_projector.retrieve(
        "global support window",
        include_global=True,
    )
    assert [note.note_id for note in opted_in.selected_notes] == [note_id_for("key-decisions", global_note.text)]
    live_runs.append(global_run)
    observations.append(global_run.observation)
    stores.append(global_reader.memory_store)

    summary = summarize_memory_observations(observations)
    artifact_dir = Path(os.getenv(LIVE_ARTIFACT_DIR_ENV, "").strip() or (tmp_path / "memory-live-artifacts"))
    artifact_dir = artifact_dir.expanduser().resolve()
    _assert_artifact_is_separate(artifact_dir, tuple(stores))
    paths = write_memory_eval_artifacts(
        {
            "schema_version": 1,
            "mode": "live_smoke",
            "case_count": len(observations),
            "case_categories": dict(
                Counter(case.category for case in (*selected_cases, workspace_case, session_case, global_case))
            ),
            "provider": provider.name,
            "model": provider.model,
            "usage_source": "actual",
            "provider_calls": sum(run.provider_calls for run in live_runs),
            "input_tokens": sum(run.input_tokens for run in live_runs),
            "output_tokens": sum(run.output_tokens for run in live_runs),
            "variants": {"memory_on": summary},
        },
        artifact_dir,
    )

    artifact_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in paths.values()
    )
    assert "private launch codename is aurora" not in artifact_text
    assert "release train is blue" not in artifact_text
    assert all(Path(path).resolve().is_relative_to(artifact_dir) for path in paths.values())

    semantic_failures = [
        observation.case_id
        for observation in observations
        if not observation.answer_semantically_correct
    ]
    assert not semantic_failures, (
        "real provider memory smoke semantic failures: "
        + ", ".join(semantic_failures)
        + f"; inspect artifacts under {artifact_dir}"
    )
