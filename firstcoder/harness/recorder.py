"""普通 AgentLoop 的 run-level harness 接线。

P1/P4 的 harness 模块分别定义了持久化、trace、证据 reducer 和最终报告契约；
本模块只负责把这些契约组合成一个单轮运行记录器。它不参与 provider 请求构造，
也不改变 AgentLoop 的控制流，所有业务事件仍由 AgentLoop 在窄边界上主动发出。

这样做有两个目的：一是让普通 AgentLoop 也能产生可审计的 run 目录，二是让
verification、context usage、provider cost 和 final readiness 使用同一份 trace 事实。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from firstcoder.agent.stop_reason_mapping import map_turn_outcome
from firstcoder.agent.user_input import AgentTurnStatus
from firstcoder.context.token_budget import ContextBudget, estimate_text_tokens
from firstcoder.context.usage_calibration import ContextPressureController
from firstcoder.harness.evidence import update_evidence_summaries
from firstcoder.harness.final_readiness import evaluate_final_readiness
from firstcoder.harness.ports import RunArtifactStore
from firstcoder.harness.provider_call import ProviderCallMetadata, UsageSnapshot
from firstcoder.harness.report import build_report
from firstcoder.harness.run_store import RunStore
from firstcoder.harness.task_state import (
    STATUS_FAILED,
    STATUS_STOPPED,
    STOP_REASON_FINAL_ANSWER_RETURNED,
    STOP_REASON_FINAL_GATE_BLOCKED,
    TaskState,
)
from firstcoder.harness.trace import TraceWriter, now_iso
from firstcoder.harness.verification import build_verifier_suggestions
from firstcoder.memory.provenance import workspace_fingerprint
from firstcoder.memory.security import StaticSecurityPolicy

_MUTATING_TOOL_NAMES = frozenset(
    {
        "apply_patch",
        "delete",
        "edit",
        "write",
    }
)


class EvidenceSummaryConsumer:
    """把已落盘的 trace event 折叠进当前 TaskState。"""

    def handle(self, task_state: TaskState, event: dict[str, Any]) -> None:
        """消费单条脱敏 trace，保持 live TaskState 与 report reducer 同步。"""

        task_state.evidence_summaries = update_evidence_summaries(
            task_state.evidence_summaries,
            event,
            changed_paths=task_state.changed_paths,
        )


class RunRecorder:
    """一个 AgentLoop turn 的 run artifact 编排器。

    记录器默认把产物写入 session store 下的 ``runs/``，因此 CLI、TUI、测试和
    直接创建的 AgentLoop 都使用同一个相对布局。权限、memory 和 session JSONL
    仍由原有模块负责，harness 只是旁路事实视图。
    """

    def __init__(
        self,
        *,
        session: Any,
        user_request: str,
        readiness_mode: str = "warn",
        artifact_store_factory: Callable[[Path], RunArtifactStore] | None = None,
    ) -> None:
        self.session = session
        self.user_request = str(user_request)
        self.readiness_mode = readiness_mode
        self.security = StaticSecurityPolicy()
        self.workspace_root = _session_workspace_root(session)
        # RunStore 的构造函数会立即创建 runs 目录。普通 AgentLoop 可能在
        # 首次 provider 调用前收到取消信号，因此这里只保存根路径，避免
        # harness 的初始化写盘抢在真正的 provider 请求之前。
        self._run_store_root = Path(session.store.root) / "runs"
        self._artifact_store_factory = artifact_store_factory or RunStore
        self.run_store: RunArtifactStore | None = None
        self.task_state = TaskState.create(
            task_id=_task_id(session),
            user_request=self.user_request,
            session_id=str(session.session_id),
            workspace_fingerprint=(
                _workspace_fingerprint(self.workspace_root) if self.workspace_root is not None else ""
            ),
        )
        self.trace_writer: TraceWriter | None = None
        self.pressure_controller = ContextPressureController()
        self._active_calls: dict[str, ProviderCallMetadata] = {}
        self._last_completion_metadata: dict[str, Any] | None = None
        self._last_identity: dict[str, Any] | None = None
        self._prompt_records: list[dict[str, Any]] = []
        self._last_prompt_metadata: dict[str, Any] = {}
        self._tool_requested_count = 0
        # reducer 会按 transition 的字段做增量 fold；因此这里记录上一条
        # transition 已经报告过的累计值，避免并行工具的累计计数被重复相加。
        self._reported_tool_requested_count = 0
        self._reported_tool_executed_count = 0
        self._started = False
        self._materialized = False
        self._pending_events: list[tuple[str, dict[str, Any]]] = []
        self._finished = False
        self._harness_write_disabled = False

    def _mark_harness_degraded(self, error: Exception) -> None:
        """记录旁路 artifact 故障，并永久停止后续写盘尝试。

        主 AgentLoop 的 provider、工具和 session 都不依赖这个标记；一旦
        artifact 写入失败，继续重试只会让同一个故障反复覆盖主流程异常。
        """

        self._harness_write_disabled = True
        if not self.task_state.harness_degraded:
            self.task_state.harness_degraded = True
            self.task_state.harness_degradation_reason = "artifact_persistence_failed"

    def _safe_artifact_write(self, operation: Callable[[], Any]) -> bool:
        """执行一个持久化动作；失败时只降级 harness。"""

        if self._harness_write_disabled:
            return False
        try:
            operation()
        except Exception as exc:  # noqa: BLE001 - 旁路故障不能替换主流程异常
            self._mark_harness_degraded(exc)
            return False
        return True

    @property
    def run_id(self) -> str:
        """返回当前 run id，供调试和测试读取。"""

        return self.task_state.run_id

    @property
    def finished(self) -> bool:
        """返回当前 run 是否已经写出终局 artifact。"""

        return self._finished

    def start(self) -> None:
        """只标记逻辑启动，并把首个事实事件放入待落盘队列。"""

        if self._started:
            return
        self._started = True
        self._pending_events.append(
            (
                "run_started",
                {
                    "session_id": self.task_state.session_id,
                    "task_id": self.task_state.task_id,
                    "user_request": self.user_request,
                },
            )
        )

    def _materialize(self) -> bool:
        """首次需要持久化时创建 run store，并按原顺序刷出内存事件。"""

        self.start()
        if self._harness_write_disabled:
            return False
        if self._materialized:
            return True
        try:
            run_store = self._artifact_store_factory(self._run_store_root)
            trace_writer = TraceWriter(
                run_store,
                self.security,
                consumers=[EvidenceSummaryConsumer()],
            )
            pending_events = self._pending_events
            self._pending_events = []
            run_store.start_run(
                self.task_state,
                task_state_payload=self.security.redact_artifact(self.task_state.to_dict()),
            )
            self.run_store = run_store
            self.trace_writer = trace_writer
            self._materialized = True
            for event, payload in pending_events:
                trace_writer.emit(self.task_state, event, payload)
        except Exception as exc:  # noqa: BLE001 - artifact 故障必须降级
            self._mark_harness_degraded(exc)
            return False
        return True

    def _emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """统一写入入口：未物化时排队，物化后交给脱敏 trace writer。"""

        self.start()
        if not self._materialized:
            self._pending_events.append((str(event), dict(payload or {})))
            return
        if self._harness_write_disabled:
            return
        trace_writer = self.trace_writer
        if trace_writer is None:  # pragma: no cover - 物化状态由本类内部维护
            self._mark_harness_degraded(RuntimeError("run recorder trace writer is not initialized"))
            return
        self._safe_artifact_write(lambda: trace_writer.emit(self.task_state, event, payload))

    def record_prompt_built(self, prepared: Any, provider: Any) -> None:
        """记录一次已经完成预算计算的 provider prompt。"""

        self.start()
        budget: ContextBudget = prepared.context_budget
        prompt_hash = str(prepared.projection_fingerprint)
        request_id = str(getattr(prepared, "request_id", "") or "")
        identity = _provider_identity(provider, budget.context_window, prompt_hash)
        pressure = self.pressure_controller.evaluate(
            estimated_input_tokens=budget.input_tokens,
            context_window=budget.context_window,
            budget_tokens=budget.input_capacity,
            current_identity=identity,
            last_completion_metadata=self._last_completion_metadata,
            last_identity=self._last_identity,
        )
        context_usage = {
            "context_window": budget.context_window,
            "reserved_output_tokens": budget.output_reserve,
            "input_capacity": budget.input_capacity,
            "total_estimated_tokens": budget.input_tokens,
            "estimated_input_tokens": budget.input_tokens,
            "estimation_method": "firstcoder_context_budget",
            "prompt_hash": prompt_hash,
            **pressure.to_context_usage_fields(),
        }
        metadata = {
            "schema_version": "firstcoder.prompt_metadata.v1",
            "provider": identity["provider"],
            "provider_base_url": identity["provider_base_url"],
            "model": identity["model"],
            "request_id": request_id,
            "projection_fingerprint": prompt_hash,
            "prompt_hash": prompt_hash,
            "context_usage": context_usage,
        }
        self._last_prompt_metadata = metadata
        self._prompt_records.append(metadata)
        self._emit(
            "prompt_built",
            {
                "prompt_metadata": metadata,
                "request_id": request_id,
                "projection_fingerprint": prompt_hash,
                "estimated_input_tokens": budget.input_tokens,
                "input_chars": sum(len(message.content or "") for message in prepared.request.messages),
            },
        )

    def record_provider_requested(self, prepared: Any, provider: Any) -> None:
        """在 provider 调用前建立 request 与 response 的配对记录。"""

        self.start()
        budget: ContextBudget = prepared.context_budget
        protocol = _provider_protocol(provider)
        metadata = ProviderCallMetadata(
            call_id=str(prepared.request_id),
            session_id=self.task_state.session_id,
            turn_id=self.task_state.task_id,
            provider=str(provider.name),
            model=str(provider.model),
            projection_fingerprint=str(prepared.projection_fingerprint),
            protocol=protocol,
            base_url=str(getattr(provider, "base_url", "") or ""),
            request_at=now_iso(),
            prompt_estimated_tokens=budget.input_tokens,
            prompt_estimation_source="firstcoder_context_budget",
            call_kind="main",
        )
        self._active_calls[metadata.call_id] = metadata
        self.task_state.record_attempt()
        self._emit(
            "model_requested",
            {
                "provider_call": metadata.to_dict(),
                "request_id": metadata.call_id,
                "projection_fingerprint": metadata.projection_fingerprint,
                "provider_protocol": protocol,
                "provider_model": metadata.model,
            },
        )

    def record_provider_response(self, prepared: Any, provider: Any, response: Any) -> None:
        """记录 provider 响应、usage 和 finish reason。"""

        self.start()
        self._materialize()
        metadata = self._active_calls.pop(str(prepared.request_id), None)
        if metadata is None:
            self.record_provider_requested(prepared, provider)
            metadata = self._active_calls.pop(str(prepared.request_id))
        metadata.response_at = now_iso()
        metadata.finish_reason = str(response.finish_reason or "")
        metadata.usage = UsageSnapshot.from_usage(response.usage)
        completion = _completion_metadata(metadata, provider)
        self._last_completion_metadata = completion
        self._last_identity = _provider_identity(
            provider,
            prepared.context_budget.context_window,
            str(prepared.projection_fingerprint),
        )
        self._emit(
            "model_parsed",
            {
                "provider_call": metadata.to_dict(),
                "provider_call_metadata": metadata.to_dict(),
                "request_id": metadata.call_id,
                "projection_fingerprint": metadata.projection_fingerprint,
                "completion_metadata": completion,
                "finish_reason": metadata.finish_reason,
                "output_chars": len(str(response.content or "")),
                "estimated_output_tokens": estimate_text_tokens(str(response.content or "")),
            },
        )

    def record_provider_error(
        self,
        prepared: Any,
        provider: Any,
        error: Exception,
        *,
        error_type: str = "provider",
    ) -> None:
        """记录没有产生 ChatResponse 的 provider 失败。"""

        self.start()
        self._materialize()
        metadata = self._active_calls.pop(str(prepared.request_id), None)
        if metadata is None:
            self.record_provider_requested(prepared, provider)
            metadata = self._active_calls.pop(str(prepared.request_id))
        metadata.response_at = now_iso()
        metadata.finish_reason = (
            error_type if error_type in {"cancelled", "interrupted"} else "error"
        )
        metadata.error = str(error)
        completion = _completion_metadata(metadata, provider)
        self._emit(
            "model_parsed",
            {
                # 失败请求也必须带与成功请求相同的顶层关联键。provider_call
                # 嵌套对象适合保留完整事实，但 benchmark/evaluator 的公共事件
                # 契约应能在不展开 provider-specific payload 时完成 request 关联。
                "request_id": metadata.call_id,
                "projection_fingerprint": metadata.projection_fingerprint,
                "provider_call": metadata.to_dict(),
                "provider_call_metadata": metadata.to_dict(),
                "completion_metadata": completion,
                "finish_reason": metadata.finish_reason,
                "error_type": error_type,
                "error": str(error),
            },
        )

    def record_auxiliary_provider_requested(
        self,
        *,
        request: Any,
        request_id: str,
        projection_fingerprint: str,
        provider: Any,
        call_kind: str,
    ) -> None:
        """记录隐藏 provider 请求，但不把它投影为 session 消息。

        task-boundary classifier 等内部调用没有 ``PreparedMainRequest`` 和
        主上下文预算对象，因此这里使用同一套 request/fingerprint/call
        metadata 契约，按消息字符数生成可比较的估算值。
        """

        self.start()
        estimated_tokens = estimate_text_tokens(
            "\n".join(str(getattr(message, "content", "") or "") for message in getattr(request, "messages", []))
        )
        metadata = ProviderCallMetadata(
            call_id=str(request_id),
            session_id=self.task_state.session_id,
            turn_id=self.task_state.task_id,
            provider=str(provider.name),
            model=str(provider.model),
            projection_fingerprint=str(projection_fingerprint),
            protocol=_provider_protocol(provider),
            base_url=str(getattr(provider, "base_url", "") or ""),
            request_at=now_iso(),
            prompt_estimated_tokens=estimated_tokens,
            prompt_estimation_source="firstcoder_auxiliary_message_chars",
            call_kind=str(call_kind),
        )
        self._active_calls[metadata.call_id] = metadata
        prompt_metadata = {
            "schema_version": "firstcoder.prompt_metadata.v1",
            "provider": metadata.provider,
            "provider_base_url": metadata.base_url,
            "model": metadata.model,
            "request_id": metadata.call_id,
            "projection_fingerprint": metadata.projection_fingerprint,
            "call_kind": metadata.call_kind,
            "context_usage": {
                "total_estimated_tokens": estimated_tokens,
                "estimated_input_tokens": estimated_tokens,
                "estimation_method": "firstcoder_auxiliary_message_chars",
                "usage_source": "estimated",
            },
        }
        self._emit(
            "prompt_built",
            {
                "prompt_metadata": prompt_metadata,
                "request_id": metadata.call_id,
                "projection_fingerprint": metadata.projection_fingerprint,
                "call_kind": metadata.call_kind,
                "estimated_input_tokens": estimated_tokens,
                "input_chars": sum(len(str(getattr(message, "content", "") or "")) for message in getattr(request, "messages", [])),
            },
        )
        self.task_state.record_attempt()
        self._emit(
            "model_requested",
            {
                "provider_call": metadata.to_dict(),
                "request_id": metadata.call_id,
                "projection_fingerprint": metadata.projection_fingerprint,
                "call_kind": metadata.call_kind,
                "provider_protocol": metadata.protocol,
                "provider_model": metadata.model,
            },
        )

    def record_auxiliary_provider_response(
        self,
        *,
        request_id: str,
        projection_fingerprint: str,
        provider: Any,
        response: Any | None = None,
        error: Exception | None = None,
        error_type: str = "provider",
    ) -> None:
        """完成隐藏请求的 request/response 配对，失败也只写旁路事实。"""

        self.start()
        metadata = self._active_calls.pop(str(request_id), None)
        if metadata is None:
            return
        metadata.response_at = now_iso()
        metadata.projection_fingerprint = str(projection_fingerprint)
        if error is not None:
            metadata.finish_reason = str(error_type or "error")
            metadata.error = str(error)
            self._emit(
                "model_parsed",
                {
                    "request_id": metadata.call_id,
                    "projection_fingerprint": metadata.projection_fingerprint,
                    "call_kind": metadata.call_kind,
                    "provider_call": metadata.to_dict(),
                    "provider_call_metadata": metadata.to_dict(),
                    "error_type": str(error_type),
                    "error": str(error),
                },
            )
            return
        metadata.finish_reason = str(getattr(response, "finish_reason", "") or "")
        metadata.usage = UsageSnapshot.from_usage(getattr(response, "usage", None))
        completion = _completion_metadata(metadata, provider)
        self._emit(
            "model_parsed",
            {
                "request_id": metadata.call_id,
                "projection_fingerprint": metadata.projection_fingerprint,
                "call_kind": metadata.call_kind,
                "provider_call": metadata.to_dict(),
                "provider_call_metadata": metadata.to_dict(),
                "completion_metadata": completion,
                "finish_reason": metadata.finish_reason,
                "output_chars": len(str(getattr(response, "content", "") or "")),
                "estimated_output_tokens": estimate_text_tokens(str(getattr(response, "content", "") or "")),
            },
        )

    def record_context_decision(self, result: Any, budget: ContextBudget) -> None:
        """把 L1-L4 压缩结果投影为 context budget reducer 的事实事件。"""

        self.start()
        before_tokens = int(getattr(result, "before_tokens", budget.input_tokens) or 0)
        after_tokens = int(getattr(result, "after_tokens", before_tokens) or 0)
        programmatic = getattr(result, "programmatic_event", None)
        l4_event = getattr(result, "l4_event", None)
        previous_usage = dict(self._last_prompt_metadata.get("context_usage", {}) or {})
        orchestrator = {
            "status": str(getattr(result, "status", "") or ""),
            "reason": str(getattr(result, "reason", "") or ""),
            "pre_compact_estimated_tokens": before_tokens,
            "post_compact_estimated_tokens": after_tokens,
            "summary_called": l4_event is not None,
            "summary_mode": "llm" if l4_event is not None else "",
            "summary_delta_event_count": 0,
            "fallback_steps": list(getattr(result, "fallback_steps", None) or []),
            "compact_call_usage": (
                dict(getattr(l4_event, "compact_call_usage", {}) or {}) or None
            ),
        }
        usage = {
            "context_window": budget.context_window,
            "reserved_output_tokens": budget.output_reserve,
            "budget_tokens": budget.input_capacity,
            "total_estimated_tokens": after_tokens,
            "actual_input_tokens": None,
            # context reducer 是增量更新；把 prompt 阶段已有的压力字段带过来，
            # 避免一个没有新实测 usage 的压缩事件清空 pressure tier。
            "pressure_ratio": previous_usage.get("pressure_ratio", 0),
            "pressure_tier": previous_usage.get("pressure_tier", ""),
            "usage_source": previous_usage.get("usage_source", "estimated"),
        }
        payload = {
            "context_orchestrator": orchestrator,
            "context_usage": usage,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "programmatic_event": _dataclass_payload(programmatic),
            "l4_event": _dataclass_payload(l4_event),
        }
        self._emit(
            "context_orchestrator_decision",
            payload,
        )

    def record_tool_event(self, event: Any) -> None:
        """把本地工具执行转成 verification 可消费的 trace event。"""

        self.start()
        kind = str(event.kind)
        if kind in {"started", "background_started"}:
            self._tool_requested_count += 1
            return
        if kind in {"permission_requested", "denied"}:
            self._emit(
                "governance_decision",
                {
                    "decision": "ask" if kind == "permission_requested" else "deny",
                    "reason_code": kind,
                    "tool_name": str(event.tool_call.name),
                },
            )
        if kind not in {"finished", "denied", "skipped", "interrupted"}:
            return

        result = event.result
        data = dict(getattr(result, "data", {}) or {}) if result is not None else {}
        dry_run = bool(data.get("dry_run") is True)
        tool_name = str(event.tool_call.name)
        paths = (
            _changed_paths(
                data,
                workspace_root=self.workspace_root,
                include_single_path=tool_name in _MUTATING_TOOL_NAMES,
            )
            if not dry_run
            else []
        )
        ok = bool(result is not None and result.ok)
        if kind == "finished":
            self.task_state.record_tool(tool_name)
        requested_count = self._tool_requested_count
        executed_count = self.task_state.tool_steps
        tool_requested_delta = requested_count - self._reported_tool_requested_count
        tool_executed_delta = executed_count - self._reported_tool_executed_count
        self._reported_tool_requested_count = requested_count
        self._reported_tool_executed_count = executed_count
        payload = {
            "name": tool_name,
            "args": event.tool_call.arguments,
            "data": data,
            "output": str(getattr(result, "content", "") or "")[:20_000],
            "error": str(getattr(result, "error", "") or ""),
            "ok": ok,
            "tool_status": "success" if ok else "failed",
            "status": "success" if ok else "failed",
            "affected_paths": paths,
            "workspace_changed": not dry_run
            and (
                bool(paths)
                or (kind == "finished" and ok and tool_name in _MUTATING_TOOL_NAMES)
            ),
        }
        self._emit("tool_executed", payload)
        self._emit(
            "loop_transition",
            {
                "kind": "continue",
                "reason": "tool_completed",
                "attempt_index": max(0, self.task_state.attempts - 1),
                "tool_requested_count": tool_requested_delta,
                "tool_executed_count": tool_executed_delta,
            },
        )

    def finish(self, response: Any) -> None:
        """完成正常或 guardrail 终局，并原子写出 report.json。"""

        if self._finished:
            return
        try:
            self.start()
            stop_reason = map_turn_outcome(
                status=AgentTurnStatus.COMPLETED.value,
                finish_reason=response.finish_reason,
            ) or STOP_REASON_FINAL_ANSWER_RETURNED
            if stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED:
                self.task_state.finish_success(str(response.content or ""))
            else:
                self.task_state.stop(stop_reason, status=STATUS_STOPPED, final_answer=str(response.content or ""))
            stop_reason = self._write_terminal_events(stop_reason=stop_reason, response=response)
            self._write_report()
        except Exception as exc:  # noqa: BLE001 - harness 故障不得改变主回答
            self._mark_harness_degraded(exc)
        finally:
            self._finished = True

    def fail(self, error: Exception, *, error_type: str = "provider") -> None:
        """为未产生最终 ChatResponse 的异常写出失败 run。"""

        if self._finished:
            return
        try:
            self.start()
            finish_reason = (
                error_type if error_type in {"cancelled", "interrupted"} else "error"
            )
            stop_reason = map_turn_outcome(
                status=AgentTurnStatus.COMPLETED.value,
                finish_reason=finish_reason,
                error_type=error_type,
            )
            self.task_state.stop(
                stop_reason or "model_error",
                status=STATUS_FAILED,
                final_answer="",
            )
            self._emit("run_error", {"error": str(error), "error_type": error_type})
            self._write_terminal_events(stop_reason=self.task_state.stop_reason, response=None)
            self._write_report()
        except Exception as exc:  # noqa: BLE001 - 保留原始 provider/tool 异常
            self._mark_harness_degraded(exc)
        finally:
            self._finished = True

    def _write_terminal_events(self, *, stop_reason: str, response: Any | None) -> str:
        # readiness 需要看到此前排队的 tool_executed 变更路径，所以终局判定
        # 前必须先物化 trace；这同时保证 task_state 的 consumer 已经同步。
        self._materialize()
        decision = evaluate_final_readiness(
            self.task_state,
            mode=self.readiness_mode,
            workspace_root=self.workspace_root,
        )
        if (
            decision.get("decision") == "block"
            and response is not None
            and stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
        ):
            # strict gate 的 block 必须改变最终 TaskState，而不是只留下一个
            # 审计事件；否则 report 会同时声称“已完成”和“final gate 阻断”。
            stop_reason = map_turn_outcome(
                status=AgentTurnStatus.COMPLETED.value,
                error_type="final_gate",
            ) or STOP_REASON_FINAL_GATE_BLOCKED
            self.task_state.stop(
                stop_reason,
                status=STATUS_STOPPED,
                final_answer=str(getattr(response, "content", "") or ""),
            )
        self._emit("final_readiness_decision", decision)
        self._emit(
            "loop_transition",
            {
                "kind": "terminal",
                "reason": stop_reason,
                "stop_reason": stop_reason,
                "attempt_index": max(0, self.task_state.attempts - 1),
                "tool_requested_count": self._tool_requested_count,
                "tool_executed_count": self.task_state.tool_steps,
            },
        )
        self._emit(
            "run_finished",
            {
                "run_status": self.task_state.status,
                "stop_reason": stop_reason,
                "output_chars": len(str(getattr(response, "content", "") or "")),
            },
        )
        return stop_reason

    def _write_report(self) -> None:
        """从终局 TaskState 聚合 report，避免报告重新读取未脱敏 trace。"""

        if not self._materialize():
            return
        run_store = self.run_store
        if run_store is None:  # pragma: no cover - _materialize 已保证初始化
            self._mark_harness_degraded(RuntimeError("run recorder store is not initialized"))
            return
        root = self.workspace_root or Path.cwd()
        self.task_state.verifier_suggestions = build_verifier_suggestions(
            root,
            self.task_state.changed_paths,
        )
        # suggestions 是终局阶段才知道的字段，必须在 report 之外同步回 TaskState，
        # 否则 live inspector 看到的 task_state.json 会落后于 report.json。
        if not self._safe_artifact_write(
            lambda: run_store.write_task_state(
                self.task_state,
                payload=self.security.redact_artifact(self.task_state.to_dict()),
            )
        ):
            return
        prompt_metadata = dict(self._last_prompt_metadata)
        prompt_metadata["request_count"] = len(self._prompt_records)
        prompt_metadata["requests"] = list(self._prompt_records)
        compactions = [
            asdict(event)
            for event in getattr(self.session.runtime_state, "recent_compaction_events", [])
        ]
        report = build_report(
            self.task_state,
            prompt_metadata=prompt_metadata,
            compactions=compactions,
            redacted_env={"secret_env_count": len(self.security.detected_secret_env_items())},
        )
        self._safe_artifact_write(
            lambda: run_store.write_report(self.task_state, self.security.redact_artifact(report))
        )


def _session_workspace_root(session: Any) -> Path | None:
    """从 memory store 或权限策略取得当前 session 的 workspace root。"""

    root = getattr(getattr(session, "memory_store", None), "workspace_root", None)
    if root is None:
        policy = getattr(getattr(session, "permission_manager", None), "policy", None)
        root = getattr(policy, "project_root", None)
    return Path(root).resolve() if root is not None else None


def _task_id(session: Any) -> str:
    benchmark_task = str(getattr(session, "benchmark_task", "") or "").strip()
    if benchmark_task:
        return benchmark_task
    return f"{session.session_id}-turn-{int(getattr(session, 'current_turn', 0)) + 1}"


def _provider_protocol(provider: Any) -> str:
    """只读取 ChatProvider 统一协议属性，不让 harness 依赖 adapter 模块名。"""

    return str(getattr(provider, "protocol", "custom") or "custom").strip() or "custom"


def _workspace_fingerprint(root: Path) -> str:
    """为非 Git 临时目录避免启动 git 子进程，仍保持稳定 workspace 身份。"""

    if not (root / ".git").exists():
        return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return workspace_fingerprint(root)


def _provider_identity(provider: Any, context_window: int, prompt_hash: str) -> dict[str, Any]:
    """构造 usage calibration 所需的六项 identity。"""

    return {
        "provider": str(provider.name),
        "provider_base_url": str(getattr(provider, "base_url", "") or ""),
        "model": str(provider.model),
        "context_window": int(context_window),
        "prompt_cache_key": str(getattr(provider, "prompt_cache_key", "") or ""),
        "prompt_hash": prompt_hash,
    }


def _completion_metadata(metadata: ProviderCallMetadata, provider: Any) -> dict[str, Any]:
    """把嵌套契约展平为 context-cost reducer 兼容的 completion metadata。"""

    usage = metadata.usage.to_dict()
    return {
        "call_id": metadata.call_id,
        "request_id": metadata.call_id,
        "projection_fingerprint": metadata.projection_fingerprint,
        "provider": metadata.provider,
        "provider_model": metadata.model,
        "provider_protocol": metadata.protocol,
        "provider_base_url": metadata.base_url,
        "protocol": metadata.protocol,
        "base_url": metadata.base_url,
        "model": metadata.model,
        "request_at": metadata.request_at,
        "response_at": metadata.response_at,
        "finish_reason": metadata.finish_reason,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "cached_tokens": usage.get("cached_input_tokens"),
        "usage": usage,
        "error": metadata.error,
        "synthetic": False,
        "provider_class": provider.__class__.__name__,
    }


def _dataclass_payload(value: Any) -> dict[str, Any] | None:
    """将 context 层 dataclass 事件转成可安全写入 JSONL 的对象。"""

    return asdict(value) if value is not None else None


def _changed_paths(
    data: dict[str, Any],
    *,
    workspace_root: Path | None,
    include_single_path: bool = False,
) -> list[str]:
    """提取并规范化 workspace 内路径，供 readiness/verification 使用。

    dry-run 在调用方先被排除；这里仍执行边界检查，避免恶意工具结果把绝对路径
    或上级路径写进 run 证据，后续 readiness 再把它当作真实改动。
    """

    values: list[Any] = []
    keys = ("changed_files", "created_files", "deleted_files", "moved_files", "affected_paths")
    if include_single_path:
        # view/review/git_log 等只读工具也返回 path；只有明确的写工具才可
        # 把这个单值字段解释为 workspace 变更。
        keys = ("path", *keys)
    for key in keys:
        value = data.get(key)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, dict):
                    values.extend(item.get(name) for name in ("source", "destination", "path"))
                else:
                    values.append(item)
        elif value:
            if isinstance(value, dict):
                values.extend(value.get(name) for name in ("source", "destination", "path"))
            else:
                values.append(value)
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized = _normalize_changed_path(text, workspace_root=workspace_root)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _normalize_changed_path(text: str, *, workspace_root: Path | None) -> str:
    """把工具路径转成 workspace-relative POSIX 形式，越界值直接丢弃。"""

    candidate = Path(text)
    if workspace_root is None:
        if candidate.is_absolute() or ".." in candidate.parts:
            return ""
        return text.replace("\\", "/").removeprefix("./")
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
        relative = resolved.relative_to(workspace_root)
    except (OSError, ValueError):
        return ""
    return relative.as_posix() if relative != Path(".") else ""
