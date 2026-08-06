"""Structured trace events and redacting emitter (fusion P1, H3/H8).

Trace schema from pico `core/runtime_events.py` (H8): events are
normalized with stable trace/turn/phase/span fields before they enter the
run trace; event-specific semantics stay with the caller.

The emitter (H3) redacts payloads *before* persistence (the M8 read/write
two-sided redaction), tracks affected paths on the TaskState, and fans out
to consumers without letting a failing consumer break the run (H11).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from firstcoder.harness.run_store import RunStore

PHASE_BY_EVENT = {
    "run_started": "runtime",
    "context_orchestrator_decision": "prompt",
    "prompt_built": "prompt",
    "model_requested": "model",
    "model_parsed": "parse",
    "loop_transition": "loop",
    "governance_decision": "governance",
    "final_readiness_decision": "final_gate",
    "tool_executed": "tool",
    "checkpoint_created": "checkpoint",
    "compaction_started": "compact",
    "compaction_finished": "compact",
    "runtime_identity_mismatch": "runtime",
    "run_finished": "runtime",
}


def now_iso() -> str:
    """UTC ISO-8601 with Z suffix (FirstCoder convention, context/models.py)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_trace_event(
    event: str,
    payload: dict[str, Any] | None,
    *,
    trace_id: str,
    turn_id: str,
    span_seq: int = 0,
    parent_span_id: str = "",
) -> dict[str, Any]:
    """Normalize one runtime event into a stable trace record (H8).

    Sets the stable trace/turn/phase/span fields; leaves event-specific
    semantics to the caller. Pure — no I/O, no runtime state.
    """
    payload = dict(payload or {})
    payload["event"] = str(event)
    payload["created_at"] = now_iso()
    payload.setdefault("trace_id", trace_id)
    payload.setdefault("turn_id", turn_id)
    payload.setdefault("phase", PHASE_BY_EVENT.get(str(event), "runtime"))
    payload.setdefault("status", _status_for(event, payload))
    payload.setdefault("duration_ms", int(payload.get("duration_ms", 0) or 0))
    payload.setdefault("input_chars", int(payload.get("input_chars", 0) or 0))
    payload.setdefault("output_chars", int(payload.get("output_chars", 0) or 0))
    payload.setdefault("estimated_input_tokens", int(payload.get("estimated_input_tokens", 0) or 0))
    payload.setdefault("estimated_output_tokens", int(payload.get("estimated_output_tokens", 0) or 0))
    payload.setdefault("artifact_paths", list(payload.get("affected_paths", []) or []))
    payload.setdefault("error_type", "")
    if parent_span_id:
        payload["parent_span_id"] = str(parent_span_id)
    payload["span_id"] = f"span_{span_seq:06d}"
    return payload


def _status_for(event: str, payload: dict[str, Any]) -> str:
    if "status" in payload:
        return str(payload.get("status") or "")
    if event == "tool_executed":
        return str(payload.get("tool_status") or "ok")
    if event == "run_finished":
        return str(payload.get("run_status") or "")
    return ""


class TraceConsumer(Protocol):
    """Read-only observer of emitted trace events (H11 consumer boundary)."""

    def handle(self, task_state: Any, event: dict[str, Any]) -> None: ...


class TraceWriter:
    """Redact -> build -> append trace events for one run (H3 emitter).

    `redactor` follows the harness `Redactor` port (memory StaticSecurityPolicy
    satisfies it). Consumers are called after persistence; a consumer failure
    is recorded in `evidence_summaries.runtime_consumer_errors` and never
    breaks the run unless the consumer is marked critical.
    """

    def __init__(
        self,
        run_store: RunStore,
        redactor: Any,
        consumers: list[TraceConsumer] | None = None,
    ) -> None:
        self.run_store = run_store
        self.redactor = redactor
        self.consumers = list(consumers or [])
        self._span_seq = 0
        self._last_span_id = ""

    def emit(self, task_state: Any, event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.redactor.redact_artifact(payload or {})  # 先脱敏再持久化
        for path in payload.get("affected_paths", []) or []:
            if path not in task_state.changed_paths:
                task_state.changed_paths.append(path)
        self._span_seq += 1
        built = build_trace_event(
            event,
            payload,
            trace_id=task_state.run_id,
            turn_id=task_state.task_id,
            span_seq=self._span_seq,
            parent_span_id=self._last_span_id,
        )
        self._last_span_id = built["span_id"]
        self.run_store.append_trace(task_state, built)
        for consumer in self.consumers:
            try:
                consumer.handle(task_state, built)
            except Exception as exc:  # noqa: BLE001 — consumer must not break the run
                error = {
                    "consumer": consumer.__class__.__name__,
                    "event": str(event),
                    "span_id": built["span_id"],
                    "message": str(exc)[:200],
                    "critical": bool(getattr(consumer, "critical", False)),
                }
                task_state.evidence_summaries.setdefault("runtime_consumer_errors", []).append(error)
        self.run_store.write_task_state(task_state)
        return built
