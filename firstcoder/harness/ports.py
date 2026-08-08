"""Stable protocol ports for the harness domain (fusion P0).

Run-level evidence contracts: trace events, run reports, session facts
read-only access, and artifact redaction. Contracts only — implementations
land in P1 (RunStore/TaskState/trace/report). See docs/fusion-plan-review.md §3.
"""

from __future__ import annotations

from typing import Any, Protocol


class SessionSource(Protocol):
    """Read-only access to session facts for run projection (fusion §5.4.1)."""

    def list_events(self, session_id: str) -> list[Any]: ...


class TraceSink(Protocol):
    """Emit run-level trace events (redacted before persistence, fusion H3).

    参数顺序与 ``TraceWriter.emit`` 对齐，避免协议只在文档中存在、运行时
    却不能替换实际实现。
    """

    def emit(
        self,
        task_state: Any,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class RunReportSink(Protocol):
    """Write terminal run reports (fusion H4)."""

    def write_report(self, task_state: Any, report: dict[str, Any]) -> Any: ...


class RunArtifactStore(Protocol):
    """一个 run artifact store 的最小写入边界。

    ``RunRecorder`` 只依赖这四个方法，因此测试可以注入故障或内存实现，
    普通 AgentLoop 仍默认使用 ``RunStore``。协议不暴露目录布局，避免
    harness 调用方复制第二套路径和原子写规则。
    """

    def start_run(
        self,
        task_state: Any,
        *,
        task_state_payload: dict[str, Any] | None = None,
    ) -> Any: ...

    def write_task_state(
        self,
        task_state: Any,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any: ...

    def append_trace(self, task_state: Any, event: dict[str, Any]) -> Any: ...

    def write_report(self, task_state: Any, report: dict[str, Any]) -> Any: ...


class Redactor(Protocol):
    """Redact secrets from text and artifacts before persistence (fusion H3/M8).

    Aligned with the actual recursive interface of
    `memory.security.StaticSecurityPolicy` (Codex P1 review fix): `redact`
    handles plain text; `redact_artifact` recursively redacts any value,
    which is what `TraceWriter.emit` calls before persistence.

    与 `memory.security.StaticSecurityPolicy` 的实际接口对齐：
    `redact` 处理纯文本，`redact_artifact` 递归脱敏任意值
    （`TraceWriter.emit` 持久化前调用）。
    """

    def redact(self, text: str) -> str: ...
    def redact_artifact(self, value: Any, key: str | None = None) -> Any: ...
