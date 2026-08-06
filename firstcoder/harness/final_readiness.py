"""Final-answer readiness gate over FirstCoder run evidence.

gate 是一个纯决策层：它读取 TaskState 已经持久化的 changed paths、verification、
governance 和 context summaries，返回 allow/warn/remind/block 决策；它不执行命令、
不修改工作区，也不直接改变 AgentLoop。严格模式的 hard reason 可由接线层映射到
``final_gate_blocked``，而 warn/soft 模式只提供可审计提示。

语义来源：pico ``core/final_readiness.py`` 及其 reason/artifact helpers；这里将
依赖合并到一个 FirstCoder 模块，避免引入 pico runtime。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

FINAL_READINESS_SUMMARY_SCHEMA = "firstcoder.final_readiness_summary.v1"
REQUIRED_ARTIFACT_SUMMARY_SCHEMA = "firstcoder.required_artifact_summary.v1"
CONTEXT_HARD_PRESSURE_RATIO = 0.95
VALID_MODES = {"off", "warn", "soft", "strict"}
UNRESOLVED_TODO_STATUS = {"pending", "in_progress"}

READINESS_REASONS = {
    "changed_paths_without_verification": ("hard", "Files changed, but no successful verification was recorded."),
    "failed_verification": ("hard", "The latest verification command failed."),
    "governance_denial": ("hard", "A runtime governance decision denied a requested tool action."),
    "partial_success_workspace_changed": ("hard", "A tool partially succeeded and changed the workspace."),
    "missing_required_artifact": ("hard", "A required output artifact mentioned in the request is still missing."),
    "unresolved_high_priority_todo": ("soft", "A current-run high priority todo is still unresolved."),
    "context_pressure_without_reduction": ("soft", "Context pressure is high and no successful reduction was recorded."),
    "tier3_summary_without_delta": ("soft", "Tier 3 context summary ran but had no new delta to summarize."),
    "replacement_ledger_disabled_under_pressure": ("soft", "Context pressure is high but the replacement ledger is disabled."),
    "provider_real_token_usage_unavailable": ("soft", "Provider real token usage was unavailable; context pressure used estimates."),
    "compact_net_negative": ("soft", "LLM compaction cost more tokens than it saved."),
    "compact_summary_quality_low": ("soft", "Compaction summary lacks concrete next steps or file references."),
    "context_pressure_compaction_failed": ("hard", "Context pressure is extreme but compaction yielded no token savings."),
}

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_OUTPUT_MARKERS = ("产出", "产物", "生成", "创建", "写入", "保存", "output", "artifact", "create", "write", "produce")
_INPUT_MARKERS = ("输入文件", "input file", "input files")
_NON_OUTPUT_MARKERS = ("约束", "评分", "评估", "constraints", "scoring", "evaluation")
_NEGATED_MARKERS = ("do not create", "don't create", "do not write", "don't write", "do not modify", "don't modify", "不要创建", "不要生成", "不要写入", "不要修改", "不创建", "不生成", "不修改")
_FILE_SUFFIXES = frozenset(
    [
        ".csv",
        ".html",
        ".json",
        ".jsonl",
        ".js",
        ".jsx",
        ".md",
        ".py",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    ]
)


def evaluate_final_readiness(
    task_state: Any,
    mode: str = "warn",
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """根据当前证据返回最终回答是否具备 readiness。"""

    resolved_mode = str(mode or "warn")
    if resolved_mode not in VALID_MODES:
        resolved_mode = "warn"
    reasons = readiness_reasons(task_state, workspace_root=workspace_root)
    signature = _reason_signature(reasons)
    state = _state(task_state)
    reminded = set(state.get("reminded_reason_signatures", []))
    already_sent = bool(signature and signature in reminded)
    decision = "allow"
    action = "none"
    if reasons and resolved_mode == "warn":
        decision = "warn"
    elif reasons and resolved_mode == "soft":
        decision, action = (("warn", "none") if already_sent else ("remind", "runtime_notice"))
        if not already_sent:
            reminded.add(signature)
    elif reasons and resolved_mode == "strict":
        decision, action = (("block", "block") if any(reason_severity(reason) == "hard" for reason in reasons) else ("warn", "none"))
    state["reminded_reason_signatures"] = sorted(reminded)
    return {
        "mode": resolved_mode,
        "decision": decision,
        "reasons": reasons,
        "reason_signature": signature,
        "reminder_already_sent": already_sent,
        "action": action,
        "required_artifact_summary": dict(
            (getattr(task_state, "evidence_summaries", {}) or {}).get("required_artifact_summary", {}) or {}
        ),
    }


def readiness_reasons(task_state: Any, workspace_root: str | Path | None = None) -> list[str]:
    """计算所有当前 readiness reason；顺序固定，方便报告和测试比较。"""

    summaries = dict(getattr(task_state, "evidence_summaries", {}) or {})
    reasons: list[str] = []
    required_artifacts = summarize_required_artifacts(task_state, workspace_root)
    if required_artifacts.get("declared_paths"):
        summaries["required_artifact_summary"] = required_artifacts
        task_state.evidence_summaries = summaries
    if required_artifacts.get("missing_paths"):
        reasons.append("missing_required_artifact")

    changed_paths = list(getattr(task_state, "changed_paths", []) or [])
    verification = dict(summaries.get("verification_signal", {}) or {})
    if changed_paths and verification.get("state") != "passed":
        reasons.append("changed_paths_without_verification")
    if verification.get("state") == "failed":
        reasons.append("failed_verification")
    if _has_partial_success_workspace_change(task_state):
        reasons.append("partial_success_workspace_changed")
    governance = dict(summaries.get("governance_summary", {}) or {})
    if int(governance.get("deny_count", 0) or 0):
        reasons.append("governance_denial")
    if _has_unresolved_high_priority_todo(task_state):
        reasons.append("unresolved_high_priority_todo")

    context = dict(summaries.get("context_budget_summary", {}) or {})
    if _context_pressure_without_reduction(context):
        reasons.append("context_pressure_without_reduction")
    if _tier3_summary_without_delta(context):
        reasons.append("tier3_summary_without_delta")
    if _replacement_ledger_disabled_under_pressure(context):
        reasons.append("replacement_ledger_disabled_under_pressure")
    if _provider_usage_unavailable(context):
        reasons.append("provider_real_token_usage_unavailable")
    if _compact_net_negative(context):
        reasons.append("compact_net_negative")
    if _compact_summary_quality_low(context):
        reasons.append("compact_summary_quality_low")
    if _context_pressure_compaction_failed(context):
        reasons.append("context_pressure_compaction_failed")
    return reasons


def readiness_notice(decision: dict[str, Any]) -> str:
    """将 gate 决策渲染为运行时可显示的提示文本。"""

    messages = [reason_message(reason) for reason in decision.get("reasons", [])]
    text = "\n".join(f"- {message}" for message in messages) or "- Readiness warning."
    if decision.get("action") == "block":
        return f"Final answer blocked by runtime readiness gate:\n{text}"
    return "Before final answer, address this runtime readiness issue:\n" f"{text}\nReturn final again only after addressing it or explaining why it is unavailable."


def reduce_final_readiness_summary(
    summary: dict[str, Any] | None,
    event: dict[str, Any],
) -> dict[str, Any]:
    """统计 final_readiness_decision 事件，保留最近原因。"""

    summary = dict(summary or {})
    summary.setdefault("schema_version", FINAL_READINESS_SUMMARY_SCHEMA)
    decision = str(event.get("decision", ""))
    summary[f"{decision}_count"] = int(summary.get(f"{decision}_count", 0) or 0) + 1
    for missing in ("allow_count", "warn_count", "remind_count", "block_count"):
        summary.setdefault(missing, 0)
    summary["last_decision"] = decision
    summary["last_reasons"] = list(event.get("reasons", []) or [])
    return summary


def reason_severity(reason: str) -> str:
    return READINESS_REASONS.get(str(reason), ("soft", str(reason)))[0]


def reason_message(reason: str) -> str:
    return READINESS_REASONS.get(str(reason), ("soft", str(reason)))[1]


def summarize_required_artifacts(task_state: Any, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """提取请求中明确声明的输出文件，并检查 workspace 内是否存在。"""

    root = Path(workspace_root).resolve() if workspace_root else None
    paths = extract_required_artifact_paths(str(getattr(task_state, "user_request", "")), root)
    missing = [path for path in paths if root and not (root / path).exists()]
    return {"schema_version": REQUIRED_ARTIFACT_SUMMARY_SCHEMA, "declared_paths": paths, "missing_paths": missing}


def extract_required_artifact_paths(text: str, workspace_root: str | Path | None = None) -> list[str]:
    """保守提取“生成/写入/保存”语境中的反引号文件路径。"""

    root = Path(workspace_root).resolve() if workspace_root else None
    paths: list[str] = []
    output_context = False
    output_dir = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if any(marker in lowered for marker in _INPUT_MARKERS) or _starts_non_output_section(line, lowered):
            output_context = False
            output_dir = ""
        if any(marker in lowered for marker in _NEGATED_MARKERS):
            continue
        marker_index = _first_marker_index(lowered, _OUTPUT_MARKERS)
        if marker_index >= 0:
            output_context = True
        line_output_dir = output_dir
        for match in _BACKTICK_RE.finditer(line):
            token = match.group(1).strip()
            normalized = _normalize_declared_path(token, root)
            if normalized and _looks_like_directory(token) and _line_declares_output_dir(line) and _token_has_output_scope(output_context, marker_index, match.start()):
                line_output_dir = normalized
                output_dir = normalized
        for match in _BACKTICK_RE.finditer(line):
            token = match.group(1).strip()
            normalized = _normalize_declared_path(token, root)
            if not normalized or _looks_like_directory(token) or not _token_has_output_scope(output_context, marker_index, match.start()):
                continue
            candidate = normalized
            if line_output_dir and "/" not in candidate and "\\" not in candidate:
                candidate = f"{line_output_dir.rstrip('/')}/{candidate}"
            if candidate not in paths:
                paths.append(candidate)
    return paths


def _state(task_state: Any) -> dict[str, Any]:
    summaries = dict(getattr(task_state, "evidence_summaries", {}) or {})
    state = dict(summaries.get("final_readiness_state", {}) or {})
    summaries["final_readiness_state"] = state
    task_state.evidence_summaries = summaries
    return state


def _reason_signature(reasons: list[str]) -> str:
    if not reasons:
        return ""
    return hashlib.sha256("|".join(sorted(reasons)).encode("utf-8")).hexdigest()[:16]


def _has_unresolved_high_priority_todo(task_state: Any) -> bool:
    latest: dict[str, dict[str, Any]] = {}
    for change in list(getattr(task_state, "todo_changes", []) or []):
        todo = dict(change.get("todo", {}) or {}) if isinstance(change, dict) else {}
        todo_id = str(todo.get("id", ""))
        if todo_id:
            latest[todo_id] = todo
    return any(todo.get("priority") == "high" and todo.get("status") in UNRESOLVED_TODO_STATUS for todo in latest.values())


def _has_partial_success_workspace_change(task_state: Any) -> bool:
    return any(
        isinstance(item, dict) and item.get("status") == "partial_success" and item.get("workspace_changed") is True
        for item in list(getattr(task_state, "runtime_reminders", []) or [])
    )


def _context_pressure_without_reduction(context: dict[str, Any]) -> bool:
    try:
        pressure = float(context.get("pressure_ratio", 0) or 0)
    except (TypeError, ValueError):
        pressure = 0.0
    return pressure >= CONTEXT_HARD_PRESSURE_RATIO and not any(int(item.get("saved_chars", 0) or 0) > 0 for item in context.get("reductions", []) or [])


def _tier3_summary_without_delta(context: dict[str, Any]) -> bool:
    return str(context.get("pressure_tier", "")) == "tier3_summary" and bool(context.get("summary_called", False)) and int(context.get("summary_delta_event_count", 0) or 0) == 0


def _replacement_ledger_disabled_under_pressure(context: dict[str, Any]) -> bool:
    return str(context.get("pressure_tier", "")) in {"tier2_prune", "tier3_summary"} and context.get("replacement_ledger_enabled") is False


def _provider_usage_unavailable(context: dict[str, Any]) -> bool:
    if not context:
        return False
    high_pressure = str(context.get("pressure_tier", "")) in {"tier2_prune", "tier3_summary"}
    try:
        ratio = float(context.get("pressure_ratio", 0) or 0)
    except (TypeError, ValueError):
        ratio = 0.0
    return context.get("provider_usage_available") is False and (high_pressure or ratio >= 0.8)


def _compact_net_negative(context: dict[str, Any]) -> bool:
    try:
        return context.get("compact_net_benefit_tokens") is not None and int(context.get("compact_net_benefit_tokens")) < 0
    except (TypeError, ValueError):
        return False


def _compact_summary_quality_low(context: dict[str, Any]) -> bool:
    return str(context.get("summary_mode", "")) == "llm" and (context.get("compact_summary_has_next_steps") is False or context.get("compact_summary_has_file_references") is False)


def _context_pressure_compaction_failed(context: dict[str, Any]) -> bool:
    if str(context.get("pressure_tier", "")) != "tier3_summary":
        return False
    try:
        pre = int(context.get("pre_compact_estimated_tokens", 0) or 0)
        post = int(context.get("post_compact_estimated_tokens", 0) or 0)
    except (TypeError, ValueError):
        return False
    return pre > 0 and post >= pre


def _normalize_declared_path(token: str, root: Path | None) -> str:
    value = str(token or "").strip().strip("\"'")
    if not value or any(part in value for part in ("*", "{", "}", "\n")) or value.startswith(("http://", "https://")):
        return ""
    path = Path(value).expanduser()
    if path.is_absolute():
        if root is None:
            return ""
        try:
            return str(path.resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            return ""
    return value.lstrip("./").replace("\\", "/")


def _looks_like_directory(token: str) -> bool:
    value = str(token or "").strip()
    return value.endswith(("/", "\\")) or Path(value).suffix.lower() not in _FILE_SUFFIXES


def _line_declares_output_dir(line: str) -> bool:
    lowered = str(line or "").lower()
    return any(marker in lowered for marker in ("写入", "output", "under", "保存到"))


def _starts_non_output_section(line: str, lowered: str) -> bool:
    sectionish = line.startswith("#") or line.endswith(":")
    return sectionish and any(marker in lowered for marker in _NON_OUTPUT_MARKERS)


def _first_marker_index(line: str, markers: tuple[str, ...]) -> int:
    positions = [position for marker in markers if (position := line.find(marker)) >= 0]
    return min(positions) if positions else -1


def _token_has_output_scope(output_context: bool, marker_index: int, token_start: int) -> bool:
    return token_start >= marker_index if marker_index >= 0 else output_context
