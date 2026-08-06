"""Context-cost pairing reports for FirstCoder run artifacts.

本模块迁移 pico ``evaluation/context_cost.py`` 中的数据结构和统计规则，刻意
删除 Pico/ScriptedModelClient/真实 provider 的实验运行时。调用方只需提供一组
``report.json``、``trace.jsonl``，即可得到 actual-only、estimated-proxy-only 和
mixed/invalid 三个证据桶。

重要约束：estimated proxy 不是 provider 账单，mixed pair 不能用于 headline cost
claim；只有两边都有 actual usage、验证通过且没有质量回归时，``claimable_cost_win``
才可能为真。
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderPricing:
    """实验使用的配置价格；不是 provider 身份认证后的真实价格。"""

    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


@dataclass(frozen=True, slots=True)
class CostUsage:
    """一次 run 的 token 统计及来源。"""

    input_tokens: int
    cached_tokens: int
    output_tokens: int
    usage_source: str
    model_call_count: int

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, int(self.input_tokens) - int(self.cached_tokens))


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    """一条 treatment/control 实验行，字段与 report/trace 可审计来源对应。"""

    task_id: str
    layer: str
    variant: str
    repeat: int
    status: str
    verification_status: str
    tool_steps: int
    attempts: int
    prompt_estimated_tokens: int
    usage: CostUsage
    cost_usd: float
    saved_chars: int
    replacement_cache_hits: int
    summary_called: bool
    summary_delta_event_count: int
    report_path: str
    trace_path: str
    compact_summary_mode: str = ""
    compact_call_input_tokens: int = 0
    compact_call_output_tokens: int = 0
    compact_net_benefit_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usage"] = asdict(self.usage)
        return payload


DEFAULT_PROXY_PRICING = ProviderPricing(input_per_1m=2.0, cached_input_per_1m=0.2, output_per_1m=8.0)


def compute_cost_usd(usage: CostUsage, pricing: ProviderPricing) -> float:
    """按配置价格计算成本；cached input 从 uncached input 中扣除。"""

    return (
        usage.uncached_input_tokens * pricing.input_per_1m
        + int(usage.cached_tokens) * pricing.cached_input_per_1m
        + int(usage.output_tokens) * pricing.output_per_1m
    ) / 1_000_000


def extract_usage_from_artifacts(
    report_path: str | Path,
    trace_path: str | Path,
    *,
    task_id: str,
    layer: str,
    variant: str,
    repeat: int,
    pricing: ProviderPricing | None,
    verification_status: str | None = None,
    allow_verification_override: bool = False,
) -> ExperimentRow:
    """从一个 run 的 JSON artifacts 提取实验行。"""

    report_file = Path(report_path)
    trace_file = Path(trace_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    trace_usage = _usage_from_trace(trace_file)
    compact_metrics = _compact_metrics_from_trace(trace_file)
    evidence = dict(report.get("evidence_summaries", {}) or {})
    summary = dict(evidence.get("context_budget_summary", {}) or {})
    prompt_metadata = dict(report.get("prompt_metadata", {}) or {})
    orchestrator = dict(prompt_metadata.get("context_orchestrator", {}) or {})
    compact_call_usage = dict(
        compact_metrics.get("compact_call_usage")
        or summary.get("compact_call_usage")
        or orchestrator.get("compact_call_usage")
        or {}
    )
    if compact_call_usage and variant == "full_orchestrator_with_llm_handoff":
        trace_usage["usage"] = _usage_with_compact_call(trace_usage["usage"], compact_call_usage)

    derived_verification = _verification_status(report)
    if allow_verification_override and verification_status is not None:
        derived_verification = str(verification_status)
    usage = trace_usage["usage"]
    return ExperimentRow(
        task_id=str(task_id),
        layer=str(layer),
        variant=str(variant),
        repeat=int(repeat),
        status=str(report.get("status", "")),
        verification_status=derived_verification,
        tool_steps=int(report.get("tool_steps", 0) or 0),
        attempts=int(report.get("attempts", 0) or 0),
        prompt_estimated_tokens=trace_usage["estimated_input_tokens"],
        usage=usage,
        cost_usd=compute_cost_usd(usage, pricing) if pricing else 0.0,
        saved_chars=int(summary.get("saved_chars", 0) or 0),
        replacement_cache_hits=int(summary.get("replacement_cache_hits", 0) or 0),
        summary_called=bool(compact_metrics.get("summary_called") or summary.get("summary_called", False)),
        summary_delta_event_count=int(compact_metrics.get("summary_delta_event_count") or summary.get("summary_delta_event_count", 0) or 0),
        compact_summary_mode=str(compact_metrics.get("compact_summary_mode") or orchestrator.get("summary_mode", "") or ""),
        compact_call_input_tokens=int(compact_call_usage.get("input_tokens", 0) or 0),
        compact_call_output_tokens=int(compact_call_usage.get("output_tokens", 0) or 0),
        compact_net_benefit_tokens=(
            compact_metrics.get("compact_net_benefit_tokens")
            if compact_metrics.get("compact_net_benefit_tokens") is not None
            else summary.get("compact_net_benefit_tokens")
        ),
        report_path=report_file.as_posix(),
        trace_path=trace_file.as_posix(),
    )


def summarize_paired_rows(
    rows: Iterable[ExperimentRow],
    *,
    treatment: str = "full_orchestrator",
    control: str = "no_context_reduction",
) -> dict[str, Any]:
    """按 task/repeat/layer 配对并分别汇总三种 usage-source bucket。"""

    rows = list(rows)
    pairs = _paired_rows(rows, treatment=treatment, control=control)
    actual_pairs = [pair for pair in pairs if _pair_usage_source(pair, treatment, control) == "actual"]
    proxy_pairs = [pair for pair in pairs if _pair_usage_source(pair, treatment, control) == "estimated_proxy"]
    mixed_pairs = [pair for pair in pairs if _pair_usage_source(pair, treatment, control) == "mixed_or_invalid"]
    return {
        "actual_only": _summarize_pair_bucket(actual_pairs, treatment=treatment, control=control),
        "estimated_proxy_only": _summarize_pair_bucket(proxy_pairs, treatment=treatment, control=control),
        "mixed_or_invalid": _summarize_pair_bucket(mixed_pairs, treatment=treatment, control=control),
        "real_usage_row_count": sum(1 for row in rows if row.usage.usage_source == "actual"),
        "estimated_proxy_row_count": sum(1 for row in rows if row.usage.usage_source == "estimated_proxy"),
    }


def build_result_payload(
    rows: Iterable[ExperimentRow],
    *,
    pricing_profile: str,
    pricing: ProviderPricing | None = None,
    treatment: str = "full_orchestrator",
    control: str = "no_context_reduction",
) -> dict[str, Any]:
    """构造可序列化的实验结果 payload。"""

    rows = list(rows)
    return {
        "artifact_type": "context-cost-experiment",
        "pricing_profile": str(pricing_profile),
        "pricing": asdict(pricing) if pricing else None,
        "summary": summarize_paired_rows(rows, treatment=treatment, control=control),
        "rows": [row.to_dict() for row in rows],
    }


def collect_rows_from_run_manifest(
    manifest: dict[str, Any],
    *,
    pricing: ProviderPricing | None,
) -> list[ExperimentRow]:
    """从 provider-free manifest 收集多条 run row，不启动实验运行时。"""

    rows: list[ExperimentRow] = []
    for item in manifest.get("runs", []) or []:
        rows.append(
            extract_usage_from_artifacts(
                item["report_path"],
                item["trace_path"],
                task_id=item["task_id"],
                layer=item.get("layer", "live"),
                variant=item["variant"],
                repeat=item.get("repeat", 0),
                pricing=pricing,
            )
        )
    return rows


def write_experiment_artifacts(
    payload: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """写入 JSON、CSV、Markdown 三种报告格式。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    csv_path = output / "paired_rows.csv"
    markdown_path = output / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows_csv(payload.get("rows", []) or [], csv_path)
    markdown_path.write_text(generate_report(payload) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def generate_report(payload: dict[str, Any], include_llm_handoff_comparison: bool = False) -> str:
    """生成面向审查者的短 Markdown 报告。"""

    summary = dict(payload.get("summary", {}) or {})
    pricing = dict(payload.get("pricing", {}) or {})
    actual = dict(summary.get("actual_only", {}) or {})
    proxy = dict(summary.get("estimated_proxy_only", {}) or {})
    mixed = dict(summary.get("mixed_or_invalid", {}) or {})
    lines = [
        "# Context Cost Experiment",
        "",
        "## Summary",
        "",
        f"- Actual-only paired tasks: {actual.get('paired_task_count', 0)}",
        f"- Actual-only quality regressions: {actual.get('quality_regression_count', 0)}",
        f"- Actual-only unknown verification pairs: {actual.get('unknown_verification_count', 0)}",
        f"- Actual-only claimable cost win: {actual.get('claimable_cost_win', False)}",
        f"- Actual-only median cost delta: {float(actual.get('median_cost_delta_pct', 0) or 0):.2%}",
        f"- Estimated-proxy paired tasks: {proxy.get('paired_task_count', 0)}",
        f"- Mixed/invalid paired tasks: {mixed.get('paired_task_count', 0)}",
        f"- Real provider rows: {summary.get('real_usage_row_count', 0)}",
        f"- Estimated proxy rows: {summary.get('estimated_proxy_row_count', 0)}",
        "- Pricing basis: configured, not provider-authenticated",
        f"- Input $/1M: {pricing.get('input_per_1m', '-')}",
        f"- Cached input $/1M: {pricing.get('cached_input_per_1m', '-')}",
        f"- Output $/1M: {pricing.get('output_per_1m', '-')}",
    ]
    report = "\n".join(lines)
    if include_llm_handoff_comparison:
        report += "\n\n" + _render_llm_handoff_comparison(payload)
    return report


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Pico 兼容命名；报告内容仍由 FirstCoder 数据聚合器生成。"""

    return generate_report(payload)


def _usage_from_trace(trace_path: str | Path) -> dict[str, Any]:
    """解析 trace：全量 provider call 有真实 usage 才进入 actual bucket。"""

    estimated_input_tokens = 0
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    model_call_count = 0
    provider_metadata_count = 0
    for event in _read_jsonl(trace_path):
        if event.get("event") == "prompt_built":
            metadata = dict(event.get("prompt_metadata", {}) or {})
            usage = dict(metadata.get("context_usage", {}) or {})
            estimated_input_tokens += int(
                usage.get("total_estimated_tokens", event.get("estimated_input_tokens", 0)) or 0
            )
        if event.get("event") != "model_parsed":
            continue
        metadata = _completion_metadata(event)
        model_call_count += 1
        if _is_provider_usage_metadata(metadata):
            provider_metadata_count += 1
            input_tokens += int(metadata.get("input_tokens", 0) or 0)
            cached_tokens += int(metadata.get("cached_tokens", metadata.get("cached_input_tokens", 0)) or 0)
            output_tokens += int(metadata.get("output_tokens", 0) or 0)
    if model_call_count > 0 and provider_metadata_count == model_call_count:
        usage = CostUsage(input_tokens, cached_tokens, output_tokens, "actual", model_call_count)
    else:
        usage = CostUsage(estimated_input_tokens, 0, 0, "estimated_proxy", model_call_count)
    return {"estimated_input_tokens": estimated_input_tokens, "usage": usage}


def _compact_metrics_from_trace(trace_path: str | Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "summary_called": False,
        "summary_delta_event_count": 0,
        "compact_summary_mode": "",
        "compact_call_usage": None,
        "compact_net_benefit_tokens": None,
    }
    for event in _read_jsonl(trace_path):
        if event.get("event") != "context_orchestrator_decision":
            continue
        orchestrator = dict(event.get("context_orchestrator", {}) or {})
        if orchestrator.get("summary_mode"):
            metrics["compact_summary_mode"] = str(orchestrator.get("summary_mode", ""))
        metrics["summary_called"] = bool(metrics["summary_called"] or orchestrator.get("summary_called", False))
        metrics["summary_delta_event_count"] = max(
            int(metrics["summary_delta_event_count"] or 0),
            int(orchestrator.get("summary_delta_event_count", 0) or 0),
        )
        usage = orchestrator.get("compact_call_usage")
        if isinstance(usage, dict):
            metrics["compact_call_usage"] = dict(usage)
            pre_tokens = int(orchestrator.get("pre_compact_estimated_tokens", 0) or 0)
            post_tokens = int(orchestrator.get("post_compact_estimated_tokens", 0) or 0)
            compact_tokens = int(usage.get("total_tokens", 0) or 0)
            metrics["compact_net_benefit_tokens"] = pre_tokens - post_tokens - compact_tokens
    return metrics


def _usage_with_compact_call(usage: CostUsage, compact_call_usage: dict[str, Any]) -> CostUsage:
    return CostUsage(
        input_tokens=int(usage.input_tokens) + int(compact_call_usage.get("input_tokens", 0) or 0),
        cached_tokens=int(usage.cached_tokens) + int(compact_call_usage.get("cached_tokens", 0) or 0),
        output_tokens=int(usage.output_tokens) + int(compact_call_usage.get("output_tokens", 0) or 0),
        usage_source=usage.usage_source,
        model_call_count=int(usage.model_call_count) + 1,
    )


def _completion_metadata(event: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(event.get("completion_metadata", {}) or {})
    for key in ("provider_call", "provider_call_metadata"):
        nested = event.get(key)
        if isinstance(nested, dict):
            for nested_key, value in nested.items():
                metadata.setdefault(nested_key, value)
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        for key, value in usage.items():
            metadata.setdefault(key, value)
    if "cached_tokens" not in metadata and "cached_input_tokens" in metadata:
        metadata["cached_tokens"] = metadata["cached_input_tokens"]
    return metadata


def _is_provider_usage_metadata(metadata: dict[str, Any]) -> bool:
    return (
        (metadata.get("provider_protocol") is not None or metadata.get("protocol") is not None)
        and (metadata.get("provider_model") is not None or metadata.get("model") is not None)
        and metadata.get("input_tokens") is not None
        and metadata.get("output_tokens") is not None
        and metadata.get("synthetic") is not True
    )


def _verification_status(report: dict[str, Any]) -> str:
    signal = dict((report.get("evidence_summaries", {}) or {}).get("verification_signal", {}) or {})
    return str(signal.get("state", "")) or "unknown"


def _paired_rows(rows: Iterable[ExperimentRow], *, treatment: str, control: str) -> list[dict[str, ExperimentRow]]:
    by_key: dict[tuple[str, int, str], dict[str, ExperimentRow]] = {}
    for row in rows:
        by_key.setdefault((row.task_id, row.repeat, row.layer), {})[row.variant] = row
    return [variants for variants in by_key.values() if treatment in variants and control in variants]


def _quality_regressed(treatment: ExperimentRow, control: ExperimentRow) -> bool:
    if control.status == "completed" and treatment.status != "completed":
        return True
    if control.verification_status == "passed" and treatment.verification_status != "passed":
        return True
    if treatment.verification_status == "unknown" and control.verification_status != "unknown":
        return True
    if treatment.tool_steps > max(control.tool_steps + 2, int(control.tool_steps * 1.10)):
        return True
    return treatment.attempts > max(control.attempts + 1, int(control.attempts * 1.10))


def _summarize_pair_bucket(
    pairs: list[dict[str, ExperimentRow]],
    *,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    uncached_deltas = [_delta_pct(pair[treatment].usage.uncached_input_tokens, pair[control].usage.uncached_input_tokens) for pair in pairs]
    cost_deltas = [_delta_pct(pair[treatment].cost_usd, pair[control].cost_usd) for pair in pairs]
    return {
        "paired_task_count": len(pairs),
        "quality_regression_count": sum(1 for pair in pairs if _quality_regressed(pair[treatment], pair[control])),
        "unknown_verification_count": sum(1 for pair in pairs if pair[treatment].verification_status == "unknown" or pair[control].verification_status == "unknown"),
        "success_rate_treatment": _rate(pair[treatment].status == "completed" for pair in pairs),
        "success_rate_control": _rate(pair[control].status == "completed" for pair in pairs),
        "verifier_pass_rate_treatment": _rate(pair[treatment].verification_status == "passed" for pair in pairs),
        "verifier_pass_rate_control": _rate(pair[control].verification_status == "passed" for pair in pairs),
        "avg_tool_steps_treatment": _mean_rounded(pair[treatment].tool_steps for pair in pairs),
        "avg_tool_steps_control": _mean_rounded(pair[control].tool_steps for pair in pairs),
        "avg_attempts_treatment": _mean_rounded(pair[treatment].attempts for pair in pairs),
        "avg_attempts_control": _mean_rounded(pair[control].attempts for pair in pairs),
        "cost_per_successful_task_treatment": _cost_per_successful_task(pair[treatment] for pair in pairs),
        "cost_per_successful_task_control": _cost_per_successful_task(pair[control] for pair in pairs),
        "billable_input_tokens_per_task_treatment": _mean_rounded(pair[treatment].usage.uncached_input_tokens for pair in pairs),
        "billable_input_tokens_per_task_control": _mean_rounded(pair[control].usage.uncached_input_tokens for pair in pairs),
        "total_input_tokens_per_task_treatment": _mean_rounded(pair[treatment].usage.input_tokens for pair in pairs),
        "total_input_tokens_per_task_control": _mean_rounded(pair[control].usage.input_tokens for pair in pairs),
        "output_tokens_per_task_treatment": _mean_rounded(pair[treatment].usage.output_tokens for pair in pairs),
        "output_tokens_per_task_control": _mean_rounded(pair[control].usage.output_tokens for pair in pairs),
        "median_uncached_input_delta_pct": _median_rounded(uncached_deltas),
        "p95_uncached_input_delta_pct": _p95_rounded(uncached_deltas),
        "median_cost_delta_pct": _median_rounded(cost_deltas),
        # 只有 actual-only bucket 能成为成本声明；proxy/mixed 只能提供方向性证据。
        "claimable_cost_win": (
            _claimable_cost_win(pairs, treatment=treatment, control=control, cost_deltas=cost_deltas)
            if pairs
            and all(_pair_usage_source(pair, treatment, control) == "actual" for pair in pairs)
            else False
        ),
    }


def _pair_usage_source(pair: dict[str, ExperimentRow], treatment: str, control: str) -> str:
    sources = {pair[treatment].usage.usage_source, pair[control].usage.usage_source}
    if sources == {"actual"}:
        return "actual"
    if sources == {"estimated_proxy"}:
        return "estimated_proxy"
    return "mixed_or_invalid"


def _delta_pct(treatment: float, control: float) -> float:
    if not control:
        return 0.0
    return round((float(treatment) - float(control)) / float(control), 4)


def _median_rounded(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.median(values), 4) if values else 0.0


def _mean_rounded(values: Iterable[float | int]) -> float:
    values = list(values)
    return round(statistics.mean(values), 4) if values else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round(sum(1 for value in values if value) / len(values), 4) if values else 0.0


def _cost_per_successful_task(rows: Iterable[ExperimentRow]) -> float:
    rows = list(rows)
    successful = [row for row in rows if row.status == "completed" and row.verification_status == "passed"]
    if not successful:
        return 0.0
    return round(sum(row.cost_usd for row in rows) / len(successful), 8)


def _claimable_cost_win(
    pairs: list[dict[str, ExperimentRow]],
    *,
    treatment: str,
    control: str,
    cost_deltas: list[float],
) -> bool:
    if not pairs or not cost_deltas or _median_rounded(cost_deltas) >= 0:
        return False
    if any(
        pair[treatment].compact_net_benefit_tokens is not None
        and int(pair[treatment].compact_net_benefit_tokens) < 0
        for pair in pairs
    ):
        return False
    if any(_quality_regressed(pair[treatment], pair[control]) for pair in pairs):
        return False
    return all(pair[treatment].verification_status == "passed" and pair[control].verification_status == "passed" for pair in pairs)


def _p95_rounded(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
    return round(ordered[index], 4)


def _write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    fieldnames = sorted(
        {key for row in rows for key in row}
        | {"usage_input_tokens", "usage_cached_tokens", "usage_output_tokens", "usage_source"}
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            usage = dict(flat.pop("usage", {}) or {})
            flat["usage_input_tokens"] = usage.get("input_tokens", "")
            flat["usage_cached_tokens"] = usage.get("cached_tokens", "")
            flat["usage_output_tokens"] = usage.get("output_tokens", "")
            flat["usage_source"] = usage.get("usage_source", "")
            writer.writerow(flat)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    return [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render_llm_handoff_comparison(payload: dict[str, Any]) -> str:
    """渲染可选的 handoff 对比，不影响默认报告和成本判定。"""

    rows = [dict(row) for row in payload.get("rows", []) or []]
    by_pair: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("task_id", "")), int(row.get("repeat", 0) or 0), str(row.get("layer", "")))
        by_pair.setdefault(key, {})[str(row.get("variant", ""))] = row
    lines = ["## LLM Handoff vs Deterministic Comparison", ""]
    lines.append("| Task | Deterministic Cost | LLM Handoff Cost | Net Benefit |")
    lines.append("|------|-------------------|------------------|-------------|")
    for (task_id, _repeat, _layer), variants in sorted(by_pair.items()):
        deterministic = variants.get("full_orchestrator")
        handoff = variants.get("full_orchestrator_with_llm_handoff")
        if not deterministic or not handoff:
            continue
        net = handoff.get("compact_net_benefit_tokens")
        net_text = "n/a" if net is None else f"{int(net)} tokens"
        lines.append(
            f"| {task_id} | {float(deterministic.get('cost_usd', 0.0)):.8f} | "
            f"{float(handoff.get('cost_usd', 0.0)):.8f} | {net_text} |"
        )
    return "\n".join(lines)
