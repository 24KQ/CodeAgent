"""P4 harness tests: verification, readiness, usage calibration and cost report."""

from __future__ import annotations

import json
from pathlib import Path

from firstcoder.agent.stop_reason_mapping import map_finish_reason, map_turn_outcome
from firstcoder.context.budget_summary import context_budget_summary
from firstcoder.context.usage_calibration import ContextPressureController
from firstcoder.harness.evidence import update_evidence_summaries
from firstcoder.harness.experiments.context_cost import (
    CostUsage,
    ExperimentRow,
    ProviderPricing,
    extract_usage_from_artifacts,
    summarize_paired_rows,
    write_experiment_artifacts,
)
from firstcoder.harness.final_readiness import evaluate_final_readiness
from firstcoder.harness.report import build_report
from firstcoder.harness.task_state import TaskState
from firstcoder.harness.verification import (
    build_verifier_suggestions,
    classify_verification_command,
    reduce_verification_signal,
)


def _identity() -> dict[str, object]:
    return {
        "provider": "test-provider",
        "provider_base_url": "https://example.invalid/v1",
        "model": "test-model",
        "context_window": 1000,
        "prompt_cache_key": "prefix-1",
        "prompt_hash": "prompt-1",
    }


def _row(variant: str, *, usage_source: str = "actual", cost: float = 1.0) -> ExperimentRow:
    return ExperimentRow(
        task_id="task-1",
        layer="fixture",
        variant=variant,
        repeat=0,
        status="completed",
        verification_status="passed",
        tool_steps=1,
        attempts=1,
        prompt_estimated_tokens=100,
        usage=CostUsage(100, 10, 20, usage_source, 1),
        cost_usd=cost,
        saved_chars=10,
        replacement_cache_hits=0,
        summary_called=False,
        summary_delta_event_count=0,
        report_path="report.json",
        trace_path="trace.jsonl",
    )


def test_verification_command_classification_is_conservative() -> None:
    assert classify_verification_command("python -m pytest -q") == "test"
    assert classify_verification_command("uv run --quiet python -m compileall .") == "compile"
    assert classify_verification_command("ruff check firstcoder") == "lint"
    assert classify_verification_command("python -c 'print(1)'") == ""


def test_verification_signal_resets_after_workspace_change_and_recovers() -> None:
    signal = reduce_verification_signal(
        None,
        {
            "event": "tool_executed",
            "name": "diagnostics",
            "workspace_changed": True,
            "span_id": "span-write",
            "affected_paths": ["src/a.py"],
        },
    )
    assert signal["state"] == "missing"
    signal = reduce_verification_signal(
        signal,
        {
            "event": "tool_executed",
            "name": "diagnostics",
            "args": {"command": "python -m pytest -q"},
            "status": "ok",
            "span_id": "span-test",
        },
        ["src/a.py"],
    )
    assert signal["state"] == "passed"
    assert signal["after_last_workspace_change"] is True


def test_evidence_reducer_dispatches_p4_leaves() -> None:
    summaries = update_evidence_summaries(
        None,
        {
            "event": "prompt_built",
            "prompt_metadata": {
                "context_usage": {
                    "context_window": 1000,
                    "reserved_output_tokens": 100,
                    "total_estimated_tokens": 700,
                }
            },
        },
    )
    summaries = update_evidence_summaries(
        summaries,
        {
            "event": "governance_decision",
            "decision": "deny",
            "reason_code": "permission_denied",
        },
    )
    summaries = update_evidence_summaries(
        summaries,
        {"event": "final_readiness_decision", "decision": "warn", "reasons": ["x"]},
    )
    assert summaries["context_budget_summary"]["estimated_tokens"] == 700
    assert summaries["governance_summary"]["deny_count"] == 1
    assert summaries["final_readiness_summary"]["warn_count"] == 1


def test_final_readiness_strict_blocks_changed_workspace_without_verification() -> None:
    state = TaskState.create("task-1", "生成 `result.json` 产物")
    state.changed_paths.append("src/a.py")
    decision = evaluate_final_readiness(state, mode="strict", workspace_root=Path.cwd())
    assert decision["decision"] == "block"
    assert "changed_paths_without_verification" in decision["reasons"]
    assert "missing_required_artifact" in decision["reasons"]


def test_final_readiness_soft_reminder_is_deduplicated() -> None:
    state = TaskState.create("task-1", "继续工作")
    state.evidence_summaries["context_budget_summary"] = {
        "pressure_ratio": 0.9,
        "pressure_tier": "tier2_prune",
        "provider_usage_available": False,
        "replacement_ledger_enabled": False,
    }
    first = evaluate_final_readiness(state, mode="soft")
    second = evaluate_final_readiness(state, mode="soft")
    assert first["decision"] == "remind"
    assert second["decision"] == "warn"
    assert second["reminder_already_sent"] is True


def test_context_pressure_uses_actual_only_for_matching_identity() -> None:
    controller = ContextPressureController()
    pressure = controller.evaluate(
        estimated_input_tokens=80,
        context_window=1000,
        budget_tokens=900,
        current_identity=_identity(),
        last_identity=_identity(),
        last_completion_metadata={
            **_identity(),
            "input_tokens": 120,
            "cached_tokens": 20,
        },
    )
    assert pressure.input_tokens == 120
    assert pressure.usage_source == "actual"
    assert pressure.cached_tokens == 20

    mismatch_metadata = dict(_identity(), prompt_hash="different")
    pressure = controller.evaluate(
        estimated_input_tokens=80,
        context_window=1000,
        current_identity=_identity(),
        last_identity=_identity(),
        last_completion_metadata={**mismatch_metadata, "input_tokens": 120},
    )
    assert pressure.input_tokens == 80
    assert pressure.calibration_source == "last_completion_identity_mismatch"


def test_context_budget_summary_tracks_reductions() -> None:
    summary = context_budget_summary(
        {
            "context_usage": {
                "context_window": 1000,
                "reserved_output_tokens": 100,
                "total_estimated_tokens": 700,
                "actual_input_tokens": 700,
            },
            "budget_reductions": [{"section": "history", "before_chars": 100, "after_chars": 20}],
        }
    )
    assert summary["effective_window"] == 900
    assert summary["saved_chars"] == 80
    assert summary["provider_usage_available"] is True


def test_context_cost_extracts_actual_usage_and_writes_artifacts(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "trace.jsonl"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "tool_steps": 1,
                "attempts": 1,
                "evidence_summaries": {"verification_signal": {"state": "passed"}},
            }
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "prompt_built", "prompt_metadata": {"context_usage": {"total_estimated_tokens": 90}}}),
                json.dumps(
                    {
                        "event": "model_parsed",
                        "completion_metadata": {
                            "provider_protocol": "openai-compatible",
                            "provider_model": "test-model",
                            "input_tokens": 100,
                            "cached_tokens": 20,
                            "output_tokens": 30,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    row = extract_usage_from_artifacts(
        report_path,
        trace_path,
        task_id="task-1",
        layer="fixture",
        variant="full_orchestrator",
        repeat=0,
        pricing=ProviderPricing(2.0, 0.2, 8.0),
    )
    assert row.usage.usage_source == "actual"
    assert row.usage.uncached_input_tokens == 80

    output = tmp_path / "artifacts"
    written = write_experiment_artifacts(
        {"summary": {}, "rows": [row.to_dict()], "pricing": None}, output
    )
    assert all(Path(path).is_file() for path in written.values())


def test_context_cost_claimable_win_requires_actual_and_passed_pairs() -> None:
    summary = summarize_paired_rows(
        [_row("no_context_reduction", cost=2.0), _row("full_orchestrator", cost=1.0)]
    )
    assert summary["actual_only"]["claimable_cost_win"] is True

    mixed = summarize_paired_rows(
        [_row("no_context_reduction", usage_source="estimated_proxy", cost=2.0), _row("full_orchestrator", cost=1.0)]
    )
    assert mixed["mixed_or_invalid"]["claimable_cost_win"] is False


def test_verifier_suggestions_use_project_tests(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_one.py").write_text("def test_one(): pass\n", encoding="utf-8")
    suggestions = build_verifier_suggestions(tmp_path, ["firstcoder/a.py"])
    assert suggestions[0]["command"] == "python -m pytest -q"


def test_p4_stop_reason_mapping_covers_retry_and_final_gate() -> None:
    assert map_finish_reason("retry_limit_reached") == "retry_limit_reached"
    assert map_turn_outcome(status="completed", error_type="final_gate") == "final_gate_blocked"


def test_task_state_and_report_persist_verifier_suggestions() -> None:
    state = TaskState.create("task-1", "request")
    state.verifier_suggestions.append({"command": "python -m pytest -q"})
    restored = TaskState.from_dict(state.to_dict())
    report = build_report(restored)
    assert report["verifier_suggestions"] == [{"command": "python -m pytest -q"}]
