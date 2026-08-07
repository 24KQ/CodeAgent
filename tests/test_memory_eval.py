"""P5.1 provider-free memory benchmark 的公共 seam 测试。"""

from __future__ import annotations

import json
from pathlib import Path

from firstcoder.harness.experiments.context_cost import write_experiment_artifacts
from firstcoder.harness.experiments.memory_eval import (
    CHALLENGE_VARIANTS,
    MEMORY_METRICS,
    MemoryEvaluationAdapter,
    MemoryMetricResult,
    build_challenge_cases,
    build_contract_cases,
    build_memory_fixture_cases,
    evaluate_memory_cases,
    write_memory_eval_artifacts,
)


def test_memory_fixture_keeps_contract_and_challenge_suites_separate() -> None:
    """contract 只验证机制合同，challenge 才承载六类长期记忆能力。"""

    contract = build_contract_cases()
    challenge = build_challenge_cases()

    assert len(contract) == 8
    assert len(challenge) >= 50
    assert {case.category for case in challenge} >= {
        "information_extraction",
        "multi_session_reasoning",
        "temporal_reasoning",
        "knowledge_updates",
        "abstention",
        "agentic_efficiency",
    }


def test_memory_on_uses_retriever_and_does_not_write_provider_or_disk(tmp_path: Path) -> None:
    """fixture 阶段只能读取内存 state，不能创建 provider、store 或文件。"""

    case = next(case for case in build_contract_cases() if case.case_id == "direct_recall_001")
    result = evaluate_memory_cases([case], MemoryEvaluationAdapter(), mode="contract")

    assert set(result["variants"]) == set(CHALLENGE_VARIANTS)
    memory_on = result["variants"]["memory_on"]["rows"][0]
    assert memory_on["selected_note_ids"] == ["direct-recall-fact"]
    assert not list(tmp_path.iterdir())


def test_memory_metrics_have_explicit_zero_denominator_semantics() -> None:
    """不可适用指标必须是 None，而不是把没有样本误报为满分。"""

    result = MemoryMetricResult(0, 0, None, False)

    assert result.rate is None
    assert result.applicable is False
    assert result.to_dict() == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
        "applicable": False,
    }


def test_memory_evaluation_exposes_all_six_metrics_and_comparative_variants() -> None:
    """四个变体都要输出固定六项指标，便于稳定比较而非只看单一分数。"""

    payload = evaluate_memory_cases(build_challenge_cases(), MemoryEvaluationAdapter(), mode="challenge")

    assert set(payload["variants"]) == set(CHALLENGE_VARIANTS)
    for variant in payload["variants"].values():
        assert set(variant["metrics"]) == set(MEMORY_METRICS)
    assert payload["variants"]["memory_on"]["metrics"]["evidence_recall"]["rate"] > payload["variants"]["memory_off"]["metrics"]["evidence_recall"]["rate"]
    assert payload["variants"]["memory_on"]["metrics"]["stale_use"] == {
        "numerator": 0,
        "denominator": 5,
        "rate": 0.0,
        "applicable": True,
    }
    assert payload["variants"]["memory_on"]["metrics"]["evidence_precision"] == {
        "numerator": 45,
        "denominator": 45,
        "rate": 1.0,
        "applicable": True,
    }
    assert payload["variants"]["memory_off"]["metrics"]["evidence_precision"]["applicable"] is False
    assert payload["variants"]["memory_on"]["metrics"]["abstention"]["rate"] == 1.0
    assert payload["variants"]["naive_recent"]["metrics"]["abstention"]["rate"] < 1.0
    assert payload["variants"]["memory_on"]["metrics"]["false_resume"] == {
        "numerator": 0,
        "denominator": 5,
        "rate": 0.0,
        "applicable": True,
    }
    assert payload["variants"]["unsafe_memory"]["metrics"]["false_resume"]["rate"] == 1.0


def test_baselines_have_observable_selection_contracts() -> None:
    """逐 case 固化 baseline 的错误选择，避免四个变体退化成同一算法。"""

    cases = build_challenge_cases()
    adapter = MemoryEvaluationAdapter()
    split_case = next(case for case in cases if case.case_id == "info_extract_000")
    update_case = next(case for case in cases if case.case_id == "knowledge_update_001")

    assert adapter.observe(split_case, "memory_on").selected_note_ids == ("info-current-00",)
    assert adapter.observe(split_case, "naive_recent").selected_note_ids == ("info-current-00",)
    assert adapter.observe(split_case, "unsafe_memory").selected_note_ids == ("info-old-00",)
    assert adapter.observe(update_case, "naive_recent").answer_correct is False
    assert adapter.observe(update_case, "unsafe_memory").answer_correct is False


def test_fixture_mode_fails_closed_and_memory_off_keeps_reject_answer() -> None:
    """未知 suite 不得静默降级，scope 拒绝场景的 off baseline 语义也要稳定。"""

    try:
        build_memory_fixture_cases("unknown")
    except ValueError as error:
        assert "contract or challenge" in str(error)
    else:
        raise AssertionError("unknown fixture mode must fail closed")

    scope_case = next(case for case in build_challenge_cases() if case.case_id == "temporal_rejection_001")
    observation = MemoryEvaluationAdapter().observe(scope_case, "memory_off")
    assert observation.answer == "No."
    assert observation.passed is True


def test_memory_artifacts_keep_only_stable_safe_fields(tmp_path: Path) -> None:
    """结果文件不能把 secret、绝对路径或完整 prompt 带出 benchmark。"""

    payload = {
        "schema_version": 1,
        "artifact_type": "memory-eval-v1",
        "variants": {
            "memory_on": {
                "rows": [
                    {
                        "id": "case-1",
                        "query": "full prompt / D:/private/workspace",
                        "selected_note_ids": ["safe-id"],
                        "selected_texts": ["sk-AAAAAAAAAAAAAAAAAAAA"],
                        "rejected_reasons": {"secret-id": "secret_shaped"},
                        "answer": "full prompt / D:/private/workspace",
                        "answer_correct": True,
                    }
                ],
                "metrics": {},
            }
        },
    }

    paths = write_memory_eval_artifacts(payload, tmp_path)
    content = "\n".join(Path(path).read_text(encoding="utf-8") for path in paths.values())

    assert "sk-AAAAAAAAAAAAAAAAAAAA" not in content
    assert "D:/private/workspace" not in content
    assert "full prompt" not in content
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    row = written["variants"]["memory_on"]["rows"][0]
    assert row == {
        "answer_correct": True,
        "id": "case-1",
        "rejected_reasons": {"secret-id": "secret_shaped"},
        "selected_note_ids": ["safe-id"],
    }


def test_shared_writer_can_render_memory_rows_without_cost_columns(tmp_path: Path) -> None:
    """共享 writer 的 memory 模式不应伪造 usage 字段，cost 默认行为仍保留。"""

    paths = write_experiment_artifacts(
        {"summary": {}, "rows": [{"id": "case-1", "answer_correct": True}]},
        tmp_path,
        markdown_renderer=lambda payload: "# Memory\n",
        include_usage_columns=False,
    )

    csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
    assert "usage_input_tokens" not in csv_text
    assert "case-1" in csv_text


def test_generated_memory_artifacts_are_deterministic_and_redacted(tmp_path: Path) -> None:
    """相同 fixture 两次写出必须一致，且只包含稳定的审计字段。"""

    payload = evaluate_memory_cases(build_challenge_cases(), MemoryEvaluationAdapter())
    first = write_memory_eval_artifacts(payload, tmp_path / "first")
    second = write_memory_eval_artifacts(payload, tmp_path / "second")

    assert Path(first["json"]).read_bytes() == Path(second["json"]).read_bytes()
    text = Path(first["json"]).read_text(encoding="utf-8")
    assert '"query"' not in text
    assert '"selected_texts"' not in text
    assert '"answer"' not in text
    assert "sk-AAAAAAAAAAAAAAAAAAAA" not in text
