"""P5.1 provider-free memory benchmark 的公共 seam 测试。"""

from __future__ import annotations

import json
from pathlib import Path

from firstcoder.harness.experiments.context_cost import write_experiment_artifacts
from firstcoder.harness.experiments.memory_eval import (
    CHALLENGE_VARIANTS,
    MEMORY_METRICS,
    MemoryEvaluationAdapter,
    MemoryFixtureCase,
    MemoryFixtureNote,
    MemoryMetricResult,
    MemoryObservation,
    assess_memory_answer,
    build_challenge_cases,
    build_contract_cases,
    build_memory_fixture_cases,
    evaluate_memory_cases,
    summarize_memory_observations,
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
    assert observation.answer == "unknown"
    assert observation.passed is True


def test_live_answer_assessment_accepts_verbose_fact_and_multilingual_abstention() -> None:
    """真实 provider 的完整句和中英文拒答不能被整句比较误判。"""

    fact_case = next(case for case in build_contract_cases() if case.case_id == "irrelevant_distractor_001")
    fact = assess_memory_answer(
        fact_case,
        "Per durable memory, the deploy key is red. No other details are stored.",
    )
    assert fact.answer_correct is True
    assert fact.answer_class == "fact"
    assert fact.matched_expected_tokens == fact.expected_token_count

    incomplete = assess_memory_answer(fact_case, "Y")
    assert incomplete.answer_correct is False
    negated = assess_memory_answer(fact_case, "The deploy key is not red.")
    assert negated.answer_correct is False

    abstention_case = next(
        case for case in build_challenge_cases() if case.case_id == "temporal_rejection_000"
    )
    chinese = assess_memory_answer(abstention_case, "无法确定，当前没有相关证据。")
    english = assess_memory_answer(abstention_case, "I cannot determine the answer from the evidence.")
    assert chinese.answer_correct is True
    assert english.answer_correct is True
    assert chinese.answer_class == "abstain"
    assert english.answer_class == "abstain"
    for answer in ("I don't know.", "I'm not sure.", "不确定", "我不知道", "无法给出答案"):
        assessment = assess_memory_answer(abstention_case, answer)
        assert assessment.answer_correct is True
        assert assessment.abstained is True


def test_answer_assessment_rejects_chinese_negated_facts_and_handles_stopword_fact() -> None:
    """语义评分要识别中文反事实，也不能把全停用词答案当成空答案。"""

    chinese_case = MemoryFixtureCase(
        "chinese-fact",
        "information_extraction",
        "部署环境",
        "部署环境是生产",
    )
    positive = assess_memory_answer(chinese_case, "部署环境是生产环境。")
    assert positive.answer_correct is True
    for negation in ("不是", "并非", "没有", "无", "未"):
        negative = assess_memory_answer(
            chinese_case,
            f"部署环境{negation}生产，而是测试环境。",
        )
        assert negative.answer_correct is False
    for negative in (
        "The deploy key isn't red.",
        "The deploy key is no longer red.",
        "The deploy key is without red.",
    ):
        assert assess_memory_answer(
            next(case for case in build_contract_cases() if case.case_id == "irrelevant_distractor_001"),
            negative,
        ).answer_correct is False

    contrastive = assess_memory_answer(
        next(case for case in build_contract_cases() if case.case_id == "irrelevant_distractor_001"),
        "The deploy key is not blue, but red.",
    )
    assert contrastive.answer_correct is True
    chinese_contrastive = assess_memory_answer(
        chinese_case,
        "部署环境是生产，无生产风险。",
    )
    assert chinese_contrastive.answer_correct is True
    token_case = MemoryFixtureCase("token-fact", "information_extraction", "q", "生产")
    assert assess_memory_answer(token_case, "未来将生产。 ").answer_correct is True
    assert assess_memory_answer(token_case, "无条件生产。 ").answer_correct is True

    stopword_case = MemoryFixtureCase(
        "stopword-fact",
        "information_extraction",
        "status",
        "is the",
    )
    assert assess_memory_answer(stopword_case, "is the").answer_correct is True
    assert assess_memory_answer(stopword_case, "is a").answer_correct is False
    empty_case = MemoryFixtureCase("empty-fact", "information_extraction", "status", "")
    assert assess_memory_answer(empty_case, "").answer_correct is False


def test_fixture_and_live_accept_correct_abstention_after_unrelated_selection() -> None:
    """误选普通无关 note 时，fixture/live 都按安全拒答通过但保留选择记录。"""

    case = MemoryFixtureCase(
        "semantic-abstention",
        "abstention",
        "production incident",
        "unknown",
        (
            MemoryFixtureNote(
                "unrelated-note",
                "production incident runbook is unavailable",
                tags=("production", "incident"),
            ),
        ),
        no_evidence=True,
    )
    adapter = MemoryEvaluationAdapter()
    fixture = adapter.observe(case, "memory_on")
    live = adapter.observe_provider_answer(
        case,
        "I cannot determine the answer from the available evidence.",
        fixture.selected_note_ids,
    )

    assert fixture.selected_note_ids == ("unrelated-note",)
    assert fixture.answer_correct is True
    assert fixture.passed is True
    assert live.answer_correct is True
    assert live.passed is True


def test_abstention_metric_does_not_require_empty_retrieval() -> None:
    """模型拒答与 Retriever 是否误选无关 note 是两个独立指标。"""

    observation = MemoryObservation(
        case_id="abstention-fixture",
        variant="live",
        selected_note_ids=("unrelated-note",),
        answer="cannot determine",
        answer_correct=True,
        answer_semantically_correct=True,
        abstained=True,
        no_evidence=True,
        passed=True,
    )

    assert observation.abstained is True
    assert summarize_memory_observations([observation])["metrics"]["abstention"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
        "applicable": True,
    }


def test_summarize_memory_observations_redacts_untrusted_rejection_reasons() -> None:
    """live runner 直接序列化 summarize 结果时也不能泄露任意 reason。"""

    observation = MemoryObservation(
        case_id="safe-case",
        variant="memory_on",
        selected_note_ids=("sk-AAAAAAAAAAAAAAAAAAAA",),
        rejected_reasons={"sk-BBBBBBBBBBBBBBBBBBBB": "secret-sketch"},
    )

    row = summarize_memory_observations([observation])["rows"][0]

    assert row["selected_note_ids"] != ["sk-AAAAAAAAAAAAAAAAAAAA"]
    assert len(row["selected_note_ids"][0]) == 12
    assert row["rejected_reasons"] == {}


def test_semantic_assessment_artifact_keeps_only_safe_fields(tmp_path: Path) -> None:
    """语义评分可以写入分类和计数，但不能把 provider 原文带出 artifact。"""

    case = next(case for case in build_contract_cases() if case.case_id == "irrelevant_distractor_001")
    assessment = assess_memory_answer(case, "Per durable memory, the deploy key is red.")
    payload = {
        "schema_version": 1,
        "mode": "live",
        "case_count": 1,
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "usage_source": "trace",
        "variants": {
            "memory_on": {
                "summary": {
                    "total_cases": 1,
                    "answer_semantic_accuracy": 1.0,
                },
                "rows": [
                    {
                        "id": case.case_id,
                        "answer": "Per durable memory, the deploy key is red.",
                        **assessment.to_artifact_fields(),
                    }
                ],
                "metrics": {},
            }
        },
    }

    paths = write_memory_eval_artifacts(payload, tmp_path)
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    row = written["variants"]["memory_on"]["rows"][0]
    assert written["provider"] == "deepseek"
    assert written["model"] == "deepseek-v4-flash"
    assert written["usage_source"] == "trace"
    assert row["answer_semantically_correct"] is True
    assert row["answer_class"] == "fact"
    assert row["expected_answer_class"] == "fact"
    assert row["matched_expected_tokens"] == row["expected_token_count"]
    assert "Per durable memory" not in "\n".join(
        Path(path).read_text(encoding="utf-8") for path in paths.values()
    )


def test_live_observation_reuses_answer_and_memory_safety_contracts() -> None:
    """live runner 的回答评分必须复用 required、reject 和 false-resume 规则。"""

    adapter = MemoryEvaluationAdapter()
    fact_case = next(case for case in build_contract_cases() if case.case_id == "irrelevant_distractor_001")
    fact = adapter.observe_provider_answer(
        fact_case,
        "Per durable memory, the deploy key is red.",
        ["distractor-red"],
    )
    assert fact.passed is True
    assert fact.answer_correct is True

    scope_case = next(case for case in build_challenge_cases() if case.case_id == "temporal_rejection_001")
    safe_rejection = adapter.observe_provider_answer(
        scope_case,
        "I cannot determine the answer from the available evidence.",
        [],
        rejected_reasons={"temporal-invalid-01": "scope_mismatch"},
    )
    assert safe_rejection.passed is True
    assert safe_rejection.abstained is True
    assert safe_rejection.false_resume_accepted is False

    unsafe_resume = adapter.observe_provider_answer(
        scope_case,
        "Yes.",
        ["temporal-invalid-01"],
    )
    assert unsafe_resume.passed is False
    assert unsafe_resume.false_resume_accepted is True

    no_evidence_case = next(case for case in build_challenge_cases() if case.case_id == "abstention_002")
    no_evidence = adapter.observe_provider_answer(
        no_evidence_case,
        "无法确定，当前没有相关证据。",
        ["unrelated-note"],
    )
    summary = summarize_memory_observations([no_evidence])
    assert summary["metrics"]["abstention"]["rate"] == 1.0


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
