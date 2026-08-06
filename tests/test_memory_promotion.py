"""P2 tests: durable promotion heuristics (promotion.py, fusion M5)."""

from __future__ import annotations

from firstcoder.memory.promotion import (
    extract_durable_promotions,
    reject_durable_reason,
)


def test_reject_empty() -> None:
    assert reject_durable_reason("") == "empty"
    assert reject_durable_reason("   ") == "empty"


def test_reject_secret_shaped() -> None:
    assert reject_durable_reason("key is sk-abc123def456") == "secret_shaped"
    assert reject_durable_reason("the api key is <redacted>") == "secret_shaped"
    assert reject_durable_reason("password: hunter2") == "secret_shaped"


def test_reject_transient_task_state() -> None:
    for prefix in ("current goal", "current blocker", "next step", "current phase", "key files", "freshness"):
        assert reject_durable_reason(f"{prefix}: finish the port") == "transient_task_state"
    for prefix in ("当前目标", "当前卡点", "下一步", "当前阶段", "关键文件", "已完成", "已排除"):
        assert reject_durable_reason(f"{prefix}：完成移植") == "transient_task_state"


def test_reject_noisy_output() -> None:
    assert reject_durable_reason("stdout: done") == "noisy_output"
    assert reject_durable_reason("traceback: boom") == "noisy_output"
    assert reject_durable_reason("exit_code: 1") == "noisy_output"
    assert reject_durable_reason("x" * 221) == "noisy_output"


def test_accept_durable_text() -> None:
    assert reject_durable_reason("We use pytest for tests") == ""
    assert reject_durable_reason("Use ruff with line-length 100") == ""


def test_no_intent_returns_nothing() -> None:
    assert extract_durable_promotions("what is the weather?", "Project convention: x") == ([], [])


def test_intent_without_matching_line() -> None:
    assert extract_durable_promotions("remember this", "plain answer") == ([], [])


def test_extract_english_line() -> None:
    promotions, rejections = extract_durable_promotions(
        "please remember", "Project convention: use pytest for tests"
    )
    assert promotions == [("project-conventions", "use pytest for tests")]
    assert rejections == []


def test_extract_zh_line() -> None:
    promotions, _ = extract_durable_promotions("请记住", "项目约定：用 pytest 写测试")
    assert promotions == [("project-conventions", "用 pytest 写测试")]


def test_extract_strips_list_prefix() -> None:
    promotions, _ = extract_durable_promotions(
        "remember", "- Project convention: use ruff\n* Decision: keep sync engine"
    )
    assert promotions == [("project-conventions", "use ruff"), ("key-decisions", "keep sync engine")]


def test_extract_all_topics() -> None:
    promotions, _ = extract_durable_promotions(
        "remember",
        "Preference: dark mode\nDependency: python >= 3.11",
    )
    assert promotions == [
        ("user-preferences", "dark mode"),
        ("dependency-facts", "python >= 3.11"),
    ]


def test_extract_marks_secret_rejection() -> None:
    promotions, rejections = extract_durable_promotions(
        "remember", "Project convention: never log the api key sk-abcdef123456"
    )
    assert promotions == []
    assert rejections == ["project-conventions:secret_shaped"]


def test_extract_skips_redacted_lines() -> None:
    promotions, rejections = extract_durable_promotions("remember", "- Project convention: <redacted>")
    assert promotions == []
    assert rejections == []


def test_extract_empty_answer() -> None:
    assert extract_durable_promotions("remember this", "") == ([], [])
