"""P7 CI 门禁脚本的纯函数契约测试。"""

from __future__ import annotations

from pathlib import Path

from scripts.check_pytest_skips import is_allowed_skip, skip_reasons, unexpected_skips
from scripts.check_release_version import validate_release_tag
from scripts.check_ruff_baseline import diagnostic_key, new_violations, violation_counts


def test_ruff_baseline_ignores_line_changes_but_detects_new_count() -> None:
    diagnostics = [
        {"filename": "firstcoder\\example.py", "code": "I001", "message": "Import block is un-sorted", "location": {"row": 10}},
        {"filename": "firstcoder/example.py", "code": "I001", "message": "Import block is un-sorted", "location": {"row": 50}},
    ]

    counts = violation_counts(diagnostics)

    assert diagnostic_key(diagnostics[0]) == diagnostic_key(diagnostics[1])
    assert new_violations(counts, {diagnostic_key(diagnostics[0]): 1}) == {
        diagnostic_key(diagnostics[0]): 1,
    }


def test_ruff_baseline_key_strips_machine_specific_absolute_prefix() -> None:
    diagnostic = {
        "filename": r"D:\python\projects\FirstCoder\firstcoder\example.py",
        "code": "F401",
        "message": "unused import",
    }

    assert diagnostic_key(diagnostic).startswith("firstcoder/example.py|")


def test_pytest_skip_allowlist_rejects_unknown_reason(tmp_path: Path) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        "<testsuite><testcase classname='x' name='known'><skipped message=\"could not import 'harbor'\" /></testcase>"
        "<testcase classname='x' name='unknown'><skipped message='new environment issue' /></testcase></testsuite>",
        encoding="utf-8",
    )

    reasons = skip_reasons(report)

    assert is_allowed_skip(reasons[0])
    assert unexpected_skips(reasons) == ["new environment issue"]
    assert is_allowed_skip(
        "collection skipped Skipped: could not import 'harbor': No module named 'harbor'"
    )
    assert is_allowed_skip("live provider requires a configured model: detail")
    assert not is_allowed_skip("known prefix was mentioned after an unrelated failure")


def test_release_tag_must_match_project_version() -> None:
    validate_release_tag("v0.1.12", "0.1.12")

    try:
        validate_release_tag("v0.1.13", "0.1.12")
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("a mismatched release tag must be rejected")
