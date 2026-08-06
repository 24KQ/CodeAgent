"""Verification evidence reducers for run-level harness reports.

本模块只把已经发生的工具事件折叠成可审计的 verification signal，不执行命令，
也不替代权限系统。这样既能让 final-readiness gate 判断“改动后是否验证”，又能让
实验报告在没有真实 provider 的情况下复用同一份 trace 数据。

语义来源：pico ``core/verification.py``。FirstCoder 的工具名是 ``diagnostics``，
因此这里同时兼容 pico 的 ``run_shell`` 和 FirstCoder 的诊断工具事件。
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

VERIFICATION_SIGNAL_SCHEMA = "firstcoder.verification_signal.v1"
VERIFICATION_TOOL_NAMES = frozenset({"diagnostics", "run_shell"})


def reduce_verification_signal(
    previous: dict[str, Any] | None,
    event: dict[str, Any],
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """把一个工具事件折叠进 verification signal。

    工作区发生变更后，旧的 ``passed`` 不能继续代表当前状态，所以先把 signal
    置为 ``missing``；只有变更之后成功执行可识别的测试、lint、编译或类型检查
    命令，状态才会重新变成 ``passed``。未知工具和普通读写命令不会伪造验证证据。
    """

    signal = dict(previous or {})
    if event.get("event") != "tool_executed":
        return signal

    paths = list(
        changed_paths
        or event.get("changed_paths")
        or event.get("affected_paths")
        or event.get("artifact_paths")
        or []
    )
    if event.get("workspace_changed") is True:
        signal = {
            "schema_version": VERIFICATION_SIGNAL_SCHEMA,
            "state": "missing",
            "last_workspace_change_span_id": str(event.get("span_id", "")),
            "changed_paths": paths,
        }

    tool_name = _tool_name(event)
    if tool_name not in VERIFICATION_TOOL_NAMES:
        return signal
    command = _command_from_event(event)
    command_class = classify_verification_command(command)
    if not command_class:
        return signal

    passed = _event_passed(event)
    signal.update(
        {
            "schema_version": VERIFICATION_SIGNAL_SCHEMA,
            "state": "passed" if passed else "failed",
            "source_span_id": str(event.get("span_id", "")),
            "command": command,
            "command_class": command_class,
            "after_last_workspace_change": bool(
                signal.get("last_workspace_change_span_id") or paths
            ),
            "changed_paths_present": bool(paths),
            # 命令分类只能证明执行了验证，不能安全地推断它覆盖了哪些文件。
            "covers_changed_paths": False,
            "coverage_confidence": "unknown",
            "changed_paths": paths,
        }
    )
    return signal


def classify_verification_command(command: str) -> str:
    """将诊断命令归类为 ``test``/``lint``/``compile``/``typecheck``。

    这里只做保守识别：无法解析或不在白名单内的命令返回空串，避免把任意 shell
    命令误记成成功验证。``python -m pytest`` 与 ``uv run`` 是仓库现有工作流常见
    形式，Windows 路径分隔符也在这里统一处理。
    """

    try:
        tokens = shlex.split(str(command))
    except ValueError:
        tokens = str(command).split()
    tokens = [token.lower() for token in tokens]
    if not tokens or tokens[0] in {"echo", "printf", "grep", "rg", "cat"}:
        return ""

    if tokens[0] == "uv" and len(tokens) > 2 and tokens[1] == "run":
        while len(tokens) > 2 and tokens[2].startswith("-"):
            tokens = tokens[:2] + tokens[3:]
        tokens = tokens[2:]
    if not tokens:
        return ""

    executable = tokens[0].replace("\\", "/").rsplit("/", 1)[-1]
    if len(tokens) > 2 and _is_python_command(executable) and tokens[1] == "-m":
        return {"pytest": "test", "compileall": "compile", "unittest": "test"}.get(
            tokens[2], ""
        )
    if tokens[0] in {"pytest", "tox", "nose", "nose2"}:
        return "test"
    if tokens[0] == "ruff" and len(tokens) > 1 and tokens[1] == "check":
        return "lint"
    if tokens[0] in {"mypy", "pyright"}:
        return "typecheck"
    if tokens[:2] in (
        ["yarn", "test"],
        ["go", "test"],
        ["cargo", "test"],
        ["make", "test"],
    ):
        return "test"
    if tokens[:2] in (["npm", "test"], ["pnpm", "test"]):
        return "test"
    if len(tokens) > 2 and tokens[:2] in (["npm", "run"], ["pnpm", "run"]):
        return {"test": "test", "build": "build", "lint": "lint"}.get(tokens[2], "")
    return ""


def build_verifier_suggestions(
    root: str | Path,
    changed_paths: list[str] | tuple[str, ...] | dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """根据项目文件和已改路径生成有限的验证建议。

    建议是报告数据，不会自动执行，也不绕过 ``diagnostics`` 的权限确认。读取项目
    文件时只检查给定 root 内的常规文件，最多返回八条稳定排序的建议。
    """

    root_path = Path(root)
    # 兼容 H11 consumer 传入的 artifact graph，以及直接传 changed_paths 的轻量调用。
    raw_paths = (
        changed_paths.get("changed_paths", [])
        if isinstance(changed_paths, dict)
        else (changed_paths or [])
    )
    paths = [str(path) for path in raw_paths if str(path).strip()]
    suggestions: list[dict[str, str]] = []

    package_path = root_path / "package.json"
    if package_path.is_file():
        try:
            scripts = dict(json.loads(package_path.read_text(encoding="utf-8")).get("scripts", {}) or {})
        except (OSError, UnicodeError, json.JSONDecodeError):
            scripts = {}
        if "test" in scripts:
            suggestions.append({"command": "npm test", "reason": "package.json defines a test script"})
        if "build" in scripts:
            suggestions.append({"command": "npm run build", "reason": "package.json defines a build script"})

    tests_dir = root_path / "tests"
    has_python_tests = tests_dir.is_dir() and any(t.suffix == ".py" for t in tests_dir.rglob("*.py"))
    if has_python_tests:
        suggestions.append({"command": "python -m pytest -q", "reason": "Python tests are present"})
    elif any(path.lower().endswith(".py") for path in paths):
        suggestions.append(
            {"command": "python -m compileall .", "reason": "Python files changed and no tests were found"}
        )
    return suggestions[:8]


def _tool_name(event: dict[str, Any]) -> str:
    return str(event.get("name") or event.get("tool_name") or event.get("tool") or "")


def _command_from_event(event: dict[str, Any]) -> str:
    args = event.get("args")
    data = event.get("data")
    for payload in (args, data):
        if isinstance(payload, dict) and payload.get("command") is not None:
            return str(payload.get("command") or "").strip()
    return str(event.get("command") or "").strip()


def _event_passed(event: dict[str, Any]) -> bool:
    if event.get("ok") is False:
        return False
    status = str(event.get("status") or event.get("tool_status") or "").lower()
    if status:
        return status in {"ok", "success", "passed", "completed"}
    return not bool(event.get("error"))


def _is_python_command(command: str) -> bool:
    suffix = command.removeprefix("python3.")
    return command in {"python", "python3", "py"} or (
        suffix != command and suffix.replace(".", "").isdigit()
    )
