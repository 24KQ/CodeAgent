"""P7 Ruff 基线门禁。

仓库历史上已经存在少量 Ruff 违规，P7 不把这些旧问题伪装成新回归。脚本按
文件、规则和诊断文本统计违规数量；行号移动不影响基线，但同类违规数量增加
或新文件出现违规会阻断 CI。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "scripts" / "ruff_baseline.json"
DEFAULT_TARGETS = ("firstcoder", "tests", "scripts")
RUFF_RULE_SELECT = ("E4", "E7", "E9", "F", "I", "UP", "DTZ", "BLE")


def diagnostic_key(diagnostic: Mapping[str, Any]) -> str:
    """返回不含行号的稳定诊断键，兼容 Windows 和 Linux 路径。"""

    filename = _relative_filename(str(diagnostic.get("filename") or ""))
    code = str(diagnostic.get("code") or "").strip()
    message = " ".join(str(diagnostic.get("message") or "").split())
    return f"{filename}|{code}|{message}"


def _relative_filename(filename: str) -> str:
    """把 Ruff 的绝对路径折叠到仓库相对路径，保持跨平台基线一致。"""

    normalized = filename.replace("\\", "/")
    root = ROOT.as_posix().rstrip("/")
    if normalized.casefold().startswith(root.casefold() + "/"):
        normalized = normalized[len(root) + 1 :]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    # Windows 基线与 Ubuntu runner 的仓库绝对路径前缀不同；目标目录名是
    # 稳定的，所以从第一个已知目标目录开始截取，避免把机器路径写进基线。
    lowered = normalized.casefold()
    for prefix in ("firstcoder/", "tests/", "scripts/"):
        marker = "/" + prefix
        index = lowered.find(marker)
        if index >= 0:
            return normalized[index + 1 :]
        if lowered.startswith(prefix):
            return normalized
    return normalized


def violation_counts(diagnostics: Iterable[Mapping[str, Any]]) -> Counter[str]:
    """统计 Ruff 诊断，Counter 便于识别同类违规数量增加。"""

    return Counter(diagnostic_key(item) for item in diagnostics)


def new_violations(
    current: Mapping[str, int],
    baseline: Mapping[str, int],
) -> dict[str, int]:
    """只返回超过历史基线的违规数量。"""

    return {
        key: count - int(baseline.get(key, 0))
        for key, count in current.items()
        if count > int(baseline.get(key, 0))
    }


def _run_ruff(targets: Sequence[str]) -> list[dict[str, Any]]:
    """执行固定规则集合；Ruff 返回 1 表示发现违规，仍需解析 JSON。"""

    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *targets,
        "--select",
        ",".join(RUFF_RULE_SELECT),
        "--output-format",
        "json",
    ]
    # 不把 Ruff 的正常“发现违规”返回码误当作执行失败；只有无法运行或无法
    # 解析结果时才让门禁报错，避免基线检查静默放过工具故障。
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or "ruff failed to run")
    if not completed.stdout.strip():
        return []
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise RuntimeError("ruff returned an unexpected JSON payload")
    return payload


def _load_baseline(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    violations = payload.get("violations") if isinstance(payload, dict) else None
    if not isinstance(violations, dict):
        raise ValueError(f"invalid Ruff baseline: {path}")
    return {str(key): int(value) for key, value in violations.items()}


def _write_baseline(path: Path, counts: Mapping[str, int], targets: Sequence[str]) -> None:
    payload = {
        "schema_version": 1,
        "ruff_rule_select": list(RUFF_RULE_SELECT),
        "targets": list(targets),
        "violations": dict(sorted(counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Ruff diagnostics against the repository baseline.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write", action="store_true", help="Write the current diagnostics as a new baseline.")
    parser.add_argument("targets", nargs="*", default=list(DEFAULT_TARGETS))
    args = parser.parse_args(argv)
    targets = tuple(str(item) for item in args.targets) or DEFAULT_TARGETS
    current = violation_counts(_run_ruff(targets))
    baseline_path = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    if args.write:
        _write_baseline(baseline_path, current, targets)
        print(f"Wrote Ruff baseline: {baseline_path} ({sum(current.values())} diagnostics)")
        return 0
    if not baseline_path.is_file():
        print(f"Ruff baseline is missing: {baseline_path}", file=sys.stderr)
        return 2
    baseline = _load_baseline(baseline_path)
    added = new_violations(current, baseline)
    if added:
        print("New Ruff violations detected:", file=sys.stderr)
        for key, count in sorted(added.items()):
            print(f"  +{count} {key}", file=sys.stderr)
        return 1
    print(
        f"Ruff baseline OK: {sum(current.values())} current diagnostics; "
        f"{sum(baseline.values())} diagnostics in baseline"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
