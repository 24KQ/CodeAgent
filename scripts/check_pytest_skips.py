"""P7 pytest skip allowlist 门禁。

pytest 的 skip 本身不是失败，但未知 skip 可能把测试悄悄移出验收范围。脚本读取
JUnit XML，只允许项目已知的可选依赖、live provider 和平台能力差异原因。
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Sequence
from pathlib import Path

ALLOWED_REASON_MARKERS = (
    "could not import 'harbor'",
    "live provider requires",
    "live provider credential is not configured",
    "set firstcoder_live_memory_test=1",
    "symlink creation not permitted on this host",
    "junction creation not available on this host",
    "symlink/junction creation not available on this host",
    "windows 非管理员账户创建符号链接受限",
    "当前环境无法创建 symlink",
    "进程组断言使用 posix 进程组语义",
)


def skip_reasons(xml_path: Path) -> list[str]:
    """提取 JUnit 中的 skip 原因；无原因的 skip 也必须显式暴露。"""

    root = ElementTree.parse(xml_path).getroot()
    reasons: list[str] = []
    for testcase in root.iter("testcase"):
        skipped = testcase.find("skipped")
        if skipped is None:
            continue
        message = str(skipped.attrib.get("message") or "").strip()
        body = " ".join((skipped.text or "").split())
        reasons.append(" ".join(part for part in (message, body) if part) or "<missing reason>")
    return reasons


def is_allowed_skip(reason: str) -> bool:
    normalized = reason.casefold()
    return any(marker.casefold() in normalized for marker in ALLOWED_REASON_MARKERS)


def unexpected_skips(reasons: Iterable[str]) -> list[str]:
    return [reason for reason in reasons if not is_allowed_skip(reason)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject unknown pytest skip reasons in JUnit XML.")
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args(argv)
    reasons = skip_reasons(args.junit_xml)
    unexpected = unexpected_skips(reasons)
    if unexpected:
        print("Unexpected pytest skips:", file=sys.stderr)
        for reason in unexpected:
            print(f"  {reason}", file=sys.stderr)
        return 1
    print(f"Pytest skip allowlist OK: {len(reasons)} skips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
