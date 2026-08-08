"""P7 发布 tag 与 pyproject 版本一致性检查。"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_version(pyproject_path: Path = ROOT / "pyproject.toml") -> str:
    """从项目唯一版本源读取版本，避免 workflow 自己维护第二份版本号。"""

    with pyproject_path.open("rb") as stream:
        payload = tomllib.load(stream)
    version = payload.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"project version is missing: {pyproject_path}")
    return version.strip()


def validate_release_tag(tag: str, version: str) -> None:
    """要求发布 tag 严格使用 v<project.version> 格式。"""

    expected = f"v{version}"
    if tag != expected:
        raise ValueError(f"release tag {tag!r} does not match project version {expected!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the release tag against pyproject.toml.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    args = parser.parse_args(argv)
    try:
        version = project_version(args.pyproject)
        validate_release_tag(args.tag, version)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Release tag OK: {args.tag} ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
