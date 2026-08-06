"""P0 slice 3 tests: memory root path constraints (paths.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from firstcoder.memory.paths import (
    DefaultWorkspaceScope,
    default_memory_root,
    resolve_memory_path,
    validate_memory_root,
)
from firstcoder.memory.ports import WorkspaceScope


def test_default_memory_root_follows_firstcoder_convention(tmp_path: Path) -> None:
    assert default_memory_root(tmp_path) == tmp_path / ".firstcoder" / "memory"


def test_validate_memory_root_accepts_inside_workspace(tmp_path: Path) -> None:
    root = tmp_path / ".firstcoder" / "memory"
    assert validate_memory_root(root, tmp_path) == root.resolve()


def test_validate_memory_root_rejects_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError):
        validate_memory_root(outside, tmp_path)


def test_resolve_memory_path_accepts_relative(tmp_path: Path) -> None:
    root = tmp_path / ".firstcoder" / "memory"
    resolved = resolve_memory_path(root, "topics/build.md")
    assert resolved == (root / "topics" / "build.md").resolve()
    assert root.resolve() in resolved.parents


def test_resolve_memory_path_rejects_absolute(tmp_path: Path) -> None:
    root = tmp_path / ".firstcoder" / "memory"
    with pytest.raises(ValueError):
        resolve_memory_path(root, str(tmp_path / "absolute.md"))


def test_resolve_memory_path_rejects_dotdot_escape(tmp_path: Path) -> None:
    root = tmp_path / ".firstcoder" / "memory"
    with pytest.raises(ValueError):
        resolve_memory_path(root, "../../escape.md")


def test_default_workspace_scope_memory_root(tmp_path: Path) -> None:
    scope = DefaultWorkspaceScope(tmp_path)
    assert Path(scope.memory_root()) == (tmp_path / ".firstcoder" / "memory").resolve()


def test_default_workspace_scope_within_workspace(tmp_path: Path) -> None:
    scope = DefaultWorkspaceScope(tmp_path)
    assert scope.is_within_workspace(str(tmp_path))
    assert scope.is_within_workspace(str(tmp_path / "src" / "main.py"))
    assert not scope.is_within_workspace(str(tmp_path.parent / "outside"))


def test_default_workspace_scope_rejects_outside_memory_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DefaultWorkspaceScope(tmp_path, memory_root=tmp_path.parent / "elsewhere")


def test_workspace_scope_satisfies_port() -> None:
    required = {name for name in vars(WorkspaceScope) if not name.startswith("_")}
    present = {
        name for name, member in vars(DefaultWorkspaceScope).items() if callable(member)
    }
    assert required <= present
