"""Memory root path constraints (fusion P0 slice 3).

Codex concurrency finding (docs/fusion-plan-review.md §7.3): the memory
root must be pinned inside the workspace, and every stored path must
resolve within it — absolute candidates and `..` escapes are rejected
before any write reaches the store.

Provides `DefaultWorkspaceScope`, a concrete implementation of the
`WorkspaceScope` port shape (see memory/ports.py).
"""

from __future__ import annotations

from pathlib import Path


def default_memory_root(workspace_root: Path) -> Path:
    """`.firstcoder/memory` under the project root (factory.py:126 convention)."""
    return Path(workspace_root) / ".firstcoder" / "memory"


def validate_memory_root(root: Path, workspace_root: Path) -> Path:
    """Resolve `root` and require it to live inside `workspace_root`.

    Raises ValueError when the root escapes the workspace (absolute path
    outside it, or `..` traversal out of it).
    """
    workspace = Path(workspace_root).resolve()
    resolved = Path(root).resolve()
    if not (resolved == workspace or workspace in resolved.parents):
        raise ValueError(f"memory root {resolved} is outside workspace {workspace}")
    return resolved


def resolve_memory_path(memory_root: Path, candidate: str | Path) -> Path:
    """Resolve a store-relative `candidate` strictly inside `memory_root`.

    Rejects absolute candidates and any resolution that escapes the root.
    """
    root = Path(memory_root).resolve()
    raw = Path(candidate)
    if raw.is_absolute():
        raise ValueError(f"memory paths must be relative, got {candidate!r}")
    resolved = (root / raw).resolve()
    if not (resolved == root or root in resolved.parents):
        raise ValueError(f"memory path {candidate!r} escapes memory root {root}")
    return resolved


class DefaultWorkspaceScope:
    """Concrete `WorkspaceScope`: memory root pinned under the workspace.

    `memory_root` is validated at construction; `is_within_workspace`
    checks the workspace residency of an arbitrary (possibly absolute)
    path, as the dream runner's write scope will require.
    """

    def __init__(self, workspace_root: Path, memory_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._memory_root = validate_memory_root(
            memory_root or default_memory_root(self.workspace_root),
            self.workspace_root,
        )

    def memory_root(self) -> str:
        return str(self._memory_root)

    def is_within_workspace(self, path: str) -> bool:
        resolved = Path(path).resolve()
        return resolved == self.workspace_root or self.workspace_root in resolved.parents
