"""Evidence provenance tracking (fusion P2, M6).

Ported from pico `features/memory.py:1046-1101, 1170-1190`: workspace-path
canonicalization, file freshness, anchor hashing, workspace fingerprint,
and evidence staleness. Pure functions over paths — no store coupling.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

MAX_ANCHOR_HASH_BYTES = 10 * 1024 * 1024

_WORKSPACE_FINGERPRINT_CACHE: dict[str, str] = {}


def resolve_workspace_path(raw_path: str | Path, workspace_root: str | Path | None = None) -> Path | None:
    path = Path(str(raw_path))
    if workspace_root is None:
        return path
    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path: str | Path, workspace_root: str | Path | None = None) -> str:
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path: str | Path, workspace_root: str | Path | None = None) -> str | None:
    """内容哈希 freshness：文件当前是否与记忆时的证据一致。"""
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def compute_anchor_hash(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > MAX_ANCHOR_HASH_BYTES:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_fingerprint(workspace_root: str | Path) -> str:
    """工作区身份指纹：git root 优先，退化到绝对路径哈希。"""
    root = str(Path(workspace_root).resolve())
    cached = _WORKSPACE_FINGERPRINT_CACHE.get(root)
    if cached:
        return cached
    try:
        git_root = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).strip()
        fingerprint = hashlib.sha256(git_root).hexdigest()[:12]
    except Exception:
        fingerprint = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
    _WORKSPACE_FINGERPRINT_CACHE[root] = fingerprint
    return fingerprint


def _source_path_for_evidence(workspace_root: str | Path | None, source_path: str | None) -> Path | None:
    if not source_path:
        return None
    path = Path(source_path)
    if path.is_absolute():
        return path
    if workspace_root is None:
        return path
    return Path(workspace_root) / path


def apply_evidence_staleness(
    note: dict,
    workspace_root: str | Path | None,
) -> dict:
    """证据失效追踪：文件变了但 anchor 没更新，标记 stale_evidence。

    只读判定，不修改任何文件；返回副本或原对象（未变化时保持同一引用）。
    """
    evidence = note.get("evidence") if isinstance(note.get("evidence"), dict) else {}
    stored_hash = str(evidence.get("evidence_anchor_hash", "") or "").strip()
    source_path = evidence.get("source_path")
    if not stored_hash or not source_path:
        return note
    current_hash = compute_anchor_hash(_source_path_for_evidence(workspace_root, source_path))
    if current_hash and current_hash != stored_hash:
        note = dict(note)
        note["stale_evidence"] = True
    return note
