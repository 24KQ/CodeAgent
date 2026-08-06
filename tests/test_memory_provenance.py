"""P2 tests: evidence provenance tracking (provenance.py, fusion M6)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from firstcoder.memory.provenance import (
    apply_evidence_staleness,
    canonicalize_path,
    compute_anchor_hash,
    file_freshness,
    resolve_workspace_path,
    workspace_fingerprint,
)


def test_resolve_workspace_path_relative(tmp_path: Path) -> None:
    resolved = resolve_workspace_path("src/a.py", tmp_path)
    assert resolved == (tmp_path / "src" / "a.py").resolve()


def test_resolve_workspace_path_absolute_inside(tmp_path: Path) -> None:
    target = tmp_path / "b.py"
    assert resolve_workspace_path(str(target), tmp_path) == target.resolve()


def test_resolve_workspace_path_escape_returns_none(tmp_path: Path) -> None:
    assert resolve_workspace_path("../outside.py", tmp_path) is None
    assert resolve_workspace_path(str(tmp_path.parent / "outside.py"), tmp_path) is None


def test_resolve_workspace_path_without_root_passthrough() -> None:
    assert resolve_workspace_path("raw/path.py") == Path("raw/path.py")


def test_canonicalize_path_relative(tmp_path: Path) -> None:
    assert canonicalize_path("src/a.py", tmp_path) == "src/a.py"


def test_canonicalize_path_absolute_inside(tmp_path: Path) -> None:
    target = tmp_path / "src" / "b.py"
    assert canonicalize_path(str(target), tmp_path) == "src/b.py"


def test_canonicalize_path_escape_falls_back_to_posix(tmp_path: Path) -> None:
    # 逃逸路径无法相对化：退回原始路径的 posix 形式（不丢信息）。
    escaped = str(tmp_path.parent / "outside.py")
    assert canonicalize_path(escaped, tmp_path) == escaped.replace("\\", "/")


def test_file_freshness_returns_sha256(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("content", encoding="utf-8")
    assert file_freshness("a.py", tmp_path) == hashlib.sha256(b"content").hexdigest()


def test_file_freshness_missing_returns_none(tmp_path: Path) -> None:
    assert file_freshness("missing.py", tmp_path) is None


def test_compute_anchor_hash_missing_returns_none(tmp_path: Path) -> None:
    assert compute_anchor_hash(tmp_path / "missing") is None


def test_compute_anchor_hash_oversized_returns_none(tmp_path: Path) -> None:
    target = tmp_path / "huge.bin"
    target.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    assert compute_anchor_hash(target) is None


def test_workspace_fingerprint_is_deterministic(tmp_path: Path) -> None:
    first = workspace_fingerprint(tmp_path)
    second = workspace_fingerprint(tmp_path)
    assert first == second
    assert len(first) == 12


def test_apply_evidence_staleness_unchanged_returns_same_object(tmp_path: Path) -> None:
    note = {"text": "x", "evidence": {}}
    assert apply_evidence_staleness(note, tmp_path) is note


def test_apply_evidence_staleness_fresh_file(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("v1", encoding="utf-8")
    note = {
        "text": "x",
        "evidence": {
            "source_path": "a.py",
            "evidence_anchor_hash": hashlib.sha256(b"v1").hexdigest(),
        },
    }
    result = apply_evidence_staleness(note, tmp_path)
    assert result is note
    assert result.get("stale_evidence") is None


def test_apply_evidence_staleness_changed_file(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("v1", encoding="utf-8")
    old_hash = hashlib.sha256(b"v1").hexdigest()
    target.write_text("v2", encoding="utf-8")
    note = {
        "text": "x",
        "evidence": {"source_path": "a.py", "evidence_anchor_hash": old_hash},
    }
    result = apply_evidence_staleness(note, tmp_path)
    assert result is not note
    assert result["stale_evidence"] is True


def test_apply_evidence_staleness_missing_file_untouched(tmp_path: Path) -> None:
    note = {
        "text": "x",
        "evidence": {"source_path": "gone.py", "evidence_anchor_hash": "abc"},
    }
    assert apply_evidence_staleness(note, tmp_path) is note
