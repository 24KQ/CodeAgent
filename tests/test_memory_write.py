"""P0 slice 3 tests: atomic writes and cross-process locking (write.py)."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from firstcoder.memory.write import atomic_write_bytes, atomic_write_text, locked_append_line


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "note.md"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def test_atomic_write_text_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    atomic_write_bytes(target, b"data")
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_locked_append_line_appends_sequentially(tmp_path: Path) -> None:
    target = tmp_path / "daily" / "2026-08-06.md"
    locked_append_line(target, "first")
    locked_append_line(target, "second")
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


def test_locked_append_line_strips_embedded_newlines(tmp_path: Path) -> None:
    target = tmp_path / "daily.md"
    locked_append_line(target, "a\nb")
    assert target.read_text(encoding="utf-8") == "a\nb\n"


def _concurrent_worker(prefix: str, target: str, count: int) -> None:
    from firstcoder.memory.write import locked_append_line

    for i in range(count):
        locked_append_line(Path(target), f"{prefix}-{i}")


def test_concurrent_appends_across_processes(tmp_path: Path) -> None:
    """The Codex TOCTOU finding: concurrent appenders must not lose lines."""
    target = tmp_path / "daily.md"
    workers = 4
    appends_per_worker = 3
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        pool.starmap(
            _concurrent_worker,
            [(f"p{w}", str(target), appends_per_worker) for w in range(workers)],
        )
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == workers * appends_per_worker
    assert len(set(lines)) == workers * appends_per_worker
