"""P2 tests: daily log primitives (logs.py, fusion M1)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from firstcoder.memory.logs import ENTRYPOINT_NAME, append_to_daily_log, daily_log_path, ensure_memory_dir


def test_ensure_memory_dir_creates_layout(tmp_path: Path) -> None:
    root = ensure_memory_dir(tmp_path / "memory")
    assert (root / "logs").is_dir()
    assert (root / "topics").is_dir()
    index = root / ENTRYPOINT_NAME
    assert index.exists()
    assert "# Durable Memory Index" in index.read_text(encoding="utf-8")


def test_ensure_memory_dir_is_idempotent(tmp_path: Path) -> None:
    root = ensure_memory_dir(tmp_path / "memory")
    index = root / ENTRYPOINT_NAME
    ensure_memory_dir(root)
    # 幂等：不覆盖已有索引（empty 标记保持原样）
    assert index.read_text(encoding="utf-8").startswith("# Durable Memory Index")


def test_daily_log_path_year_month_day_layout(tmp_path: Path) -> None:
    today = date(2026, 8, 6)
    path = daily_log_path(tmp_path / "memory", today=today)
    assert path == tmp_path / "memory" / "logs" / "2026" / "08" / "2026-08-06.md"
    assert path.parent.is_dir()


def test_append_to_daily_log_formats_entry(tmp_path: Path) -> None:
    today = date(2026, 8, 6)
    path = append_to_daily_log(tmp_path / "memory", "remember: use pytest", today=today)
    assert path is not None
    line = path.read_text(encoding="utf-8").splitlines()[0]
    assert re.match(r"^- \[\d{2}:\d{2}\] remember: use pytest$", line)


def test_append_to_daily_log_appends_multiple_entries(tmp_path: Path) -> None:
    today = date(2026, 8, 6)
    append_to_daily_log(tmp_path / "memory", "first", today=today)
    append_to_daily_log(tmp_path / "memory", "second", today=today)
    lines = (tmp_path / "memory" / "logs" / "2026" / "08" / "2026-08-06.md").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 2


def test_append_to_daily_log_empty_entry_returns_none(tmp_path: Path) -> None:
    assert append_to_daily_log(tmp_path / "memory", "   ") is None


def test_append_to_daily_log_strips_whitespace(tmp_path: Path) -> None:
    today = date(2026, 8, 6)
    path = append_to_daily_log(tmp_path / "memory", "  note text  ", today=today)
    assert path is not None
    assert "note text" in path.read_text(encoding="utf-8")
    assert "note text  " not in path.read_text(encoding="utf-8")


def test_append_to_daily_log_with_source_writes_sidecar(tmp_path: Path) -> None:
    """source 给定时在同一把 .daily.lock 内追加 evidence 侧车行（Codex P2 review #4）。"""
    import json

    from firstcoder.memory.models import MemoryEvidence

    today = date(2026, 8, 6)
    source = MemoryEvidence(source_path="src/a.py", session_id="s1", anchor_hash="h1", scope="workspace")
    path = append_to_daily_log(tmp_path / "memory", "remember: use pytest", today=today, source=source)
    assert path is not None

    evidence_path = path.with_name(path.stem + ".evidence.jsonl")
    rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["text"] == "remember: use pytest"
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["evidence_anchor_hash"] == "h1"
    assert rows[0]["scope"] == "workspace"


def test_append_to_daily_log_without_source_no_sidecar(tmp_path: Path) -> None:
    today = date(2026, 8, 6)
    path = append_to_daily_log(tmp_path / "memory", "plain entry", today=today)
    assert path is not None
    assert not path.with_name(path.stem + ".evidence.jsonl").exists()
