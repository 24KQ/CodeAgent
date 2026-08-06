"""P2 tests: working-memory pure transforms (working.py, fusion M2)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from firstcoder.memory.durable import DurableMemoryStore
from firstcoder.memory.working import (
    EPISODIC_NOTE_LIMIT,
    WORKING_FILE_LIMIT,
    append_note,
    clip,
    default_memory_state,
    invalidate_file_summary,
    invalidate_stale_file_summaries,
    normalize_memory_state,
    remember_file,
    set_file_summary,
    set_task_summary,
    summarize_read_result,
    utc_now_iso,
)


def test_clip_short_text_unchanged() -> None:
    assert clip("hello", 100) == "hello"


def test_clip_truncates_with_marker() -> None:
    result = clip("a" * 10, 5)
    assert result == "aaaaa\n...[truncated 5 chars]"


def test_utc_now_iso_is_utc_iso() -> None:
    parsed = datetime.fromisoformat(utc_now_iso())
    assert parsed.utcoffset() is not None


def test_default_state_shape() -> None:
    state = default_memory_state()
    assert state["working"] == {"task_summary": "", "recent_files": []}
    assert state["episodic_notes"] == []
    assert state["file_summaries"] == {}
    assert state["next_note_index"] == 0


def test_normalize_none_returns_default() -> None:
    state = normalize_memory_state(None)
    assert state["working"]["task_summary"] == ""


def test_normalize_non_mapping_raises() -> None:
    import pytest

    with pytest.raises(TypeError):
        normalize_memory_state("not a state")


def test_normalize_migrates_legacy_fields(tmp_path: Path) -> None:
    state = normalize_memory_state(
        {
            "task": "port the memory layer",
            "files": ["src/a.py", "src/b.py", "src/a.py"],
            "notes": ["first thought", "second thought"],
        },
        workspace_root=tmp_path,
    )
    assert state["working"]["task_summary"] == "port the memory layer"
    assert state["working"]["recent_files"] == ["src/a.py", "src/b.py"]  # 去重保序
    assert state["task"] == "port the memory layer"
    assert state["files"] == ["src/a.py", "src/b.py"]
    assert [note["text"] for note in state["episodic_notes"]] == ["first thought", "second thought"]
    assert state["notes"] == ["first thought", "second thought"]
    assert state["next_note_index"] == 2


def test_normalize_caps_recent_files(tmp_path: Path) -> None:
    files = [f"src/f{i}.py" for i in range(12)]
    state = normalize_memory_state({"files": files}, workspace_root=tmp_path)
    assert len(state["working"]["recent_files"]) == WORKING_FILE_LIMIT
    assert state["working"]["recent_files"][0] == "src/f4.py"


def test_normalize_caps_episodic_notes(tmp_path: Path) -> None:
    notes = [f"note {i}" for i in range(20)]
    state = normalize_memory_state({"notes": notes}, workspace_root=tmp_path)
    assert len(state["episodic_notes"]) == EPISODIC_NOTE_LIMIT
    assert state["episodic_notes"][0]["text"] == "note 8"


def test_normalize_normalizes_note_shape(tmp_path: Path) -> None:
    state = normalize_memory_state(
        {
            "episodic_notes": [
                {"text": "  spaced  ", "tags": ["a", "a", " b "], "note_index": 3},
                "plain string note",
            ]
        },
        workspace_root=tmp_path,
    )
    first = state["episodic_notes"][0]
    assert first["text"] == "spaced"
    assert first["tags"] == ["a", "b"]  # 去重保序
    assert first["note_index"] == 3
    second = state["episodic_notes"][1]
    assert second["text"] == "plain string note"
    assert second["kind"] == "episodic"


def test_normalize_durable_topics_injected(tmp_path: Path) -> None:
    store = DurableMemoryStore(tmp_path / "memory")
    store.promote([("key-decisions", "pytest is the runner")])
    state = normalize_memory_state(default_memory_state(), store=store)
    assert state["durable_topics"] == ["key-decisions"]


def test_set_task_summary_updates_mirrors(tmp_path: Path) -> None:
    state = set_task_summary(default_memory_state(), "   new summary   ", workspace_root=tmp_path)
    assert state["working"]["task_summary"] == "new summary"
    assert state["task"] == "new summary"


def test_set_task_summary_clips(tmp_path: Path) -> None:
    state = set_task_summary(default_memory_state(), "x" * 400, workspace_root=tmp_path)
    assert state["working"]["task_summary"].startswith("x" * 300)
    assert "[truncated" in state["working"]["task_summary"]


def test_remember_file_moves_duplicate_to_end(tmp_path: Path) -> None:
    state = remember_file(default_memory_state(), "src/a.py", workspace_root=tmp_path)
    state = remember_file(state, "src/b.py", workspace_root=tmp_path)
    state = remember_file(state, "src/a.py", workspace_root=tmp_path)
    assert state["working"]["recent_files"] == ["src/b.py", "src/a.py"]
    assert state["files"] == ["src/b.py", "src/a.py"]


def test_append_note_dedupes_and_increments_index(tmp_path: Path) -> None:
    state = default_memory_state()
    state = append_note(state, "first", tags=("a", "a"), workspace_root=tmp_path)
    state = append_note(state, "second", workspace_root=tmp_path)
    state = append_note(state, "first", workspace_root=tmp_path)  # 重复：移动到末尾
    assert [note["text"] for note in state["episodic_notes"]] == ["second", "first"]
    assert [note["note_index"] for note in state["episodic_notes"]] == [1, 2]
    assert state["next_note_index"] == 3
    assert state["notes"] == ["second", "first"]


def test_append_note_empty_is_noop(tmp_path: Path) -> None:
    state = append_note(default_memory_state(), "   ", workspace_root=tmp_path)
    assert state["episodic_notes"] == []


def test_append_note_caps_limit(tmp_path: Path) -> None:
    state = default_memory_state()
    for i in range(20):
        state = append_note(state, f"note {i}", workspace_root=tmp_path)
    assert len(state["episodic_notes"]) == EPISODIC_NOTE_LIMIT
    assert state["episodic_notes"][0]["text"] == "note 8"


def test_set_file_summary_with_freshness(tmp_path: Path) -> None:
    target = tmp_path / "src" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("code", encoding="utf-8")
    state = set_file_summary(default_memory_state(), "src/a.py", "has the helper", workspace_root=tmp_path)
    summary = state["file_summaries"]["src/a.py"]
    assert summary["summary"] == "has the helper"
    assert summary["freshness"] is not None


def test_set_file_summary_empty_skipped(tmp_path: Path) -> None:
    state = set_file_summary(default_memory_state(), "src/a.py", "   ", workspace_root=tmp_path)
    assert state["file_summaries"] == {}


def test_invalidate_file_summary(tmp_path: Path) -> None:
    state = set_file_summary(default_memory_state(), "src/a.py", "s", workspace_root=tmp_path)
    state = invalidate_file_summary(state, "src/a.py", workspace_root=tmp_path)
    assert state["file_summaries"] == {}


def test_invalidate_stale_file_summaries(tmp_path: Path) -> None:
    target = tmp_path / "src" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("v1", encoding="utf-8")
    state = set_file_summary(default_memory_state(), "src/a.py", "s1", workspace_root=tmp_path)
    target.write_text("v2", encoding="utf-8")
    state, invalidated = invalidate_stale_file_summaries(state, workspace_root=tmp_path)
    assert invalidated == ["src/a.py"]
    assert state["file_summaries"] == {}


def test_invalidate_stale_keeps_fresh(tmp_path: Path) -> None:
    target = tmp_path / "src" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("v1", encoding="utf-8")
    state = set_file_summary(default_memory_state(), "src/a.py", "s1", workspace_root=tmp_path)
    state, invalidated = invalidate_stale_file_summaries(state, workspace_root=tmp_path)
    assert invalidated == []
    assert "src/a.py" in state["file_summaries"]


def test_summarize_read_result_joins_lines() -> None:
    result = summarize_read_result("# Header\nline one\nline two\nline three\nline four")
    assert result == "line one | line two | line three"


def test_summarize_read_result_empty() -> None:
    assert summarize_read_result("   \n ") == "(empty)"


def test_summarize_read_result_clips() -> None:
    long = "\n".join(f"line {i}" for i in range(10))
    result = summarize_read_result(long, limit=20)
    assert "[truncated" in result
    assert len(result) < len(long)
