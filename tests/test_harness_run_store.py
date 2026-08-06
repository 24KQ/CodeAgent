"""P1 slice 1 tests: RunStore artifact persistence (run_store.py)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from firstcoder.harness.run_store import RunStore
from firstcoder.harness.task_state import TaskState


def _state(tmp_path: Path, run_id: str = "run_abc") -> TaskState:
    return TaskState.create("t1", "user request", run_id=run_id)


def _try_make_junction(link: Path, target: Path) -> bool:
    """Windows 上用 mklink /J 创建 junction（普通用户即可，无需管理员）。
    非 Windows 或创建失败返回 False，调用方据此 skip。"""
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_start_run_creates_layout(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    run_dir = store.start_run(_state(tmp_path))
    assert run_dir == store.run_dir("run_abc")
    assert (run_dir / "task_state.json").exists()
    assert store.load_task_state("run_abc")["status"] == "running"


def test_write_task_state_updates(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = _state(tmp_path)
    store.start_run(state)
    state.finish_success("done")
    store.write_task_state(state)
    assert store.load_task_state("run_abc")["stop_reason"] == "final_answer_returned"


def test_append_trace_appends_jsonl_lines(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = _state(tmp_path)
    store.start_run(state)
    store.append_trace(state, {"event": "run_started", "turn_id": "t1"})
    store.append_trace(state, {"event": "loop_transition", "kind": "continue"})
    lines = store.trace_path("run_abc").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "run_started"


def test_write_and_load_report(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = _state(tmp_path)
    store.start_run(state)
    report = {"run_id": "run_abc", "status": "completed", "stop_reason": "final_answer_returned"}
    store.write_report(state, report)
    assert store.load_report("run_abc") == report


def test_artifacts_dir(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    state = _state(tmp_path)
    assert store.artifacts_dir(state).name == "artifacts"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../escape",
        "a/b",
        ".",
        "..",
        "run x",
        "run\\x",
        "run..x/..",
        "run_abc\n",  # 末尾换行逃逸（fullmatch 拒绝，Codex P1 review fix）
        "run_abc.",  # Windows 剥离末尾点 -> 与其他 id 撞目录
        "run_abc ",
        "CON",  # Windows 保留名
        "con.txt",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "lpt9",
    ],
)
def test_run_id_guard_rejects_unsafe_names(tmp_path: Path, bad: str) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(ValueError):
        store.run_dir(bad)


def test_run_id_guard_accepts_safe_names(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    assert store.run_dir("run_20260806-121314-abc123") == tmp_path / "runs" / "run_20260806-121314-abc123"
    assert store.run_dir("task.a-1") == tmp_path / "runs" / "task.a-1"


def test_run_dir_rejects_symlink_escape(tmp_path: Path) -> None:
    """已存在的 run 目录若解析出 store root，必须拒绝（symlink/junction 逃逸）。"""
    store = RunStore(tmp_path / "runs")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "runs" / "run_evil"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")
    with pytest.raises(ValueError):
        store.run_dir("run_evil")


def test_run_dir_rejects_symlink_to_store_root(tmp_path: Path) -> None:
    """run 目录是指向 store root 本身的链接也必须拒绝（写入会落进根目录）。"""
    store = RunStore(tmp_path / "runs")
    link = tmp_path / "runs" / "run_root"
    try:
        link.symlink_to(tmp_path / "runs", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")
    with pytest.raises(ValueError):
        store.run_dir("run_root")


def test_run_dir_rejects_symlink_aliasing_another_run(tmp_path: Path) -> None:
    """run 目录 symlink 指向 store 内另一个 run：不出 root 但破坏 run 隔离，
    同样必须拒绝（Codex P1 review fix 复验）。"""
    store = RunStore(tmp_path / "runs")
    store.start_run(_state(tmp_path, run_id="run_a"))
    link = tmp_path / "runs" / "run_b"
    try:
        link.symlink_to(tmp_path / "runs" / "run_a", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")
    with pytest.raises(ValueError):
        store.run_dir("run_b")


def test_run_dir_rejects_junction_aliasing_another_run(tmp_path: Path) -> None:
    """Windows junction 指向 store 内另一个 run 也必须拒绝（Codex re-review #4）：
    Python 3.12 起 is_symlink 与 is_junction 分离，junction 对 is_symlink()
    返回 False，需显式检查；且 junction 普通用户即可创建，无需管理员。"""
    store = RunStore(tmp_path / "runs")
    store.start_run(_state(tmp_path, run_id="run_a"))
    link = tmp_path / "runs" / "run_b"
    if not _try_make_junction(link, tmp_path / "runs" / "run_a"):
        pytest.skip("junction creation not available on this host")
    with pytest.raises(ValueError):
        store.run_dir("run_b")


def test_append_trace_rejects_file_symlink(tmp_path: Path) -> None:
    """trace 文件本身是 symlink 时拒绝跟随（Codex P1 review fix 复验）。"""
    store = RunStore(tmp_path / "runs")
    state = _state(tmp_path, run_id="run_link")
    store.start_run(state)
    target = tmp_path / "leaked.jsonl"
    try:
        store.trace_path("run_link").symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")
    with pytest.raises(ValueError):
        store.append_trace(state, {"event": "run_started"})
