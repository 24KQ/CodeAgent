"""Daily log primitives (fusion P2, M1).

Ported from pico `features/memory.py:78-110` with the M1 write-upgrade:
the append is now a locked read-modify-write through
`firstcoder.memory.write.locked_append_line` (Codex TOCTOU finding), so
concurrent processes cannot lose or interleave entries.

声明边界：本模块只管 daily log 的目录/路径/追加，不做任何内容判定；
secrets 判定与 quarantine 在 security.py，evidence 侧车在 durable.py。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from firstcoder.memory.write import atomic_write_text, locked_append_line

ENTRYPOINT_NAME = "MEMORY.md"

_EMPTY_INDEX = (
    "# Durable Memory Index\n\n"
    "_Empty. `/remember` writes a daily log entry; `/dream` consolidates "
    "logs into topic files and adds entries here._\n"
)


def ensure_memory_dir(memory_dir: str | Path) -> Path:
    """确保 memory 目录骨架存在（logs/topics/index），缺失时原子创建。"""
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "logs").mkdir(parents=True, exist_ok=True)
    (memory_dir / "topics").mkdir(parents=True, exist_ok=True)
    index_path = memory_dir / ENTRYPOINT_NAME
    if not index_path.exists():
        atomic_write_text(index_path, _EMPTY_INDEX)
    return memory_dir


def daily_log_path(memory_dir: str | Path, today: date | None = None) -> Path:
    """当日日志路径：`logs/<year>/<month>/<date>.md`，父目录自动创建。"""
    today = today or date.today()
    memory_dir = ensure_memory_dir(memory_dir)
    path = memory_dir / "logs" / str(today.year) / f"{today.month:02d}" / f"{today.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_to_daily_log(
    memory_dir: str | Path,
    entry: str,
    today: date | None = None,
) -> Path | None:
    """追加一条带时间戳的日志行（加锁读改写，防并发丢失）；空 entry 返回 None。"""
    entry = str(entry).strip()
    if not entry:
        return None
    path = daily_log_path(memory_dir, today=today)
    timestamp = datetime.now().strftime("%H:%M")
    locked_append_line(path, f"- [{timestamp}] {entry}")
    return path
