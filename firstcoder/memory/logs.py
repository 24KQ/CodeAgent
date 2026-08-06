"""Daily log primitives (fusion P2, M1).

Ported from pico `features/memory.py:78-110` with the M1 write-upgrade:
the append is now a locked read-modify-write through a shared `.daily.lock`
(Codex TOCTOU finding), so concurrent processes cannot lose or interleave
entries. The optional `source` evidence is written to a per-day sidecar
(`<date>.evidence.jsonl`) under the same lock — the Store's
`append_daily_log` delegates here, so both APIs serialize on one lock
(Codex P2 review #3: previously two different locks could interleave).

声明边界：本模块只管 daily log 的目录/路径/追加，不做任何内容判定；
secrets 判定与 quarantine 在 security.py，evidence 侧车行由
`source` 参数原样记录（不校验）。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from firstcoder.memory.models import MemoryEvidence
from firstcoder.memory.paths import ensure_no_link_or_junction
from firstcoder.memory.write import atomic_write_bytes, atomic_write_text, cross_process_lock

ENTRYPOINT_NAME = "MEMORY.md"

_EMPTY_INDEX = (
    "# Durable Memory Index\n\n"
    "_Empty. `/remember` writes a daily log entry; `/dream` consolidates "
    "logs into topic files and adds entries here._\n"
)


def daily_lock_path(memory_dir: str | Path) -> Path:
    """daily log 的唯一互斥锁：日志行与 evidence 侧车共用（见模块 docstring）。"""
    return Path(memory_dir) / ".daily.lock"


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
    *,
    source: MemoryEvidence | None = None,
) -> Path | None:
    """追加一条带时间戳的日志行；`source` 给定时在同一把 `.daily.lock` 内
    追加当天 evidence 侧车行（原子读改写，行序与日志一致）。空 entry 返回 None。

    崩溃一致性（recovery 语义，Codex P2 review #3）：日志与侧车是两个
    文件、两次替换，进程在两次替换之间崩溃会留下孤儿日志行（无侧车行）
    或孤儿侧车行（有侧车无日志行）。恢复策略：孤儿日志行按"无证据的普通
    记忆"处理（P3 /remember 正常提升）；孤儿侧车行被读取方忽略
    （load_daily_log_evidence 只按侧车行读，不承诺配对完整性）。
    """
    entry = str(entry).strip()
    if not entry:
        return None
    memory_dir = Path(memory_dir)
    ensure_no_link_or_junction(memory_dir)
    # logs 目录本身也可能被预置为 symlink/junction（Codex P2 review #3）：
    # _transaction 只覆盖 store 写路径，这里必须覆盖 standalone/委托入口。
    ensure_no_link_or_junction(memory_dir / "logs")
    path = daily_log_path(memory_dir, today=today)
    timestamp = datetime.now().strftime("%H:%M")
    evidence_path = path.with_name(path.stem + ".evidence.jsonl")
    with cross_process_lock(daily_lock_path(memory_dir)):
        existing_log = path.read_text(encoding="utf-8") if path.exists() else ""
        atomic_write_bytes(path, (existing_log + f"- [{timestamp}] {entry}" + "\n").encode("utf-8"))
        if source is not None:
            row = {
                "text": entry,
                "session_id": source.session_id,
                "source_path": source.source_path,
                "evidence_anchor_hash": source.anchor_hash,
                "scope": source.scope,
                "at": datetime.now().astimezone().isoformat(),
            }
            existing_evidence = evidence_path.read_text(encoding="utf-8") if evidence_path.exists() else ""
            atomic_write_bytes(
                evidence_path,
                (existing_evidence + json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
            )
    return path
