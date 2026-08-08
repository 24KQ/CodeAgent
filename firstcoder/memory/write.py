"""Cross-process safe memory writes (fusion P0 slice 3).

M3 write-upgrade: pico wrote daily logs / metadata / MEMORY.md with plain
`write_text` and a PID lock (TOCTOU, memory.py:320-337). These primitives
replace that with temp+rename atomic publication plus a portalocker
cross-process lock, mirroring the task-plan lock idiom in
planning/service.py. Daily-log appends are read-modify-write under the
lock so concurrent processes cannot lose each other's lines.

Pure infrastructure — existing behavior must remain unchanged until a real
writer is wired in (P2 memory data plane).
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import portalocker


@contextmanager
def cross_process_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive cross-process lock (portalocker LOCK_EX), blocking.

    `lock_path` parent directories are created on demand. Follows the
    task-plan lock idiom (planning/service.py) without its per-session
    thread-lock/digest layer — callers that need both combine them.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a+b", flags=portalocker.LOCK_EX):
        yield


@contextmanager
def try_cross_process_lock(lock_path: Path) -> Iterator[bool]:
    """尝试取得独占锁，忙时立即返回 ``False`` 而不是阻塞调用方。

    普通 durable 写入必须等待锁以保证提交顺序；scheduler 的触发入口则位于
    主 turn 收口路径，不能为了等待另一个进程的 provider 调用而阻塞用户。两种
    语义共用同一 portalocker 文件，只在等待策略上分开。
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(
        lock_path,
        mode="a+b",
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    )
    try:
        # LOCK_NB 只改变底层锁标志；portalocker 默认仍会按 5 秒超时重试。
        # scheduler 触发位于普通 turn 收口路径，必须把“忙”变成一次即时结果。
        lock.acquire(timeout=0, fail_when_locked=True)
    except portalocker.exceptions.AlreadyLocked:
        yield False
        return
    try:
        yield True
    finally:
        lock.release()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically: same-dir temp file + fsync + rename.

    A crash or concurrent reader sees either the old content or the new
    content, never a partial write. On failure the temp file is removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """`atomic_write_bytes` for text content."""
    atomic_write_bytes(path, text.encode(encoding))


def locked_append_line(path: Path, line: str, encoding: str = "utf-8") -> None:
    """Append one line to `path` under a cross-process lock.

    Read-modify-write: the lock serializes concurrent appenders so the
    daily log cannot lose lines or interleave. Lock file lives next to the
    target (`<path>.lock`).
    """
    lock_path = path.with_name(path.name + ".lock")
    with cross_process_lock(lock_path):
        existing = path.read_text(encoding=encoding) if path.exists() else ""
        atomic_write_text(path, existing + line.rstrip("\n") + "\n", encoding=encoding)
