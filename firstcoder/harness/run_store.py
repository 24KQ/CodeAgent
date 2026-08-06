"""Per-run artifact persistence (fusion P1, H1).

Ported from pico `core/run_store.py`. Session JSONL stores resumable
conversation state; RunStore stores audit artifacts for one run
(task_state.json / trace.jsonl / report.json / artifacts/) so recovery
state and review evidence stay separate.

Write hardening (P0 slice 3): JSON payloads go through the atomic
temp+rename primitive (`firstcoder.memory.write.atomic_write_bytes`);
trace appends stay plain-append because a trace is single-writer by
invariant (one runtime, one run). Run ids are validated to keep run
directory paths inside the store root.
"""

from __future__ import annotations

import errno
import json
import os
import re
from pathlib import Path

from firstcoder.memory.write import atomic_write_bytes

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")

#: Windows 保留名（含扩展名形式如 `CON.txt`），拒绝以避免写盘重定向到设备。
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _is_link_or_junction(path: Path) -> bool:
    """路径是 symlink 或 Windows junction（目录重解析点）即为链接。

    Python 3.12 起 `Path.is_symlink()` 与 `Path.is_junction()` 分离：
    3.12 上 junction 对 `is_symlink()` 返回 False，必须显式检查
    （Codex re-review #4 指出）。3.11 及以前 `os.path.islink` 已涵盖
    junction，`is_junction` 不存在，用 getattr 兼容。
    """
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _check_run_id(self, run_id: object) -> str:
        value = str(getattr(run_id, "run_id", run_id) or "")
        # fullmatch + \Z：字符集不含路径分隔符，且拒绝末尾换行的逃逸；
        # 整体等于 "." / ".." 也拒绝。
        if not _RUN_ID_PATTERN.fullmatch(value) or value in (".", ".."):
            raise ValueError(f"run id {value!r} is not a safe directory name")
        # Windows 会静默剥离目录名末尾的点和空格，导致两个 id 指向同一目录。
        if value.rstrip(". ") != value:
            raise ValueError(f"run id {value!r} must not end with '.' or whitespace")
        if value.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"run id {value!r} is a Windows reserved name")
        return value

    def run_dir(self, run_id: object) -> Path:
        value = self._check_run_id(run_id)
        candidate = self.root / value
        if _is_link_or_junction(candidate):
            # is_symlink()（lexists 语义）能抓住断裂链接——exists() 对断裂
            # symlink 返回 False；is_junction() 补上 Python 3.12 起与
            # symlink 分离的 Windows junction（普通用户可创建、无需管理员）。
            # run 目录不允许是任何形式的链接：既防逃逸出 store root，也防
            # 指向 store 内其他 run 的别名破坏 run 隔离（Codex P1 review fix）。
            raise ValueError(f"run dir {value!r} must not be a symlink or junction")
        if candidate.exists():
            root = self.root.resolve()
            resolved = candidate.resolve()
            if root not in resolved.parents:
                raise ValueError(f"run dir {value!r} resolves outside the store root")
        # 威胁模型边界（Codex re-review #4 正式接受）：本方法防御的是预置的
        # 静态路径欺骗——恶意 run_id、store 内被预置的 symlink/junction
        # 别名或逃逸。不防御"检查后、使用前"的并发路径替换（目录级 TOCTOU）：
        # 那要求攻击者能在 store root 内创建或替换目录条目，而 store root
        # 的写入者只有 agent 运行时与用户；能这么做的攻击者已可直接替换
        # store root 本身（root 无法自证），超出本类防御范围。文件级
        # TOCTOU 由 `_open_no_follow`（O_NOFOLLOW）缩窗。
        return candidate

    def task_state_path(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "task_state.json"

    def trace_path(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "trace.jsonl"

    def report_path(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "report.json"

    def artifacts_dir(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "artifacts"

    def start_run(self, task_state: object) -> Path:
        """One user request maps to one run directory of independent artifacts."""
        run_dir = self.run_dir(task_state)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.write_task_state(task_state)
        return run_dir

    def write_task_state(self, task_state: object) -> Path:
        path = self.task_state_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, task_state.to_dict())
        return path

    def append_trace(self, task_state: object, event: dict) -> Path:
        path = self.trace_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        # trace 采用 jsonl 追加写入：agent 运行是流式事件序列，逐条落盘
        # 比最后一次性写整份 trace 更稳，也更适合调试。单 writer 不变量
        # 保证追加不需要跨进程锁。O_CREAT：首次写入时文件尚不存在
        # （Codex P1 review fix 复验：_open_no_follow 不再吞掉创建）。
        fd = self._open_no_follow(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True))
            handle.write("\n")
        return path

    def write_report(self, task_state: object, report: dict) -> Path:
        path = self.report_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, report)
        return path

    def load_task_state(self, task_id: str) -> dict:
        return json.loads(self._read_no_follow(self.task_state_path(task_id)))

    def load_report(self, task_id: str) -> dict:
        return json.loads(self._read_no_follow(self.report_path(task_id)))

    @staticmethod
    def _open_no_follow(path: Path, flags: int) -> int:
        """链接前置检查 + POSIX O_NOFOLLOW 双保险（Codex P1 review fix）。

        残留窗口如实记录，不声称完全闭合：
        - POSIX：O_NOFOLLOW 只保护最终文件组件，不保护父目录——攻击者把
          run 目录替换为指向外部的链接时，open 仍会跟随（目录级 TOCTOU，
          威胁模型边界见 `run_dir`，已正式接受）。
        - Windows：无 O_NOFOLLOW flag，前置检查与 open 之间对最终文件
          组件也存在理论 TOCTOU（需攻击者并发替换文件）。
        """
        if _is_link_or_junction(path):
            raise ValueError(f"{path} is a symlink or junction; refusing to follow")
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(path, flags | no_follow)
        except OSError as exc:
            if no_follow and exc.errno == errno.ELOOP:
                raise ValueError(f"{path} is a symlink; refusing to follow") from exc
            raise

    def _read_no_follow(self, path: Path) -> str:
        """以 O_RDONLY 读取文件，链接前置检查 + O_NOFOLLOW（同 `_open_no_follow`）。"""
        fd = self._open_no_follow(path, os.O_RDONLY)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            return handle.read()

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        # 原子写：先写临时文件，再 replace（P0 原语，memory/write.py）。
        # 即使中途异常，也不容易留下半截 JSON。
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        atomic_write_bytes(path, text.encode("utf-8"))
