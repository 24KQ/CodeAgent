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

import json
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
        if candidate.exists():
            # 已存在的目录必须是 store root 内的真实路径：resolve() 解析
            # symlink / junction 后做 containment 校验，防止 run 目录被
            # 重定向到 root 之外。新建目录不检查（mkdir 不会跟随不存在的
            # 链接；预置 symlink 的 TOCTOU 在 Windows 上需要特权，残留
            # 风险在此处用注释记录）。
            root = self.root.resolve()
            resolved = candidate.resolve()
            if not (resolved == root or root in resolved.parents):
                raise ValueError(f"run dir {value!r} resolves outside the store root")
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
        # 保证追加不需要跨进程锁。
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True))
            handle.write("\n")
        return path

    def write_report(self, task_state: object, report: dict) -> Path:
        path = self.report_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, report)
        return path

    def load_task_state(self, task_id: str) -> dict:
        return json.loads(self.task_state_path(task_id).read_text(encoding="utf-8"))

    def load_report(self, task_id: str) -> dict:
        return json.loads(self.report_path(task_id).read_text(encoding="utf-8"))

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        # 原子写：先写临时文件，再 replace（P0 原语，memory/write.py）。
        # 即使中途异常，也不容易留下半截 JSON。
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        atomic_write_bytes(path, text.encode("utf-8"))
