"""显式 memory slash commands。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from firstcoder.app.commands import CommandResult
from firstcoder.memory.prompt import MemoryProjector, load_memory_index_text
from firstcoder.memory.runtime import MemoryRuntime


class MemoryCommandSession(Protocol):
    """命令只依赖当前 session 的 memory facade，不直接拼文件路径。"""

    memory_runtime: MemoryRuntime
    memory_projector: MemoryProjector


@dataclass(slots=True)
class MemoryCommandHandler:
    """处理 ``/remember`` 和 ``/memory [query]``。"""

    session: MemoryCommandSession

    def handle(self, text: str) -> CommandResult:
        command = text.strip()
        if not command.startswith("/"):
            return CommandResult(handled=False)

        if command == "/memory":
            return CommandResult(handled=True, output=self._show_index())
        if command.startswith("/memory "):
            query = command.split(" ", 1)[1].strip()
            if not query:
                return CommandResult(handled=True, output=self._show_index())
            return CommandResult(
                handled=True,
                output=self.session.memory_projector.render_retrieval(query),
            )
        if command == "/remember":
            return CommandResult(handled=True, output="Usage: /remember <text> [--promote <topic>]")
        if command.startswith("/remember "):
            return CommandResult(handled=True, output=self._remember(command[10:].strip()))
        return CommandResult(handled=False)

    def _show_index(self) -> str:
        index = load_memory_index_text(
            self.session.memory_runtime.store.root,
            security=self.session.memory_runtime.security,
        )
        if not index:
            return "No durable memories yet. Use /remember <text> to capture one."
        return index

    def _remember(self, raw: str) -> str:
        text, topic = _split_promote_syntax(raw)
        if not text:
            return "Usage: /remember <text> [--promote <topic>]"

        try:
            captured = self.session.memory_runtime.record(text, source="slash_command")
        except Exception:  # noqa: BLE001 - slash command 只返回安全摘要
            return "Memory note could not be saved."
        if not captured.ok:
            return captured.error or "Memory note could not be saved."
        if topic is None:
            if captured.quarantined:
                return "Saved to the daily log; the note is quarantined from durable memory."
            return "Saved to the daily log."

        try:
            promoted = self.session.memory_runtime.promote(topic, text, source="slash_command")
        except Exception:  # noqa: BLE001 - 不把原始输入交给 UI 错误文本
            return "Saved to the daily log; durable promotion was blocked."
        if not promoted.ok:
            return "Saved to the daily log; durable promotion was blocked."
        return "Saved to the daily log and promoted to durable memory."


def _split_promote_syntax(raw: str) -> tuple[str, str | None]:
    """解析两种显式提升写法，同时保留普通 note 的原始空格。

    支持 ``/remember text --promote topic`` 和 ``/remember --promote topic text``；
    另支持 ``promote topic text`` 作为便于工具用户输入的别名。topic 最终仍由
    DurableMemoryStore 校验，不能借命令解析绕过安全 slug 约束。
    """

    value = str(raw or "").strip()
    if not value:
        return "", None

    prefix = re.match(r"^(?:promote|--promote)\s+([^\s]+)\s+(.+)$", value, re.IGNORECASE)
    if prefix:
        return prefix.group(2).strip(), prefix.group(1).strip()

    suffix = re.match(r"^(.+?)\s+--promote(?:\s+([^\s]+))?$", value, re.IGNORECASE)
    if suffix:
        return suffix.group(1).strip(), (suffix.group(2) or "key-decisions").strip()
    return value, None


__all__ = ["MemoryCommandHandler"]
