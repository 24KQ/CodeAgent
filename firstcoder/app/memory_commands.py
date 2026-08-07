"""显式 memory slash commands。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from firstcoder.app.commands import CommandResult
from firstcoder.memory.durable import DURABLE_TOPIC_DEFAULTS
from firstcoder.memory.prompt import MemoryProjector
from firstcoder.memory.runtime import MemoryRuntime

_PROMOTABLE_TOPICS = frozenset(DURABLE_TOPIC_DEFAULTS)
_INVALID_PROMOTE = ""


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
            include_global, query = _split_global_flag(command.split(" ", 1)[1].strip())
            if not query:
                return CommandResult(
                    handled=True,
                    output=self._show_index(include_global=include_global),
                )
            return CommandResult(
                handled=True,
                output=self.session.memory_projector.render_retrieval(
                    query,
                    include_global=include_global,
                ),
            )
        if command == "/remember":
            return CommandResult(
                handled=True,
                output="Usage: /remember <text> [--promote <topic>] [--global]",
            )
        if command.startswith("/remember "):
            return CommandResult(handled=True, output=self._remember(command[10:].strip()))
        return CommandResult(handled=False)

    def _show_index(self, *, include_global: bool = False) -> str:
        # index 展示也必须走 projector 的 visibility 过滤，不能直接读取
        # MEMORY.md；否则 session-only/global 条目会绕过读取边界。
        index = self.session.memory_projector.render_index(include_global=include_global)
        if not index:
            return "No visible durable memories for this session."
        return index

    def _remember(self, raw: str) -> str:
        include_global, scoped_raw = _split_global_flag(raw)
        text, topic = _split_promote_syntax(scoped_raw)
        if topic == _INVALID_PROMOTE:
            return "Usage: /remember <text> [--promote <topic>] [--global]"
        if not text:
            return "Usage: /remember <text> [--promote <topic>] [--global]"
        if include_global and topic is None:
            # global 只能通过明确 promotion 进入独立 store，普通 capture
            # 仍固定属于当前 session，避免一个 flag 意外改变 capture 语义。
            return "Usage: /remember <text> [--promote <topic>] [--global]"

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
            promoted = self.session.memory_runtime.promote(
                topic,
                text,
                source="slash_command",
                visibility="global" if include_global else "workspace",
            )
        except Exception:  # noqa: BLE001 - 不把原始输入交给 UI 错误文本
            return "Saved to the daily log; durable promotion was blocked."
        if not promoted.ok:
            return "Saved to the daily log; durable promotion was blocked."
        if include_global:
            return "Saved to the daily log and promoted to global memory."
        return "Saved to the daily log and promoted to durable memory."


def _split_global_flag(raw: str) -> tuple[bool, str]:
    """解析命令首尾的 ``--global``，保留正文中的同名自然语言 token。

    flag 只允许出现在命令开头或结尾；如果无条件删除正文中所有同名词，
    ``/remember note about --global behavior`` 会在写入前悄悄改变原文。
    """

    value = str(raw or "").strip()
    if not value:
        return False, ""
    parts = value.split()
    if parts and parts[0] == "--global":
        return True, " ".join(parts[1:]).strip()
    if parts and parts[-1] == "--global":
        return True, " ".join(parts[:-1]).strip()
    return False, value


def _split_promote_syntax(raw: str) -> tuple[str, str | None]:
    """解析显式提升写法，同时保留普通 note 的原始文本。

    支持 ``/remember text --promote topic`` 和 ``/remember --promote topic text``；
    另支持已知 topic 的 ``promote topic text`` 兼容别名。别名必须命中 topic
    白名单，避免普通句子 ``promote the ...`` 被误当成命令而丢失原文。
    """

    value = str(raw or "").strip()
    if not value:
        return "", None

    prefix = re.match(r"^(promote|--promote)\s+([^\s]+)\s+(.+)$", value, re.IGNORECASE)
    if prefix and (prefix.group(1).lower() == "--promote" or prefix.group(2) in _PROMOTABLE_TOPICS):
        return prefix.group(3).strip(), prefix.group(2).strip()

    if re.match(r"^--promote(?:\s|$)", value, re.IGNORECASE):
        # 显式标记但缺少合法的 topic/text 时返回特殊空 topic；handler 会在
        # 写入 daily log 前给出 usage，避免把用户笔误静默保存成普通 note。
        return "", _INVALID_PROMOTE

    suffix = re.match(r"^(.+?)\s+--promote(?:\s+([^\s]+))?$", value, re.IGNORECASE)
    if suffix:
        return suffix.group(1).strip(), (suffix.group(2) or _INVALID_PROMOTE).strip()
    return value, None


__all__ = ["MemoryCommandHandler"]
