"""会话级 ``memory_note`` 工具。

工具只接受记忆文本和受控 topic，不接受任意文件路径；真正的脱敏、quarantine、
daily log 写入和 audit 由绑定 workspace 的 ``MemoryRuntime`` 统一完成。
"""

from __future__ import annotations

from firstcoder.memory.runtime import MemoryRuntime
from firstcoder.providers.types import ToolDefinition
from firstcoder.tools.types import Tool, ToolResult, make_error_result, make_text_result
from firstcoder.utils.schema import object_schema


def create_memory_note_tool(runtime: MemoryRuntime) -> Tool:
    """创建写 daily log 的 session-scoped memory 工具。"""

    def memory_note(*, text: str, promote: bool = False, topic: str = "key-decisions") -> ToolResult:
        if not isinstance(text, str) or not text.strip():
            return make_error_result("memory_note", "text 不能为空")
        if not isinstance(promote, bool):
            return make_error_result("memory_note", "promote 必须是布尔值")

        try:
            captured = runtime.record(text, source="tool:memory_note")
        except Exception:  # noqa: BLE001 - 工具边界必须返回无敏感参数的安全错误
            return make_error_result("memory_note", "记忆写入失败")
        if not captured.ok:
            return make_error_result("memory_note", captured.error or "记忆写入失败", **captured.public_data())

        data = captured.public_data()
        if not promote:
            return make_text_result("memory_note", "Memory note saved to the daily log.", **data)

        try:
            promoted = runtime.promote(topic, text, source="tool:memory_note")
        except Exception:  # noqa: BLE001 - 不让通用 registry 把原始参数放进 error data
            return make_text_result(
                "memory_note",
                "Memory note saved to the daily log; durable promotion was blocked.",
                **data,
                promote_ok=False,
            )
        data.update({"promote_ok": promoted.ok, "promote_error": promoted.error})
        if not promoted.ok:
            return make_text_result(
                "memory_note",
                "Memory note saved to the daily log; durable promotion was blocked.",
                **data,
            )
        data.update(promoted.public_data())
        return make_text_result(
            "memory_note",
            "Memory note saved and promoted to durable memory.",
            **data,
        )

    parameters = object_schema(
        {
            "text": {"type": "string"},
            "promote": {"type": "boolean", "default": False},
            "topic": {
                "type": "string",
                "enum": ["project-conventions", "key-decisions", "dependency-facts", "user-preferences"],
                "default": "key-decisions",
            },
        },
        required=["text"],
    )
    parameters["additionalProperties"] = False
    return Tool(
        definition=ToolDefinition(
            name="memory_note",
            description="Capture a redacted note in the current project's daily memory log.",
            parameters=parameters,
        ),
        executor=memory_note,
    )
