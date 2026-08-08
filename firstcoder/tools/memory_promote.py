"""会话级 ``memory_promote`` 工具。"""

from __future__ import annotations

from firstcoder.memory.runtime import MemoryRuntime
from firstcoder.permissions.types import PermissionAction
from firstcoder.providers.types import ToolDefinition
from firstcoder.tools.types import (
    Tool,
    ToolPermissionSpec,
    ToolResult,
    make_error_result,
    make_text_result,
)
from firstcoder.utils.schema import object_schema


def create_memory_promote_tool(runtime: MemoryRuntime) -> Tool:
    """创建显式提升 durable topic 的 session-scoped 工具。"""

    def memory_promote(*, topic: str, text: str, visibility: str = "workspace") -> ToolResult:
        if not isinstance(topic, str) or not topic.strip():
            return make_error_result("memory_promote", "topic 不能为空")
        if not isinstance(text, str) or not text.strip():
            return make_error_result("memory_promote", "text 不能为空")
        if visibility not in {"session", "workspace", "global"}:
            return make_error_result("memory_promote", "visibility 必须是 session、workspace 或 global")
        try:
            receipt = runtime.promote(
                topic,
                text,
                source="tool:memory_promote",
                visibility=visibility,
            )
        except Exception:  # noqa: BLE001 - 工具结果不能泄漏原始 secret 或路径
            return make_error_result("memory_promote", "durable memory 提升失败")
        if not receipt.ok:
            return make_error_result(
                "memory_promote",
                receipt.error or "durable memory 提升失败",
                **receipt.public_data(),
            )
        return make_text_result(
            "memory_promote",
            "Memory promoted to durable memory.",
            **receipt.public_data(),
        )

    parameters = object_schema(
        {
            "topic": {
                "type": "string",
                "enum": ["project-conventions", "key-decisions", "dependency-facts", "user-preferences"],
            },
            "text": {"type": "string"},
            "visibility": {
                "type": "string",
                "enum": ["session", "workspace", "global"],
                "default": "workspace",
            },
        },
        required=["topic", "text"],
    )
    parameters["additionalProperties"] = False
    return Tool(
        definition=ToolDefinition(
            name="memory_promote",
            description="Promote one safe note into the current project's durable memory topics.",
            parameters=parameters,
        ),
        executor=memory_promote,
        permission=ToolPermissionSpec(
            action=PermissionAction.WRITE_PATH,
            target_builder=lambda arguments: runtime.permission_target(
                visibility=arguments.get("visibility", "workspace"),
            ),
            reason="提升 memory note 需要确认。",
            allow_auto=False,
        ),
    )
