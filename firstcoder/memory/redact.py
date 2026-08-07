"""记忆层的安全策略适配。

P0/P2 已经提供了 :class:`StaticSecurityPolicy`，其中包含环境变量值替换、
静态 secret 正则和 quarantine 判定。本模块不复制这些规则，而是提供 memory
runtime 使用的窄适配层，明确约束所有进入记忆文件、提示词、工具结果和审计
事件的字符串都必须先经过同一套规则。
"""

from __future__ import annotations

from typing import Any

from firstcoder.memory.security import REDACTED_VALUE, StaticSecurityPolicy


class MemoryRedactor(StaticSecurityPolicy):
    """面向 P3 runtime 的三层记忆安全策略。

    ``StaticSecurityPolicy`` 是数据面级别的纯规则实现；这个子类只补充
    runtime 侧更容易读懂的别名和输入规范化。这样 memory 工具、命令、
    prompt projector 和 audit writer 可以共享一个对象，而不会因为调用方
    忘记使用 ``redact`` 导致 secret 从旁路输出泄漏。
    """

    def __init__(self, secret_env_names: set[str] | None = None) -> None:
        # 环境变量名比较在底层策略中按大写进行，因此这里先规范化显式配置，
        # 避免调用方传入小写名称时出现“配置了但没有脱敏”的不一致行为。
        super().__init__({str(name).upper() for name in (secret_env_names or set())})

    def redact_text(self, text: object) -> str:
        """脱敏普通文本；保留 ``redact`` 作为 MemorySecurityPolicy 契约入口。"""

        return self.redact(str(text))

    def redact_value(self, value: Any, *, key: str | None = None) -> Any:
        """递归脱敏工具结果或审计 artifact。

        ``redact_artifact`` 已经实现了 key-name、环境变量值和静态正则三层
        组合；这里提供一个语义更明确的 runtime 别名，调用方不需要直接依赖
        security.py 的具体类名。
        """

        return self.redact_artifact(value, key=key)


# 允许审查和调用方使用更具领域含义的名字，同时保持唯一实现来源。
RedactingMemorySecurityPolicy = MemoryRedactor


def create_memory_redactor(secret_env_names: set[str] | None = None) -> MemoryRedactor:
    """创建默认的记忆脱敏策略。"""

    return MemoryRedactor(secret_env_names=secret_env_names)


__all__ = [
    "REDACTED_VALUE",
    "MemoryRedactor",
    "RedactingMemorySecurityPolicy",
    "create_memory_redactor",
]
