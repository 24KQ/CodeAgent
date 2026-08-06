"""Durable promotion heuristics (fusion P2, M5).

Ported verbatim from pico `features/memory.py:38-50, 562-613`. Pure static
rules: intent patterns gate promotion, line patterns pick note text,
rejection reasons classify bad candidates. Zero runtime dependencies.

判定链路：用户消息带记忆意图（英文或中文关键词）→ 逐行匹配主题模式
（Decision:/决策：等）→ 拒绝原因分类（secret/瞬态/噪声）→ 通过的
(topic, note_text) 交给 durable.promote 落盘。
"""

from __future__ import annotations

import re

#: 记忆意图触发词：英文 + 中文（任一命中才开始提取）。
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
#: 列表前缀（"- " / "* " / "1. "），提取正文前剥掉。
DURABLE_MEMORY_LIST_PREFIX_PATTERN = re.compile(r"^(?:[-*]|\d+[.)])\s+")
#: 主题行模式：英文六条 + 中文四条，顺序即匹配优先级（同一行只会命中第一条）。
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)

SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")


def reject_durable_reason(note_text: str, redacted_value: str = "<redacted>") -> str:
    """分类拒绝原因：空 / secret_shaped / transient_task_state / noisy_output。"""
    text = str(note_text or "").strip()
    lowered = text.lower()
    if not text:
        return "empty"
    if redacted_value in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
        return "secret_shaped"
    checkpoint_like_prefixes = (
        "current goal",
        "current blocker",
        "next step",
        "current phase",
        "key files",
        "freshness",
        "当前目标",
        "当前卡点",
        "下一步",
        "当前阶段",
        "关键文件",
        "已完成",
        "已排除",
    )
    if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
        return "transient_task_state"
    if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
        return "noisy_output"
    return ""


def extract_durable_promotions(
    user_message: str,
    final_answer: str,
    redacted_value: str = "<redacted>",
) -> tuple[list[tuple[str, str]], list[str]]:
    """从一轮对话里提取 durable 提升候选与拒绝原因。

    只有 user 消息带记忆意图时才启动；对最终回答逐行匹配
    DURABLE_MEMORY_LINE_PATTERNS，返回 (promotions, rejections)。
    """
    user_text = str(user_message or "")
    if not (
        DURABLE_MEMORY_INTENT_PATTERN.search(user_text)
        or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)
    ):
        return [], []
    promotions = []
    rejections = []
    for line in str(final_answer or "").splitlines():
        text = DURABLE_MEMORY_LIST_PREFIX_PATTERN.sub("", line.strip(), count=1)
        if not text or redacted_value in text:
            continue
        for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue
            note_text = match.group(1).strip()
            if note_text:
                reason = reject_durable_reason(note_text, redacted_value=redacted_value)
                if reason:
                    rejections.append(f"{topic}:{reason}")
                    break
                promotions.append((topic, note_text))
            break
    return promotions, rejections
