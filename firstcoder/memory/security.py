"""Static memory security rules (fusion P0 slice 3).

Ported from pico `features/memory_lint.py`, `features/memory_quarantine.py`
and `core/runtime_secrets.py` as pure static rules (M7/M8): secret patterns,
quarantine injection signatures, env-value + regex redaction. Zero runtime
dependencies. Provides `StaticSecurityPolicy`, a concrete implementation of
the `MemorySecurityPolicy` port shape (see memory/ports.py).

Existing behavior must remain unchanged until a real policy is wired in
(P2 memory data plane). See docs/fusion-plan-review.md §1.2 M7/M8, §7.4.
"""

from __future__ import annotations

import os
import re

from firstcoder.memory.models import MemoryNote

_KEYWORD = r"(?:key|token|secret|password|api)"
_LONG_HEX = r"[A-Fa-f0-9]{32,}"
_LONG_BASE64 = r"[A-Za-z0-9+/]{40,}={0,2}"

#: Five secret shapes (pico memory_lint.py:13-23, verbatim) plus the
#: `sk-proj-...` class (Codex P2 review #10: the P0 comment noted the gap
#: for the data-plane wiring, which is now live — `sk-proj-` OpenAI project
#: keys must be caught before `promote` writes them).
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(
        rf"(?i)(?:{_KEYWORD}.{{0,20}}(?:{_LONG_HEX}|{_LONG_BASE64})|(?:{_LONG_HEX}|{_LONG_BASE64}).{{0,20}}{_KEYWORD})"
    ),
]

# 最后一条模式只在文本同时出现 secret 关键词时才有可能命中。先做这个
# cheap pre-check，避免在超长普通字符串上让“关键词 + 长 base64”组合模式
# 反复回溯；这不改变命中语义，只把无关输入快速排除。
_SECRET_CONTEXT_PATTERN = re.compile(r"(?i)(?:key|token|secret|password|api)")

#: Relative-date phrases that make a note decay (pico memory_lint.py:25).
RELATIVE_DATE_PATTERN = re.compile(
    r"(?i)\b(tomorrow|yesterday|next week|last week)\b|今天|明天|昨天|下周|上周"
)

#: Prompt-injection signatures for the quarantine gate (pico memory_quarantine.py:7-12).
QUARANTINE_PATTERN = re.compile(
    r"ignore (?:previous|prior) instructions|</?(?:system|assistant)>|disregard all earlier|new instructions:|you are now",
    re.I,
)

REDACTED_VALUE = "<redacted>"

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def should_quarantine(text: str) -> bool:
    """True when text carries a quarantine signature or looks secret-shaped."""
    text = str(text)
    if QUARANTINE_PATTERN.search(text):
        return True
    if any(pattern.search(text) for pattern in SECRET_PATTERNS[:-1]):
        return True
    return bool(_SECRET_CONTEXT_PATTERN.search(text) and SECRET_PATTERNS[-1].search(text))


def looks_sensitive_env_name(name: str) -> bool:
    """Heuristic: env var names that typically carry secrets."""
    upper = str(name).upper()
    return any(
        upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}")
        for marker in SENSITIVE_ENV_NAME_MARKERS
    )


class StaticSecurityPolicy:
    """Concrete `MemorySecurityPolicy`: env-value + regex redaction, quarantine gate.

    Env-value replacement runs first (longest value first, so a value that is
    a substring of another cannot survive), then the static `SECRET_PATTERNS`.
    This is the first of the three-layer redaction design (env + regex + gate,
    M8); the gate is `passes_quarantine`.
    """

    def __init__(self, secret_env_names: set[str] | None = None) -> None:
        self.secret_env_names = set(secret_env_names or ())

    def detected_secret_env_items(self) -> list[tuple[str, str]]:
        items = [
            (name, value)
            for name, value in os.environ.items()
            if value and (str(name).upper() in self.secret_env_names or looks_sensitive_env_name(name))
        ]
        items.sort(key=lambda item: item[0])
        return items

    def redact(self, text: str) -> str:
        text = str(text)
        for _, value in sorted(self.detected_secret_env_items(), key=lambda item: len(item[1]), reverse=True):
            text = text.replace(value, REDACTED_VALUE)
        for pattern in SECRET_PATTERNS[:-1]:
            text = pattern.sub(REDACTED_VALUE, text)
        if _SECRET_CONTEXT_PATTERN.search(text):
            text = SECRET_PATTERNS[-1].sub(REDACTED_VALUE, text)
        return text

    def redact_artifact(self, value: object, key: str | None = None) -> object:
        """Recursive artifact redaction (M8 layer one: key-name + env values)."""
        if key and looks_sensitive_env_name(key):
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self.redact_artifact(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, str):
            return self.redact(value)
        return value

    def passes_quarantine(self, note: MemoryNote) -> bool:
        return not should_quarantine(note.text)
