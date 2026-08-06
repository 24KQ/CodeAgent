"""Provider usage calibration for FirstCoder context pressure reporting.

FirstCoder 的 ContextBudget 继续负责固定/历史消息估算和 L1-L4 压缩；本模块只
回答一个更窄的问题：上一轮 provider 返回的实际 input token 是否能安全地校准
当前 prompt。只有 provider、base URL、model、窗口、cache key、prompt hash 六项
identity 全部匹配时才采用 actual 值，否则保留估算并明确记录原因。

算法来源：pico ``core/context_pressure.py``；实现不持有 AgentLoop，也不触发
任何第二套 compaction。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IDENTITY_KEYS = (
    "provider",
    "provider_base_url",
    "model",
    "context_window",
    "prompt_cache_key",
    "prompt_hash",
)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ContextPressure:
    """一次 prompt 的估算/实测 token 和压力分档。"""

    input_tokens: int
    context_window: int
    budget_tokens: int
    actual_input_tokens: int | None = None
    last_actual_input_tokens: int | None = None
    usage_source: str = "estimated"
    calibration_source: str = "missing_last_completion_metadata"
    cached_tokens: int | None = None

    @property
    def pressure_ratio(self) -> float:
        budget = max(1, int(self.budget_tokens or 0))
        return round(max(0, int(self.input_tokens or 0)) / budget, 6)

    @property
    def window_ratio(self) -> float:
        window = max(1, int(self.context_window or 0))
        return round(max(0, int(self.input_tokens or 0)) / window, 6)

    @property
    def pressure_tier(self) -> str:
        ratio = self.pressure_ratio
        if ratio >= 0.95:
            return "tier3_summary"
        if ratio >= 0.8:
            return "tier2_prune"
        if ratio >= 0.6:
            return "tier1_snip"
        return "tier0_observe"

    def to_context_usage_fields(self) -> dict[str, Any]:
        """返回可直接放进 prompt_metadata 的稳定字段。"""

        return {
            "pressure_ratio": self.pressure_ratio,
            "window_ratio": self.window_ratio,
            "pressure_tier": self.pressure_tier,
            "budget_tokens": self.budget_tokens,
            "usage_source": self.usage_source,
            "actual_input_tokens": self.actual_input_tokens,
            "last_actual_input_tokens": self.last_actual_input_tokens,
            "calibration_source": self.calibration_source,
            "cached_tokens": self.cached_tokens,
        }


class ContextPressureController:
    """将 prompt 估算值与上一轮 provider usage 做 identity-safe 校准。"""

    def evaluate(
        self,
        *,
        estimated_input_tokens: int,
        context_window: int,
        budget_tokens: int | None = None,
        current_identity: dict[str, Any] | None,
        last_completion_metadata: dict[str, Any] | None = None,
        last_identity: dict[str, Any] | None = None,
    ) -> ContextPressure:
        estimated = max(0, int(estimated_input_tokens or 0))
        window = max(1, int(context_window or 0))
        budget = int(budget_tokens or 0) or window
        metadata = _flatten_completion_metadata(last_completion_metadata)
        last_actual = _optional_int(metadata.get("input_tokens"))
        cached = None
        calibration_source = "missing_last_completion_metadata"
        usage_source = "estimated"
        actual = None
        input_tokens = estimated

        if metadata and last_actual is None:
            calibration_source = "missing_actual_input_tokens"
        elif last_actual is not None:
            if self._identity_matches(current_identity, metadata, last_identity):
                input_tokens = last_actual
                actual = last_actual
                cached = _optional_int(
                    metadata.get("cached_tokens", metadata.get("cached_input_tokens"))
                )
                usage_source = "actual"
                calibration_source = "current_identity_match"
            else:
                calibration_source = "last_completion_identity_mismatch"

        return ContextPressure(
            input_tokens=input_tokens,
            context_window=window,
            budget_tokens=budget,
            actual_input_tokens=actual,
            last_actual_input_tokens=last_actual,
            usage_source=usage_source,
            calibration_source=calibration_source,
            cached_tokens=cached,
        )

    def _identity_matches(
        self,
        current_identity: dict[str, Any] | None,
        metadata: dict[str, Any],
        last_identity: dict[str, Any] | None,
    ) -> bool:
        current = dict(current_identity or {})
        previous = dict(last_identity or {})
        previous.update({key: metadata[key] for key in IDENTITY_KEYS if key in metadata})
        return all(
            key in current and key in previous and current.get(key) == previous.get(key)
            for key in IDENTITY_KEYS
        )


def _flatten_completion_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """兼容扁平 completion metadata 与 ProviderCallMetadata.to_dict()。"""

    flattened = dict(metadata or {})
    usage = flattened.get("usage")
    if isinstance(usage, dict):
        for key, value in usage.items():
            flattened.setdefault(key, value)
    if "cached_tokens" not in flattened and "cached_input_tokens" in flattened:
        flattened["cached_tokens"] = flattened["cached_input_tokens"]
    return flattened
