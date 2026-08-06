"""Provider-call metadata contract (fusion P1, §7.3 H5 prerequisite).

Codex finding (docs/fusion-plan-review.md §7.3): `TokenUsage` only carries
input/output/total — no cached tokens, no unified request trace, no
prompt-estimation persistence, no protocol/base URL, no request<->response
pairing. Before the H5 context-cost pairing experiment can exist, every
provider call must record this metadata.

This module is the contract. The wiring at the loop call site lands with
the stop_reason mapping slice (P1 切片 4); until then nothing calls it and
existing behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from firstcoder.providers.types import TokenUsage


@dataclass(slots=True)
class UsageSnapshot:
    """Token usage with cache accounting, flattened for report/trace JSON."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None

    @classmethod
    def from_usage(cls, usage: TokenUsage | None) -> "UsageSnapshot":
        if usage is None:
            return cls()
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ProviderCallMetadata:
    """One request<->response pair for one provider call (H5 input row)."""

    call_id: str
    session_id: str
    turn_id: str
    provider: str
    model: str
    protocol: str = ""
    base_url: str = ""
    request_at: str = ""
    response_at: str = ""
    finish_reason: str = ""
    usage: UsageSnapshot = field(default_factory=UsageSnapshot)
    prompt_estimated_tokens: int = 0
    prompt_estimation_source: str = ""
    error: str = ""
    retry_of: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["usage"] = self.usage.to_dict()
        return payload
