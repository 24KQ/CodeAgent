"""P1 slice 3 tests: provider-call metadata contract + cached-token parsing."""

from __future__ import annotations

from firstcoder.harness.provider_call import ProviderCallMetadata, UsageSnapshot
from firstcoder.providers.openai_compatible import _parse_usage as parse_openai
from firstcoder.providers.anthropic_provider import _parse_usage as parse_anthropic
from firstcoder.providers.streaming import merge_usage, token_usage
from firstcoder.providers.types import TokenUsage


# --- cached-token parsing ---------------------------------------------------


def test_openai_usage_parses_cached_from_details() -> None:
    usage = parse_openai(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
        }
    )
    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 80


def test_openai_usage_without_details_keeps_none() -> None:
    usage = parse_openai({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert usage.cached_input_tokens is None


def test_anthropic_usage_parses_cache_read() -> None:
    usage = parse_anthropic(
        {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 90}
    )
    assert usage is not None
    assert usage.cached_input_tokens == 90


def test_token_usage_and_merge_keep_cached() -> None:
    left = token_usage(100, 20, 120, 90)
    right = token_usage(50, 10, 60)
    merged = merge_usage(left, right)
    assert merged.cached_input_tokens == 90
    assert merged.input_tokens == 50
    # 全空 -> None（原语义保留）
    assert token_usage(None, None, None, None) is None
    # 只有 cached 也算有值
    assert token_usage(None, None, None, 5) is not None


# --- ProviderCallMetadata contract ------------------------------------------


def test_usage_snapshot_round_trip() -> None:
    snapshot = UsageSnapshot.from_usage(
        TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, cached_input_tokens=80)
    )
    assert snapshot.to_dict() == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cached_input_tokens": 80,
    }


def test_usage_snapshot_from_none() -> None:
    assert UsageSnapshot.from_usage(None).to_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
    }


def test_provider_call_metadata_to_dict() -> None:
    call = ProviderCallMetadata(
        call_id="call_1",
        session_id="s1",
        turn_id="t1",
        provider="deepseek",
        model="deepseek-chat",
        protocol="openai-compatible",
        base_url="https://api.deepseek.com/v1",
        request_at="2026-08-06T12:00:00Z",
        finish_reason="stop",
        usage=UsageSnapshot(input_tokens=10, output_tokens=5, total_tokens=15),
        prompt_estimated_tokens=9,
        prompt_estimation_source="estimator",
    )
    data = call.to_dict()
    assert data["call_id"] == "call_1"
    assert data["protocol"] == "openai-compatible"
    assert data["base_url"] == "https://api.deepseek.com/v1"
    assert data["usage"]["cached_input_tokens"] is None
    assert data["prompt_estimation_source"] == "estimator"
