"""Tests for LLM error classification helpers in agent.core.agent_loop.

Covers two regressions on 2026-04-25:

1. Non-Anthropic context overflow (Kimi 365k > 262k) was not classified as
   ``_is_context_overflow_error``, so the recovery path didn't fire and
   session 62ccfdcb died with 68 wasted compaction events.

2. Bedrock TPM rate limit (`Too many tokens, please wait before trying
   again.`) needs the longer rate-limit retry schedule. The old schedule
   ([5, 15, 30] = 50s) burned through 6 sessions costing >$2,400 combined
   on the same day.
"""

from agent.core.agent_loop import (
    _is_cloud_overloaded,
    _is_context_overflow_error,
    _is_rate_limit_error,
    _is_transient_error,
    _persistent_retry_delay,
)


# ── context overflow ────────────────────────────────────────────────────


def test_kimi_prompt_too_long_is_context_overflow():
    # Verbatim error text from session 62ccfdcb (2026-04-25, Kimi K2.6).
    err = Exception(
        "litellm.BadRequestError: OpenAIException - The prompt is too long: "
        "365407, model maximum context length: 262143"
    )
    assert _is_context_overflow_error(err)


def test_openai_context_length_exceeded_is_context_overflow():
    err = Exception("Error: This model's maximum context length is 8192 tokens.")
    assert _is_context_overflow_error(err)


def test_random_error_is_not_context_overflow():
    err = Exception("connection reset by peer")
    assert not _is_context_overflow_error(err)


# ── rate limit ──────────────────────────────────────────────────────────


def test_bedrock_too_many_tokens_is_rate_limit():
    # Verbatim from sessions b37a3823, c4d7a831, b63c4933 (2026-04-25).
    err = Exception(
        'litellm.RateLimitError: BedrockException - {"message":"Too many '
        'tokens, please wait before trying again."}'
    )
    assert _is_rate_limit_error(err)
    # Rate-limit errors are also classified as transient.
    assert _is_transient_error(err)


def test_429_is_rate_limit():
    err = Exception("HTTP 429 Too Many Requests")
    assert _is_rate_limit_error(err)


def test_timeout_is_transient_but_not_rate_limit():
    err = Exception("Request timed out after 600s")
    assert _is_transient_error(err)
    assert not _is_rate_limit_error(err)


# ── retry delay (persistent backoff) ───────────────────────────────────


def test_rate_limit_uses_longer_base_delay():
    err = Exception("Too many tokens, please wait before trying again.")
    assert _is_rate_limit_error(err)
    # Rate-limited: base 30s → attempt 0 = 30s
    assert _persistent_retry_delay(err, 0) == 30


def test_cloud_overloaded_uses_medium_base_delay():
    err = Exception("Server overloaded, try again later")
    assert _is_cloud_overloaded(err)
    assert _is_transient_error(err)
    # Cloud overload: base 15s → attempt 0 = 15s
    assert _persistent_retry_delay(err, 0) == 15


def test_other_transient_uses_short_base_delay():
    err = Exception("503 service unavailable")
    assert _is_transient_error(err)
    assert not _is_rate_limit_error(err)
    assert not _is_cloud_overloaded(err)
    # Transient: base 5s → attempt 0 = 5s
    assert _persistent_retry_delay(err, 0) == 5


def test_persistent_retry_grows_exponentially():
    err = Exception("503 service unavailable")
    # 5 * 2^0 = 5, 5 * 2^1 = 10, 5 * 2^2 = 20, 5 * 2^3 = 40
    assert _persistent_retry_delay(err, 0) == 5
    assert _persistent_retry_delay(err, 1) == 10
    assert _persistent_retry_delay(err, 2) == 20
    assert _persistent_retry_delay(err, 3) == 40


def test_persistent_retry_caps_at_300_seconds():
    err = Exception("503 service unavailable")
    # 5 * 2^6 = 320 → capped at 300
    assert _persistent_retry_delay(err, 6) == 300
    # 5 * 2^10 = 5120 → capped at 300
    assert _persistent_retry_delay(err, 10) == 300


def test_non_transient_not_detected():
    err = Exception("invalid request: bad parameter")
    assert not _is_transient_error(err)
    assert not _is_rate_limit_error(err)
    assert not _is_cloud_overloaded(err)
