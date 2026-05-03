"""
Context overflow detection for LLM responses.

Detects both explicit overflow errors (provider returns an error message)
and silent overflow (provider accepts the request but input exceeds context).

Based on pi's overflow detection system.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Regex patterns to detect context overflow errors from different providers
OVERFLOW_PATTERNS = [
    re.compile(r"prompt is too long", re.IGNORECASE),  # Anthropic
    re.compile(r"request_too_large", re.IGNORECASE),  # Anthropic (HTTP 413)
    re.compile(r"input is too long for requested model", re.IGNORECASE),  # Bedrock
    re.compile(r"exceeds the context window", re.IGNORECASE),  # OpenAI
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),  # Google Gemini
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),  # xAI Grok
    re.compile(r"reduce the length of the messages", re.IGNORECASE),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),  # OpenRouter
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),  # GitHub Copilot
    re.compile(r"exceeds the available context size", re.IGNORECASE),  # llama.cpp
    re.compile(r"greater than the context length", re.IGNORECASE),  # LM Studio
    re.compile(r"context window exceeds limit", re.IGNORECASE),  # MiniMax
    re.compile(r"exceeded model token limit", re.IGNORECASE),  # Kimi
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),  # Mistral
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),  # Ollama
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),  # Generic
    re.compile(r"too many tokens", re.IGNORECASE),  # Generic
    re.compile(r"token limit exceeded", re.IGNORECASE),  # Generic
]

# Patterns that indicate non-overflow errors (e.g. rate limiting)
NON_OVERFLOW_PATTERNS = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


def is_context_overflow(error_message: str | None) -> bool:
    """Check if an error message indicates context overflow."""
    if not error_message:
        return False

    # Skip non-overflow errors
    for pattern in NON_OVERFLOW_PATTERNS:
        if pattern.search(error_message):
            return False

    # Check overflow patterns
    for pattern in OVERFLOW_PATTERNS:
        if pattern.search(error_message):
            return True

    return False


def is_silent_overflow(
    input_tokens: int | None,
    context_window: int | None,
) -> bool:
    """Detect silent overflow where the model accepts the request but
    input exceeds the context window.

    This handles providers like z.ai that accept overflow silently.
    """
    if input_tokens is None or context_window is None:
        return False
    return input_tokens > context_window


def estimate_tokens_from_content(content: str | None) -> int:
    """Rough token estimate from text content.

    Uses ~4 characters per token as a conservative estimate.
    """
    if not content:
        return 0
    return len(content) // 4
