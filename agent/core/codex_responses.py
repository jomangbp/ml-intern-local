"""ChatGPT Codex Responses API client used for Codex-subscription GPT models.

LiteLLM's OpenAI adapter talks to ``/chat/completions``. The current Codex
subscription backend for GPT-5.x uses the Responses API at
``https://chatgpt.com/backend-api/codex/responses`` and requires streaming.
This module implements the small subset ml-intern needs: text deltas,
function-call output items, and usage accounting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx


DeltaCallback = Callable[[str], Awaitable[None]] | Callable[[str], None]


@dataclass
class CodexResponsesResult:
    content: str | None = None
    tool_calls_acc: dict[int, dict] = field(default_factory=dict)
    token_count: int = 0
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


def is_codex_responses_params(params: dict[str, Any] | None) -> bool:
    return bool(params and params.get("_codex_responses"))


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _messages_to_codex(messages: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    inputs: list[dict[str, Any]] = []

    for msg in messages:
        role = str(_get_field(msg, "role", "user") or "user")
        content = _content_to_text(_get_field(msg, "content", ""))

        if role == "system":
            if content:
                instructions.append(content)
            continue

        if role == "tool":
            tool_call_id = _get_field(msg, "tool_call_id", "") or ""
            if tool_call_id:
                inputs.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": content,
                })
            elif content:
                inputs.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"Tool result:\n{content}"}],
                })
            continue

        if role == "assistant":
            if content:
                inputs.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            tool_calls = _get_field(msg, "tool_calls", None)
            for tc in tool_calls or []:
                fn = _get_field(tc, "function", {})
                name = _get_field(fn, "name", "")
                args = _get_field(fn, "arguments", "") or "{}"
                call_id = _get_field(tc, "id", "") or f"call_{len(inputs)}"
                if name:
                    inputs.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": args,
                    })
            continue

        if role not in {"user", "developer"}:
            role = "user"
        if not content:
            continue
        wire_role = "user" if role == "developer" else role
        inputs.append({
            "role": wire_role,
            "content": [{"type": "input_text", "text": content}],
        })

    if not inputs:
        inputs.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})
    return "\n\n".join(instructions), inputs


def _tools_to_codex(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        name = fn.get("name") or tool.get("name")
        if not name:
            continue
        item = {
            "type": "function",
            "name": name,
            "description": fn.get("description") or tool.get("description") or "",
            "parameters": fn.get("parameters") or tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if "strict" in fn:
            item["strict"] = fn["strict"]
        elif "strict" in tool:
            item["strict"] = tool["strict"]
        out.append(item)
    return out


def _usage_total(usage: dict[str, Any]) -> int:
    for key in ("total_tokens", "total_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    total = 0
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total += value
    return total


async def _maybe_call_delta(callback: DeltaCallback | None, text: str) -> None:
    if not callback or not text:
        return
    result = callback(text)
    if hasattr(result, "__await__"):
        await result  # type: ignore[misc]


async def codex_responses_completion(
    *,
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    params: dict[str, Any],
    stream: bool,
    max_output_tokens: int | None = None,
    on_delta: DeltaCallback | None = None,
    timeout: float = 600,
) -> CodexResponsesResult:
    """Call ChatGPT Codex Responses API and return ml-intern-style result."""
    instructions, input_items = _messages_to_codex(messages)
    codex_tools = _tools_to_codex(tools)

    api_key = str(params.get("api_key") or "")
    account_id = str(params.get("account_id") or "")
    url = str(params.get("api_base") or "https://chatgpt.com/backend-api/codex/responses")
    model = str(params.get("model") or "gpt-5.5")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "chatgpt-account-id": account_id,
        "originator": "ml-intern",
        "OpenAI-Beta": "responses=experimental",
        "version": "0.125.0",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "codex_cli_rs/0.125.0",
    }

    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "stream": True,  # Codex backend requires stream=true.
        "store": False,
        "instructions": instructions or "You are a helpful assistant.",
        "tool_choice": "auto",
    }
    if codex_tools:
        body["tools"] = codex_tools
    # Codex backend currently rejects public Responses API's
    # max_output_tokens field, so we intentionally do not forward it.
    effort = params.get("reasoning_effort")
    if effort:
        body["reasoning"] = {"effort": effort}

    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    raw_response: dict[str, Any] | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code >= 400:
                text = await response.aread()
                raise RuntimeError(
                    f"Codex Responses API error {response.status_code}: "
                    f"{text.decode('utf-8', errors='replace')[:2000]}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                et = event.get("type")
                if et == "response.output_text.delta":
                    delta = event.get("delta") or ""
                    if delta:
                        content_parts.append(delta)
                        if stream:
                            await _maybe_call_delta(on_delta, delta)
                elif et == "response.output_text.done":
                    text = event.get("text")
                    if isinstance(text, str) and not content_parts:
                        content_parts.append(text)
                elif et == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        idx = int(event.get("output_index") or len(tool_calls_acc))
                        tool_calls_acc[idx] = {
                            "id": item.get("call_id") or item.get("id") or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                elif et == "response.completed":
                    raw_response = event.get("response") or {}
                    usage = raw_response.get("usage") or {}
                    finish_reason = "stop"
                elif et in {"response.failed", "response.incomplete"}:
                    raw_response = event.get("response") or {}
                    error = raw_response.get("error") or event.get("error") or event
                    raise RuntimeError(f"Codex Responses API failed: {error}")

    content = "".join(content_parts) or None
    return CodexResponsesResult(
        content=content,
        tool_calls_acc=tool_calls_acc,
        token_count=_usage_total(usage),
        finish_reason=finish_reason,
        usage=usage,
        raw_response=raw_response,
    )
