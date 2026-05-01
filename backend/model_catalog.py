"""Shared model catalog for Web UI and Telegram bot.

Static + dynamic models:
  - ``AVAILABLE_MODELS`` — always-present cloud models.
  - ``get_all_models()`` — static models plus Ollama models detected at runtime.
  - ``scan_ollama_models()`` — force a fresh scan of the local Ollama instance.
"""

from __future__ import annotations

import time
from typing import Any

_HAS_HTTPX: bool = True
try:
    import httpx  # noqa: F401
except ImportError:
    _HAS_HTTPX = False

# ---------------------------------------------------------------------------
# Static models (always present)
# ---------------------------------------------------------------------------
AVAILABLE_MODELS: list[dict[str, Any]] = [
    {
        "id": "anthropic/claude-opus-4-6",
        "label": "Claude Opus 4.6",
        "provider": "anthropic",
        "recommended": True,
    },
    {
        "id": "MiniMaxAI/MiniMax-M2.7",
        "label": "MiniMax M2.7",
        "provider": "minimax",
        "recommended": True,
    },
    {
        "id": "openai/gpt-5.3-codex",
        "label": "GPT-5.3 Codex",
        "provider": "openai",
    },
    {
        "id": "openai/gpt-5.4",
        "label": "GPT-5.4",
        "provider": "openai",
    },
    {
        "id": "openai/gpt-5.5",
        "label": "GPT-5.5",
        "provider": "openai",
    },
    {
        "id": "moonshotai/Kimi-K2.6",
        "label": "Kimi K2.6",
        "provider": "huggingface",
    },
    {
        "id": "zai-org/GLM-5.1",
        "label": "GLM 5.1",
        "provider": "zai",
    },
    {
        "id": "xiaomi/MiMo",
        "label": "Xiaomi MiMo",
        "provider": "xiaomi",
    },
]

# ---------------------------------------------------------------------------
# Dynamic Ollama model scanning
# ---------------------------------------------------------------------------
# Cache so we don't hit Ollama on every UI model-list fetch.
_OLLAMA_CACHE: list[dict[str, Any]] | None = None
_OLLAMA_CACHE_AT: float = 0.0
_OLLAMA_CACHE_TTL: float = 30.0  # seconds


def _ollama_default_base() -> str:
    """Return the default Ollama API base URL."""
    import os
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _detect_ollama_models(*, force: bool = False) -> list[dict[str, Any]]:
    """Query ``/api/tags`` on the local Ollama instance.

    Returns a list of model dicts with keys ``id``, ``label``, ``provider``,
    plus ``local`` / ``cloud`` tagging. Results are cached for
    ``_OLLAMA_CACHE_TTL`` seconds.

    On failure (Ollama not running, connection refused, etc.) returns an
    empty list so the catalog degrades gracefully.
    """
    global _OLLAMA_CACHE, _OLLAMA_CACHE_AT

    now = time.monotonic()
    if not force and _OLLAMA_CACHE is not None and (now - _OLLAMA_CACHE_AT) < _OLLAMA_CACHE_TTL:
        return _OLLAMA_CACHE

    if not _HAS_HTTPX:
        _OLLAMA_CACHE = []
        _OLLAMA_CACHE_AT = now
        return _OLLAMA_CACHE

    base = _ollama_default_base()
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        _OLLAMA_CACHE = []
        _OLLAMA_CACHE_AT = now
        return _OLLAMA_CACHE

    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for entry in data.get("models") or []:
        raw_name: str = (entry.get("name") or "").strip()
        if not raw_name:
            continue

        # Parse name:tag
        tag = "latest"
        display_name = raw_name
        if ":" in raw_name:
            display_name, tag = raw_name.rsplit(":", 1)

        # Detect cloud vs local
        is_cloud = bool(
            tag == "cloud"
            or entry.get("remote_host")
            or entry.get("remote_model")
        )
        size_bytes = entry.get("size", 0)
        size_gb = size_bytes / (1024**3) if isinstance(size_bytes, (int, float)) else 0

        # Family / parameter info from details
        details: dict = entry.get("details") or {}
        family: str = details.get("family") or ""
        param_size: str = details.get("parameter_size") or ""
        quant: str = details.get("quantization_level") or ""

        model_id = f"ollama/{raw_name}"
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        # Build a descriptive label
        if is_cloud:
            prefix = "☁️ "
        elif size_gb > 1:
            prefix = "🖥️ "
        else:
            prefix = "🖥️ "

        label_parts = [f"{prefix}{display_name}"]
        if param_size:
            label_parts.append(param_size)
        if quant and quant not in ("unknown", ""):
            label_parts.append(quant)
        label = f" ({tag})".join(label_parts) if tag != "latest" else " ".join(label_parts)

        models.append({
            "id": model_id,
            "label": label,
            "provider": "ollama",
            "ollama_local": not is_cloud,
            "ollama_cloud": is_cloud,
            "ollama_size_gb": round(size_gb, 2) if size_gb > 0 else None,
        })

    _OLLAMA_CACHE = models
    _OLLAMA_CACHE_AT = now
    return models


def scan_ollama(**kwargs: Any) -> list[dict[str, Any]]:
    """Force a fresh Ollama scan and return models.  Keyword args are
    forwarded to ``_detect_ollama_models`` — for now only ``force=True``
    is meaningful and is always implied."""
    return _detect_ollama_models(force=True)


# ---------------------------------------------------------------------------
# Combined catalog
# ---------------------------------------------------------------------------
def get_all_models(*, force_ollama_scan: bool = False) -> list[dict[str, Any]]:
    """Return the full model catalog: static models + dynamically detected
    Ollama models (when available).

    Pass ``force_ollama_scan=True`` to force a fresh scan (used by the
    /api/providers/ollama/scan endpoint).
    """
    result = list(AVAILABLE_MODELS)
    ollama_models = _detect_ollama_models(force=force_ollama_scan)
    result.extend(ollama_models)
    return result


def model_ids() -> set[str]:
    return {m["id"] for m in get_all_models()}


def format_models_for_text() -> str:
    all_models = get_all_models()
    return "\n".join(
        f"{idx}. {m['label']} — {m['id']}" for idx, m in enumerate(all_models, start=1)
    )


def resolve_model_choice(choice: str) -> str | None:
    """Resolve a Telegram/UI model choice by id, numeric index, or label."""
    raw = (choice or "").strip()
    if not raw:
        return None
    all_models = get_all_models()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(all_models):
            return all_models[idx]["id"]
    lowered = raw.lower()
    for model in all_models:
        if lowered in {model["id"].lower(), model["label"].lower()}:
            return model["id"]
    # Convenience: allow bare slugs like gpt-5.5, ollama/llama3.2.
    for model in all_models:
        if model["id"].lower().endswith("/" + lowered):
            return model["id"]
    # Also match Ollama models by display name without prefix/emoji
    for model in all_models:
        if model.get("provider") == "ollama":
            label_no_emoji = model["label"].replace("☁️ ", "").replace("🖥️ ", "")
            if lowered == label_no_emoji.lower():
                return model["id"]
    return None
