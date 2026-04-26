"""Shared model catalog for Web UI and Telegram bot."""

AVAILABLE_MODELS = [
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
        "id": "openai/gpt-5.3",
        "label": "GPT-5.3",
        "provider": "openai",
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
        "id": "openai/gpt-5.4-codex",
        "label": "GPT-5.4 Codex",
        "provider": "openai",
    },
    {
        "id": "openai/gpt-5.5",
        "label": "GPT-5.5",
        "provider": "openai",
    },
    {
        "id": "openai/gpt-5.5-codex",
        "label": "GPT-5.5 Codex",
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
]


def model_ids() -> set[str]:
    return {m["id"] for m in AVAILABLE_MODELS}


def format_models_for_text() -> str:
    return "\n".join(
        f"{idx}. {m['label']} — {m['id']}" for idx, m in enumerate(AVAILABLE_MODELS, start=1)
    )


def resolve_model_choice(choice: str) -> str | None:
    """Resolve a Telegram/UI model choice by id, numeric index, or label."""
    raw = (choice or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(AVAILABLE_MODELS):
            return AVAILABLE_MODELS[idx]["id"]
    lowered = raw.lower()
    for model in AVAILABLE_MODELS:
        if lowered in {model["id"].lower(), model["label"].lower()}:
            return model["id"]
    # Convenience: allow bare GPT slugs like gpt-5.5-codex.
    for model in AVAILABLE_MODELS:
        if model["id"].lower().endswith("/" + lowered):
            return model["id"]
    return None
