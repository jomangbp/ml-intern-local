"""Tests for Ollama & Xiaomi MiMo provider integration.

Covers:
  - Dynamic Ollama model scanning (with mocked HTTPX)
  - Xiaomi MiMo routing via llm_params
  - Ollama model routing via llm_params
  - resolve_model_choice for both new providers
  - Model guidance for both new providers
  - Provider key env var handling
"""

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Ollama dynamic model scanning
# ---------------------------------------------------------------------------

def _ollama_tags_response() -> dict:
    """Simulate a typical Ollama /api/tags response mixing local GGUF and cloud models."""
    return {
        "models": [
            {
                "name": "deepseek-v4-pro:cloud",
                "model": "deepseek-v4-pro:cloud",
                "remote_model": "deepseek-v4-pro",
                "remote_host": "https://ollama.com:443",
                "size": 344,
                "details": {"family": "", "parameter_size": "", "quantization_level": ""},
            },
            {
                "name": "llama3.2:latest",
                "model": "llama3.2:latest",
                "size": 4_123_456_789,
                "details": {"family": "llama", "parameter_size": "3B", "quantization_level": "Q4_K_M"},
            },
            {
                "name": "qwen3.5:4b",
                "model": "qwen3.5:4b",
                "size": 3_389_983_735,
                "details": {"family": "qwen35", "parameter_size": "4.21B", "quantization_level": "Q4_K_M"},
            },
        ]
    }


@pytest.fixture(autouse=True)
def _clear_ollama_cache():
    """Clear the Ollama cache between tests so each test gets a fresh scan."""
    from backend.model_catalog import _OLLAMA_CACHE, _OLLAMA_CACHE_AT

    _OLLAMA_CACHE = None
    _OLLAMA_CACHE_AT = 0.0
    yield


def test_ollama_scan_detects_cloud_and_local():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = _ollama_tags_response()

        from backend.model_catalog import _detect_ollama_models

        models = _detect_ollama_models(force=True)
        assert len(models) == 3

        cloud = [m for m in models if m.get("ollama_cloud")]
        local = [m for m in models if m.get("ollama_local")]
        assert len(cloud) == 1
        assert len(local) == 2
        assert cloud[0]["id"] == "ollama/deepseek-v4-pro:cloud"
        assert local[0]["id"] == "ollama/llama3.2:latest"


def test_ollama_scan_empty_when_not_running():
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = ConnectionError("Ollama not running")

        from backend.model_catalog import _detect_ollama_models

        models = _detect_ollama_models(force=True)
        assert models == []


def test_ollama_scan_caches_results():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = _ollama_tags_response()

        from backend.model_catalog import _detect_ollama_models

        # First call hits Ollama
        models1 = _detect_ollama_models(force=True)
        assert len(models1) == 3

        # Second call within TTL returns cache
        mock_get.reset_mock()
        models2 = _detect_ollama_models(force=False)
        assert len(models2) == 3
        mock_get.assert_not_called()


def test_get_all_models_includes_ollama():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = _ollama_tags_response()

        from backend.model_catalog import get_all_models, AVAILABLE_MODELS

        all_m = get_all_models()
        # Static models + 3 Ollama models
        assert len(all_m) == len(AVAILABLE_MODELS) + 3
        ollama_ids = {m["id"] for m in all_m if m.get("provider") == "ollama"}
        assert "ollama/deepseek-v4-pro:cloud" in ollama_ids
        assert "ollama/qwen3.5:4b" in ollama_ids


def test_resolve_model_choice_includes_ollama():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = _ollama_tags_response()

        from backend.model_catalog import resolve_model_choice

        assert resolve_model_choice("ollama/qwen3.5:4b") == "ollama/qwen3.5:4b"
        assert resolve_model_choice("ollama/deepseek-v4-pro:cloud") == "ollama/deepseek-v4-pro:cloud"

        # Unknown Ollama model should not match
        result = resolve_model_choice("ollama/nonexistent")
        assert result is None


def test_resolve_model_choice_includes_xiaomi():
    from backend.model_catalog import resolve_model_choice

    assert resolve_model_choice("xiaomi/MiMo") == "xiaomi/MiMo"
    # Labels without emoji
    assert resolve_model_choice("Xiaomi MiMo") == "xiaomi/MiMo"


# ---------------------------------------------------------------------------
# Provider routing (llm_params)
# ---------------------------------------------------------------------------

def test_ollama_routing_via_llm_params():
    """Ollama models route through litellm's native ollama/ provider."""
    from agent.core.llm_params import _resolve_llm_params

    params = _resolve_llm_params("ollama/qwen3.5:4b")
    assert params["model"] == "ollama/qwen3.5:4b"
    # Native ollama provider uses /api/chat, not /v1 — no /v1 suffix
    assert "localhost:11434" in params.get("api_base", "")
    assert "/v1" not in params.get("api_base", "")
    # No API key needed for local Ollama (no auth by default)
    assert params.get("api_key") is None or params.get("api_key") == ""


def test_xiaomi_routing_via_llm_params():
    """Xiaomi MiMo routes through the local provider registry to MIMO_API_BASE."""
    from agent.core.llm_params import _resolve_llm_params

    params = _resolve_llm_params(
        "xiaomi/MiMo",
        provider_keys={"xiaomi": "mimo_test_key"},
    )
    assert params["model"] == "openai/MiMo"
    assert "mimo.xiaomi.com" in params.get("api_base", "")
    assert params.get("api_key") == "mimo_test_key"


def test_xiaomi_routing_uses_dynamic_key_env_fallback():
    """Xiaomi routing should pick up MIMO_API_KEY from env when not in provider_keys."""
    original = os.environ.get("MIMO_API_KEY")
    os.environ["MIMO_API_KEY"] = "env_mimo_key"
    try:
        from agent.core.llm_params import _resolve_llm_params

        params = _resolve_llm_params("xiaomi/MiMo")
        assert params.get("api_key") == "env_mimo_key"
    finally:
        if original:
            os.environ["MIMO_API_KEY"] = original
        else:
            os.environ.pop("MIMO_API_KEY", None)


# ---------------------------------------------------------------------------
# Model guidance
# ---------------------------------------------------------------------------

def test_ollama_guidance():
    from agent.prompts.model_guidance import model_guidance

    g = model_guidance("ollama/qwen3.5:4b")
    assert "Ollama" in g
    assert "local" in g.lower()
    assert "no sandbox" in g.lower()


def test_ollama_guidance_any_model():
    """All ollama/<model> variants should get the same generic guidance."""
    from agent.prompts.model_guidance import model_guidance

    for model_name in ["ollama/llama3.2", "ollama/qwen3.5:4b", "ollama/deepseek-v4-pro:cloud"]:
        g = model_guidance(model_name)
        assert "Ollama" in g


def test_xiaomi_mimo_guidance():
    from agent.prompts.model_guidance import model_guidance

    g = model_guidance("xiaomi/MiMo")
    assert "Xiaomi MiMo" in g
    assert "OpenAI-compatible" in g or "MiMo" in g


# ---------------------------------------------------------------------------
# Canonical model id resolution
# ---------------------------------------------------------------------------

def test_canonical_id_ollama():
    from agent.prompts.model_guidance import canonical_model_id

    assert canonical_model_id("ollama/llama3.2") == "ollama"
    assert canonical_model_id("ollama/qwen3.5:4b") == "ollama"
    assert canonical_model_id("OLLAMA/LLAMA3.2") == "ollama"


def test_canonical_id_xiaomi():
    from agent.prompts.model_guidance import canonical_model_id

    assert canonical_model_id("xiaomi/MiMo") == "xiaomi-mimo"
    assert canonical_model_id("xiaomi-org/MiMo") == "xiaomi-mimo"
    assert canonical_model_id("mimo") == "xiaomi-mimo"
    assert canonical_model_id("XIAOMI/MIMO") == "xiaomi-mimo"
