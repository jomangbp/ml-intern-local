from backend.model_catalog import AVAILABLE_MODELS, resolve_model_choice
from agent.context_manager.manager import ContextManager
from agent.prompts.model_guidance import canonical_model_id, model_guidance
from agent.prompts.prompt_manager import PromptManager


def test_openai_catalog_keeps_only_requested_gpt_models():
    ids = [m["id"] for m in AVAILABLE_MODELS]
    assert "openai/gpt-5.3-codex" in ids
    assert "openai/gpt-5.4" in ids
    assert "openai/gpt-5.5" in ids
    assert "openai/gpt-5.3" not in ids
    assert "openai/gpt-5.4-codex" not in ids
    assert "openai/gpt-5.5-codex" not in ids
    assert resolve_model_choice("gpt-5.5") == "openai/gpt-5.5"
    assert resolve_model_choice("gpt-5.5-codex") is None


def test_model_guidance_aliases_legacy_openai_names():
    assert canonical_model_id("openai/gpt-5.3") == "gpt-5.3-codex"
    assert canonical_model_id("openai/gpt-5.4-codex") == "gpt-5.4"
    assert canonical_model_id("openai/gpt-5.5-codex") == "gpt-5.5"
    assert "GPT-5.3 Codex" in model_guidance("openai/gpt-5.3-codex")
    assert "GPT-5.4" in model_guidance("openai/gpt-5.4")
    assert "GPT-5.5" in model_guidance("openai/gpt-5.5")


def test_openai_guidance_matches_model_specific_prompting_advice():
    codex = model_guidance("openai/gpt-5.3-codex")
    assert "avoid unnecessary upfront narration" in codex
    assert "Avoid upfront plans, preambles" in codex
    assert "autonomous senior-engineer" in codex
    assert "smallest meaningful validation" in codex

    gpt54 = model_guidance("openai/gpt-5.4")
    assert "explicit contracts" in gpt54
    assert "dependency-aware tool flow" in gpt54
    assert "Keep outputs compact and structured" in gpt54
    assert "Treat intermediate updates as non-final" in gpt54


def test_non_openai_guidance_aliases_catalog_models():
    assert canonical_model_id("MiniMaxAI/MiniMax-M2.7") == "minimax-m2.7"
    assert canonical_model_id("moonshotai/Kimi-K2.6") == "kimi-k2.6"
    assert canonical_model_id("zai-org/GLM-5.1") == "glm-5.1"
    assert "MiniMax M2.7" in model_guidance("MiniMaxAI/MiniMax-M2.7")
    assert "Kimi K2.6" in model_guidance("moonshotai/Kimi-K2.6")
    assert "GLM-5.1" in model_guidance("zai-org/GLM-5.1")


def test_non_openai_guidance_matches_provider_prompting_advice():
    minimax = model_guidance("MiniMaxAI/MiniMax-M2.7")
    assert "interleaved reasoning" in minimax
    assert "limited set of goals" in minimax
    assert "lightweight state trackers" in minimax
    assert "context-efficient" in minimax

    kimi = model_guidance("moonshotai/Kimi-K2.6")
    assert "clear task steps" in kimi
    assert "delimiters and explicit source boundaries" in kimi
    assert "summarize or filter stale conversation state" in kimi
    assert "not over-prescribe exact tool sequences" in kimi

    glm = model_guidance("zai-org/GLM-5.1")
    assert "long-horizon loops" in glm
    assert "stepwise execution" in glm
    assert "experiment-analyze-optimize loop" in glm
    assert "strategy drift" in glm


def test_prompt_manager_builds_profiles_by_mode_and_interface():
    overlay = PromptManager().build_overlay(
        local_mode=True,
        interface="telegram",
        model_name="openai/gpt-5.5",
    )
    assert "# ML Intern Local identity" in overlay
    assert "# Local execution profile" in overlay
    assert "# Telegram interface style" in overlay
    assert "Default verbosity for this interface: low" in overlay
    assert "# Model guidance: GPT-5.5" in overlay


def test_context_manager_appends_model_guidance_and_refreshes():
    cm = ContextManager(tool_specs=[], hf_token=None, local_mode=True, model_name="openai/gpt-5.4", interface="webui")
    assert "# Model guidance: GPT-5.4" in cm.system_prompt
    assert "# Local execution profile" in cm.system_prompt
    assert "# Web UI interface style" in cm.system_prompt

    cm.refresh_system_prompt(model_name="openai/gpt-5.5")
    assert "# Model guidance: GPT-5.5" in cm.items[0].content
    assert "# Model guidance: GPT-5.4" not in cm.items[0].content
