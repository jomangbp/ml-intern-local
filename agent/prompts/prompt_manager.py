"""Composable prompt profile manager for ML Intern Local.

The goal is to avoid one giant legacy prompt by layering small, explicit
profiles for identity, execution mode, interface, tool policy, and task style.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from agent.prompts.model_guidance import model_guidance

_PROFILE_DIR = Path(__file__).parent / "profiles"

_INTERFACE_PROFILES = {
    "telegram": "telegram_style.md",
    "webui": "webui_style.md",
    "web": "webui_style.md",
    "cli": "cli_style.md",
}

_TASK_PROFILES = {
    "coding": "coding_agent.md",
    "ml_research": "ml_research_agent.md",
    "training": "ml_research_agent.md",
}

_DEFAULT_TASK_PROFILES = ("coding_agent.md", "ml_research_agent.md")


def normalize_interface(interface: str | None = None) -> str:
    raw = (interface or os.environ.get("ML_INTERN_INTERFACE") or "webui").strip().lower()
    return raw if raw in _INTERFACE_PROFILES else "webui"


def verbosity_for_interface(interface: str | None = None) -> str:
    normalized = normalize_interface(interface)
    env_name = {
        "telegram": "TELEGRAM_VERBOSITY",
        "webui": "WEBUI_VERBOSITY",
        "web": "WEBUI_VERBOSITY",
        "cli": "CLI_VERBOSITY",
    }.get(normalized, "WEBUI_VERBOSITY")
    fallback = "low" if normalized == "telegram" else "medium"
    value = (os.environ.get(env_name) or fallback).strip().lower()
    return value if value in {"low", "medium", "high"} else fallback


@lru_cache(maxsize=64)
def _load_profile(filename: str) -> str:
    path = _PROFILE_DIR / filename
    return path.read_text(encoding="utf-8").strip()


class PromptManager:
    """Build prompt overlays from small markdown profile files."""

    def build_overlay(
        self,
        *,
        local_mode: bool,
        interface: str | None = None,
        task_type: str | None = None,
        model_name: str | None = None,
    ) -> str:
        normalized_interface = normalize_interface(interface)
        files: list[str] = [
            "base_system.md",
            "instruction_priority.md",
            "personality.md",
            "collaboration_style.md",
            "local_execution.md" if local_mode else "hf_sandbox.md",
            _INTERFACE_PROFILES[normalized_interface],
            "tool_policy.md",
            "approval_policy.md",
            "research_budget.md",
        ]

        if task_type:
            task_file = _TASK_PROFILES.get(task_type.strip().lower())
            if task_file:
                files.append(task_file)
        else:
            files.extend(_DEFAULT_TASK_PROFILES)

        parts = [_load_profile(name) for name in files]
        parts.append(f"# Verbosity\n\nDefault verbosity for this interface: {verbosity_for_interface(normalized_interface)}.")

        guidance = model_guidance(model_name)
        if guidance:
            parts.append(guidance)

        return "\n\n".join(part for part in parts if part)
