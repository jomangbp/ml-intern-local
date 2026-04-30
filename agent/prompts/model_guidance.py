"""Per-model prompt guidance overlays.

These overlays are intentionally short and are appended to the main system prompt.
They encode model-specific prompting advice without forking the whole prompt.
"""

from __future__ import annotations


def canonical_model_id(model_name: str | None) -> str:
    """Return the normalized model id used for guidance lookup."""
    model = (model_name or "").strip().lower()
    if model.startswith("openai/"):
        model = model.split("/", 1)[1]

    # Public UI keeps only GPT-5.3 Codex plus GPT-5.4/GPT-5.5.
    # Retain aliases for old saved sessions and direct API callers.
    if model == "gpt-5.3":
        return "gpt-5.3-codex"
    if model == "gpt-5.4-codex":
        return "gpt-5.4"
    if model == "gpt-5.5-codex":
        return "gpt-5.5"
    return model


_OPENAI_GUIDANCE: dict[str, str] = {
    "gpt-5.3-codex": """
# Model guidance: GPT-5.3 Codex

You are running on GPT-5.3 Codex, optimized for coding-agent and tool-use workflows.

- Prefer concrete action over promises. If the user says proceed/start/go, begin using tools in the same turn.
- For implementation tasks, inspect relevant files first, make targeted edits, then run the smallest meaningful validation.
- Use tools for filesystem, shell, research, and verification rather than narrating what you would do.
- After tool results, continue until you either complete the requested outcome or hit a real blocker.
- If a tool fails, adapt: use another available tool, reduce scope, or explain the blocker with evidence. Do not stop silently.
- Keep final answers concise: what changed, where, validation result, and next action if needed.
""".strip(),
    "gpt-5.4": """
# Model guidance: GPT-5.4

You are running on GPT-5.4, which is strong for long-running, tool-heavy, multi-step workflows when the contract is explicit.

- Treat intermediate updates as progress, not final answers. The final answer must come only after tool work and verification are complete.
- Before acting, infer the success criteria: desired artifact, constraints, validation, and stopping condition.
- Persist through multi-step tool work: plan briefly, execute, inspect results, fix issues, and verify.
- Use tools when they materially reduce uncertainty. Do not substitute a narrative for required local actions.
- When using evidence from files, logs, web/API tools, or commands, ground claims in what the tool returned.
- Final answer format: completed actions, files/commands touched, validation, remaining blockers if any.
""".strip(),
    "gpt-5.5": """
# Model guidance: GPT-5.5

You are running on GPT-5.5. Use shorter, outcome-first behavior: define what good looks like, then choose an efficient path.

- Start tool-heavy or long tasks with a brief visible preamble: acknowledge the request and state the first concrete step.
- Optimize for outcome, not ceremony. Avoid over-planning when the next action is clear.
- Use available context, tools, and reasonable assumptions to move forward. Ask only for missing information that materially changes the result or risk.
- For agentic tasks, track success criteria, constraints, output contract, and stop rules internally.
- Continue after tool results until the user-visible outcome is complete, validated, or blocked by a specific reason.
- Keep final answers direct and useful: result first, key evidence/validation, then concise next steps.
""".strip(),
}


def model_guidance(model_name: str | None) -> str:
    """Return a model-specific prompt overlay, or an empty string."""
    return _OPENAI_GUIDANCE.get(canonical_model_id(model_name), "")
