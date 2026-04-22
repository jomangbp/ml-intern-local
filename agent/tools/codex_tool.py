"""
Codex OAuth Tool — device-code authentication via the OpenAI Codex CLI.

Wraps `codex login --device-auth` for non-interactive use and detects whether
a session is already active. This lets ml-intern (and its parent agent) piggyback
on a Codex login rather than maintaining its own OAuth flow.

Usage in ml-intern:
    codex_login()          — check status, return "logged_in" or "not_logged_in"
    codex_login(force=True) — re-authenticate even if already logged in
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Union

from agent.tools.types import ToolResult

# ── Config paths Codex uses ────────────────────────────────────────────

def _codex_config_dir() -> Path:
    """Return Codex config directory (platform-aware)."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "codex"
    elif os.name == "darwin":
        return Path.home() / "Library" / "Application Support" / "codex"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        return Path(xdg) / "codex"


def _codex_auth_token() -> str | None:
    """Return the stored Codex auth token if present."""
    cfg = _codex_config_dir() / "auth.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
        return data.get("access_token") or data.get("token")
    except Exception:
        return None


async def codex_login_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Check or initiate Codex OAuth (device code flow).

    Args:
        force (bool): If True, force a new login even if already authenticated.
            Default: False.

    Returns:
        ToolResult describing login state and next steps.
    """
    force = arguments.get("force", False)

    # 1. Check if codex binary is available
    codex_path = shutil.which("codex")
    if not codex_path:
        return (
            "Codex CLI is not installed.\n\n"
            "Install it with:\n"
            "  npm install -g @openai/codex\n"
            "  # or\n"
            "  brew install --cask codex\n\n"
            "Then run `codex login --device-auth` manually and retry.",
            False,
        )

    # 2. Check existing auth token
    if not force:
        token = _codex_auth_token()
        if token:
            username = _detect_codex_user(codex_path)
            return (
                f"Codex is already logged in as: {username or 'unknown'}\n"
                f"Token found at: {_codex_config_dir() / 'auth.json'}",
                True,
            )

    # 3. Detect terminal capability for device code auth
    #    Codex uses a localhost redirect for OAuth; if we can't handle that,
    #    we fall back to instructing the user.
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    # 4. Attempt device code flow in a subprocess
    #    Codex --device-auth opens no browser — it prints a URL + code.
    tmp_code: list[str] = [""]

    def _capture_device_code(line: str) -> None:
        m = re.search(r"(https?://[^\s]+)\s+with\s+code\s+([A-Z0-9]+)", line)
        if m:
            tmp_code[0] = f"URL: {m.group(1)}\nCode: {m.group(2)}"
        elif "successfully" in line.lower() or "logged in" in line.lower():
            tmp_code[0] = "SUCCESS"

    print_url: list[str] = [""]

    def _capture_url(line: str) -> None:
        if line.startswith("http"):
            print_url[0] = line.strip()

    output_lines: list[str] = []
    has_codex_com = False

    def _collect(line: str) -> None:
        output_lines.append(line)
        if "codex.ai" in line or "platform.openai.com" in line:
            has_codex_com = True
        _capture_device_code(line)
        _capture_url(line)

    try:
        result = _run_codex_command(
            ["login", "--device-auth", "--verbose"],
            on_line=_collect,
            timeout=60,
        )
    except Exception as e:
        return f"Failed to run `codex login --device-auth`: {e}", False

    # 5. Parse result
    if tmp_code[0] == "SUCCESS":
        username = _detect_codex_user(codex_path)
        return f"Codex login successful. User: {username or 'unknown'}", True

    if print_url[0]:
        # Device code URL was printed — surface it to the agent
        return (
            f"Codex device code flow started.\n\n"
            f"Open this URL in your browser:\n  {print_url[0]}\n\n"
            f"Complete the sign-in there, then this tool will detect the session.\n"
            f"Or run `codex login --device-auth` manually in a terminal first.",
            True,
        )

    # 6. Fallback: check if it opened a browser automatically
    if has_codex_com:
        return (
            "Codex browser-based login was initiated.\n\n"
            "Complete the sign-in in the opened browser, then run this tool again "
            "to confirm authentication.",
            True,
        )

    return (
        "Could not determine Codex login state automatically.\n\n"
        "Manual steps:\n"
        "1. Run in a terminal: `codex login --device-auth`\n"
        "2. Open the URL it prints, sign in with ChatGPT or API key\n"
        "3. Retry this tool\n\n"
        f"Codex binary: {codex_path}",
        True,  # not an error — just needs manual step
    )


def _detect_codex_user(codex_path: str) -> str | None:
    """Ask codex for the current username if logged in."""
    try:
        result = subprocess.run(
            [codex_path, "whoami"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _run_codex_command(
    args: list[str],
    on_line: Union[Callable[[str], None], None] = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a codex subcommand, optionally streaming lines to on_line."""
    merge_env = {**os.environ, **(env or {})}
    # Suppress Codex telemetry / analytics noise
    merge_env["CODEX_NO_ANALYTICS"] = "1"

    if on_line is not None:
        proc = subprocess.Popen(
            [shutil.which("codex"), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merge_env,
        )
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                on_line(line.rstrip("\n"))
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        return subprocess.CompletedProcess(proc.args, proc.returncode)

    result = subprocess.run(
        [shutil.which("codex"), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merge_env,
    )
    return result


# ── Tool specification ─────────────────────────────────────────────────

CODEX_LOGIN_TOOL_SPEC = {
    "name": "codex_login",
    "description": (
        "Check Codex OAuth login status or initiate device-code authentication.\n\n"
        "Codex is OpenAI's coding agent CLI. This tool wraps `codex login --device-auth`\n"
        "to detect an existing session or kick off a new device-code flow.\n\n"
        "Use this when:\n"
        "  • You need Codex credentials for another tool (codex_run, Codex IDE integration)\n"
        "  • You're bridging ml-intern with a Codex-backed workflow\n"
        "  • You want to verify that a Codex session is active\n\n"
        "If the tool reports 'not logged in', the user should run in a terminal:\n"
        "  codex login --device-auth\n"
        "then retry this tool.\n\n"
        "Requires `codex` to be installed (npm install -g @openai/codex)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": (
                    "If True, force a new login even if already authenticated. "
                    "Default: False (check status only)."
                ),
                "default": False,
            },
        },
        "additionalProperties": False,
    },
}
