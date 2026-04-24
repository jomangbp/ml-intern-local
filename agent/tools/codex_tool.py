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
from pathlib import Path
from typing import Any, Callable, Union

# ── Config paths Codex uses ────────────────────────────────────────────

def _codex_config_candidates() -> list[Path]:
    """Return candidate Codex config dirs in priority order."""
    home = Path.home()
    candidates: list[Path] = []

    # Newer Codex CLI default on Linux/macOS tends to be ~/.codex
    candidates.append(home / ".codex")

    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        candidates.append(base / "codex")
    elif os.name == "darwin":
        candidates.append(home / "Library" / "Application Support" / "codex")
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", home / ".config")
        candidates.append(Path(xdg) / "codex")

    # Deduplicate while preserving order
    seen = set()
    out: list[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _codex_auth_file() -> Path:
    """Return the best auth.json path (existing one preferred)."""
    for cfg_dir in _codex_config_candidates():
        p = cfg_dir / "auth.json"
        if p.exists():
            return p
    # Fallback path for UI display when no auth file exists yet
    return _codex_config_candidates()[0] / "auth.json"


def _read_codex_auth() -> dict[str, Any]:
    p = _codex_auth_file()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _codex_auth_token() -> str | None:
    """Return the stored Codex auth token if present."""
    data = _read_codex_auth()
    if not data:
        return None
    # Legacy flat keys
    token = data.get("access_token") or data.get("token")
    if token:
        return token
    # Newer schema
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    return tokens.get("access_token")


def codex_auth_token() -> str | None:
    """Public helper: return current Codex OAuth token if present."""
    return _codex_auth_token()


def _is_codex_logged_in(codex_path: str | None) -> bool:
    """Check login state via Codex CLI status command."""
    if not codex_path:
        return False
    try:
        result = subprocess.run(
            [codex_path, "login", "status"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        return result.returncode == 0 and "logged in" in text.lower()
    except Exception:
        return False


def codex_auth_status() -> dict[str, Any]:
    """Public helper: lightweight Codex auth status for API/UI use."""
    codex_path = shutil.which("codex")
    logged_in = _is_codex_logged_in(codex_path)
    # Fallback to token presence in case status command fails in older/newer CLIs
    if not logged_in:
        logged_in = bool(codex_path and _codex_auth_token())

    username = _detect_codex_user(codex_path) if logged_in else None
    return {
        "installed": bool(codex_path),
        "logged_in": logged_in,
        "username": username,
        "config_path": str(_codex_auth_file()),
    }


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

    # 2. Check existing auth status
    if not force and _is_codex_logged_in(codex_path):
        username = _detect_codex_user(codex_path)
        return (
            f"Codex is already logged in as: {username or 'unknown'}\n"
            f"Auth file: {_codex_auth_file()}",
            True,
        )

    # 3. Attempt device-code flow in a subprocess. Codex prints a URL + code,
    # then waits for browser completion. We capture the prompt and timeout early.
    login_status: list[str] = [""]
    print_url: list[str] = [""]
    device_code: list[str] = [""]
    output_lines: list[str] = []
    has_openai_auth_url: list[bool] = [False]

    ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def _collect(line: str) -> None:
        clean = ansi_re.sub("", line).strip()
        if not clean:
            return
        output_lines.append(clean)

        lower = clean.lower()
        if "successfully" in lower or "logged in" in lower:
            login_status[0] = "SUCCESS"

        url_match = re.search(r"https?://\S+", clean)
        if url_match:
            url = url_match.group(0)
            if "auth.openai.com" in url or "openai.com" in url:
                has_openai_auth_url[0] = True
            print_url[0] = url

        code_match = re.search(r"\b[A-Z0-9]{4,}-[A-Z0-9]{4,}\b", clean)
        if code_match:
            device_code[0] = code_match.group(0)

    try:
        _run_codex_command(
            ["login", "--device-auth"],
            on_line=_collect,
            timeout=20,
        )
    except Exception as e:
        return f"Failed to run `codex login --device-auth`: {e}", False

    # 4. Parse result
    if login_status[0] == "SUCCESS":
        username = _detect_codex_user(codex_path)
        return f"Codex login successful. User: {username or 'unknown'}", True

    if print_url[0] or device_code[0]:
        code_block = f"\nCode:\n  {device_code[0]}\n" if device_code[0] else ""
        return (
            "Codex device code flow started.\n\n"
            f"Open this URL in your browser:\n  {print_url[0] or 'https://auth.openai.com/codex/device'}\n"
            f"{code_block}\n"
            "Complete sign-in, then run this tool again to confirm authentication.\n"
            "(You can also run `codex login --device-auth` manually in your terminal.)",
            True,
        )

    # 5. Fallback: check if browser/OAuth path was initiated
    if has_openai_auth_url[0]:
        return (
            "Codex browser/device login was initiated.\n\n"
            "Complete sign-in in your browser, then run this tool again to confirm.",
            True,
        )

    tail = "\n".join(output_lines[-6:]) if output_lines else "(no output captured)"
    return (
        "Could not determine Codex login state automatically.\n\n"
        "Manual steps:\n"
        "1. Run in a terminal: `codex login --device-auth`\n"
        "2. Open the URL it prints, sign in with ChatGPT or API key\n"
        "3. Retry this tool\n\n"
        f"Codex binary: {codex_path}\n"
        f"Last CLI output:\n{tail}",
        True,  # not an error — just needs manual step
    )


def _detect_codex_user(codex_path: str | None) -> str | None:
    """Best-effort user identity from auth metadata / CLI."""
    # Newer auth schema stores account id in tokens.account_id
    data = _read_codex_auth()
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    account_id = tokens.get("account_id")
    if isinstance(account_id, str) and account_id.strip():
        return account_id.strip()

    if codex_path:
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
            # Expected for device-code flow (Codex waits for user browser auth).
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            return subprocess.CompletedProcess(proc.args, 124)
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
