"""
Local training tool — run training experiments on the host machine.

In local mode, hf_jobs is disabled so training/experiments run entirely
on the local machine via this tool, using bash + trackio locally.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent.core.tools import ToolSpec

MAX_OUTPUT_CHARS = 30_000
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


def _extract_trackio_url(output: str) -> str | None:
    """Extract local Trackio dashboard URL from training output."""
    patterns = [
        re.compile(r"(https?://(?:127\.0\.0\.1|localhost):\d+)"),
        re.compile(r"(https?://[^\s]*huggingface\.co[^\s]*spaces[^\s]*)", re.IGNORECASE),
        re.compile(r"Running on\s+(https?://[^\s]+)", re.IGNORECASE),
        re.compile(r"(https?://\w+\.gradio\.live)", re.IGNORECASE),
    ]
    for pattern in patterns:
        m = pattern.search(output)
        if m:
            url = m.group(1).rstrip("/")
            if url.startswith("http"):
                return url
    return None


def _detect_training_started(output: str) -> bool:
    """Detect if training has started from output indicators."""
    output_lower = output.lower()
    indicators = [
        "trackio.init", "training started", "run started", "starting training",
        "starting experiment", "step 0", "epoch 0", "steps: 0",
        "global step 0", "loaded config", "loading dataset", "loading model",
    ]
    for indicator in indicators:
        if indicator in output_lower:
            return True
    if re.search(r"step\s+0\b", output_lower) or re.search(r"epoch\s+0\b", output_lower):
        return True
    return False


async def _stream_output(stream: asyncio.StreamReader) -> list[str]:
    """Read all lines from a stream until EOF."""
    lines = []
    while True:
        line = await stream.readline()
        if not line:
            break
        lines.append(line.decode("utf-8", errors="replace"))
    return lines


async def _local_training_handler(
    arguments: dict[str, Any], session: Any = None, **_
) -> tuple[str, bool]:
    """
    Run a training or experiment job on the local machine.

    In local mode (no hf_jobs), training runs directly via bash.
    If trackio is in the training command, the local Trackio URL is extracted.
    """
    hf_token = session.hf_token if session else None

    command: str = arguments.get("command", "")
    project: str = arguments.get("project", "ml-intern-local")
    work_dir: str = arguments.get("work_dir", ".")
    timeout: int = min(arguments.get("timeout") or 300, 36000)

    if not command:
        return "No training command provided.", False

    clean_command = _strip_ansi(command)

    import os
    env = dict(os.environ)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    # Check if trackio is installed
    try:
        import trackio as _t
        trackio_available = True
    except ImportError:
        trackio_available = False

    result_msg = ""

    try:
        proc = await asyncio.create_subprocess_shell(
            f"{clean_command} 2>&1",
            shell=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=work_dir,
        )

        # Stream output until command finishes or timeout hits
        output_lines: list[str] = []
        timed_out = False
        try:
            output_lines = await asyncio.wait_for(
                _stream_output(proc.stdout),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            timed_out = True
            # Timed out — kill the process, read whatever was produced
            try:
                proc.kill()
            except Exception:
                pass
            try:
                remaining = await asyncio.wait_for(proc.stdout.read(), timeout=2)
                if remaining:
                    output_lines.append(remaining.decode("utf-8", errors="replace"))
            except Exception:
                pass

        return_code = await proc.wait()
        full_output = _strip_ansi("".join(output_lines))
        trackio_url = _extract_trackio_url(full_output)
        training_started = _detect_training_started(full_output)

        result_msg = full_output

        # Append trackio URL if found
        if trackio_url:
            result_msg += f"\n\nLocal Trackio dashboard: {trackio_url}"
        elif training_started and not trackio_available:
            result_msg += (
                "\n\nNote: Training appears to be running. "
                "To enable local Trackio monitoring: pip install trackio"
            )

    except asyncio.TimeoutError:
        result_msg = (
            f"Training command timed out after {timeout}s.\n"
            "For long-running jobs use nohup:\n"
            "  nohup <command> > /tmp/training.log 2>&1 & echo $!\n"
            "Then check: kill -0 <PID> && echo running || echo done"
        )
        return result_msg, False

    except Exception as e:
        return f"local_training error: {e}", False

    if len(result_msg) > MAX_OUTPUT_CHARS:
        head = result_msg[: int(MAX_OUTPUT_CHARS * 0.25)]
        tail = result_msg[-int(MAX_OUTPUT_CHARS * 0.75):]
        result_msg = (
            f"{head}\n\n[... {len(result_msg) - MAX_OUTPUT_CHARS:,} chars truncated ...]\n"
            f"Run with nohup for full output."
        )
        if trackio_url:
            result_msg += f"\nLocal Trackio: {trackio_url}"

    success = (not timed_out) and (return_code == 0)
    if not success and result_msg:
        result_msg += f"\n\nExit code: {return_code}" if return_code is not None else "\n\nExit code: unknown"
    return result_msg, success


_LOCAL_TRAINING_SPEC = {
    "name": "local_training",
    "description": (
        "Run a training or experiment job on the local machine.\n"
        "\n"
        "Use this instead of hf_jobs when operating in local mode. "
        "Runs training/experiments directly on the host machine with optional local Trackio.\n"
        "\n"
        "Usage:\n"
        "- Provide a bash command that runs your training script.\n"
        "- For long-running jobs, use nohup to background the process.\n"
        "- Set HF_TOKEN in your environment or pass via the session.\n"
        "\n"
        "Local Trackio:\n"
        "- If `import trackio` is in your training script, a local Trackio dashboard starts.\n"
        "- The local Trackio URL (e.g. http://127.0.0.1:7861) is returned in the output.\n"
        "- No HF Space is created when using local Trackio.\n"
        "\n"
        "Examples:\n"
        "  python train.py --config configs/train.yaml\n"
        "  nohup accelerate launch train.py --config train.yaml > /tmp/train.log 2>&1 & echo $!\n"
    ),
    "parameters": {
        "type": "object",
        "required": ["command"],
        "additionalProperties": False,
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run the training (e.g., 'python train.py --config config.yaml').",
            },
            "project": {
                "type": "string",
                "description": "Project name for local Trackio logging (default: 'ml-intern-local').",
            },
            "work_dir": {
                "type": "string",
                "description": "Working directory for the training command (default: current directory).",
            },
            "timeout": {
                "type": "integer",
                "description": "Max time in seconds to stream output before returning (default: 300, max 36000).",
            },
        },
    },
}


def get_local_training_tool() -> ToolSpec:
    """Return the local_training tool spec."""
    return ToolSpec(
        name=_LOCAL_TRAINING_SPEC["name"],
        description=_LOCAL_TRAINING_SPEC["description"],
        parameters=_LOCAL_TRAINING_SPEC["parameters"],
        handler=_local_training_handler,
    )