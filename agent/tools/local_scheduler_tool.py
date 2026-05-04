"""
Local scheduled task / training watchdog tool.

This tool is available in local mode and lets the agent schedule a background
loop/checker on the host machine. The primary use case is a training watchdog:
wait a user-configured number of minutes, check whether matching training
processes are still running, and stop them safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.tools import ToolSpec

STATE_DIR = Path(os.environ.get("ML_INTERN_SCHEDULER_DIR", "~/.cache/ml-intern/scheduled_tasks")).expanduser()
LOG_DIR = STATE_DIR / "logs"
MAX_OUTPUT_TAIL = 12_000
MIN_INTERVAL_SECONDS = 1
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
DEFAULT_GRACE_SECONDS = 30

_SIGNAL_MAP = {
    "TERM": signal.SIGTERM,
    "INT": signal.SIGINT,
    "KILL": signal.SIGKILL,
}

_TOO_BROAD_PATTERNS = {
    "python",
    "python3",
    "bash",
    "sh",
    "zsh",
    "fish",
    "train",
    "scripts/train.py",
    "nohup",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _task_paths(task_id: str) -> tuple[Path, Path, Path]:
    _ensure_dirs()
    config_path = STATE_DIR / f"{task_id}.json"
    status_path = STATE_DIR / f"{task_id}.status.json"
    cancel_path = STATE_DIR / f"{task_id}.cancel"
    return config_path, status_path, cancel_path


def _delete_task_safe(task_id: str) -> tuple[str, bool]:
    """Cancel and remove all files for a local scheduler task."""
    config_path, status_path, cancel_path = _task_paths(task_id)
    if not status_path.exists():
        return f"No such scheduled task: {task_id}", False
    # Cancel runner if alive
    status = _read_json(status_path)
    runner_pid = status.get("runner_pid")
    if runner_pid and _pid_is_alive(int(runner_pid)):
        try:
            os.kill(int(runner_pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Remove all files
    for p in (config_path, status_path, cancel_path):
        p.unlink(missing_ok=True)
    return f"Deleted scheduled task {task_id}.", True


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _tail_text(text: str, max_chars: int = MAX_OUTPUT_TAIL) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    # A completed child may remain briefly as a zombie until its parent reaps it;
    # report that as not alive for scheduler status/stop purposes.
    try:
        stat = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        ).stdout.strip()
        if stat.startswith("Z"):
            return False
    except Exception:
        pass
    return True


def _process_rows() -> list[dict[str, Any]]:
    """Return process table rows with pid, ppid, stat, and command."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        rows.append({"pid": pid, "ppid": ppid, "stat": parts[2], "command": parts[3]})
    return rows


def _current_ancestor_pids() -> set[int]:
    rows = _process_rows()
    parent_by_pid = {int(r["pid"]): int(r["ppid"]) for r in rows}
    protected = {os.getpid(), os.getppid()}
    pid = os.getpid()
    for _ in range(64):
        parent = parent_by_pid.get(pid)
        if not parent or parent in protected:
            break
        protected.add(parent)
        pid = parent
    return protected


def _descendants(root_pids: set[int], rows: list[dict[str, Any]]) -> set[int]:
    children_by_ppid: dict[int, list[int]] = {}
    for row in rows:
        children_by_ppid.setdefault(int(row["ppid"]), []).append(int(row["pid"]))

    result: set[int] = set()
    stack = list(root_pids)
    while stack:
        pid = stack.pop()
        for child in children_by_ppid.get(pid, []):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def _find_matching_pids(pattern: str, match_mode: str = "substring") -> list[dict[str, Any]]:
    """Find processes whose command line matches pattern, excluding this runner."""
    rows = _process_rows()
    protected = _current_ancestor_pids()
    protected.add(os.getpid())

    matches: list[dict[str, Any]] = []
    regex: re.Pattern[str] | None = None
    if match_mode == "regex":
        regex = re.compile(pattern)
    lower_pattern = pattern.lower()

    for row in rows:
        pid = int(row["pid"])
        cmd = str(row["command"])
        if pid in protected:
            continue
        # Never let the scheduler stop itself or another scheduler task.
        if "local_scheduler_tool.py" in cmd or "scheduled_tasks" in cmd:
            continue
        matched = bool(regex.search(cmd)) if regex else lower_pattern in cmd.lower()
        if matched:
            matches.append(row)
    return matches


def _stop_pids(pids: list[int], stop_signal: str = "TERM", grace_seconds: int = DEFAULT_GRACE_SECONDS) -> dict[str, Any]:
    """Stop pids and their descendants, escalating to SIGKILL after grace period."""
    rows = _process_rows()
    pid_set = set(pids)
    pid_set |= _descendants(pid_set, rows)
    pid_set.discard(os.getpid())
    pid_set.discard(os.getppid())

    sig = _SIGNAL_MAP.get(stop_signal.upper(), signal.SIGTERM)
    attempted: list[int] = []
    errors: list[str] = []

    # Children first, then parents. This keeps training workers from surviving
    # after a shell wrapper exits.
    ordered = sorted(pid_set, reverse=True)
    for pid in ordered:
        try:
            os.kill(pid, sig)
            attempted.append(pid)
        except ProcessLookupError:
            pass
        except Exception as e:  # pragma: no cover - platform/permission dependent
            errors.append(f"pid {pid}: {e}")

    deadline = time.monotonic() + max(0, grace_seconds)
    while time.monotonic() < deadline:
        alive = [pid for pid in attempted if _pid_is_alive(pid)]
        if not alive:
            break
        time.sleep(0.5)

    escalated: list[int] = []
    for pid in attempted:
        if _pid_is_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                escalated.append(pid)
            except ProcessLookupError:
                pass
            except Exception as e:  # pragma: no cover - platform/permission dependent
                errors.append(f"pid {pid} SIGKILL: {e}")

    return {
        "attempted_pids": attempted,
        "sigkill_pids": escalated,
        "errors": errors,
    }


def _run_check(config: dict[str, Any], log_path: Path) -> dict[str, Any]:
    started = _utc_now()
    pieces: list[str] = [f"[{started}] scheduled check started"]
    result: dict[str, Any] = {
        "started_at": started,
        "command_exit_code": None,
        "matching_processes": [],
        "stop_result": None,
        "output_tail": "",
    }

    command = config.get("command") or ""
    if command:
        timeout = int(config.get("command_timeout_seconds") or DEFAULT_COMMAND_TIMEOUT_SECONDS)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=config.get("work_dir") or ".",
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            output = proc.stdout + proc.stderr
            result["command_exit_code"] = proc.returncode
            result["output_tail"] = _tail_text(output)
            pieces.append(f"command exit={proc.returncode}")
            if output:
                pieces.append(output)
        except subprocess.TimeoutExpired as e:
            output = (e.stdout or "") + (e.stderr or "")
            result["command_exit_code"] = "timeout"
            result["output_tail"] = _tail_text(output)
            pieces.append(f"command timed out after {timeout}s")
            if output:
                pieces.append(output)
        except Exception as e:
            result["command_exit_code"] = "error"
            result["output_tail"] = str(e)
            pieces.append(f"command error: {e}")

    training_match = config.get("training_match") or ""
    if training_match:
        matches = _find_matching_pids(training_match, config.get("match_mode", "substring"))
        result["matching_processes"] = matches
        pieces.append(f"matched {len(matches)} process(es) for {training_match!r}")
        for match in matches:
            pieces.append(f"  pid={match['pid']} ppid={match['ppid']} cmd={match['command']}")

        if matches and config.get("stop_if_running", True):
            stop_result = _stop_pids(
                [int(m["pid"]) for m in matches],
                stop_signal=config.get("stop_signal", "TERM"),
                grace_seconds=int(config.get("grace_seconds") or DEFAULT_GRACE_SECONDS),
            )
            result["stop_result"] = stop_result
            pieces.append(f"stop result: {json.dumps(stop_result, sort_keys=True)}")

    finished = _utc_now()
    result["finished_at"] = finished
    pieces.append(f"[{finished}] scheduled check finished")
    _append_log(log_path, "\n".join(pieces) + "\n")
    return result


def _runner(config_path: Path) -> int:
    config = _read_json(config_path)
    task_id = config["task_id"]
    _, status_path, cancel_path = _task_paths(task_id)
    log_path = Path(config["log_path"])

    status = {
        "task_id": task_id,
        "status": "running",
        "runner_pid": os.getpid(),
        "started_at": _utc_now(),
        "runs_completed": 0,
        "config": config,
        "log_path": str(log_path),
    }
    _atomic_write_json(status_path, status)

    interval = max(MIN_INTERVAL_SECONDS, int(config["interval_seconds"]))
    repeat = bool(config.get("repeat", False))
    max_runs = int(config.get("max_runs") or (0 if repeat else 1))

    try:
        while True:
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline:
                if cancel_path.exists():
                    status.update({"status": "cancelled", "finished_at": _utc_now()})
                    _atomic_write_json(status_path, status)
                    return 0
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

            status["status"] = "checking"
            status["last_check_started_at"] = _utc_now()
            _atomic_write_json(status_path, status)

            check = _run_check(config, log_path)
            status["runs_completed"] = int(status.get("runs_completed") or 0) + 1
            status["last_check"] = check
            status["last_check_finished_at"] = check.get("finished_at")
            status["status"] = "running" if repeat else "completed"
            if not repeat:
                status["finished_at"] = _utc_now()
            _atomic_write_json(status_path, status)

            if not repeat:
                break
            if max_runs > 0 and int(status["runs_completed"]) >= max_runs:
                status["status"] = "completed"
                status["finished_at"] = _utc_now()
                _atomic_write_json(status_path, status)
                break
            if cancel_path.exists():
                status.update({"status": "cancelled", "finished_at": _utc_now()})
                _atomic_write_json(status_path, status)
                break
    except Exception as e:
        status.update({"status": "failed", "error": str(e), "finished_at": _utc_now()})
        _atomic_write_json(status_path, status)
        _append_log(log_path, f"[{_utc_now()}] scheduler failed: {e}\n")
        return 1
    return 0


def _validate_create_args(arguments: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    interval_minutes = arguments.get("interval_minutes")
    interval_seconds = arguments.get("interval_seconds")
    if interval_seconds is None:
        if interval_minutes is None:
            return None, "Provide interval_minutes (or interval_seconds for tests/short checks)."
        try:
            interval_seconds = int(float(interval_minutes) * 60)
        except (TypeError, ValueError):
            return None, "interval_minutes must be a number."
    try:
        interval_seconds = int(interval_seconds)
    except (TypeError, ValueError):
        return None, "interval_seconds must be an integer."
    if interval_seconds < MIN_INTERVAL_SECONDS:
        return None, f"interval_seconds must be >= {MIN_INTERVAL_SECONDS}."

    command = (arguments.get("command") or "").strip()
    training_match = (arguments.get("training_match") or "").strip()
    if not command and not training_match:
        return None, "Provide command and/or training_match."

    match_mode = arguments.get("match_mode") or "substring"
    if match_mode not in {"substring", "regex"}:
        return None, "match_mode must be 'substring' or 'regex'."
    if match_mode == "regex" and training_match:
        try:
            re.compile(training_match)
        except re.error as e:
            return None, f"Invalid training_match regex: {e}"

    stop_if_running = bool(arguments.get("stop_if_running", True))
    if training_match and stop_if_running:
        normalized = training_match.strip().lower()
        if len(normalized) < 6 or normalized in _TOO_BROAD_PATTERNS:
            return None, (
                "training_match is too broad for a stop watchdog. Use a more specific "
                "substring/regex, e.g. 'scripts/train.py --resume outputs/run-a' or an output directory name."
            )

    stop_signal = (arguments.get("stop_signal") or "TERM").upper()
    if stop_signal not in _SIGNAL_MAP:
        return None, "stop_signal must be TERM, INT, or KILL."

    task_id = str(uuid.uuid4())[:12]
    task_name = arguments.get("task_name") or f"training-watchdog-{task_id}"
    log_path = Path(arguments.get("log_path") or (LOG_DIR / f"{task_id}.log")).expanduser()

    config = {
        "task_id": task_id,
        "task_name": task_name,
        "description": arguments.get("description") or "",
        "created_at": _utc_now(),
        "interval_seconds": interval_seconds,
        "interval_minutes": interval_seconds / 60,
        "repeat": bool(arguments.get("repeat", False)),
        "max_runs": int(arguments.get("max_runs") or 0),
        "command": command,
        "command_timeout_seconds": int(arguments.get("command_timeout_seconds") or DEFAULT_COMMAND_TIMEOUT_SECONDS),
        "work_dir": arguments.get("work_dir") or ".",
        "training_match": training_match,
        "match_mode": match_mode,
        "stop_if_running": stop_if_running,
        "stop_signal": stop_signal,
        "grace_seconds": int(arguments.get("grace_seconds") or DEFAULT_GRACE_SECONDS),
        "log_path": str(log_path),
    }
    return config, None


def _format_task(status: dict[str, Any]) -> str:
    cfg = status.get("config") or {}
    parts = [
        f"- {status.get('task_id')} ({cfg.get('task_name', 'unnamed')}): {status.get('status')}",
        f"  pid={status.get('runner_pid')} interval={cfg.get('interval_minutes')} min repeat={cfg.get('repeat')}",
    ]
    if cfg.get("training_match"):
        parts.append(f"  training_match={cfg.get('training_match')!r} stop_if_running={cfg.get('stop_if_running')}")
    if cfg.get("command"):
        parts.append(f"  command={cfg.get('command')!r}")
    if status.get("runs_completed") is not None:
        parts.append(f"  runs_completed={status.get('runs_completed')}")
    if status.get("last_check"):
        matches = status["last_check"].get("matching_processes") or []
        stop = status["last_check"].get("stop_result") or {}
        parts.append(f"  last_check={status['last_check'].get('finished_at')} matches={len(matches)} stopped={stop.get('attempted_pids', [])}")
    parts.append(f"  log={status.get('log_path') or cfg.get('log_path')}")
    return "\n".join(parts)


async def _local_scheduler_handler(arguments: dict[str, Any], session: Any = None, **_) -> tuple[str, bool]:
    action = arguments.get("action", "list")
    _ensure_dirs()

    if action == "create":
        config, error = _validate_create_args(arguments)
        if error:
            return error, False
        assert config is not None
        config_path, status_path, cancel_path = _task_paths(config["task_id"])
        if cancel_path.exists():
            cancel_path.unlink()
        _atomic_write_json(config_path, config)
        initial_status = {
            "task_id": config["task_id"],
            "status": "scheduled",
            "created_at": _utc_now(),
            "config": config,
            "log_path": config["log_path"],
        }
        _atomic_write_json(status_path, initial_status)

        env = dict(os.environ)
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--run", str(config_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        initial_status["runner_pid"] = proc.pid
        initial_status["status"] = "scheduled"
        _atomic_write_json(status_path, initial_status)

        return (
            "Scheduled training watchdog.\n"
            f"task_id: {config['task_id']}\n"
            f"runner_pid: {proc.pid}\n"
            f"first_check_in: {config['interval_seconds']}s ({config['interval_minutes']:.2f} min)\n"
            f"repeat: {config['repeat']}\n"
            f"training_match: {config['training_match'] or '(none)'}\n"
            f"stop_if_running: {config['stop_if_running']}\n"
            f"log: {config['log_path']}\n"
            "Use local_scheduler action='status' with this task_id to inspect it, or action='cancel' to stop the watchdog."
        ), True

    if action == "list":
        statuses = []
        for path in sorted(STATE_DIR.glob("*.status.json")):
            try:
                statuses.append(_read_json(path))
            except Exception:
                continue
        if not statuses:
            return "No scheduled local tasks.", True
        return "Scheduled local tasks:\n" + "\n".join(_format_task(s) for s in statuses), True

    if action == "status":
        task_id = arguments.get("task_id")
        if not task_id:
            return "Provide task_id for status.", False
        _, status_path, _ = _task_paths(str(task_id))
        if not status_path.exists():
            return f"No such scheduled task: {task_id}", False
        status = _read_json(status_path)
        runner_pid = status.get("runner_pid")
        if runner_pid:
            status["runner_alive"] = _pid_is_alive(int(runner_pid))
        return _format_task(status) + "\n\nRaw status:\n" + json.dumps(status, indent=2, sort_keys=True), True

    if action == "cancel":
        task_id = arguments.get("task_id")
        if not task_id:
            return "Provide task_id for cancel.", False
        _, status_path, cancel_path = _task_paths(str(task_id))
        if not status_path.exists():
            return f"No such scheduled task: {task_id}", False
        cancel_path.write_text(_utc_now(), encoding="utf-8")
        status = _read_json(status_path)
        runner_pid = status.get("runner_pid")
        if runner_pid and _pid_is_alive(int(runner_pid)):
            try:
                os.kill(int(runner_pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        status.update({"status": "cancelled", "cancelled_at": _utc_now()})
        _atomic_write_json(status_path, status)
        return f"Cancelled scheduled task {task_id}.", True

    if action == "run_once":
        config, error = _validate_create_args({**arguments, "interval_seconds": 1})
        if error:
            return error, False
        assert config is not None
        log_path = Path(config["log_path"])
        check = _run_check(config, log_path)
        return "Ran scheduled check once:\n" + json.dumps(check, indent=2, sort_keys=True), True

    return "Unknown action. Use create, list, status, cancel, or run_once.", False


_LOCAL_SCHEDULER_SPEC = {
    "name": "local_scheduler",
    "description": (
        "Schedule, list, inspect, or cancel a local background watchdog task. "
        "Use this in local mode to run a command after a user-configured interval, "
        "or to check whether training is still running and stop it safely.\n\n"
        "Common training watchdog usage: action='create', interval_minutes=<minutes>, "
        "training_match=<specific substring or regex from ps command line>, "
        "stop_if_running=true. The watchdog waits for the interval, finds matching "
        "processes, sends TERM, waits grace_seconds, then escalates to KILL if needed.\n\n"
        "Use action='list' to see tasks, action='status' with task_id to inspect, "
        "and action='cancel' to cancel a watchdog before it fires. Set repeat=true "
        "to make the check loop every interval; otherwise it runs once."
    ),
    "parameters": {
        "type": "object",
        "required": ["action"],
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "status", "cancel", "run_once"],
                "description": "Operation to perform.",
            },
            "task_id": {
                "type": "string",
                "description": "Task id for status or cancel.",
            },
            "task_name": {
                "type": "string",
                "description": "Human-readable task name for create.",
            },
            "description": {
                "type": "string",
                "description": "Human-readable purpose of the scheduled task.",
            },
            "interval_minutes": {
                "type": "number",
                "description": "Minutes to wait before each check. Required for create unless interval_seconds is provided.",
            },
            "interval_seconds": {
                "type": "integer",
                "description": "Seconds to wait before each check. Prefer interval_minutes for user-facing schedules; useful for short tests.",
            },
            "repeat": {
                "type": "boolean",
                "description": "If true, keep checking every interval until cancelled or max_runs is reached. Default false.",
                "default": False,
            },
            "max_runs": {
                "type": "integer",
                "description": "Maximum checks for repeat mode. 0 means unlimited. Default 0.",
            },
            "command": {
                "type": "string",
                "description": "Optional shell command to run at each check before process matching/stopping.",
            },
            "command_timeout_seconds": {
                "type": "integer",
                "description": "Timeout for command at each check. Default 120 seconds.",
            },
            "work_dir": {
                "type": "string",
                "description": "Working directory for command. Default current directory.",
            },
            "training_match": {
                "type": "string",
                "description": "Specific substring or regex to match training process command lines, e.g. 'scripts/train.py --resume outputs/run'. Required for stopping training.",
            },
            "match_mode": {
                "type": "string",
                "enum": ["substring", "regex"],
                "description": "How to interpret training_match. Default substring.",
                "default": "substring",
            },
            "stop_if_running": {
                "type": "boolean",
                "description": "If true, stop matching training processes when found. Default true.",
                "default": True,
            },
            "stop_signal": {
                "type": "string",
                "enum": ["TERM", "INT", "KILL"],
                "description": "Signal to send first. Default TERM.",
                "default": "TERM",
            },
            "grace_seconds": {
                "type": "integer",
                "description": "Seconds to wait after stop_signal before SIGKILL. Default 30.",
            },
            "log_path": {
                "type": "string",
                "description": "Optional path for scheduler logs.",
            },
        },
    },
}


def get_local_scheduler_tool() -> ToolSpec:
    """Return local scheduler / training watchdog tool spec."""
    from agent.core.tools import ToolSpec

    return ToolSpec(
        name=_LOCAL_SCHEDULER_SPEC["name"],
        description=_LOCAL_SCHEDULER_SPEC["description"],
        parameters=_LOCAL_SCHEDULER_SPEC["parameters"],
        handler=_local_scheduler_handler,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a ml-intern local scheduled task")
    parser.add_argument("--run", type=Path, required=True, help="Path to task config JSON")
    args = parser.parse_args()
    return _runner(args.run)


if __name__ == "__main__":
    raise SystemExit(main())
