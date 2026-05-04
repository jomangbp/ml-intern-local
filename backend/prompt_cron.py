"""In-process prompt cron scheduler for ML Intern sessions.

Persists cron state to disk so crons survive gateway restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

SubmitResult = bool | tuple[bool, str | None] | dict[str, Any]
SubmitPrompt = Callable[[str, str], Awaitable[SubmitResult]]

CRON_STATE_DIR = Path(os.environ.get(
    "ML_INTERN_CRON_DIR",
    str(Path.home() / ".cache" / "ml-intern" / "crons"),
))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_submit_result(result: SubmitResult) -> tuple[bool, str | None]:
    """Normalize submit callback return into (ok, error)."""
    if isinstance(result, bool):
        return result, None
    if isinstance(result, tuple) and len(result) >= 1:
        ok = bool(result[0])
        err = str(result[1]) if len(result) > 1 and result[1] else None
        return ok, err
    if isinstance(result, dict):
        ok = bool(result.get("ok"))
        err = result.get("error")
        return ok, str(err) if err else None
    return bool(result), None


def _persist_cron(task_id: str, config: dict, status: dict) -> None:
    """Save cron state to disk."""
    try:
        CRON_STATE_DIR.mkdir(parents=True, exist_ok=True)
        data = {"config": config, "status": status}
        path = CRON_STATE_DIR / f"{task_id}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("Failed to persist cron %s: %s", task_id, e)


def _delete_cron_file(task_id: str) -> None:
    try:
        (CRON_STATE_DIR / f"{task_id}.json").unlink(missing_ok=True)
    except Exception:
        pass


def _load_persisted_crons() -> list[dict]:
    """Load all persisted cron configs from disk."""
    crons = []
    if not CRON_STATE_DIR.exists():
        return crons
    for path in CRON_STATE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            config = data.get("config", {})
            status = data.get("status", {})
            # Only restore active crons
            if status.get("status") not in ("cancelled", "completed", "failed"):
                crons.append(data)
        except Exception as e:
            logger.warning("Failed to load cron %s: %s", path, e)
    return crons


class PromptCronManager:
    """Schedule prompts to be submitted to an agent session at intervals.

    Crons persist to disk and are restored on gateway restart.
    When a cron fires, it calls the submit_prompt callback which should
    handle submitting to the agent and optionally reporting results.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._asyncio_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._submit_factory: Callable[[str, str, dict], SubmitPrompt] | None = None

    def set_submit_factory(self, factory: Callable[[str, str, dict], SubmitPrompt]) -> None:
        """Set factory that creates submit functions for restored crons.

        Called once at startup by the component that knows how to route
        prompts (e.g. Telegram bot or web UI). Factory signature:
            factory(chat_id_str, session_id, config) -> submit_fn
        """
        self._submit_factory = factory

    async def restore(self) -> int:
        """Restore persisted crons. Returns count of restored crons.

        Must be called after set_submit_factory() and after the event
        loop is running.
        """
        if not self._submit_factory:
            logger.warning("No submit_factory set, cannot restore crons")
            return 0

        crons = _load_persisted_crons()
        restored = 0
        for data in crons:
            config = data["config"]
            status = data.get("status", {})
            task_id = config["task_id"]

            # Extract Telegram chat_id from user_id
            user_id = status.get("user_id", "")
            chat_id = user_id.replace("telegram:", "") if user_id.startswith("telegram:") else ""

            if not chat_id:
                logger.warning("Skipping cron %s: no chat_id", task_id)
                continue

            try:
                # Backfill defaults for older persisted cron records.
                interval_seconds = int(config.get("interval_seconds") or max(60, int(float(config.get("interval_minutes") or 1) * 60)))
                config.setdefault("interval_seconds", interval_seconds)
                config.setdefault("max_consecutive_failures", 5)
                config.setdefault("failure_retry_delay_seconds", max(30, min(300, interval_seconds // 2 or 30)))

                submit_fn = self._submit_factory(chat_id, config.get("session_id", ""), config)
                # Reset run count and status
                status["status"] = "scheduled"
                status["runs_completed"] = status.get("runs_completed", 0)
                status["consecutive_failures"] = int(status.get("consecutive_failures") or 0)
                status["runner_alive"] = True

                async with self._lock:
                    self._tasks[task_id] = dict(status)
                    self._tasks[task_id]["config"] = config
                    self._asyncio_tasks[task_id] = asyncio.create_task(
                        self._runner(task_id, submit_fn),
                        name=f"ml-intern-prompt-cron-{task_id}",
                    )

                _persist_cron(task_id, config, self._tasks[task_id])
                restored += 1
                logger.info("Restored cron %s: every %gm prompt=%s",
                            task_id, config.get("interval_minutes"), config.get("prompt", "")[:50])
            except Exception as e:
                logger.error("Failed to restore cron %s: %s", task_id, e)

        if restored:
            logger.info("Restored %d crons", restored)
        return restored

    async def create(
        self,
        *,
        session_id: str,
        user_id: str,
        interval_minutes: float,
        prompt: str,
        submit_prompt: SubmitPrompt,
        task_name: str | None = None,
        repeat: bool = True,
        max_runs: int = 0,
        run_immediately: bool = False,
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required for prompt cron tasks")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required for prompt cron tasks")
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than 0")
        if max_runs < 0:
            raise ValueError("max_runs must be 0 or greater")

        task_id = f"cron-{uuid.uuid4().hex[:10]}"
        interval_seconds = int(interval_minutes * 60)
        config = {
            "task_id": task_id,
            "task_name": task_name or f"Prompt cron {task_id}",
            "kind": "prompt_cron",
            "session_id": session_id,
            "interval_minutes": interval_minutes,
            "interval_seconds": interval_seconds,
            "prompt": prompt.strip(),
            "repeat": repeat,
            "max_runs": max_runs,
            "run_immediately": run_immediately,
            # Robustness: tolerate transient provider/network hiccups.
            "max_consecutive_failures": 5,
            "failure_retry_delay_seconds": max(30, min(300, interval_seconds // 2 or 30)),
        }
        status = {
            "task_id": task_id,
            "status": "scheduled",
            "created_at": _utc_now(),
            "user_id": user_id,
            "runs_completed": 0,
            "consecutive_failures": 0,
            "runner_alive": True,
            "config": config,
        }

        async with self._lock:
            self._tasks[task_id] = status
            self._asyncio_tasks[task_id] = asyncio.create_task(
                self._runner(task_id, submit_prompt), name=f"ml-intern-prompt-cron-{task_id}"
            )

        _persist_cron(task_id, config, status)
        return dict(status)

    async def _runner(self, task_id: str, submit_prompt: SubmitPrompt) -> None:
        _logger = logging.getLogger(f"prompt_cron.{task_id[:8]}")
        try:
            while True:
                async with self._lock:
                    status = self._tasks.get(task_id)
                    if not status or status.get("status") == "cancelled":
                        _logger.info("Runner exiting: cancelled or missing")
                        return
                    config = status["config"]
                    runs_completed = int(status.get("runs_completed") or 0)

                if not (config.get("run_immediately") and runs_completed == 0):
                    interval = max(1, int(config.get("interval_seconds") or 60))
                    _logger.info("Sleeping %ds until next run (run #%d)", interval, runs_completed + 1)
                    await asyncio.sleep(interval)

                async with self._lock:
                    status = self._tasks.get(task_id)
                    if not status or status.get("status") == "cancelled":
                        _logger.info("Runner exiting: cancelled during sleep")
                        return
                    status["status"] = "running"
                    status["last_check_started_at"] = _utc_now()

                _persist_cron(task_id, config, status)

                _logger.info("Executing run #%d: %s", runs_completed + 1, config.get("prompt", "")[:50])
                submit_error: str | None = None
                try:
                    raw_result = await submit_prompt(config["session_id"], config["prompt"])
                    ok, submit_error = _coerce_submit_result(raw_result)
                except Exception as e:
                    ok = False
                    submit_error = str(e)
                    _logger.exception("Run #%d submit raised", runs_completed + 1)

                _logger.info("Run #%d finished ok=%s", runs_completed + 1, ok)
                finished_at = _utc_now()

                retry_delay = 0
                async with self._lock:
                    status = self._tasks.get(task_id)
                    if not status:
                        return
                    runs_completed = int(status.get("runs_completed") or 0) + 1
                    status["runs_completed"] = runs_completed
                    status["last_check_finished_at"] = finished_at
                    status["last_check"] = {
                        "finished_at": finished_at,
                        "prompt_submitted": ok,
                        "prompt": config["prompt"],
                    }

                    if not ok:
                        status["consecutive_failures"] = int(status.get("consecutive_failures") or 0) + 1
                        status["last_error"] = submit_error or "Submit returned False"
                        status["last_error_at"] = finished_at

                        # One-shot tasks fail immediately. Repeating tasks get retries.
                        if not config.get("repeat", True):
                            status["status"] = "failed"
                            status["error"] = status["last_error"]
                            status["finished_at"] = finished_at
                            status["runner_alive"] = False
                            _persist_cron(task_id, config, status)
                            return

                        max_failures = int(config.get("max_consecutive_failures") or 5)
                        if int(status["consecutive_failures"]) >= max_failures:
                            status["status"] = "failed"
                            status["error"] = (
                                f"{status['last_error']} (consecutive failures: {status['consecutive_failures']})"
                            )
                            status["finished_at"] = finished_at
                            status["runner_alive"] = False
                            _persist_cron(task_id, config, status)
                            return

                        status["status"] = "scheduled"
                        _persist_cron(task_id, config, status)
                        retry_delay = int(config.get("failure_retry_delay_seconds") or 30)
                        _logger.warning(
                            "Run #%d failed (%s). consecutive_failures=%d/%d; retrying in %ds",
                            runs_completed,
                            status.get("last_error", "unknown"),
                            status["consecutive_failures"],
                            max_failures,
                            retry_delay,
                        )
                    else:
                        # Success path
                        status["consecutive_failures"] = 0
                        status.pop("last_error", None)
                        status.pop("last_error_at", None)
                        status.pop("error", None)

                        if not config.get("repeat", True):
                            status["status"] = "completed"
                            status["finished_at"] = finished_at
                            status["runner_alive"] = False
                            _persist_cron(task_id, config, status)
                            return
                        max_runs = int(config.get("max_runs") or 0)
                        if max_runs > 0 and runs_completed >= max_runs:
                            status["status"] = "completed"
                            status["finished_at"] = finished_at
                            status["runner_alive"] = False
                            _persist_cron(task_id, config, status)
                            return
                        status["status"] = "scheduled"
                        _persist_cron(task_id, config, status)

                if retry_delay > 0:
                    await asyncio.sleep(max(1, retry_delay))
                    continue

        except asyncio.CancelledError:
            async with self._lock:
                status = self._tasks.get(task_id)
                if status:
                    status["status"] = "cancelled"
                    status["cancelled_at"] = _utc_now()
                    status["runner_alive"] = False
                    _persist_cron(task_id, status.get("config", {}), status)
            raise
        except Exception as e:
            _logger.exception("Runner crashed")
            async with self._lock:
                status = self._tasks.get(task_id)
                if status:
                    status["status"] = "failed"
                    status["error"] = str(e)
                    status["finished_at"] = _utc_now()
                    status["runner_alive"] = False
                    _persist_cron(task_id, status.get("config", {}), status)

    async def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            tasks = []
            for task_id, status in self._tasks.items():
                if user_id not in (None, "dev") and status.get("user_id") not in (None, user_id):
                    continue
                item = dict(status)
                task = self._asyncio_tasks.get(task_id)
                item["runner_alive"] = bool(task and not task.done() and item.get("status") not in {"cancelled", "completed", "failed"})
                tasks.append(item)
        return sorted(tasks, key=lambda s: s.get("created_at", ""), reverse=True)

    async def get(self, task_id: str) -> dict[str, Any] | None:
        async with self._lock:
            status = self._tasks.get(task_id)
            if not status:
                return None
            item = dict(status)
            task = self._asyncio_tasks.get(task_id)
            item["runner_alive"] = bool(task and not task.done() and item.get("status") not in {"cancelled", "completed", "failed"})
            return item

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            status = self._tasks.get(task_id)
            if not status:
                return False
            status["status"] = "cancelled"
            status["cancelled_at"] = _utc_now()
            status["runner_alive"] = False
            config = status.get("config", {})
            task = self._asyncio_tasks.get(task_id)
            if task and not task.done():
                task.cancel()
        _delete_cron_file(task_id)
        return True

    async def delete(self, task_id: str) -> bool:
        """Permanently delete a cron — cancel it first, then remove from memory."""
        await self.cancel(task_id)
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
            self._asyncio_tasks.pop(task_id, None)
        _delete_cron_file(task_id)
        return True


prompt_cron_manager = PromptCronManager()
