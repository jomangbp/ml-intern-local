# ml-intern local mode — HF Spaces prevention changelog
**Date:** 2026-04-22
**Goal:** Fix ml-intern local mode to run training/experiments on the local machine without creating HF Spaces

---

## PASS ✅

### 1. Removed `hf_jobs` from local mode tools
- **File:** `agent/core/tools.py`
- **Change:** In `local_mode`, filter out `hf_jobs` from available tools before prepending local tools
- **Effect:** Agent no longer has access to HF Jobs launch in local sessions
- **Verified:** "Loaded 18 built-in tools: bash, read, write, edit, local_training, ..." (no hf_jobs)

### 2. Updated system prompt for local mode
- **File:** `agent/context_manager/manager.py`
- **Change:** Extended the local mode prompt section to:
  - State that `hf_jobs` is NOT available
  - Instruct the agent to run training via local bash commands
  - Tell the agent to install/use trackio locally for experiment tracking
  - Return local log paths and dashboard URLs instead of HF Space links

### 3. README updated
- **File:** `README.md`
- **Change:** Clarified that `local` mode disables `hf_jobs` so training runs locally

### 4. Added `local_training` tool
- **File:** `agent/tools/local_training_tool.py` (new file)
- **Purpose:** Drop-in for `hf_jobs` in local mode — runs training on the host machine
- **Features:**
  - Streams stdout/stderr from training command
  - Extracts local Trackio dashboard URL from output
  - Detects training start indicators
  - Handles long-running jobs with background (`nohup`) patterns
  - Accepts `session` parameter (matches handler signature pattern)
  - Passes `HF_TOKEN` from session to training environment
- **Spec name:** `local_training`
- **Params:** `command` (required), `project`, `timeout`

### 5. Registered `local_training` in tool creation
- **File:** `agent/core/tools.py`
- **Change:** In `local_mode`, added `get_local_training_tool()` alongside `get_local_tools()`
- **Result:** 18 tools loaded in local mode (vs 17 in HF mode, no hf_jobs + added local_training)

### 6. Backend restart verified
- Backend started cleanly with no import/syntax errors
- `curl http://127.0.0.1:7860/api/health` → `{"status":"ok","active_sessions":0}`
- Local session created with `"mode=local"` confirmed in logs

### 7. Frontend build synced to static/
- Built and deployed `index-_uCi3GRZ.js` (1.4MB) to `static/`
- Previous React #185 fix (ActivityStatusBar selectors) also included

---

## 2026-04-24 addendum — local scheduler / training watchdog

- **File:** `agent/tools/local_scheduler_tool.py` (new file)
- **Tool name:** `local_scheduler`
- **Purpose:** Let the agent program a cron/loop-style local watchdog from chat.
- **Actions:** `create`, `list`, `status`, `cancel`, `run_once`
- **Training watchdog behavior:** wait a user-configured interval, match training processes by a specific command-line substring/regex, send `TERM`, wait `grace_seconds`, then escalate to `SIGKILL` if needed.
- **Registered in:** `agent/core/tools.py` for local mode sessions.
- **State/logs:** `~/.cache/ml-intern/scheduled_tasks/`
- **Verified:** py_compile, local-mode tool registration, one-shot scheduled command smoke test, process-stop smoke test, frontend production build, and scheduler REST API smoke test.
- **UI:** Added a top-bar clock button that opens a graphical scheduler dialog for creating, running, listing, refreshing, and cancelling prompt cron or watchdog tasks.
- **Slash command:** `/cron [interval in minutes] <prompt to send to agent or llm>` schedules a repeating prompt for the active session.
- **UI live refresh:** Active sessions now periodically hydrate/reconnect while visible, so cron-generated turns show up without manually refreshing the browser.
- **Telegram:** Added optional `TELEGRAM_BOT_TOKEN` long-polling bot integration with `/start`, `/new`, regular prompts, and `/cron [minutes] <prompt>` per chat session.

---

## FAIL ❌

### 1. Backend `nohup + disown` doesn't survive shell exit
- **Issue:** Running `nohup ... & disown` in bash doesn't fully detach from the shell, causing the uvicorn process to be killed when the command completes
- **Workaround:** Use `cd /path && /full/venv/bin/python -m uvicorn ... &` (no `nohup`) — process survives because it's backgrounded in the same shell that stays alive
- **Status:** Backend running on PID 64651 via this method; not a code issue, just execution nuance

---

## Summary of changes

### Modified files (3)
| File | Change |
|------|--------|
| `agent/core/tools.py` | `hf_jobs` removed in local mode; `local_training` added |
| `agent/context_manager/manager.py` | System prompt updated to explain local mode behavior |
| `README.md` | Clarified local mode disables hf_jobs |

### New files (2)
| File | Purpose |
|------|---------|
| `agent/tools/local_training_tool.py` | Local training tool with trackio URL extraction |
| `agent/tools/local_scheduler_tool.py` | Local cron/loop-style scheduled task and training watchdog tool |

### Build artifact
| File | Purpose |
|------|---------|
| `static/assets/index-_uCi3GRZ.js` | Production frontend with React #185 fix + local mode changes |

---

## How local mode works now

```
Local session created
        │
        ▼
create_builtin_tools(local_mode=True)
        │
        ├─ [hf_jobs] ─ ✂️ REMOVED (no HF Jobs/Spaces)
        │
        ├─ get_local_tools()     → bash, read, write, edit
        │
        ├─ get_local_training_tool() → local_training
        │       └─ runs bash command locally
        │       └─ extracts Trackio URL from output
        │       └─ returns local dashboard URL (not HF Space)
        │
        ├─ get_local_scheduler_tool() → local_scheduler
        │       └─ schedules one-shot or repeating local checks
        │       └─ can stop matching training processes after N minutes
        │       └─ supports list/status/cancel for watchdog tasks
        │
        └─ other tools (research, docs, hf_repo_*, github_*, etc.)

System prompt (local mode section):
  "hf_jobs tool is NOT available in local mode"
  "For training: run locally with bash"
  "Use trackio locally: pip install trackio; trackio.init()"
```

---

## Next steps for user

1. **Test at** `http://127.0.0.1:7860` (Ctrl+Shift+R)
   - Toggle Local ON → Start Session
   - Agent should see `local_training` tool, not `hf_jobs`

2. **Training workflow:**
   ```bash
   python your_train.py --config config.yaml
   # or for background:
   nohup python your_train.py --config config.yaml > /tmp/train.log 2>&1 & echo $!
   ```

3. **Local Trackio:**
   ```python
   import trackio
   trackio.init(project="my-experiment")
   # local dashboard starts at http://127.0.0.1:7861
   ```

4. **If you want Trackio running on a different port**, set:
   ```bash
   export TRACKIO_PORT=7861
   ```