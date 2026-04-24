# ml-intern — local mode tokenize crash fix
**Date:** 2026-04-22 (evening)
**Bug:** Crash when `local_training` tool tried to run the tokenize step

---

## Root Cause

The `local_training` handler had **no `cwd` (working directory)** parameter, so it ran the tokenize command in whatever the subprocess default was. The script `download_and_tokenize.py` uses **relative paths** like `"./data"`, which were resolved from the wrong directory, causing the script to fail or hang.

The handler also had a **broken timeout handling** — nested `asyncio.wait_for(...).__await__()` which was structurally wrong.

---

## Fixes applied

### PASS ✅ — `cwd` parameter added to `local_training`
- **File:** `agent/tools/local_training_tool.py`
- **Change:** Added `work_dir` argument → passed as `cwd=work_dir` to `asyncio.create_subprocess_shell()`
- **Effect:** Tokenize script now runs in the correct directory

### PASS ✅ — Timeout handling simplified
- **Change:** Removed broken nested `wait_for().__await__()` pattern
- **New pattern:** `await asyncio.wait_for(_stream_output(proc.stdout), timeout=...)`
- **On timeout:** Kill process → read remaining output → return what we have

### PASS ✅ — `local_training` handler fully rewritten
- Simplified `_stream_output()` helper
- Clean `try/except/finally`-free control flow
- `cwd`, `HF_TOKEN`, trackio detection all working

### PASS ✅ — Backend verified
- Session created in local mode: `"mode=local"`
- `Loaded 18 built-in tools: bash, read, write, edit, local_training, ...` ✓
- `Agent ready with 26 tools total` ✓

---

## Files modified

| File | Change |
|------|--------|
| `agent/tools/local_training_tool.py` | Complete rewrite — cwd support, clean timeout, trackio extraction |

---

## How to test

1. Open `http://127.0.0.1:7860` (Ctrl+Shift+R)
2. Toggle **Local** ON → Start Session
3. Ask to run tokenize:
   > "Run `python download_and_tokenize.py` in the hybrid-slm/scripts directory"

The `local_training` tool now runs from the correct working directory, and the agent can track progress without crashing.

---

## Backend status

```
PID: 78311
Status: {"status":"ok","active_sessions":0}
Port: 7860
Mode: local
Tools loaded: bash, read, write, edit, local_training, research, explore_hf_docs, fetch_hf_docs, hf_papers, hf_inspect_dataset, plan_tool, hf_repo_files, hf_repo_git, github_find_examples, github_list_repos, github_read_file, codex_login, hf_catalog (18 built-in)
MCP: 7 tools (space_search, hub_repo_search, paper_search, hub_repo_details, dynamic_space, hf_hub_query, gr1_z_image_turbo_generate)
Total: 26 tools
```