# Tokenizer stability fix for 6GB RAM (WSL)
Date: 2026-04-22

## Problem
Tokenization was crashing hard (and sometimes took down the whole workflow). In this environment:
- WSL usable RAM ~6GB for this task
- `datasets` streaming iterator was aborting Python at process exit (`terminate called without an active exception`)
- Previous tokenizer script also loaded massive data structures in memory

---

## Root causes found

1. **High-memory tokenization design**
   - Old script loaded full dataset splits into memory (`list`, JSON dumps, giant token lists)
   - Not safe for 6GB constrained RAM

2. **Environment-specific crash in HF datasets streaming**
   - `load_dataset(..., streaming=True)` reliably aborted interpreter on exit in this setup

3. **local_training tool masked command failures**
   - Returned success even when command exited non-zero

4. **Training dataset reader mismatch**
   - Training script expected `np.load(...)`, but tokenizer output is `.bin`

---

## Changes made

### 1) Rewrote tokenizer pipeline for low memory
**File:** `backend/hybrid-slm/scripts/download_and_tokenize.py`

- Replaced in-memory flow with **incremental chunked tokenization**
- Removed `streaming=True` path (unstable here)
- Uses dataset split slicing chunks: `split[start:end]`
- Writes tokens directly to `.bin` using small buffer (`array('I')`)
- Added crash-safe checkpoint resume:
  - `data/tokenize_checkpoint.json`
- Added RAM profiles:
  - `--profile 6gb` (default)
  - `--profile 12gb`
  - `--profile full`
- Added `--resume / --no-resume`
- Added low-memory thread env defaults:
  - `TOKENIZERS_PARALLELISM=false`
  - `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`

### 2) Fixed local training tool reliability
**File:** `agent/tools/local_training_tool.py`

- Added `work_dir` support (`cwd=...`) so tokenization runs in correct folder
- Fixed timeout/output flow
- Now returns **failure** on non-zero exit code
- Includes exit code in output when failed

### 3) Fixed token dataset loader compatibility
**File:** `backend/hybrid-slm/scripts/train.py`

- `TokenizedDataset` now supports `.bin` via `np.memmap(dtype=np.uint32)`
- Uses shifted next-token labels instead of zero labels
- Memory-mapped loading to keep RAM low

---

## PASS / FAIL log

### PASS ✅
1. `download_and_tokenize.py` compiles and runs in low-memory mode
2. No streaming-abort crash after switching to chunked split loading
3. Resume works (`--resume` continues from checkpoint)
4. local_training now reports command failures correctly
5. Backend healthy at `http://127.0.0.1:7860/api/health`

### FAIL ❌ (before fix)
1. Streaming mode (`streaming=True`) aborted interpreter on exit
2. Old tokenizer design consumed too much memory for 6GB budget
3. local_training used to report success for failing commands

---

## Recommended command for your machine (6GB budget)
Run from: `/home/jgbla/repos/ml-intern/backend/hybrid-slm`

```bash
/home/jgbla/repos/ml-intern/.venv/bin/python scripts/download_and_tokenize.py \
  --profile 6gb \
  --output-dir data \
  --resume
```

If you want smaller first run (safer sanity pass):
```bash
/home/jgbla/repos/ml-intern/.venv/bin/python scripts/download_and_tokenize.py \
  --profile 6gb \
  --train-doc-limit 50000 \
  --val-doc-limit 5000 \
  --output-dir data \
  --resume
```

---

## Notes
- This is configured for stability first (not max throughput)
- If WSL still gets OOM pressure, reduce `--train-doc-limit` and run in stages with `--resume`
- Checkpoint is at `data/tokenize_checkpoint.json`

---

## Live run started

A background tokenize run was started with the 6GB profile:

```bash
cd /home/jgbla/repos/ml-intern/backend/hybrid-slm
nohup /home/jgbla/repos/ml-intern/.venv/bin/python scripts/download_and_tokenize.py \
  --profile 6gb \
  --output-dir data \
  --resume > /tmp/tokenize.log 2>&1 &
```

Current Python PID: `104633`
