#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
LOG_FILE="${LOG_FILE:-/tmp/ml-intern-ui.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "❌ Python venv not found: $PYTHON_BIN"
  echo "Create it first: cd $ROOT_DIR && python3 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

# Use HF CLI cached token if HF_TOKEN is not already set.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi

# Stop previous backend instances so we always get a clean restart.
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

cd "$BACKEND_DIR"

if [[ "${1:-}" == "--bg" ]]; then
  nohup "$PYTHON_BIN" -m uvicorn main:app --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "✅ ml-intern started in background"
  echo "   PID: $PID"
  echo "   URL: http://127.0.0.1:$PORT"
  echo "   Log: $LOG_FILE"
else
  echo "▶️  Starting ml-intern on http://127.0.0.1:$PORT"
  exec "$PYTHON_BIN" -m uvicorn main:app --host "$HOST" --port "$PORT"
fi
