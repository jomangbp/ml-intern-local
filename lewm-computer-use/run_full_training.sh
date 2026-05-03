#!/usr/bin/env bash
# Full-scale LeWM training launch: CLI + Mind2Web → combined HDF5 → background training
# Safe for WSL: low RAM, no display deps, backgrounded

set -e
PROJECT_DIR="/home/jgbla/repos/ml-intern/lewm-computer-use"
cd "$PROJECT_DIR"

echo "============================================"
echo "  LeWM Full-Scale Training (CLI + Web)"
echo "  $(date)"
echo "============================================"
echo ""

# ── Step 1: Generate CLI dataset (pyte, headless, safe) ──
CLI_H5="data/cli_train.h5"
if [ ! -f "$CLI_H5" ]; then
    echo "[1/4] Generating CLI dataset (500 commands, pyte headless)..."
    python3 scripts/build_cli_dataset.py \
        --synthetic-only --num-commands 500 --img-size 128 \
        --output "$CLI_H5" 2>&1
    echo "  CLI done: $(du -h "$CLI_H5" | cut -f1)"
else
    echo "[1/4] CLI dataset exists, skipping ($(du -h "$CLI_H5" | cut -f1))"
fi

# ── Step 2: Convert Mind2Web (need full set) ──
MW_H5="data/mind2web_full.h5"
if [ ! -f "$MW_H5" ]; then
    echo "[2/4] Converting Mind2Web (all 27 parquet files)..."
    python3 scripts/convert_mind2web.py \
        --all --max-trajs 500 --img-size 128 \
        --output "$MW_H5" 2>&1
    echo "  Mind2Web done: $(du -h "$MW_H5" | cut -f1)"
else
    echo "[2/4] Mind2Web HDF5 exists, skipping ($(du -h "$MW_H5" | cut -f1))"
fi

# ── Step 3: Merge ──
COMBINED_H5="data/combined_train.h5"
echo "[3/4] Merging CLI + Mind2Web into combined dataset..."
python3 scripts/merge_datasets.py \
    "$MW_H5" "$CLI_H5" \
    --output "$COMBINED_H5" --img-size 128 2>&1
echo "  Combined: $(du -h "$COMBINED_H5" | cut -f1)"

# ── Step 4: Launch training in background ──
echo "[4/4] Launching training in background..."
RUN_NAME="combined_run_$(date +%Y%m%d_%H%M)"
OUT_DIR="outputs/$RUN_NAME"
mkdir -p "$OUT_DIR"

nohup python3 -u scripts/train_lewm.py \
    --data "$COMBINED_H5" \
    --epochs 100 --lr 5e-5 --batch-size 4 --grad-accum 2 \
    --img-size 128 --ctx-len 3 --encoder-scale tiny \
    --output-dir "$OUT_DIR" \
    > "$OUT_DIR/train.log" 2>&1 &

TRAIN_PID=$!
echo "  PID: $TRAIN_PID"
echo "  Log: $OUT_DIR/train.log"
echo "  Output: $OUT_DIR/"
echo ""
echo "============================================"
echo "  Training launched!"
echo "  Monitor: tail -f $OUT_DIR/train.log"
echo "  Check:   kill -0 $TRAIN_PID 2>/dev/null && echo 'running' || echo 'done'"
echo "============================================"
echo ""
echo "After training completes, evaluate with:"
echo "  python3 scripts/eval_lewm.py --checkpoint $OUT_DIR/best_model.pt --data $COMBINED_H5"
