#!/usr/bin/env bash
# Improved LeWM training with anti-overfitting measures
# Mind2Web all splits (47 parquet, ~3K trajectories)
set -e

PROJECT_DIR="/home/jgbla/repos/ml-intern/lewm-computer-use"
cd "$PROJECT_DIR"

RUN_NAME="mw_improved_$(date +%Y%m%d_%H%M)"
OUT_DIR="outputs/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  LeWM Improved Training (Mind2Web all splits)"
echo "  Run: $RUN_NAME"
echo "  $(date)"
echo "============================================"
echo ""
echo "Changes from previous run:"
echo "  - Dataset: all 47 parquet (train+test) → ~3K trajectories"
echo "  - weight_decay: 1e-2 (was 1e-3, 10× stronger)"
echo "  - dropout: 0.15 (was 0.1)"
echo "  - embed_dim: 256 (was 192)"
echo ""

nohup python3 -u scripts/train_lewm.py \
    --data data/mind2web_all.h5 \
    --epochs 100 --lr 5e-5 --weight-decay 1e-2 \
    --batch-size 4 --grad-accum 2 \
    --img-size 128 --ctx-len 3 --embed-dim 256 \
    --encoder-scale tiny --dropout 0.15 \
    --output-dir "$OUT_DIR" \
    > "$OUT_DIR/train.log" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Log: tail -f $OUT_DIR/train.log"
echo ""
echo "============================================"
echo "After training:"
echo "  python3 scripts/eval_lewm.py --checkpoint $OUT_DIR/best_model.pt --data data/mind2web_all.h5"
echo "============================================"
