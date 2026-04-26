"""
Backfill baseline (v1-hybrid-slm-baseline) training log into Trackio
for comparison with the new Gemma 4 run.
"""
import re
import sys
from pathlib import Path

import trackio

LOG_FILE = Path(__file__).parent.parent / "outputs" / "v1-hybrid-slm-baseline" / "training_log.txt"

def parse_log():
    """Parse the baseline training log into structured data."""
    entries = []
    current_step = None
    
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            # Training step line
            m = re.match(
                r"step=([\d,]+)\s*\|\s*loss=([\d.]+)\s*\|\s*lr=([\de.-]+)\s*\|\s*tokens/s=([\d,]+)\s*\|\s*epoch=(\d+)\s*\|\s*mem=([\d.]+)GB",
                line,
            )
            if m:
                step = int(m.group(1).replace(",", ""))
                entry = {
                    "step": step,
                    "train/loss": float(m.group(2)),
                    "train/learning_rate": float(m.group(3)),
                    "train/tokens_per_sec": int(m.group(4).replace(",", "")),
                    "train/epoch": int(m.group(5)),
                    "train/gpu_memory_gb": float(m.group(6)),
                }
                entries.append(entry)
                current_step = step
                continue
            
            # Val line (after eval)
            m = re.match(r"\s*val_loss=([\d.]+)\s*\|\s*val_ppl=([\d.]+)", line)
            if m and current_step is not None:
                entries[-1]["val/loss"] = float(m.group(1))
                entries[-1]["val/perplexity"] = float(m.group(2))
    
    return entries


def main():
    entries = parse_log()
    print(f"Parsed {len(entries)} log entries from baseline run")
    print(f"  Step range: {entries[0]['step']} → {entries[-1]['step']}")
    
    if entries[-1].get("val/loss"):
        print(f"  Final val_loss: {entries[-1]['val/loss']:.4f}")
        print(f"  Final val_ppl: {entries[-1]['val/perplexity']:.2f}")
    
    # Log to trackio
    run = trackio.init(
        project="hybrid-slm",
        name="v1-hybrid-slm-baseline",
        config={
            "model": "hybrid-slm-baseline",
            "architecture": "6x GatedDeltaNet + 2x FullAttention",
            "learning_rate": 4e-4,
            "min_learning_rate": 4e-5,
            "warmup_steps": 2000,
            "max_steps": 5900,
            "batch_size": 3,
            "grad_accum": 2,
            "seq_length": 1024,
            "hidden_size": 512,
            "num_layers": 8,
            "activation": "silu",
            "logit_softcapping": "none",
            "sandwich_norm": False,
            "qk_norm": False,
            "dataset": "TinyStories (25M tokens)",
        },
    )
    
    for entry in entries:
        step = entry.pop("step")
        run.log(entry, step=step)
    
    trackio.finish()
    print(f"\n✓ Backfilled {len(entries)} entries to trackio project 'hybrid-slm', run 'v1-hybrid-slm-baseline'")


if __name__ == "__main__":
    main()
