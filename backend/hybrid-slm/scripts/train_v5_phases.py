#!/usr/bin/env python3
"""
Training Script v5: Phase-Based Hybrid5 SLM Training
Optimized for 6GB VRAM (RTX 3060)

Phase-based training inspired by:
  - IMU-1 (arXiv:2602.02522): 3-stage data curriculum
  - Nanbeige4 (arXiv:2512.06266): Fine-grained WSD with progressive quality
  - Xmodel-2.5 (arXiv:2511.19496): Optimizer/data switching in decay
  - D2Z (arXiv:2502.15938): Decay-to-zero for better convergence

Phase Design for ~2.8B tokens, 86M params:
  Phase 1 (steps 0–5000):    Warmup + Early Stable
    - Full SYNTH+Hermes mix, streaming
    - LR: linear warmup 1e-5 → 4e-4 over 500 steps, then stable at 4e-4
    - Heavy checkpoint: every 1000 steps + final
    - Evaluate every 500 steps

  Phase 2 (steps 5000–10000): Mid Stable
    - Resume from Phase 1 final checkpoint
    - Same data, same LR (4e-4 stable)
    - Checkpoint every 1000 steps + final
    - Evaluate every 500 steps

  Phase 3 (steps 10000–16000): Late Stable
    - Resume from Phase 2 final checkpoint
    - Same data, same LR (4e-4 stable)
    - Checkpoint every 2000 steps + final
    - Evaluate every 1000 steps

  Phase 4 (steps 16000–20000): Decay
    - Resume from Phase 3 final checkpoint
    - 1-sqrt(t) LR decay: 4e-4 → 1e-5 (the "river valley" where loss drops)
    - More frequent evals (every 500 steps)
    - Checkpoint every 500 steps (capture the convergence curve)
    - Apply checkpoint EMA at the end

Each phase is a separate script invocation:
  python3 scripts/train_v5_phases.py --phase 1
  python3 scripts/train_v5_phases.py --phase 2
  ...etc

The LR schedule is CONTINUOUS across phases — each phase knows where it is
in the global schedule and sets the scheduler accordingly.

GPU: BS=2, grad_accum=6, seq=1024 → 12,288 tokens/step, ~95% VRAM usage
"""

import os
import sys
import math
import time
import gc
import json
import argparse
import copy
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

import trackio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemma4v2.hybrid5_config import Hybrid5Config
from src.gemma4v2.hybrid5_model import Hybrid5Model


# ── Phase Definitions ──────────────────────────────────────────
# Global schedule: total 20K steps, warmup 500, decay starts at 16K
GLOBAL_TOTAL_STEPS = 20000
GLOBAL_WARMUP_STEPS = 500
GLOBAL_DECAY_START = 16000  # 80% of total
GLOBAL_PEAK_LR = 4e-4
GLOBAL_MIN_LR = 1e-5

PHASES = {
    1: {
        "name": "Phase 1: Warmup + Early Stable",
        "start_step": 0,
        "end_step": 5000,
        "save_steps": 1000,
        "eval_steps": 500,
        "description": "Linear warmup (0→500) then early stable (500→5000)",
    },
    2: {
        "name": "Phase 2: Mid Stable",
        "start_step": 5000,
        "end_step": 10000,
        "save_steps": 1000,
        "eval_steps": 500,
        "description": "Stable LR training, model deepening",
    },
    3: {
        "name": "Phase 3: Late Stable",
        "start_step": 10000,
        "end_step": 16000,
        "save_steps": 2000,
        "eval_steps": 1000,
        "description": "Late stable, preparing for decay",
    },
    4: {
        "name": "Phase 4: LR Decay (River Valley)",
        "start_step": 16000,
        "end_step": 20000,
        "save_steps": 500,
        "eval_steps": 250,
        "description": "1-sqrt(t) decay — where the magic happens",
    },
}


# ── Data Utilities ─────────────────────────────────────────────
class TokenizedDataset(Dataset):
    """Memory-efficient token dataset for validation"""
    def __init__(self, data_path: str, max_length: int = 1024, stride: int = 256):
        self.max_length = max_length
        self.stride = stride
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tokenized data not found at {data_path}")
        self.ids = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.size = max(0, len(self.ids) - (max_length + 1))

    def __len__(self):
        return max(0, self.size // self.stride)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_length + 1
        window = self.ids[start:end].astype(np.int64)
        return {
            "input_ids": torch.from_numpy(window[:-1]),
            "labels": torch.from_numpy(window[1:]),
        }


class RandomTokenDataset:
    """Fast random-access dataset using memmap + random slicing.
    
    Much faster than DataLoader + IterableDataset for large binary files.
    Avoids multiprocessing overhead and worker deadlocks on 11GB memmaps.
    Simply picks random start positions and returns a window of tokens.
    """
    def __init__(self, data_path: str, max_length: int = 1024, seed: int = 42):
        self.max_length = max_length
        self.rng = np.random.RandomState(seed)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tokenized data not found at {data_path}")
        self.ids = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.num_positions = max(0, len(self.ids) - (max_length + 1))
        print(f"    RandomTokenDataset: {len(self.ids):,} tokens, {self.num_positions:,} positions")

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Returns a random batch of shape [batch_size, max_length]."""
        positions = self.rng.randint(0, self.num_positions, size=batch_size)
        input_ids = np.empty((batch_size, self.max_length), dtype=np.int64)
        labels = np.empty((batch_size, self.max_length), dtype=np.int64)
        for i, pos in enumerate(positions):
            window = self.ids[pos:pos + self.max_length + 1]
            input_ids[i] = window[:-1]
            labels[i] = window[1:]
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }


# ── Training State ───────────────────────────────────────────────
@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    num_tokens: int = 0
    best_val_loss: float = float('inf')
    best_step: int = 0


# ── WSD Scheduler (continuous across phases) ──────────────────
class WSDScheduler:
    """Warmup-Stable-Decay with 1-sqrt(t) decay.
    
    Initialized with GLOBAL parameters — always knows where it is
    in the full 20K step schedule regardless of which phase is running.
    """
    def __init__(self, optimizer, last_step: int = 0):
        self.optimizer = optimizer
        self.last_step = last_step

    def step(self):
        self.last_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_lr(self) -> float:
        step = self.last_step
        if step < GLOBAL_WARMUP_STEPS:
            # Linear warmup
            return GLOBAL_MIN_LR + (GLOBAL_PEAK_LR - GLOBAL_MIN_LR) * (step / GLOBAL_WARMUP_STEPS)
        elif step < GLOBAL_DECAY_START:
            # Stable phase
            return GLOBAL_PEAK_LR
        else:
            # 1-sqrt(t) decay
            decay_progress = (step - GLOBAL_DECAY_START) / (GLOBAL_TOTAL_STEPS - GLOBAL_DECAY_START)
            decay_progress = min(decay_progress, 1.0)
            decay_factor = 1.0 - math.sqrt(decay_progress)
            return max(GLOBAL_PEAK_LR * decay_factor, GLOBAL_MIN_LR)

    def get_phase(self) -> str:
        if self.last_step < GLOBAL_WARMUP_STEPS:
            return "warmup"
        elif self.last_step < GLOBAL_DECAY_START:
            return "stable"
        else:
            return "decay"

    def state_dict(self):
        return {"last_step": self.last_step}

    def load_state_dict(self, d):
        self.last_step = d["last_step"]


# ── Z-Loss ─────────────────────────────────────────────────────
def compute_z_loss(logits: torch.Tensor, weight: float = 1e-4) -> torch.Tensor:
    """Z-loss for numerical stability. Computed on a subsample for efficiency."""
    # Only compute on last token position to avoid huge memory/compute
    # (the full-sequence version is too expensive for 100K vocab)
    last_logits = logits[:, -1, :]  # [B, V]
    log_z = torch.logsumexp(last_logits, dim=-1)  # [B]
    return weight * (log_z ** 2).mean()


# ── Checkpoint EMA ─────────────────────────────────────────────
def checkpoint_ema(model, checkpoints: List[str], beta: float = 0.8):
    if not checkpoints:
        return
    print(f"\nCheckpoint EMA (β={beta}) over {len(checkpoints)} checkpoints:")
    state_dicts = []
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        state_dicts.append(ckpt["model_state_dict"])
        print(f"  Loaded: {path}")
    n = len(state_dicts)
    weights = [beta ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    averaged_state = {}
    for key in state_dicts[0].keys():
        averaged_state[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
    model.load_state_dict(averaged_state)
    print("  ✓ Checkpoint EMA applied")


# ── Save / Load ────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, state, phase, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "state": {
            "step": state.step, "epoch": state.epoch,
            "total_loss": state.total_loss, "num_tokens": state.num_tokens,
            "best_val_loss": state.best_val_loss, "best_step": state.best_step,
        },
        "phase": phase,
        "global_schedule": {
            "total_steps": GLOBAL_TOTAL_STEPS,
            "warmup_steps": GLOBAL_WARMUP_STEPS,
            "decay_start": GLOBAL_DECAY_START,
            "peak_lr": GLOBAL_PEAK_LR,
            "min_lr": GLOBAL_MIN_LR,
        },
    }, path)


def load_checkpoint(model, optimizer, path):
    if not path or not os.path.exists(path):
        return None, TrainingState(), {}
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    state = TrainingState(
        step=ckpt["state"]["step"],
        epoch=ckpt["state"]["epoch"],
        total_loss=ckpt["state"]["total_loss"],
        num_tokens=ckpt["state"]["num_tokens"],
        best_val_loss=ckpt["state"].get("best_val_loss", float('inf')),
        best_step=ckpt["state"].get("best_step", 0),
    )
    print(f"  Resumed from step {state.step} (best val: {state.best_val_loss:.4f} @ step {state.best_step})")
    return ckpt, state, ckpt.get("scheduler_state", {})


# ── Evaluation ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs['loss'].item() * input_ids.numel()
        total_tokens += input_ids.numel()
        if total_tokens >= max_batches * 1024:
            break
    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase-based Hybrid5 training")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Which phase to run (1-4)")
    parser.add_argument("--train-data", default="data/combined_synth_hermes_train_tokens.bin")
    parser.add_argument("--val-data", default="data/combined_synth_hermes_val_tokens.bin")
    parser.add_argument("--output-dir", default="outputs/v5-phased")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--z-loss-weight", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Explicit checkpoint to resume from (overrides phase default)")
    args = parser.parse_args()

    phase_cfg = PHASES[args.phase]
    start_step = phase_cfg["start_step"]
    end_step = phase_cfg["end_step"]
    phase_steps = end_step - start_step

    # Determine resume checkpoint
    resume_path = args.resume
    if resume_path is None and start_step > 0:
        # Auto-find previous phase's final checkpoint
        prev_phase = args.phase - 1
        auto_path = os.path.join(args.output_dir, f"phase{prev_phase}", "final", "model.pt")
        if os.path.exists(auto_path):
            resume_path = auto_path
            print(f"  Auto-resuming from: {resume_path}")
        else:
            print(f"  WARNING: No checkpoint found at {auto_path}")
            print(f"  Starting from scratch (this will hurt training!)")

    phase_dir = os.path.join(args.output_dir, f"phase{args.phase}")
    os.makedirs(phase_dir, exist_ok=True)

    # Model
    model_config = Hybrid5Config()
    model = Hybrid5Model(model_config)
    num_params = sum(p.numel() for p in model.parameters())

    model = model.cuda()
    model.enable_gradient_checkpointing()
    model._move_rotary_to_device('cuda')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=GLOBAL_PEAK_LR, weight_decay=0.1, betas=(0.9, 0.95))

    # Load checkpoint
    state = TrainingState()
    scheduler_state = {}
    if resume_path:
        ckpt, state, scheduler_state = load_checkpoint(model, optimizer, resume_path)
        if state.step < start_step:
            print(f"  WARNING: checkpoint is at step {state.step}, expected >= {start_step}")

    # Scheduler — always initialized with global knowledge
    scheduler = WSDScheduler(optimizer, last_step=state.step)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    scaler = GradScaler('cuda')

    # Verify data
    for path in [args.train_data, args.val_data]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found: {path}")
        tokens = os.path.getsize(path) // 4
        print(f"  {os.path.basename(path)}: {os.path.getsize(path)/1e9:.2f} GB ({tokens:,} tokens)")

    # Trackio
    run_name = f"v5-phase{args.phase}"
    trackio.init(
        project="hybrid-slm",
        name=run_name,
        resume="allow",
        config={
            "model": "v5-phased-hybrid5",
            "phase": args.phase,
            "phase_name": phase_cfg["name"],
            "steps": f"{start_step}→{end_step}",
            "dataset": "SYNTH + Hermes (~2.8B tokens)",
            "global_total_steps": GLOBAL_TOTAL_STEPS,
            "global_warmup": GLOBAL_WARMUP_STEPS,
            "global_decay_start": GLOBAL_DECAY_START,
            "peak_lr": GLOBAL_PEAK_LR,
            "min_lr": GLOBAL_MIN_LR,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum * args.seq_length,
            "seq_length": args.seq_length,
            "model_params": num_params,
        },
    )

    # Print phase header
    print("\n" + "=" * 70)
    print(f"  {phase_cfg['name']}")
    print("=" * 70)
    print(f"  Phase:        {args.phase}/4")
    print(f"  Steps:        {start_step:,} → {end_step:,} ({phase_steps:,} steps)")
    print(f"  Description:  {phase_cfg['description']}")
    print(f"  Save every:   {phase_cfg['save_steps']} steps")
    print(f"  Eval every:   {phase_cfg['eval_steps']} steps")
    print(f"  Global LR:    {scheduler.get_lr():.2e} ({scheduler.get_phase()})")
    print(f"  Batch:        {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} (×{args.seq_length} seq = {args.batch_size * args.grad_accum * args.seq_length:,} tokens/step)")
    print(f"  Resume from:  {resume_path or 'scratch'}")
    print(f"  Output:       {phase_dir}")

    # Print global schedule context
    print(f"\n  ── Global Schedule Context ──")
    for p_id, p_cfg in PHASES.items():
        marker = " ◀ YOU ARE HERE" if p_id == args.phase else ""
        s, e = p_cfg["start_step"], p_cfg["end_step"]
        if s < GLOBAL_WARMUP_STEPS:
            phase_type = "warmup+stable"
        elif e <= GLOBAL_DECAY_START:
            phase_type = "stable"
        elif s >= GLOBAL_DECAY_START:
            phase_type = "decay"
        else:
            phase_type = "stable→decay"
        print(f"    Phase {p_id}: steps {s:>5,}–{e:>5,} ({phase_type}){marker}")

    # Datasets
    print(f"\n  Loading data...")
    train_dataset = RandomTokenDataset(
        data_path=args.train_data, max_length=args.seq_length, seed=42,
    )
    val_dataset = TokenizedDataset(
        data_path=args.val_data, max_length=args.seq_length, stride=256,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    print(f"  Train positions: {train_dataset.num_positions:,}")
    print(f"  Val windows:     {len(val_dataset):,}")

    # Quick test
    print("\n  Testing forward pass...")
    test_ids = torch.randint(0, model_config.vocab_size, (1, 128)).cuda()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=test_ids, labels=test_ids)
            print(f"  Loss: {out['loss'].item():.4f}")
    del test_ids, out
    torch.cuda.empty_cache()

    # Training loop
    print("\n" + "=" * 70)
    print("  STARTING PHASE TRAINING")
    print("=" * 70)

    log_file = open(os.path.join(phase_dir, "training_log.txt"), "a")
    start_time = time.time()
    saved_checkpoints = []
    eff_batch = args.batch_size * args.grad_accum * args.seq_length

    for step in range(start_step, end_step):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_z_loss = 0.0
        step_tokens = 0

        for micro_step in range(args.grad_accum):
            batch = train_dataset.get_batch(args.batch_size)
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / args.grad_accum
                if args.z_loss_weight > 0:
                    z_loss = compute_z_loss(outputs['logits'], args.z_loss_weight) / args.grad_accum
                    loss = loss + z_loss
                    step_z_loss += z_loss.item() * args.grad_accum

            scaler.scale(loss).backward()
            step_loss += outputs['loss'].item()
            step_tokens += input_ids.numel()
            del outputs, loss, input_ids, labels

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        phase = scheduler.get_phase()

        state.step = step + 1
        state.total_loss += step_loss / args.grad_accum
        state.num_tokens += step_tokens

        # Logging (every 10 steps)
        if step % 10 == 0:
            avg_loss = state.total_loss / max(10, (step - start_step + 1) // 10 + 1)
            elapsed = time.time() - start_time
            mem = torch.cuda.memory_allocated() / 1e9
            # Correct tokens/s: total tokens processed in this logging window / elapsed
            interval_tokens = 10 * eff_batch
            correct_tps = interval_tokens / elapsed if elapsed > 0 else 0

            phase_icon = {"warmup": "🔄", "stable": "⚡", "decay": "📉"}.get(phase, "?")
            log_line = (f"step={step} | loss={avg_loss:.4f} | z_loss={step_z_loss:.6f} | "
                       f"lr={lr:.2e} | phase={phase} {phase_icon} | "
                       f"tok/s={correct_tps:.0f} | epoch={state.epoch} | "
                       f"mem={mem:.2f}GB | elapsed={elapsed:.0f}s")
            print(log_line, flush=True)
            log_file.write(log_line + "\n")
            log_file.flush()

            trackio.log({
                "loss": avg_loss, "z_loss": step_z_loss,
                "lr": lr, "tokens_per_sec": correct_tps,
                "mem_gb": mem,
            }, step=step)

            state.total_loss = 0.0
            start_time = time.time()

        # Evaluation
        if step > start_step and (step - start_step) % phase_cfg["eval_steps"] == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 'cuda')
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                state.best_step = step
                best_path = os.path.join(phase_dir, "best", "model.pt")
                save_checkpoint(model, optimizer, scheduler, state, args.phase, best_path)
                print(f"  ★ New best: {val_loss:.4f} (PPL: {val_ppl:.2f})")
            eval_line = f"[Eval @ step {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Best: {state.best_val_loss:.4f} @ {state.best_step}"
            print(eval_line, flush=True)
            log_file.write(eval_line + "\n")
            log_file.flush()
            trackio.log({"val_loss": val_loss, "val_ppl": val_ppl, "best_val_loss": state.best_val_loss}, step=step)
            model.train()

        # Checkpoint
        if step > start_step and (step - start_step) % phase_cfg["save_steps"] == 0:
            ckpt_path = os.path.join(phase_dir, f"checkpoint-{step}", "model.pt")
            save_checkpoint(model, optimizer, scheduler, state, args.phase, ckpt_path)
            saved_checkpoints.append(ckpt_path)
            print(f"  💾 Saved checkpoint: step {step}", flush=True)

        # Memory cleanup
        if step % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Final checkpoint for this phase
    final_path = os.path.join(phase_dir, "final", "model.pt")
    save_checkpoint(model, optimizer, scheduler, state, args.phase, final_path)
    saved_checkpoints.append(final_path)

    # Phase 4 special: checkpoint EMA at the end
    if args.phase == 4:
        ema_ckpts = saved_checkpoints[-5:] if len(saved_checkpoints) >= 5 else saved_checkpoints
        checkpoint_ema(model, ema_ckpts, beta=0.8)
        ema_path = os.path.join(phase_dir, "ema", "model.pt")
        os.makedirs(os.path.dirname(ema_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_checkpoints": ema_ckpts,
        }, ema_path)
        ema_val_loss, ema_val_ppl = evaluate(model, val_loader, 'cuda')
        print(f"\n  EMA Val Loss: {ema_val_loss:.4f} | EMA Val PPL: {ema_val_ppl:.2f}")
        log_file.write(f"\n[EMA] Val Loss: {ema_val_loss:.4f} | Val PPL: {ema_val_ppl:.2f}\n")

    # Phase summary
    print("\n" + "=" * 70)
    print(f"  PHASE {args.phase} COMPLETE: {phase_cfg['name']}")
    print("=" * 70)
    print(f"  Steps:  {start_step} → {state.step}")
    print(f"  Tokens: {state.num_tokens:,} ({state.num_tokens/1e9:.3f}B)")
    print(f"  Best:   val_loss={state.best_val_loss:.4f} @ step {state.best_step}")
    print(f"  Output: {phase_dir}")

    if args.phase < 4:
        print(f"\n  ▶ Next: python3 scripts/train_v5_phases.py --phase {args.phase + 1}")

    log_file.close()


if __name__ == "__main__":
    main()
