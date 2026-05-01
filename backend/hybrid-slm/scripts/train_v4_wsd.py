#!/usr/bin/env python3
"""
Training Script v4: Hybrid5 SLM with Proper WSD Schedule
Optimized for 6GB VRAM (RTX 3060)

Architecture: Hybrid5 (IMU-1 techniques + Gemma 4 interleaved pattern)
Dataset: SYNTH (2M EN docs, ~1.64B tokens) + Hermes Agent Traces (~1.15B tokens)
         Combined: ~2.8B tokens

Key fixes from v3 (overnight run that plateaued):
  1. WSD scheduler with 1-sqrt(t) decay (IMU-1 validated, NOT cosine)
  2. 20% decay fraction (IMU-1 ablated: 10% too short, 30% overshoots, 20% optimal)
  3. Single continuous run - no resume splits that break the LR schedule
  4. Z-loss (λ=1e-4) for numerical stability (IMU-1 uses this)
  5. Checkpoint EMA (β=0.8) for final model averaging
  6. Decay to near-zero LR (D2Z paper shows this matters)

Why the plateau happened:
  The old run never entered the decay phase because each resume reset max_steps.
  In WSD, the dramatic loss drop happens ONLY during decay ("river valley theory").
  Val PPL 163 during stable phase is expected and NOT a true plateau.

Based on:
  - IMU-1 (arXiv:2602.02522): Sample-Efficient Pre-training of SLMs
  - MiniCPM (arXiv:2404.06395): WSD scheduler inventors
  - D2Z (arXiv:2502.15938): Decay-to-zero for better convergence
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
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

import trackio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemma4v2.hybrid5_config import Hybrid5Config
from src.gemma4v2.hybrid5_model import Hybrid5Model


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


class StreamingTokenDataset(IterableDataset):
    """Streaming dataset with buffered shuffling"""
    def __init__(self, data_path: str, max_length: int = 1024, stride: int = 256,
                 chunk_samples: int = 64_000, seed: int = 42):
        self.data_path = data_path
        self.max_length = max_length
        self.stride = stride
        self.chunk_samples = chunk_samples
        self.seed = seed
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tokenized data not found at {data_path}")
        self.ids = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.total_size = max(0, len(self.ids) - (max_length + 1))
        self.num_windows = max(0, self.total_size // stride)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        np.random.seed(self.seed + worker_id)

        chunk_size = self.chunk_samples * self.stride
        chunk_start = 0

        while chunk_start < len(self.ids) - self.max_length - 1:
            chunk_end = min(chunk_start + chunk_size, len(self.ids) - self.max_length - 1)
            chunk_ids = self.ids[chunk_start:chunk_end]

            windows = []
            pos = 0
            while pos + self.max_length + 1 < len(chunk_ids):
                windows.append(chunk_ids[pos:pos + self.max_length + 1])
                pos += self.stride

            np.random.shuffle(windows)

            for window in windows:
                if len(window) >= self.max_length + 1:
                    yield {
                        "input_ids": torch.from_numpy(window[:self.max_length].astype(np.int64)),
                        "labels": torch.from_numpy(window[1:self.max_length + 1].astype(np.int64)),
                    }

            chunk_start += chunk_size // 2

    def __len__(self):
        return self.num_windows


# ── Training State ───────────────────────────────────────────────
@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    num_tokens: int = 0
    best_val_loss: float = float('inf')
    best_step: int = 0


# ── WSD Scheduler with 1-sqrt(t) Decay (IMU-1 validated) ───────
class WSDScheduler:
    """Warmup-Stable-Decay LR Scheduler
    
    Key differences from old version:
      - 1-sqrt(t) decay profile (IMU-1 validated, NOT cosine)
      - Single continuous schedule (no resume breaks)
      - Decay to near-zero (D2Z paper shows this is critical)
      - Configurable decay fraction (default 20% per IMU-1 ablation)
    
    Phases:
      Warmup: step 0 → warmup_steps (linear ramp from min_lr to peak_lr)
      Stable: warmup_steps → decay_start (constant at peak_lr)
      Decay:  decay_start → total_steps (1-sqrt(t) to near-zero)
    
    The "river valley" effect: loss drops dramatically ONLY during decay.
    This is by design — stable phase makes optimization progress that
    only becomes visible when LR decreases.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = 4e-4,
        min_lr: float = 1e-5,
        decay_fraction: float = 0.20,
        last_step: int = 0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.decay_fraction = decay_fraction
        self.last_step = last_step
        
        # Compute decay boundary
        self.decay_start = int(total_steps * (1 - decay_fraction))
        
    def step(self):
        self.last_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
    
    def get_lr(self) -> float:
        step = self.last_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.min_lr + (self.peak_lr - self.min_lr) * (step / self.warmup_steps)
        elif step < self.decay_start:
            # Stable phase
            return self.peak_lr
        else:
            # 1-sqrt(t) decay (IMU-1 validated)
            decay_progress = (step - self.decay_start) / (self.total_steps - self.decay_start)
            decay_progress = min(decay_progress, 1.0)
            decay_factor = 1.0 - math.sqrt(decay_progress)
            return max(self.peak_lr * decay_factor, self.min_lr)
    
    def get_phase(self) -> str:
        if self.last_step < self.warmup_steps:
            return "warmup"
        elif self.last_step < self.decay_start:
            return "stable"
        else:
            return "decay"


# ── Z-Loss (IMU-1: numerical stability, λ=1e-4) ──────────────
def compute_z_loss(logits: torch.Tensor, z_loss_weight: float = 1e-4) -> torch.Tensor:
    """Z-loss: encourages logits to stay bounded for numerical stability.
    
    From IMU-1: z_loss = λ * mean(log(sum(exp(logits))^2))
    Helps prevent attention logit explosion, especially with QK-Norm.
    """
    log_z = torch.logsumexp(logits, dim=-1)
    return z_loss_weight * (log_z ** 2).mean()


# ── Checkpoint EMA (IMU-1: β=0.8 post-hoc averaging) ─────────
def checkpoint_ema(model, checkpoints: List[str], beta: float = 0.8):
    """Exponential moving average of checkpoint weights.
    
    Later checkpoints get exponentially more weight.
    β=0.8 means the last checkpoint gets ~45% weight, second-to-last ~9%, etc.
    """
    if not checkpoints:
        return
    
    print(f"\nCheckpoint EMA (β={beta}) over {len(checkpoints)} checkpoints:")
    
    # Load all state dicts
    state_dicts = []
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        state_dicts.append(ckpt["model_state_dict"])
        print(f"  Loaded: {path}")
    
    # Compute EMA weights (exponential decay, newest gets most weight)
    n = len(state_dicts)
    weights = [beta ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    
    # Average parameters
    averaged_state = {}
    for key in state_dicts[0].keys():
        averaged_state[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
    
    model.load_state_dict(averaged_state)
    print("  ✓ Checkpoint EMA applied")


# ── Checkpointing ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, state, config_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": {
            "last_step": scheduler.last_step,
            "warmup_steps": scheduler.warmup_steps,
            "total_steps": scheduler.total_steps,
            "peak_lr": scheduler.peak_lr,
            "min_lr": scheduler.min_lr,
            "decay_fraction": scheduler.decay_fraction,
            "decay_start": scheduler.decay_start,
        },
        "state": {
            "step": state.step, "epoch": state.epoch,
            "total_loss": state.total_loss, "num_tokens": state.num_tokens,
            "best_val_loss": state.best_val_loss, "best_step": state.best_step,
        },
        "config": config_dict,
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
    print(f"  Resumed from step {state.step} (best val loss: {state.best_val_loss:.4f} @ step {state.best_step})")
    return ckpt, state, ckpt.get("scheduler_state", {})


# ── Evaluation ───────────────────────────────────────────────────
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

        if total_tokens >= max_batches * 1024:  # ~50 batches worth of tokens
            break

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


# ── Main Training ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train v4: Hybrid5 with proper WSD")
    parser.add_argument("--train-data", default="data/combined_synth_hermes_train_tokens.bin")
    parser.add_argument("--val-data", default="data/combined_synth_hermes_val_tokens.bin")
    parser.add_argument("--output-dir", default="outputs/v4-hybrid5-wsd")
    parser.add_argument("--run-name", default="v4-hybrid5-wsd")
    
    # Training schedule
    parser.add_argument("--total-steps", type=int, default=20000,
                        help="Total training steps (single run, no splits)")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup steps")
    parser.add_argument("--decay-fraction", type=float, default=0.20,
                        help="Fraction of total steps for 1-sqrt(t) decay (IMU-1: 20%%)")
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="Peak learning rate (stable phase)")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Minimum LR during decay (near-zero, D2Z)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    
    # Batch
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--seq-length", type=int, default=1024)
    
    # Checkpointing
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    
    # IMU-1 additions
    parser.add_argument("--z-loss-weight", type=float, default=1e-4,
                        help="Z-loss weight for numerical stability (IMU-1: 1e-4)")
    parser.add_argument("--checkpoint-ema-beta", type=float, default=0.8,
                        help="EMA beta for final checkpoint averaging (IMU-1: 0.8)")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()

    # Validate schedule
    decay_start = int(args.total_steps * (1 - args.decay_fraction))
    decay_steps = args.total_steps - decay_start
    
    # Load model config
    model_config = Hybrid5Config()
    os.makedirs(args.output_dir, exist_ok=True)

    # Print config
    print("\n" + "=" * 70)
    print("V4-HYBRID5-SLM: PROPER WSD TRAINING")
    print("=" * 70)
    print(f"\n  Architecture: Hybrid5 (IMU-1 + Gemma 4 pattern)")
    print(f"  Hidden: {model_config.hidden_size} | Layers: {model_config.num_hidden_layers} | Heads: {model_config.num_attention_heads}")
    print(f"  Vocab: {model_config.vocab_size:,}")
    print(f"\n  ── WSD Schedule (FIXED) ──")
    print(f"  Total steps:   {args.total_steps:,}")
    print(f"  Warmup:        {args.warmup_steps:,} steps ({100*args.warmup_steps/args.total_steps:.1f}%)")
    print(f"  Stable:        {decay_start - args.warmup_steps:,} steps ({100*(decay_start-args.warmup_steps)/args.total_steps:.1f}%)")
    print(f"  Decay start:   step {decay_start:,}")
    print(f"  Decay:         {decay_steps:,} steps ({100*args.decay_fraction:.0f}%)")
    print(f"  Decay profile: 1-sqrt(t) (IMU-1 validated)")
    print(f"  Peak LR:       {args.lr:.1e}")
    print(f"  Min LR:        {args.min_lr:.1e} (decay to near-zero)")
    print(f"\n  ── Training ──")
    print(f"  Batch:         {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Seq length:    {args.seq_length}")
    print(f"  Weight decay:  {args.weight_decay}")
    print(f"  Z-loss weight: {args.z_loss_weight}")
    print(f"\n  ── Data ──")
    print(f"  Train: {args.train_data}")
    print(f"  Val:   {args.val_data}")
    
    # Verify data files
    for path in [args.train_data, args.val_data]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        size_gb = os.path.getsize(path) / 1e9
        tokens = os.path.getsize(path) // 4
        print(f"  {os.path.basename(path)}: {size_gb:.2f} GB ({tokens:,} tokens)")

    # Create model
    model = Hybrid5Model(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {num_params/1e6:.2f}M parameters")

    model = model.cuda()
    model.enable_gradient_checkpointing()
    model._move_rotary_to_device('cuda')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                      betas=(0.9, 0.95))  # β2=0.95 per IMU-1/Llama style

    # Load checkpoint if resuming
    state = TrainingState()
    scheduler_state = {}
    if args.resume:
        ckpt, state, scheduler_state = load_checkpoint(model, optimizer, args.resume)

    # WSD Scheduler - SINGLE CONTINUOUS SCHEDULE
    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
        decay_fraction=args.decay_fraction,
        last_step=state.step,
    )
    # Restore scheduler state if resuming
    if scheduler_state:
        scheduler.last_step = scheduler_state.get("last_step", state.step)

    scaler = GradScaler('cuda')

    # Trackio
    trackio.init(
        project="hybrid-slm",
        name=args.run_name,
        config={
            "model": "v4-hybrid5-wsd",
            "architecture": "hybrid5",
            "dataset": "SYNTH + Hermes (~2.8B tokens)",
            "total_steps": args.total_steps,
            "warmup_steps": args.warmup_steps,
            "decay_fraction": args.decay_fraction,
            "decay_profile": "1-sqrt(t)",
            "peak_lr": args.lr,
            "min_lr": args.min_lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seq_length": args.seq_length,
            "weight_decay": args.weight_decay,
            "z_loss_weight": args.z_loss_weight,
            "model_params": num_params,
            "resume_from_step": state.step,
        },
    )

    # Quick test
    print("\n  Testing forward pass...")
    test_ids = torch.randint(0, model_config.vocab_size, (1, 128)).cuda()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=test_ids, labels=test_ids)
            print(f"  Initial loss: {out['loss'].item():.4f}")
    del test_ids, out
    torch.cuda.empty_cache()

    # Dataloaders
    print("\n  Loading data...")
    train_dataset = StreamingTokenDataset(
        data_path=args.train_data,
        max_length=args.seq_length,
        stride=256,
        chunk_samples=64_000,
        seed=42,
    )
    val_dataset = TokenizedDataset(
        data_path=args.val_data,
        max_length=args.seq_length,
        stride=256,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    print(f"  Train windows: {len(train_dataset):,}")
    print(f"  Val windows:   {len(val_dataset):,}")

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print(f"  Phase schedule: warmup→{args.warmup_steps} | stable→{decay_start} | decay→{args.total_steps}")
    print("=" * 70)

    log_file = open(os.path.join(args.output_dir, "training_log.txt"), "a")
    start_time = time.time()
    train_iter = iter(train_loader)
    
    # Track checkpoints for EMA
    saved_checkpoints = []
    
    config_dict = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "decay_fraction": args.decay_fraction,
        "peak_lr": args.lr,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_length": args.seq_length,
    }

    for step in range(state.step, args.total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_z_loss = 0.0
        step_tokens = 0

        for micro_step in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                state.epoch += 1

            input_ids = batch['input_ids'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / args.grad_accum
                
                # Z-loss for numerical stability (IMU-1)
                if args.z_loss_weight > 0:
                    z_loss = compute_z_loss(outputs['logits'], args.z_loss_weight) / args.grad_accum
                    loss = loss + z_loss
                    step_z_loss += z_loss.item() * args.grad_accum

            scaler.scale(loss).backward()
            step_loss += outputs['loss'].item()
            step_tokens += input_ids.numel()

            del outputs, loss, input_ids, labels

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        phase = scheduler.get_phase()

        state.step = step + 1
        state.total_loss += step_loss / args.grad_accum
        state.num_tokens += step_tokens

        # Logging
        if step % args.logging_steps == 0:
            avg_loss = state.total_loss / args.logging_steps
            tokens_per_sec = step_tokens / (time.time() - start_time) if start_time > 0 else 0
            elapsed = time.time() - start_time
            mem_used = torch.cuda.memory_allocated() / 1e9

            # Extra info during decay phase
            phase_indicator = ""
            if phase == "warmup":
                phase_indicator = "🔄"
            elif phase == "stable":
                phase_indicator = "⚡"
            elif phase == "decay":
                phase_indicator = "📉"

            log_line = (f"step={step} | loss={avg_loss:.4f} | z_loss={step_z_loss:.6f} | "
                       f"lr={lr:.2e} | phase={phase} {phase_indicator} | "
                       f"tokens/s={tokens_per_sec:.0f} | epoch={state.epoch} | "
                       f"mem={mem_used:.2f}GB | elapsed={elapsed:.0f}s")
            print(log_line, flush=True)
            log_file.write(log_line + "\n")
            log_file.flush()

            trackio.log({
                "loss": avg_loss,
                "z_loss": step_z_loss,
                "lr": lr,
                "tokens_per_sec": tokens_per_sec,
                "mem_gb": mem_used,
                "phase_warmup": 1.0 if phase == "warmup" else 0.0,
                "phase_stable": 1.0 if phase == "stable" else 0.0,
                "phase_decay": 1.0 if phase == "decay" else 0.0,
            }, step=step)

            state.total_loss = 0.0
            start_time = time.time()

        # Evaluation
        if step > 0 and step % args.eval_steps == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 'cuda')
            
            # Track best
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                state.best_step = step
                # Save best checkpoint
                best_path = os.path.join(args.output_dir, "best/model.pt")
                save_checkpoint(model, optimizer, scheduler, state, config_dict, best_path)
                print(f"  ★ New best val loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")

            eval_line = f"[Eval @ step {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Best: {state.best_val_loss:.4f} @ step {state.best_step}"
            print(eval_line, flush=True)
            log_file.write(eval_line + "\n")
            log_file.flush()

            trackio.log({
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "best_val_loss": state.best_val_loss,
            }, step=step)

            model.train()

        # Checkpoint
        if step > 0 and step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            save_checkpoint(model, optimizer, scheduler, state, config_dict, ckpt_path + "/model.pt")
            saved_checkpoints.append(ckpt_path + "/model.pt")
            print(f"  💾 Saved checkpoint: {ckpt_path}", flush=True)

        # Memory cleanup
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Final save
    final_path = os.path.join(args.output_dir, "final/model.pt")
    save_checkpoint(model, optimizer, scheduler, state, config_dict, final_path)
    saved_checkpoints.append(final_path)

    # Checkpoint EMA (IMU-1: β=0.8)
    # Use last 5 checkpoints for EMA (last ~10K steps)
    ema_checkpoints = saved_checkpoints[-5:] if len(saved_checkpoints) >= 5 else saved_checkpoints
    checkpoint_ema(model, ema_checkpoints, beta=args.checkpoint_ema_beta)
    
    # Save EMA model
    ema_path = os.path.join(args.output_dir, "ema/model.pt")
    os.makedirs(os.path.dirname(ema_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config_dict,
        "ema_checkpoints": ema_checkpoints,
        "ema_beta": args.checkpoint_ema_beta,
    }, ema_path)
    
    # Evaluate EMA model
    print("\n  Evaluating EMA model...")
    ema_val_loss, ema_val_ppl = evaluate(model, val_loader, 'cuda')
    print(f"  EMA Val Loss: {ema_val_loss:.4f} | EMA Val PPL: {ema_val_ppl:.2f}")
    log_file.write(f"\n[EMA Final] Val Loss: {ema_val_loss:.4f} | Val PPL: {ema_val_ppl:.2f}\n")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Output: {args.output_dir}")
    print(f"  Final step: {state.step}")
    print(f"  Total tokens: {state.num_tokens:,}")
    print(f"  Best val loss: {state.best_val_loss:.4f} (PPL: {math.exp(state.best_val_loss):.2f}) @ step {state.best_step}")
    print(f"  EMA val loss:  {ema_val_loss:.4f} (PPL: {ema_val_ppl:.2f})")

    log_file.close()


if __name__ == "__main__":
    main()
