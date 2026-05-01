#!/usr/bin/env python3
"""
Training Script for v3-hybrid5-slm on SYNTH + Hermes Dataset
Optimized for 6GB VRAM (RTX 3060)

Architecture: Hybrid5 (IMU-1 techniques + Gemma 4 interleaved pattern)
Dataset: SYNTH (2M EN docs, ~1.64B tokens) + Hermes Agent Traces (~1.15B tokens)
         Combined: ~2.8B tokens

Key differences from TinyStories training:
  - 100x more data → need more steps, longer warmup
  - Mixed data types (reasoning + tool-calling)
  - WSD scheduler with appropriate warmup/decay ratios
"""

import os
import sys
import math
import time
import gc
import json
import argparse
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

from src.gemma4v2.hybrid5_config import Hybrid5Config, Hybrid5TrainingConfig
from src.gemma4v2.hybrid5_model import Hybrid5Model
from src.layers import RMSNorm


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


# ── Scheduler (WSD from MiniCPM) ────────────────────────────────
class WSDScheduler:
    """Warmup-Stable-Decay LR Scheduler from MiniCPM"""
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        decay_steps: int = 10000,
        last_step: int = 0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.last_step = last_step
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self.last_step += 1
        lr = self.get_lr()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = lr
        return lr

    def get_lr(self) -> float:
        step = self.last_step
        if step < self.warmup_steps:
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * (step / self.warmup_steps)
        elif step < self.max_steps:
            return self.base_lrs[0]
        else:
            decay_step = step - self.max_steps
            progress = min(decay_step / self.decay_steps, 1.0)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ── Checkpointing ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, state, config_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scheduler_state = {
        "last_step": scheduler.last_step,
        "warmup_steps": scheduler.warmup_steps,
        "max_steps": scheduler.max_steps,
        "min_lr": scheduler.min_lr,
        "decay_steps": scheduler.decay_steps,
        "base_lrs": scheduler.base_lrs,
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "state": {"step": state.step, "epoch": state.epoch,
                  "total_loss": state.total_loss, "num_tokens": state.num_tokens},
        "config": config_dict,
    }, path)


def load_checkpoint(model, optimizer, path):
    if not path or not os.path.exists(path):
        return TrainingState(), 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    state = TrainingState(
        step=ckpt["state"]["step"],
        epoch=ckpt["state"]["epoch"],
        total_loss=ckpt["state"]["total_loss"],
        num_tokens=ckpt["state"]["num_tokens"],
    )
    print(f"  Resumed from step {state.step}")
    return state, state.step, ckpt.get("scheduler_state_dict", {})


# ── Evaluation ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)

        total_loss += outputs['loss'].item() * input_ids.numel()
        total_tokens += input_ids.numel()
        num_batches += 1

        if num_batches >= max_batches:
            break

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


# ── Main Training ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train v3-hybrid5-slm on SYNTH+Hermes")
    parser.add_argument("--train-data", default="data/combined_synth_hermes_train_tokens.bin")
    parser.add_argument("--val-data", default="data/combined_synth_hermes_val_tokens.bin")
    parser.add_argument("--output-dir", default="outputs/v3-hybrid5-slm-synth-hermes")
    parser.add_argument("--run-name", default="v3-hybrid5-slm-synth-hermes")
    parser.add_argument("--max-steps", type=int, default=5900)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--min-lr", type=float, default=4e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--decay-steps", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load model config
    model_config = Hybrid5Config()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    print("\n" + "=" * 70)
    print("V3-HYBRID5-SLM TRAINING ON SYNTH + HERMES")
    print("=" * 70)
    print(f"  Architecture: Hybrid5 (IMU-1 + Gemma 4 pattern)")
    print(f"  Hidden size:  {model_config.hidden_size}")
    print(f"  Layers:       {model_config.num_hidden_layers}")
    print(f"  Heads:        {model_config.num_attention_heads} (GQA: {model_config.num_key_value_heads} KV)")
    print(f"  Vocab:        {model_config.vocab_size:,}")
    print(f"  Train data:   {args.train_data}")
    print(f"  Val data:     {args.val_data}")
    print(f"  LR:           {args.lr:.1e} (WSD: warmup={args.warmup_steps}, max={args.max_steps})")
    print(f"  Batch:        {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Seq length:   {args.seq_length}")
    print(f"  Max steps:    {args.max_steps:,}")
    print(f"  Output dir:   {args.output_dir}")

    # Verify data files exist
    for path in [args.train_data, args.val_data]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        size_gb = os.path.getsize(path) / 1e9
        tokens = os.path.getsize(path) // 4
        print(f"  {os.path.basename(path)}: {size_gb:.2f} GB ({tokens:,} tokens)")

    model = Hybrid5Model(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {num_params/1e6:.2f}M parameters")

    model = model.cuda()
    model.enable_gradient_checkpointing()
    model._move_rotary_to_device('cuda')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Load checkpoint if resuming
    state = TrainingState()
    scheduler_ckpt = {}
    if args.resume:
        state, start_step, scheduler_ckpt = load_checkpoint(model, optimizer, args.resume)

    # WSD Scheduler
    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        min_lr=args.min_lr,
        decay_steps=args.decay_steps,
        last_step=state.step,
    )
    # Restore scheduler state if resuming
    if scheduler_ckpt:
        scheduler.last_step = scheduler_ckpt.get("last_step", state.step)
        scheduler.base_lrs = scheduler_ckpt.get("base_lrs", [args.lr])

    scaler = GradScaler('cuda')

    # Trackio
    trackio.init(
        project="hybrid-slm",
        name=args.run_name,
        config={
            "model": "v3-hybrid5-slm",
            "architecture": "hybrid5",
            "techniques": "IMU-1 (QK-Norm, Per-Head Gating, Value Residuals, LayerNorm Scaling)",
            "dataset": "SYNTH (2M EN docs) + Hermes Agent Traces",
            "train_data": args.train_data,
            "val_data": args.val_data,
            "learning_rate": args.lr,
            "min_learning_rate": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seq_length": args.seq_length,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "model_params": num_params,
            "resume_from_step": state.step,
        },
    )

    # Quick test
    print("\nTesting forward pass...")
    test_ids = torch.randint(0, model_config.vocab_size, (1, 128)).cuda()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=test_ids, labels=test_ids)
            print(f"  Loss: {out['loss'].item():.4f}")
    del test_ids, out
    torch.cuda.empty_cache()

    # Dataloaders
    print("\nLoading data...")
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
    print("=" * 70)

    log_file = open(os.path.join(args.output_dir, "training_log.txt"), "a")
    start_time = time.time()
    train_iter = iter(train_loader)

    config_dict = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_length": args.seq_length,
    }

    for step in range(state.step, args.max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
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

        state.step = step + 1
        state.total_loss += step_loss / args.grad_accum
        state.num_tokens += step_tokens

        # Logging
        if step % args.logging_steps == 0:
            avg_loss = state.total_loss / args.logging_steps
            tokens_per_sec = step_tokens / (time.time() - start_time) if start_time > 0 else 0
            elapsed = time.time() - start_time

            mem_used = torch.cuda.memory_allocated() / 1e9

            log_line = (f"step={step} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                       f"tokens/s={tokens_per_sec:.0f} | epoch={state.epoch} | "
                       f"mem={mem_used:.2f}GB | elapsed={elapsed:.0f}s")
            print(log_line, flush=True)
            log_file.write(log_line + "\n")
            log_file.flush()

            trackio.log({
                "loss": avg_loss,
                "lr": lr,
                "tokens_per_sec": tokens_per_sec,
                "mem_gb": mem_used,
            }, step=step)

            state.total_loss = 0.0
            start_time = time.time()

        # Evaluation
        if step > 0 and step % args.eval_steps == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 'cuda')
            eval_line = f"[Eval @ step {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
            print(eval_line, flush=True)
            log_file.write(eval_line + "\n")
            log_file.flush()

            trackio.log({
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, step=step)

            model.train()

        # Checkpoint
        if step > 0 and step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            save_checkpoint(model, optimizer, scheduler, state, config_dict, ckpt_path + "/model.pt")
            print(f"\n  Saved checkpoint: {ckpt_path}", flush=True)

        # Memory cleanup
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Final save
    final_path = os.path.join(args.output_dir, "final/model.pt")
    save_checkpoint(model, optimizer, scheduler, state, config_dict, final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Output: {args.output_dir}")
    print(f"  Final step: {state.step}")
    print(f"  Total tokens: {state.num_tokens:,}")

    log_file.close()


if __name__ == "__main__":
    main()
