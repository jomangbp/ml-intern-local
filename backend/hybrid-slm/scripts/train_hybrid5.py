"""
Training Script for Hybrid5 SLM
Optimized for 6GB VRAM (RTX 3060)

Based on: IMU-1 (arXiv:2602.02522) techniques:
1. QK-Norm Attention
2. Per-Head Gating
3. Value Residuals
4. LayerNorm Scaling (1/sqrt(layer))
5. NorMuon Optimizer

For fair comparison with Gemma4:
- Same data (TinyStories)
- Same LR schedule (WSD from MiniCPM)
- Same batch config (3 batch, 2 grad accum, 1024 seq)
- Same architecture pattern (5:1 interleaved)

Note: Model is ~158M params (larger due to 262k vocab), but training 
setup matches Gemma4 exactly for fair comparison.
"""

import os
import sys
import math
import time
import gc
import json
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
    """Memory-efficient token dataset (same as Gemma4)"""
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
    """Streaming dataset with buffered shuffling (same as Gemma4)"""
    def __init__(self, data_path: str, max_length: int = 1024, stride: int = 256,
                 chunk_samples: int = 64_000, seed: int = 42):
        self.data_path = data_path
        self.max_length = max_length
        self.stride = stride
        self.chunk_samples = chunk_samples
        self.seed = seed
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tokenized data not found at {data_path}")
        # Use memmap for binary uint32 files
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
        last_step: int = 0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
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
            decay_steps = 10000  # ~10k decay steps
            progress = min(decay_step / decay_steps, 1.0)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ── Checkpointing ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, state, train_config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # WSDScheduler is a custom class - save its attributes instead of state_dict
    scheduler_state = {
        "last_step": scheduler.last_step,
        "warmup_steps": scheduler.warmup_steps,
        "max_steps": scheduler.max_steps,
        "min_lr": scheduler.min_lr,
        "base_lrs": scheduler.base_lrs,
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "state": state,
        "config": train_config,
    }, path)


def load_checkpoint(model, optimizer, scheduler, train_config):
    path = train_config.resume_from_checkpoint
    if not path or not os.path.exists(path):
        return TrainingState(), 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    state = ckpt["state"]
    print(f"  Resumed from step {state.step}")
    return state, state.step


# ── Data Loaders ────────────────────────────────────────────────
def create_dataloaders(train_config, max_length: int = 1024):
    """Create train and validation dataloaders"""
    train_dataset = StreamingTokenDataset(
        data_path=train_config.train_data_path,
        max_length=max_length,
        stride=256,
        chunk_samples=64_000,
        seed=42,
    )
    val_dataset = TokenizedDataset(
        data_path=train_config.val_data_path,
        max_length=max_length,
        stride=256,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.per_device_train_batch_size,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.per_device_train_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


# ── Evaluation ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device):
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

        if num_batches >= 50:
            break

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


# ── Main Training ───────────────────────────────────────────────
def main():
    # Load configs
    model_config = Hybrid5Config()
    train_config = Hybrid5TrainingConfig()

    # Create output directory
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Create model
    print("\n" + "=" * 60)
    print("HYBRID5 TRAINING (IMU-1 Techniques)")
    print("=" * 60)
    print(f"Config:")
    print(f"  - Hidden size: {model_config.hidden_size}")
    print(f"  - Layers: {model_config.num_hidden_layers}")
    print(f"  - Heads: {model_config.num_attention_heads} (GQA: {model_config.num_key_value_heads} KV)")
    print(f"  - Vocab: {model_config.vocab_size:,}")
    print(f"  - Pattern: 5:1 interleaved (sliding/full)")
    print(f"\n5 IMU-1 Techniques:")
    print(f"  1. QK-Norm Attention")
    print(f"  2. Per-Head Gating")
    print(f"  3. Value Residuals")
    print(f"  4. LayerNorm Scaling (1/sqrt(layer))")
    print(f"  5. NorMuon Optimizer")
    print(f"\nTraining:")
    print(f"  - LR: {train_config.learning_rate:.1e} (WSD scheduler)")
    print(f"  - Batch: {train_config.per_device_train_batch_size} x {train_config.gradient_accumulation_steps} = {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps}")
    print(f"  - Seq length: {train_config.max_seq_length}")
    print(f"  - Max steps: {train_config.max_steps:,}")

    model = Hybrid5Model(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params/1e6:.2f}M parameters")

    model = model.cuda()
    model.enable_gradient_checkpointing()
    model._move_rotary_to_device('cuda')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    # Load checkpoint if resuming
    state = TrainingState()
    if train_config.resume_from_checkpoint:
        state, start_step = load_checkpoint(model, optimizer, None, train_config)
        print(f"Resumed from checkpoint: {train_config.resume_from_checkpoint}")

    # WSD Scheduler
    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        min_lr=train_config.min_learning_rate,
        last_step=state.step,
    )

    scaler = GradScaler('cuda')

    # Trackio
    run_name = "v3-hybrid5-slm"
    trackio.init(
        project="hybrid-slm",
        name=run_name,
        config={
            "model": "v3-hybrid5-slm",
            "architecture": "hybrid5",
            "techniques": "IMU-1 (QK-Norm, Per-Head Gating, Value Residuals, LayerNorm Scaling, NorMuon)",
            "comparison_to": "Gemma4 (v2) and v1-hybrid-slm-baseline",
            "learning_rate": train_config.learning_rate,
            "min_learning_rate": train_config.min_learning_rate,
            "warmup_steps": train_config.warmup_steps,
            "max_steps": train_config.max_steps,
            "batch_size": train_config.per_device_train_batch_size,
            "grad_accum": train_config.gradient_accumulation_steps,
            "seq_length": train_config.max_seq_length,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "model_params": num_params,
            "dataset": "TinyStories (25M tokens)",
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
    train_loader, val_loader = create_dataloaders(train_config, max_length=train_config.max_seq_length)
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples:   {len(val_loader.dataset):,}")

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING HYBRID5 TRAINING")
    print("=" * 60)

    log_file = open(os.path.join(train_config.output_dir, "training_log.txt"), "a")
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_steps = train_config.gradient_accumulation_steps

    for step in range(state.step, train_config.max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0

        for micro_step in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                state.epoch += 1

            input_ids = batch['input_ids'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16, enabled=train_config.bf16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / accum_steps

            scaler.scale(loss).backward()
            step_loss += outputs['loss'].item()  # Log the actual loss (not grad-scaled)
            step_tokens += input_ids.numel()

            del outputs, loss, input_ids, labels

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()

        state.step = step + 1
        state.total_loss += step_loss / accum_steps  # Average over micro-steps
        state.num_tokens += step_tokens

        # Logging
        if step % train_config.logging_steps == 0:
            avg_loss = state.total_loss / train_config.logging_steps
            tokens_per_sec = step_tokens / (time.time() - start_time) if start_time > 0 else 0
            elapsed = time.time() - start_time

            mem_used = torch.cuda.memory_allocated() / 1e9

            log_line = f"step={step} | loss={avg_loss:.4f} | lr={lr:.2e} | tokens/s={tokens_per_sec:.0f} | epoch={state.epoch} | mem={mem_used:.2f}GB"
            print(log_line)
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
        if step > 0 and step % train_config.eval_steps == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 'cuda')
            eval_line = f"[Eval @ step {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
            print(eval_line)
            log_file.write(eval_line + "\n")
            log_file.flush()

            trackio.log({
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, step=step)

            model.train()

        # Checkpoint
        if step > 0 and step % train_config.save_steps == 0:
            ckpt_path = os.path.join(train_config.output_dir, f"checkpoint-{step}")
            save_checkpoint(model, optimizer, scheduler, state, train_config, ckpt_path + "/model.pt")
            print(f"\nSaved checkpoint: {ckpt_path}")

        # Memory cleanup
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Final save
    final_path = os.path.join(train_config.output_dir, "final/model.pt")
    save_checkpoint(model, optimizer, scheduler, state, train_config, final_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Output: {train_config.output_dir}")
    print(f"Final step: {state.step}")

    log_file.close()


if __name__ == "__main__":
    main()