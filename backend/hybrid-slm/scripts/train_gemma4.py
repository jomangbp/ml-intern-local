"""
Training Script for Gemma 4-Inspired SLM
Optimized for 6GB VRAM (RTX 3060 Laptop)

Comparison run against v1-hybrid-slm-baseline (5900 steps).
Uses same data (TinyStories), same LR schedule, same batch config for fair comparison.

Gemma 4 innovations over baseline:
- Sandwich norm (4 RMSNorms per layer + layer scalar)
- GeLU(tanh) instead of SiLU
- QK-norm (L2 normalization)
- Dual head dimensions (128 for global, 64 for linear)
- Logit softcapping to [-30, 30]
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

import numpy as np

import trackio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemma4.config import Gemma4SLMConfig, Gemma4TrainingConfig
from src.gemma4.model import create_gemma4_model, Gemma4SLMModel
from src.layers import RMSNorm


# ── Reuse data utilities from V3 training ──────────────────────────
class TokenizedDataset(Dataset):
    """Memory-efficient token dataset (same as V3)"""
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
    """Streaming dataset with buffered shuffling (same as V3)"""
    def __init__(self, data_path: str, max_length: int = 1024, stride: int = 256,
                 chunk_samples: int = 64_000, seed: int = 42):
        self.data_path = data_path
        self.max_length = max_length
        self.stride = stride
        self.chunk_samples = chunk_samples
        self.seed = seed
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tokenized data not found at {data_path}")
        ids_tmp = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.num_tokens = len(ids_tmp)
        self.num_samples = max(0, (self.num_tokens - (max_length + 1)) // stride)
        del ids_tmp

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        ids = np.memmap(self.data_path, dtype=np.uint32, mode="r")
        num_samples = self.num_samples
        chunk_size = self.chunk_samples

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            token_start = chunk_start * self.stride
            token_end = (chunk_end - 1) * self.stride + self.max_length + 1
            big_chunk = ids[token_start:token_end].astype(np.int64)

            samples = []
            for i in range(chunk_start, chunk_end):
                local_start = (i - chunk_start) * self.stride
                local_end = local_start + self.max_length + 1
                window = big_chunk[local_start:local_end]
                samples.append({
                    "input_ids": torch.from_numpy(window[:-1].copy()),
                    "labels": torch.from_numpy(window[1:].copy()),
                })

            rng.shuffle(samples)
            for item in samples:
                yield item
            del big_chunk, samples


class CosineAnnealingWarmup:
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr: float = 0.0, last_step: int = 0):
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
            pg['lr'] = lr * base_lr / self.base_lrs[0] if self.base_lrs[0] > 0 else lr

    def get_lr(self) -> float:
        step = self.last_step
        if step < self.warmup_steps:
            return self.base_lrs[0] * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    num_tokens: int = 0
    best_loss: float = float('inf')


def create_dataloaders(config: Gemma4TrainingConfig, max_length: int = 1024):
    train_dataset = StreamingTokenDataset(
        data_path=config.train_data_path,
        max_length=max_length,
        chunk_samples=32_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x])
        }
    )
    val_dataset = TokenizedDataset(data_path=config.val_data_path, max_length=max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x])
        }
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, num_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        del outputs, loss, input_ids, labels
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    return {
        'val_loss': avg_loss,
        'val_ppl': math.exp(min(avg_loss, 20)),
    }


def save_checkpoint(model, optimizer, scheduler, state, config, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler.last_step,
        'state': {'step': state.step, 'epoch': state.epoch, 'best_loss': state.best_loss},
    }
    torch.save(checkpoint, f"{path}/checkpoint.pt")
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, f"{path}/pytorch_model.bin")
    model.cuda()


def load_checkpoint(model, optimizer, config):
    path = config.resume_from_checkpoint
    checkpoint = torch.load(f"{path}/checkpoint.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    state = TrainingState(
        step=checkpoint['state']['step'],
        epoch=checkpoint['state']['epoch'],
        best_loss=checkpoint['state']['best_loss'],
    )
    del checkpoint
    gc.collect()
    print(f"Loaded checkpoint from step {state.step}")
    return state, checkpoint.get('scheduler_step', state.step) if 'checkpoint' in dir() else state.step


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Gemma 4 SLM")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    model_config = Gemma4SLMConfig()
    train_config = Gemma4TrainingConfig()

    # Match baseline exactly: bs=3, accum=2 → effective=6
    train_config.per_device_train_batch_size = 3
    train_config.gradient_accumulation_steps = 2
    train_config.output_dir = "outputs/gemma4-slm"
    train_config.max_steps = 5900  # Match baseline run length

    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.resume:
        train_config.resume_from_checkpoint = args.resume

    # Count layer types
    full_layers = sum(1 for i in range(model_config.num_hidden_layers)
                      if (i + 1) % model_config.full_attention_interval == 0)
    linear_layers = model_config.num_hidden_layers - full_layers

    print("\n" + "=" * 60)
    print("Gemma 4-Inspired SLM Training")
    print("=" * 60)
    print(f"\nModel architecture:")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_hidden_layers} ({linear_layers} linear + {full_layers} full)")
    print(f"  Local head dim: {model_config.head_dim}")
    print(f"  Global head dim: {model_config.global_head_dim}")
    print(f"  Sandwich norm: {model_config.use_sandwich_norm}")
    print(f"  Layer scalar: {model_config.use_layer_scalar}")
    print(f"  QK-norm: {model_config.use_qk_norm}")
    print(f"  Logit softcapping: {model_config.final_logit_softcapping}")
    print(f"  Activation: {model_config.hidden_act}")
    print(f"\nTraining:")
    print(f"  Batch size: {train_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch: {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps * train_config.max_seq_length} tokens/step")
    print(f"  LR: {train_config.learning_rate}")
    print(f"  Warmup: {train_config.warmup_steps}")

    # Model
    print("\nInitializing model...")
    model = create_gemma4_model(model_config)
    model = model.cuda()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    state = TrainingState()
    scheduler_step = 0

    if train_config.resume_from_checkpoint:
        state, scheduler_step = load_checkpoint(model, optimizer, train_config)

    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        min_lr=train_config.min_learning_rate,
        last_step=scheduler_step,
    )

    scaler = GradScaler('cuda')
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Trackio
    run_name = "v4-gemma4-fresh-comparison"
    trackio.init(
        project="hybrid-slm",
        name=run_name,
        config={
            "model": "gemma4-slm",
            "comparison_to": "v1-hybrid-slm-baseline (5900 steps)",
            "learning_rate": train_config.learning_rate,
            "min_learning_rate": train_config.min_learning_rate,
            "warmup_steps": train_config.warmup_steps,
            "max_steps": train_config.max_steps,
            "batch_size": train_config.per_device_train_batch_size,
            "grad_accum": train_config.gradient_accumulation_steps,
            "seq_length": train_config.max_seq_length,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "global_head_dim": model_config.global_head_dim,
            "local_head_dim": model_config.head_dim,
            "sandwich_norm": model_config.use_sandwich_norm,
            "layer_scalar": model_config.use_layer_scalar,
            "qk_norm": model_config.use_qk_norm,
            "logit_softcapping": model_config.final_logit_softcapping,
            "activation": model_config.hidden_act,
            "model_params": sum(p.numel() for p in model.parameters()),
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
    print("STARTING GEMMA 4 TRAINING")
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
            step_loss += loss.item() * accum_steps
            step_tokens += input_ids.numel()

            del outputs, loss, input_ids, labels

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        state.step = step + 1
        state.total_loss += step_loss
        state.num_tokens += step_tokens

        # Periodic cache clearing
        if step % train_config.empty_cache_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # Logging
        if state.step % train_config.logging_steps == 0:
            elapsed = time.time() - start_time
            avg_loss = step_loss / accum_steps
            lr = optimizer.param_groups[0]['lr']
            tokens_per_sec = state.num_tokens / elapsed if elapsed > 0 else 0
            mem_gb = torch.cuda.memory_allocated() / 1024**3

            log_line = (f"step={state.step:,} | loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | tokens/s={tokens_per_sec:,.0f} | "
                        f"epoch={state.epoch} | mem={mem_gb:.2f}GB")
            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

            trackio.log({
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/tokens_seen": state.num_tokens,
                "train/gpu_memory_gb": mem_gb,
                "train/epoch": state.epoch,
            })

        # Evaluation
        if state.step % train_config.eval_steps == 0:
            print(f"\n--- Evaluating at step {state.step} ---")
            eval_results = evaluate(model, val_loader, num_batches=50)
            val_loss = eval_results['val_loss']
            val_ppl = eval_results['val_ppl']
            eval_line = f"  val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}"
            print(eval_line)
            log_file.write(eval_line + "\n")
            log_file.flush()

            trackio.log({
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
            })

            if val_loss < state.best_loss:
                state.best_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, state, train_config,
                                os.path.join(train_config.output_dir, "best"))
                print(f"  ★ New best model (val_loss={val_loss:.4f})")

            model.cuda()
            torch.cuda.empty_cache()

        # Checkpoint
        if state.step % train_config.save_steps == 0:
            ckpt_dir = os.path.join(train_config.output_dir, f"checkpoint-{state.step}")
            save_checkpoint(model, optimizer, scheduler, state, train_config, ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")
            model.cuda()
            torch.cuda.empty_cache()

    # Final
    final_dir = os.path.join(train_config.output_dir, "final")
    save_checkpoint(model, optimizer, scheduler, state, train_config, final_dir)
    print(f"\nTraining complete! Final model saved to {final_dir}")

    eval_results = evaluate(model, val_loader, num_batches=100)
    print(f"  Final val_loss={eval_results['val_loss']:.4f} | val_ppl={eval_results['val_ppl']:.2f}")

    log_file.close()
    print(f"Training log: {train_config.output_dir}/training_log.txt")


if __name__ == "__main__":
    main()
