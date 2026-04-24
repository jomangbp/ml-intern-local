"""
Training Script for Hybrid SLM
Optimized for 6GB VRAM (WSL)

Memory optimizations for 6GB VRAM:
1. Batch size 1 with 128 gradient accumulation steps
2. Sequence length 1024 (vs 2048)
3. Gradient checkpointing (mandatory)
4. Periodic cache clearing
5. Memory-efficient data loading
6. bf16 mixed precision
"""

import os
import sys
import math
import time
import gc
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, HybridSLMModel
from src.layers import RMSNorm
from configs.model_config import HybridSLMConfig, TrainingConfig


@dataclass
class TrainingState:
    """Track training state"""
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    num_tokens: int = 0
    best_loss: float = float('inf')


class CosineAnnealingWarmup:
    """Cosine annealing with linear warmup"""
    
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
        
        # Base LR
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self):
        """Update learning rate"""
        self.last_step += 1
        lr = self.get_lr()
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = lr * base_lr / self.base_lrs[0] if self.base_lrs[0] > 0 else lr
    
    def get_lr(self) -> float:
        """Calculate current learning rate"""
        step = self.last_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class TokenizedDataset(Dataset):
    """Memory-efficient token dataset.

    Supports either:
    - raw binary token file (.bin) written as uint32
    - numpy array file (.npy/.npz)

    Uses memory mapping so dataset stays on disk (low RAM usage).
    """

    def __init__(
        self,
        data_path: str,
        max_length: int = 1024,
        stride: int = 256,
    ):
        self.max_length = max_length
        self.stride = stride

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Tokenized data not found at {data_path}. "
                "Run download_and_tokenize.py first."
            )

        path_lower = data_path.lower()
        if path_lower.endswith(".bin"):
            self.ids = np.memmap(data_path, dtype=np.uint32, mode="r")
        else:
            self.ids = np.load(data_path, mmap_mode="r")

        # Need +1 token because labels are next-token shifted
        self.size = max(0, len(self.ids) - (max_length + 1))

    def __len__(self):
        return max(0, self.size // self.stride)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_length + 1

        window = self.ids[start:end].astype(np.int64)
        input_ids = window[:-1]
        labels = window[1:]

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }


class TextDataset(Dataset):
    """
    On-the-fly tokenization dataset
    For smaller datasets or testing
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024  # Reduced for 6GB
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


def create_dataloaders(
    config: TrainingConfig,
    tokenizer,
    max_length: int = 1024
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders - optimized for memory"""
    
    # Training dataloader with memory efficiency
    train_dataset = TokenizedDataset(
        data_path=config.train_data_path,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,  # 1 for 6GB
        shuffle=True,
        num_workers=0,  # Disable workers to save memory
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x])
        }
    )
    
    # Validation dataloader
    val_dataset = TokenizedDataset(
        data_path=config.val_data_path,
        max_length=max_length
    )
    
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


def compute_loss_scale(model: nn.Module) -> float:
    """Compute the loss scale factor based on model size"""
    num_params = sum(p.numel() for p in model.parameters())
    return 1.0


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: AdamW,
    scheduler: CosineAnnealingWarmup,
    scaler: GradScaler,
    config: TrainingConfig,
    use_compile: bool = False
) -> Dict[str, float]:
    """
    Single training step with gradient accumulation
    Optimized for 6GB VRAM
    """
    model.train()
    
    # Move to device one tensor at a time to reduce peak memory
    input_ids = batch['input_ids'].cuda(non_blocking=True)
    labels = batch['labels'].cuda(non_blocking=True)
    
    # Reset gradients
    optimizer.zero_grad(set_to_none=True)  # Use set_to_none for memory savings
    
    total_loss = 0.0
    num_tokens = 0
    
    # Micro-batch loop with gradient accumulation
    # For batch_size=1, we do 128 accumulation steps
    micro_batch_size = input_ids.shape[0]
    
    for i in range(0, micro_batch_size, 1):  # Process one at a time
        micro_input = input_ids[i:i+1]
        micro_labels = labels[i:i+1]
        
        # Forward with mixed precision
        with autocast('cuda', dtype=torch.bfloat16, enabled=config.bf16):
            outputs = model(
                input_ids=micro_input,
                labels=micro_labels
            )
            loss = outputs['loss']
            loss = loss / config.gradient_accumulation_steps
        
        # Backward with scaling
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_tokens += micro_input.numel()
        
        # Clean up intermediate tensors
        del outputs, loss
        torch.cuda.empty_cache()
    
    # Gradient clipping
    if config.gradient_clipping > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clipping
        )
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Update learning rate
    scheduler.step()
    
    return {
        'loss': total_loss,
        'tokens': num_tokens,
        'lr': optimizer.param_groups[0]['lr']
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    num_batches: int = 20,  # Reduced for 6GB
    config: TrainingConfig = None
) -> Dict[str, float]:
    """
    Evaluate model on validation set - optimized for 6GB VRAM
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    num_examples = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        
        with autocast('cuda', dtype=torch.bfloat16, enabled=True):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
        
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        num_examples += input_ids.shape[0]
        
        # Clean up
        del outputs, loss, input_ids, labels
        torch.cuda.empty_cache()
    
    return {
        'val_loss': total_loss / total_tokens,
        'val_ppl': math.exp(total_loss / total_tokens),
        'num_examples': num_examples
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingWarmup,
    state: TrainingState,
    config: TrainingConfig,
    path: str
):
    """Save checkpoint"""
    os.makedirs(path, exist_ok=True)
    
    # Save model state separately to reduce memory during save
    checkpoint = {
        'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__,
        'state': {
            'step': state.step,
            'epoch': state.epoch,
            'best_loss': state.best_loss
        },
        'config': config
    }
    
    torch.save(checkpoint, f"{path}/checkpoint.pt")
    
    # Also save just the model weights for easy loading
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, f"{path}/pytorch_model.bin")
    
    # Save config
    import json
    with open(f"{path}/config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Re-load model to GPU
    model.cuda()


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[AdamW],
    scheduler: Optional[CosineAnnealingWarmup],
    config: TrainingConfig
) -> TrainingState:
    """Load checkpoint"""
    path = config.resume_from_checkpoint
    
    checkpoint = torch.load(f"{path}/checkpoint.pt", map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    state = TrainingState(
        step=checkpoint['state']['step'],
        epoch=checkpoint['state']['epoch'],
        best_loss=checkpoint['state']['best_loss']
    )
    
    del checkpoint
    gc.collect()
    
    print(f"Loaded checkpoint from step {state.step}")
    return state


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def main():
    """Main training function - optimized for 6GB VRAM"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train Hybrid SLM")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, no training")
    args = parser.parse_args()

    # Config
    model_config = HybridSLMConfig()
    train_config = TrainingConfig()

    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.resume:
        train_config.resume_from_checkpoint = args.resume

    print("\n" + "=" * 60)
    print("Hybrid SLM Training - 6GB VRAM Optimized")
    print("=" * 60)
    print(f"\nModel config (6GB optimized):")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_hidden_layers} "
          f"({model_config.num_hidden_layers // model_config.full_attention_interval} full, "
          f"{model_config.num_hidden_layers - model_config.num_hidden_layers // model_config.full_attention_interval} linear)")
    print(f"  Sequence length: {train_config.max_seq_length}")
    print(f"  Batch size: {train_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch: {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps * train_config.max_seq_length}")

    # Model
    print("\nInitializing model...")
    model = create_model(model_config)
    model = model.cuda()

    # Enable gradient checkpointing (mandatory for 6GB)
    if train_config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Scheduler
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        min_lr=train_config.min_learning_rate
    )

    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda')

    # Training state
    state = TrainingState()

    # Load checkpoint if resuming
    if train_config.resume_from_checkpoint:
        state = load_checkpoint(model, optimizer, scheduler, train_config)

    # Output directory
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Logging
    print(f"\nTraining config:")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Warmup steps: {train_config.warmup_steps}")
    print(f"  Max steps: {train_config.max_steps}")
    print(f"  Save every: {train_config.save_steps} steps")
    print(f"  Eval every: {train_config.eval_steps} steps")
    print(f"  Output dir: {train_config.output_dir}")

    # Quick test forward pass
    print("\nTesting forward pass...")
    test_ids = torch.randint(0, model_config.vocab_size, (1, 128)).cuda()
    test_labels = test_ids.clone()

    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=test_ids, labels=test_labels)
            print(f"  Output shape: {outputs['logits'].shape}")
            if outputs['loss'] is not None:
                print(f"  Loss: {outputs['loss'].item():.4f}")

    del test_ids, test_labels, outputs
    torch.cuda.empty_cache()

    print("\n✓ Setup complete!")
    print_gpu_memory()

    if args.eval_only:
        return model, optimizer, scheduler, state

    # ── Create dataloaders ──────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(train_config, tokenizer=None,
                                                   max_length=train_config.max_seq_length)
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples:   {len(val_loader.dataset):,}")
    print(f"  Train batches: {len(train_loader):,}")

    # ── Training loop ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    log_file = open(os.path.join(train_config.output_dir, "training_log.txt"), "a")
    start_time = time.time()

    # Keep a running iterator for the training set
    train_iter = iter(train_loader)

    for step in range(state.step, train_config.max_steps):
        # Fetch next batch (re-iterate when exhausted)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            state.epoch += 1

        # ── Single training step ────────────────────────────────────
        model.train()
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        labels = batch['labels'].cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward + backward
        with autocast('cuda', dtype=torch.bfloat16, enabled=train_config.bf16):
            outputs = model(
                input_ids=input_ids,
                labels=labels
            )
            loss = outputs['loss']

        scaler.scale(loss).backward()
        total_loss = loss.item()
        del outputs, loss

        # Gradient clipping
        if train_config.gradient_clipping > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        state.step = step + 1
        state.total_loss += total_loss
        state.num_tokens += input_ids.numel()

        del input_ids, labels

        # Periodic cache clearing for 6GB VRAM
        if step % train_config.empty_cache_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # ── Logging ─────────────────────────────────────────────────
        if state.step % train_config.logging_steps == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss  # loss of this step
            lr = optimizer.param_groups[0]['lr']
            tokens_per_sec = state.num_tokens / elapsed if elapsed > 0 else 0

            log_line = (f"step={state.step:,} | loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | tokens/s={tokens_per_sec:,.0f} | "
                        f"epoch={state.epoch} | mem={torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

        # ── Evaluation ──────────────────────────────────────────────
        if state.step % train_config.eval_steps == 0:
            print(f"\n--- Evaluating at step {state.step} ---")
            eval_results = evaluate(model, val_loader, num_batches=50, config=train_config)
            val_loss = eval_results['val_loss']
            val_ppl = eval_results['val_ppl']
            eval_line = f"  val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}"
            print(eval_line)
            log_file.write(eval_line + "\n")
            log_file.flush()

            # Save best model
            if val_loss < state.best_loss:
                state.best_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, state, train_config,
                                os.path.join(train_config.output_dir, "best"))
                print(f"  ★ New best model (val_loss={val_loss:.4f})")

            model.cuda()  # ensure back on GPU after eval
            torch.cuda.empty_cache()

        # ── Checkpoint ──────────────────────────────────────────────
        if state.step % train_config.save_steps == 0:
            ckpt_dir = os.path.join(train_config.output_dir, f"checkpoint-{state.step}")
            save_checkpoint(model, optimizer, scheduler, state, train_config, ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")
            model.cuda()
            torch.cuda.empty_cache()

    # ── Final save ──────────────────────────────────────────────────
    final_dir = os.path.join(train_config.output_dir, "final")
    save_checkpoint(model, optimizer, scheduler, state, train_config, final_dir)
    print(f"\nTraining complete! Final model saved to {final_dir}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    eval_results = evaluate(model, val_loader, num_batches=100, config=train_config)
    print(f"  val_loss={eval_results['val_loss']:.4f} | val_ppl={eval_results['val_ppl']:.2f}")

    # Save training summary
    summary = {
        "total_steps": state.step,
        "epochs": state.epoch,
        "best_val_loss": state.best_loss,
        "final_val_loss": eval_results['val_loss'],
        "final_val_ppl": eval_results['val_ppl'],
        "total_tokens": state.num_tokens,
        "training_time_sec": time.time() - start_time,
        "model_params": sum(p.numel() for p in model.parameters()),
        "config": {
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "vocab_size": model_config.vocab_size,
            "max_seq_length": train_config.max_seq_length,
        }
    }
    with open(os.path.join(train_config.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log_file.close()
    print(f"\nTraining log saved to {train_config.output_dir}/training_log.txt")

    return model, optimizer, scheduler, state


if __name__ == "__main__":
    main()
