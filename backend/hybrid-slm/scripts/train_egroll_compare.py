"""
EGGROLL Comparison Training Script

Compares EGGROLL vs Backprop training on hybrid-slm over 5000 steps.
Saves checkpoints at each save_steps for later comparison.

Usage:
    python train_egroll_compare.py --method both --max-steps 5000
"""

import os
import sys
import math
import time
import gc
import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

import trackio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, HybridSLMModel
from src.egroll import EGGROLLTrainerOptimized, EGGROLLConfig
from src.layers import RMSNorm
from configs.model_config import HybridSLMConfig
from scripts.train import (
    CosineAnnealingWarmup, StreamingTokenDataset, TokenizedDataset,
    create_dataloaders, evaluate, save_checkpoint, load_checkpoint,
    print_gpu_memory
)


@dataclass
class ComparisonConfig:
    """Configuration for comparison training"""
    # Training method
    method: str = "both"  # "backprop", "egroll", "both"
    
    # Common settings
    max_steps: int = 5000
    save_steps: int = 1000  # Save checkpoints every N steps
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Data - same TinyStories-only data as v1 baseline and Gemma4
    train_data_path: str = "data/train_tokens.bin"
    val_data_path: str = "data/val_tokens.bin"
    max_seq_length: int = 1024
    
    # Batch settings
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Effective batch = 4
    
    # Output
    output_dir: str = "outputs/comparison_egroll_vs_backprop"
    
    # EGGROLL specific
    egroll_population: int = 32
    egroll_rank: int = 2
    egroll_noise_scale: float = 0.001
    egroll_lr: float = 5e-4
    
    # Backprop specific
    backprop_lr: float = 4e-4
    backprop_warmup: int = 2000
    backprop_min_lr: float = 4e-5
    
    # Mixed precision
    bf16: bool = True
    
    # Hardware
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1


def train_backprop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ComparisonConfig,
    output_subdir: str
) -> Dict:
    """Train with standard backprop"""
    print("\n" + "=" * 60)
    print("BACKPROP TRAINING")
    print("=" * 60)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.backprop_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=config.backprop_warmup,
        max_steps=config.max_steps,
        min_lr=config.backprop_min_lr
    )
    
    # Grad scaler
    scaler = GradScaler('cuda')
    
    # State
    state = {
        'step': 0,
        'epoch': 0,
        'total_loss': 0.0,
        'num_tokens': 0,
        'best_loss': float('inf'),
        'start_time': time.time(),
        'metrics': []
    }
    
    train_iter = iter(train_loader)
    
    while state['step'] < config.max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        
        for _ in range(config.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                state['epoch'] += 1
            
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
            
            # Forward
            with autocast('cuda', dtype=torch.bfloat16, enabled=config.bf16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            step_loss += loss.item() * config.gradient_accumulation_steps
            step_tokens += input_ids.numel()
            
            del outputs, loss, input_ids, labels
        
        # Step
        if config.gradient_clipping > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        state['step'] += 1
        state['total_loss'] += step_loss
        state['num_tokens'] += step_tokens
        
        # Logging
        if state['step'] % config.logging_steps == 0:
            elapsed = time.time() - state['start_time']
            lr = optimizer.param_groups[0]['lr']
            tokens_per_sec = state['num_tokens'] / elapsed if elapsed > 0 else 0
            
            log_dict = {
                'step': state['step'],
                'loss': step_loss,
                'lr': lr,
                'tokens_per_sec': tokens_per_sec,
                'epoch': state['epoch']
            }
            state['metrics'].append(log_dict)
            
            print(f"[Backprop] step={state['step']:,} | loss={step_loss:.4f} | "
                  f"lr={lr:.2e} | tokens/s={tokens_per_sec:,.0f}")
        
        # Eval & Save
        if state['step'] % config.eval_steps == 0:
            eval_results = evaluate(model, val_loader, num_batches=30)
            print(f"[Backprop Eval] val_loss={eval_results['val_loss']:.4f} | "
                  f"val_ppl={eval_results['val_ppl']:.2f}")
            
            trackio.log({
                'backprop/val_loss': eval_results['val_loss'],
                'backprop/val_ppl': eval_results['val_ppl']
            })
        
        if state['step'] % config.save_steps == 0:
            ckpt_dir = os.path.join(config.output_dir, output_subdir, f"step_{state['step']}")
            save_checkpoint_simple(model.state_dict(), ckpt_dir)
            print(f"  [Backprop] Saved: {ckpt_dir}")
    
    return state


def train_egroll(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ComparisonConfig,
    output_subdir: str
) -> Dict:
    """Train with EGGROLL"""
    print("\n" + "=" * 60)
    print("EGGROLL TRAINING")
    print("=" * 60)
    print(f"Population: {config.egroll_population}, Rank: {config.egroll_rank}")
    
    # Create EGGROLL trainer
    egroll_config = EGGROLLConfig(
        population_size=config.egroll_population,
        rank=config.egroll_rank,
        noise_scale=config.egroll_noise_scale,
        learning_rate=config.egroll_lr,
        device="cuda"
    )
    
    def fitness_fn(m, batch):
        with torch.no_grad():
            outputs = m(input_ids=batch['input_ids'], labels=batch['labels'])
            return -outputs['loss'].item()
    
    trainer = EGGROLLTrainerOptimized(model, egroll_config, fitness_fn)
    
    # State
    state = {
        'step': 0,
        'epoch': 0,
        'total_reward': 0.0,
        'num_tokens': 0,
        'best_reward': float('-inf'),
        'start_time': time.time(),
        'metrics': []
    }
    
    train_iter = iter(train_loader)
    
    while state['step'] < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            state['epoch'] += 1
        
        # Move batch to GPU
        batch = {
            'input_ids': batch['input_ids'].cuda(non_blocking=True),
            'labels': batch['labels'].cuda(non_blocking=True)
        }
        
        # EGGROLL step
        step_metrics = trainer.step(batch)
        state['step'] += 1
        state['total_reward'] += step_metrics['reward_mean']
        state['num_tokens'] += batch['input_ids'].numel() * config.egroll_population
        state['best_reward'] = max(state['best_reward'], step_metrics['reward_mean'])
        
        # Logging
        if state['step'] % config.logging_steps == 0:
            elapsed = time.time() - state['start_time']
            tokens_per_sec = state['num_tokens'] / elapsed if elapsed > 0 else 0
            
            log_dict = {
                'step': state['step'],
                'reward_mean': step_metrics['reward_mean'],
                'reward_std': step_metrics['reward_std'],
                'best_reward': state['best_reward'],
                'tokens_per_sec': tokens_per_sec,
                'epoch': state['epoch']
            }
            state['metrics'].append(log_dict)
            
            print(f"[EGGROLL] step={state['step']:,} | reward={step_metrics['reward_mean']:.4f} "
                  f"(±{step_metrics['reward_std']:.4f}) | best={state['best_reward']:.4f} | "
                  f"tokens/s={tokens_per_sec:,.0f}")
        
        # Eval & Save
        if state['step'] % config.eval_steps == 0:
            eval_results = evaluate(model, val_loader, num_batches=10)
            print(f"[EGGROLL Eval] val_loss={eval_results['val_loss']:.4f} | "
                  f"val_ppl={eval_results['val_ppl']:.2f}")
            
            trackio.log({
                'egroll/val_loss': eval_results['val_loss'],
                'egroll/val_ppl': eval_results['val_ppl']
            })
        
        if state['step'] % config.save_steps == 0:
            ckpt_dir = os.path.join(config.output_dir, output_subdir, f"step_{state['step']}")
            save_checkpoint_simple(model.state_dict(), ckpt_dir)
            print(f"  [EGGROLL] Saved: {ckpt_dir}")
        
        del batch
        torch.cuda.empty_cache()
    
    return state


def save_checkpoint_simple(state_dict: Dict, path: str):
    """Save model checkpoint"""
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
    
    with open(os.path.join(path, "config.json"), 'w') as f:
        json.dump({'source': 'egroll_comparison'}, f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EGGROLL vs Backprop Comparison")
    parser.add_argument("--method", choices=["backprop", "egroll", "both"], default="both")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--population", type=int, default=32, help="EGGROLL population size")
    args = parser.parse_args()
    
    config = ComparisonConfig(
        method=args.method,
        max_steps=args.max_steps,
        egroll_population=args.population
    )
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trackio
    trackio.init(
        project="hybrid-slm",
        name=f"egroll_vs_backprop_{config.max_steps}steps",
        config={
            'method': config.method,
            'max_steps': config.max_steps,
            'population': config.egroll_population,
            'rank': config.egroll_rank
        }
    )
    
    # Create model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
    model_config = HybridSLMConfig()
    model = create_model(model_config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Save initial checkpoint
    os.makedirs(os.path.join(config.output_dir, "initial"), exist_ok=True)
    save_checkpoint_simple(model.state_dict(), os.path.join(config.output_dir, "initial"))
    
    # Create dataloaders
    print("\nLoading data...")
    
    train_dataset = TokenizedDataset(
        data_path=config.train_data_path,
        max_length=config.max_seq_length,
        stride=256
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataset = TokenizedDataset(
        data_path=config.val_data_path,
        max_length=config.max_seq_length
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_batch_size,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    results = {}
    
    # Train backprop
    if config.method in ["backprop", "both"]:
        model_bp = create_model(model_config)
        model_bp = model_bp.cuda()
        
        results['backprop'] = train_backprop(
            model_bp, train_loader, val_loader, config, "backprop"
        )
    
    # Train EGGROLL
    if config.method in ["egroll", "both"]:
        model_es = create_model(model_config)
        model_es = model_es.cuda()
        
        results['egroll'] = train_egroll(
            model_es, train_loader, val_loader, config, "egroll"
        )
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    comparison_summary = {
        'config': {
            'max_steps': config.max_steps,
            'egroll_population': config.egroll_population,
            'egroll_rank': config.egroll_rank
        },
        'results': results,
        'checkpoints': {
            'initial': os.path.join(config.output_dir, "initial"),
            'backprop': os.path.join(config.output_dir, "backprop"),
            'egroll': os.path.join(config.output_dir, "egroll")
        }
    }
    
    with open(os.path.join(config.output_dir, "comparison_summary.json"), 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    
    print(f"\nComparison complete! Results saved to: {config.output_dir}")
    print("\nCheckpoints available at:")
    print(f"  - Initial: {os.path.join(config.output_dir, 'initial')}")
    if 'backprop' in results:
        print(f"  - Backprop final: {os.path.join(config.output_dir, 'backprop', f'step_{config.max_steps}')}")
    if 'egroll' in results:
        print(f"  - EGGROLL final: {os.path.join(config.output_dir, 'egroll', f'step_{config.max_steps}')}")
    
    trackio.finish()


if __name__ == "__main__":
    main()
