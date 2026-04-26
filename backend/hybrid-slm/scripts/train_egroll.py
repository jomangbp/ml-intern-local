"""
Exclusive EGGROLL Training Script for Hybrid SLM

Pure Evolution Strategies training with low-rank perturbations.
Based on arXiv:2511.16652 - "Evolution Strategies at the Hyperscale"

Usage:
    python scripts/train_egroll.py --max-steps 100000
    python scripts/train_egroll.py --max-steps 50000 --resume outputs/egroll-exclusive/checkpoint-10000
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
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np

import trackio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, HybridSLMModel
from src.egroll import EGGROLLTrainerOptimized, EGGROLLConfig, LowRankPerturbation
from configs.model_config import HybridSLMConfig
from scripts.train import (
    TokenizedDataset, StreamingTokenDataset,
    evaluate, print_gpu_memory
)


@dataclass
class EgRollTrainConfig:
    """Configuration for exclusive EgRoll training"""
    # Training
    max_steps: int = 100000
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 10

    # Data
    train_data_path: str = "data/combined_train_tokens.bin"
    val_data_path: str = "data/combined_val_tokens.bin"
    max_seq_length: int = 1024

    # Batch - EgRoll processes 1 sample per step (population handles diversity)
    batch_size: int = 2

    # EGGROLL hyperparameters (from paper + tuned)
    population_size: int = 32      # Number of perturbed models per step
    rank: int = 2                  # Low-rank perturbation rank
    noise_scale: float = 0.001     # Perturbation magnitude σ
    learning_rate: float = 5e-4    # Update step size α

    # Mixed precision
    bf16: bool = True

    # Output
    output_dir: str = "outputs/egroll-exclusive"

    # Resume
    resume_from: Optional[str] = None


class EgRollTrainer:
    """
    Production EgRoll trainer with:
    - Streaming data loading
    - Periodic evaluation
    - Checkpoint saving / resuming
    - Trackio logging
    - Anomaly detection (loss spikes, NaN)
    """

    def __init__(self, config: EgRollTrainConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # State
        self.step = 0
        self.epoch = 0
        self.best_reward = float('-inf')
        self.best_val_loss = float('inf')
        self.num_tokens = 0
        self.start_time = time.time()
        self.loss_history: List[float] = []
        self.anomaly_count = 0

        # Build model
        self.model_config = HybridSLMConfig()
        self.model = create_model(self.model_config).to(self.device)
        self.model.eval()  # EgRoll doesn't use gradients on the main model

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {num_params:,} params ({num_params/1e6:.2f}M)")

        # EgRoll config
        self.egroll_config = EGGROLLConfig(
            population_size=config.population_size,
            rank=config.rank,
            noise_scale=config.noise_scale,
            learning_rate=config.learning_rate,
            device=self.device,
        )

        # Build parameter structure for efficient updates
        self._build_param_structure()

        # Pre-generate random seeds
        self.rng = np.random.default_rng(42)

        # Resume if specified
        if config.resume_from:
            self._resume(config.resume_from)

    def _build_param_structure(self):
        """Cache parameter names, shapes, and references"""
        self.param_names = []
        self.param_shapes = []
        self.param_refs = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.is_leaf:
                self.param_names.append(name)
                self.param_shapes.append(tuple(param.shape))
                self.param_refs.append(param)
        print(f"Trainable parameter groups: {len(self.param_names)}")

    def _resume(self, path: str):
        """Resume from checkpoint"""
        ckpt_file = os.path.join(path, "checkpoint.pt")
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.step = ckpt.get('step', 0)
            self.epoch = ckpt.get('epoch', 0)
            self.best_reward = ckpt.get('best_reward', float('-inf'))
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.num_tokens = ckpt.get('num_tokens', 0)
            print(f"Resumed from step {self.step}")
            del ckpt
            gc.collect()
        else:
            # Try loading just the model weights
            model_file = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"Loaded model weights from {path}")
                del state_dict

    def fitness_fn(self, model: nn.Module, batch: Dict) -> float:
        """Evaluate model fitness = negative cross-entropy loss"""
        with torch.no_grad():
            with autocast('cuda', dtype=torch.bfloat16, enabled=self.config.bf16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels']
                )
                loss = outputs['loss']
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    return -20.0  # Penalty for broken models
                return -loss.item()

    def _egroll_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single EgRoll optimization step:
        1. Clone current parameters
        2. Generate N low-rank perturbations
        3. Evaluate each perturbed model
        4. Normalize rewards (z-score)
        5. Aggregate weighted perturbations
        6. Update parameters
        """
        # Snapshot current parameters
        original_state = {
            name: param.data.clone()
            for name, param in zip(self.param_names, self.param_refs)
        }

        # Collect rewards and perturbations
        all_rewards = []
        perturbations_by_layer = [[] for _ in self.param_names]

        for member_idx in range(self.egroll_config.population_size):
            # Generate perturbation and apply
            pert_state = {}
            for i, (name, shape) in enumerate(zip(self.param_names, self.param_shapes)):
                lr_perturb = LowRankPerturbation(shape, self.egroll_config.rank, self.device)
                eps = lr_perturb.sample()
                pert_state[name] = original_state[name] + self.egroll_config.noise_scale * eps
                perturbations_by_layer[i].append(eps)

            # Load and evaluate
            self.model.load_state_dict(pert_state, strict=False)
            reward = self.fitness_fn(self.model, batch)
            all_rewards.append(reward)

        # Restore original
        self.model.load_state_dict(
            {name: original_state[name] for name in self.param_names},
            strict=False
        )
        # Fill in non-param entries from current state
        full_state = self.model.state_dict()
        for name in original_state:
            full_state[name] = original_state[name]
        self.model.load_state_dict(full_state)

        # Z-score normalize rewards
        rewards_t = torch.tensor(all_rewards, device=self.device)
        r_mean = rewards_t.mean()
        r_std = rewards_t.std()
        if r_std < 1e-8:
            r_std = torch.tensor(1.0, device=self.device)
        norm_rewards = (rewards_t - r_mean) / r_std

        # Aggregate and apply update
        with torch.no_grad():
            for i, param_ref in enumerate(self.param_refs):
                update = torch.zeros_like(param_ref.data)
                for j, eps in enumerate(perturbations_by_layer[i]):
                    update += norm_rewards[j] * eps
                param_ref.data.add_(
                    self.egroll_config.learning_rate * update / self.egroll_config.population_size
                )

        # Current loss = negative best reward (approximate)
        current_loss = -r_mean.item()

        return {
            'reward_mean': r_mean.item(),
            'reward_std': r_std.item(),
            'current_loss': current_loss,
            'best_reward': max(self.best_reward, r_mean.item()),
        }

    def _save_checkpoint(self, tag: str):
        """Save training checkpoint"""
        path = os.path.join(self.config.output_dir, tag)
        os.makedirs(path, exist_ok=True)

        ckpt = {
            'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'step': self.step,
            'epoch': self.epoch,
            'best_reward': self.best_reward,
            'best_val_loss': self.best_val_loss,
            'num_tokens': self.num_tokens,
            'config': {
                'population_size': self.egroll_config.population_size,
                'rank': self.egroll_config.rank,
                'noise_scale': self.egroll_config.noise_scale,
                'learning_rate': self.egroll_config.learning_rate,
            }
        }
        torch.save(ckpt, os.path.join(path, "checkpoint.pt"))
        torch.save(
            {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            os.path.join(path, "pytorch_model.bin")
        )
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump({
                'model_params': sum(p.numel() for p in self.model.parameters()),
                'step': self.step,
                'egroll': {
                    'population': self.egroll_config.population_size,
                    'rank': self.egroll_config.rank,
                    'noise_scale': self.egroll_config.noise_scale,
                    'lr': self.egroll_config.learning_rate,
                }
            }, f, indent=2)

        self.model.to(self.device)
        print(f"  Saved checkpoint: {path}")

    def _check_anomaly(self, loss: float) -> bool:
        """Detect training anomalies: NaN, Inf, sudden spikes"""
        if math.isnan(loss) or math.isinf(loss):
            self.anomaly_count += 1
            return True

        self.loss_history.append(loss)
        if len(self.loss_history) >= 50:
            recent = self.loss_history[-10:]
            older = self.loss_history[-50:-10]
            if len(older) > 0:
                recent_mean = sum(recent) / len(recent)
                older_mean = sum(older) / len(older)
                # Spike: recent loss more than 3x the older average
                if recent_mean > older_mean * 3.0 and older_mean > 0:
                    self.anomaly_count += 1
                    return True
        return False

    def train(self):
        """Main training loop"""
        # ── Data ─────────────────────────────────────────────
        print("\nLoading data...")
        train_dataset = StreamingTokenDataset(
            data_path=self.config.train_data_path,
            max_length=self.config.max_seq_length,
            stride=256,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda x: {
                'input_ids': torch.stack([item['input_ids'] for item in x]),
                'labels': torch.stack([item['labels'] for item in x])
            }
        )

        val_dataset = TokenizedDataset(
            data_path=self.config.val_data_path,
            max_length=self.config.max_seq_length,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: {
                'input_ids': torch.stack([item['input_ids'] for item in x]),
                'labels': torch.stack([item['labels'] for item in x])
            }
        )

        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples:   {len(val_dataset):,}")

        # ── Trackio ──────────────────────────────────────────
        run_name = f"egroll-exclusive-p{self.config.population_size}-r{self.config.rank}"
        trackio.init(
            project="hybrid-slm",
            name=run_name,
            config={
                "method": "egroll",
                "population_size": self.config.population_size,
                "rank": self.config.rank,
                "noise_scale": self.config.noise_scale,
                "learning_rate": self.config.learning_rate,
                "max_steps": self.config.max_steps,
                "batch_size": self.config.batch_size,
                "seq_length": self.config.max_seq_length,
                "model_params": sum(p.numel() for p in self.model.parameters()),
                "dataset": "combined (TinyStories + FineWeb-Edu, 522M tokens)",
                "resume_from_step": self.step,
            },
        )

        # ── Open log file ────────────────────────────────────
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_file = open(os.path.join(self.config.output_dir, "training_log.txt"), "a")

        # ── Header ───────────────────────────────────────────
        print("\n" + "=" * 60)
        print("EGGROLL EXCLUSIVE TRAINING")
        print("=" * 60)
        print(f"  Population: {self.config.population_size}")
        print(f"  Rank:       {self.config.rank}")
        print(f"  Noise σ:    {self.config.noise_scale}")
        print(f"  LR α:       {self.config.learning_rate}")
        print(f"  Max steps:  {self.config.max_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Seq length: {self.config.max_seq_length}")
        print(f"  Output:     {self.config.output_dir}")
        print("=" * 60)

        # ── Training loop ────────────────────────────────────
        train_iter = iter(train_loader)

        for step in range(self.step, self.config.max_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                self.epoch += 1

            # Move to GPU
            gpu_batch = {
                'input_ids': batch['input_ids'].to(self.device, non_blocking=True),
                'labels': batch['labels'].to(self.device, non_blocking=True),
            }

            # EgRoll step
            metrics = self._egroll_step(gpu_batch)
            self.step = step + 1
            self.num_tokens += gpu_batch['input_ids'].numel()
            self.best_reward = max(self.best_reward, metrics['reward_mean'])

            # Anomaly check
            is_anomaly = self._check_anomaly(metrics['current_loss'])

            # ── Logging ──────────────────────────────────────
            if self.step % self.config.logging_steps == 0:
                elapsed = time.time() - self.start_time
                tokens_per_sec = self.num_tokens / elapsed if elapsed > 0 else 0
                mem_gb = torch.cuda.memory_allocated() / 1024**3

                log_line = (
                    f"step={self.step:,} | loss={metrics['current_loss']:.4f} | "
                    f"reward={metrics['reward_mean']:.4f} (±{metrics['reward_std']:.4f}) | "
                    f"best_reward={self.best_reward:.4f} | "
                    f"tokens/s={tokens_per_sec:,.0f} | epoch={self.epoch} | "
                    f"mem={mem_gb:.2f}GB"
                    + (" | ⚠️ ANOMALY" if is_anomaly else "")
                )
                print(log_line)
                log_file.write(log_line + "\n")
                log_file.flush()

                trackio.log({
                    "train/loss": metrics['current_loss'],
                    "train/reward_mean": metrics['reward_mean'],
                    "train/reward_std": metrics['reward_std'],
                    "train/best_reward": self.best_reward,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens_seen": self.num_tokens,
                    "train/gpu_memory_gb": mem_gb,
                    "train/epoch": self.epoch,
                    "train/anomaly_count": self.anomaly_count,
                })

            # ── Evaluation ───────────────────────────────────
            if self.step % self.config.eval_steps == 0:
                print(f"\n--- Evaluating at step {self.step} ---")
                eval_results = evaluate(self.model, val_loader, num_batches=30)
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

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
                    print(f"  ★ New best model (val_loss={val_loss:.4f})")

                self.model.to(self.device)
                torch.cuda.empty_cache()

            # ── Checkpoint ───────────────────────────────────
            if self.step % self.config.save_steps == 0:
                self._save_checkpoint(f"checkpoint-{self.step}")

            # Cleanup
            del gpu_batch, batch
            if self.step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # ── Final ────────────────────────────────────────────
        self._save_checkpoint("final")

        print("\n--- Final Evaluation ---")
        eval_results = evaluate(self.model, val_loader, num_batches=100)
        final_line = f"  val_loss={eval_results['val_loss']:.4f} | val_ppl={eval_results['val_ppl']:.2f}"
        print(final_line)
        log_file.write(final_line + "\n")

        # Summary
        summary = {
            "method": "egroll",
            "total_steps": self.step,
            "epochs": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_reward": self.best_reward,
            "final_val_loss": eval_results['val_loss'],
            "final_val_ppl": eval_results['val_ppl'],
            "total_tokens": self.num_tokens,
            "training_time_sec": time.time() - self.start_time,
            "anomaly_count": self.anomaly_count,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "egroll_config": {
                "population": self.egroll_config.population_size,
                "rank": self.egroll_config.rank,
                "noise_scale": self.egroll_config.noise_scale,
                "learning_rate": self.egroll_config.learning_rate,
            }
        }
        with open(os.path.join(self.config.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        log_file.close()
        trackio.finish()
        print(f"\nDone! Results in {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Exclusive EgRoll Training")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--noise-scale", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = EgRollTrainConfig(
        max_steps=args.max_steps,
        population_size=args.population,
        rank=args.rank,
        noise_scale=args.noise_scale,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        resume_from=args.resume,
    )

    trainer = EgRollTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
