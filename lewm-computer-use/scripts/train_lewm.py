#!/usr/bin/env python3
"""
Self-contained LeWorldModel training for computer-use GUI data.

Uses the LeWM model architecture (jepa.py, module.py) directly with PyTorch.
No dependency on stable-worldmodel or stable-pretraining.

Architecture:
    Screenshot (H,W,3) → ViT-Tiny → Projector → z_t (192-dim)
    Action (5-dim)     → Embedder → AdaLN → ARPredictor → ẑ_{t+1}
    Loss: MSE(ẑ_{t+1}, z_{t+1}) + λ·SIGReg(z)

Usage:
    # Test with dummy data (no GPU needed)
    python scripts/train_lewm.py --dummy-data --epochs 2

    # Train on real trajectories
    python scripts/train_lewm.py --data data/gui_trajectories.h5 --epochs 50 --lr 5e-5
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add le-wm repo to path
LEWM_PATH = Path("/home/jgbla/repos/le-wm")
sys.path.insert(0, str(LEWM_PATH))
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg


# ──────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────

class HDF5GuiDataset(Dataset):
    """
    Loads GUI trajectory HDF5 and returns (context_frames, actions, target_frame).
    
    Each sample:
      - context_pixels: (ctx_len, H, W, 3) — context screenshots
      - context_actions: (ctx_len, action_dim) — corresponding actions
      - target_pixels: (1, H, W, 3) — next screenshot to predict
    
    For LeWM: encoder gets context_pixels → z_ctx, predictor gets (z_ctx, acts) → ẑ_next
    """
    
    def __init__(
        self,
        h5_path: Path,
        context_len: int = 3,
        num_preds: int = 1,
        img_size: int = 224,
        normalize_pixels: bool = True,
    ):
        import h5py
        self.h5_path = h5_path
        self.context_len = context_len
        self.num_preds = num_preds
        self.img_size = img_size
        self.normalize_pixels = normalize_pixels
        
        with h5py.File(h5_path, "r") as f:
            self.total_frames = f["pixels"].shape[0]
            self.action_dim = f.attrs["action_dim"]
            self.orig_h = f.attrs.get("img_height", f["pixels"].shape[1])
            self.orig_w = f.attrs.get("img_width", f["pixels"].shape[2])
            self.pixels_dset = f["pixels"]
            self.action_dset = f["action"]
            
            # Read everything into memory for speed (GUI datasets are much smaller than video)
            print(f"  Loading {self.total_frames} frames into memory...")
            self.pixels = self.pixels_dset[:]
            self.actions = self.action_dset[:]
        
        # Resize if needed
        if self.orig_h != img_size or self.orig_w != img_size:
            print(f"  Resizing from ({self.orig_h},{self.orig_w}) to ({img_size},{img_size})...")
            resized = np.zeros((self.total_frames, img_size, img_size, 3), dtype=np.uint8)
            for i in range(self.total_frames):
                img = Image.fromarray(self.pixels[i])
                resized[i] = np.array(img.resize((img_size, img_size)))
            self.pixels = resized
        
        # Compute number of valid training samples
        self.num_samples = self.total_frames - context_len - num_preds
        print(f"  Dataset: {self.num_samples} training samples "
              f"(ctx={context_len}, preds={num_preds})")
    
    def __len__(self):
        return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        # Context: frames [idx, idx+context_len)
        ctx_pixels = self.pixels[idx:idx + self.context_len]
        ctx_actions = self.actions[idx:idx + self.context_len]
        
        # Target: next frame
        tgt_idx = idx + self.context_len + self.num_preds - 1
        tgt_pixels = self.pixels[tgt_idx:tgt_idx + 1]
        
        # Convert to float and normalize
        ctx_pixels = torch.from_numpy(ctx_pixels).float() / 255.0  # [0, 1]
        ctx_actions = torch.from_numpy(ctx_actions).float()
        tgt_pixels = torch.from_numpy(tgt_pixels).float() / 255.0
        
        # CHW format for ViT
        ctx_pixels = ctx_pixels.permute(0, 3, 1, 2)  # (T, C, H, W)
        tgt_pixels = tgt_pixels.permute(0, 3, 1, 2)
        
        return {
            "pixels": ctx_pixels,      # (ctx_len, 3, H, W)
            "action": ctx_actions,     # (ctx_len, action_dim)
            "tgt_pixels": tgt_pixels,  # (1, 3, H, W)
        }


class DummyGuiDataset(Dataset):
    """Synthetic dataset for testing the pipeline."""
    
    def __init__(self, num_samples=500, context_len=3, img_size=128, action_dim=5):
        self.num_samples = num_samples
        self.context_len = context_len
        self.img_size = img_size
        self.action_dim = action_dim
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        ctx_pixels = torch.rand(self.context_len, 3, self.img_size, self.img_size)
        ctx_actions = torch.randn(self.context_len, self.action_dim) * 0.1
        tgt_pixels = torch.rand(1, 3, self.img_size, self.img_size)
        return {"pixels": ctx_pixels, "action": ctx_actions, "tgt_pixels": tgt_pixels}


# ──────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────

def build_lewm(
    img_size: int = 128,
    patch_size: int = 14,
    embed_dim: int = 192,
    history_size: int = 3,
    action_dim: int = 5,
    encoder_scale: str = "tiny",
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dropout: float = 0.1,
):
    """
    Build the LeWM model.
    
    Uses HuggingFace ViT as encoder backbone + custom ARPredictor.
    """
    from transformers import ViTConfig, ViTModel
    
    # Map scale to config
    scale_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    }
    sc = scale_configs.get(encoder_scale, scale_configs["tiny"])
    
    vit_config = ViTConfig(
        image_size=img_size,
        patch_size=patch_size,
        hidden_size=sc["hidden_size"],
        num_hidden_layers=sc["num_hidden_layers"],
        num_attention_heads=sc["num_attention_heads"],
        intermediate_size=sc["hidden_size"] * 4,
    )
    
    encoder = ViTModel(vit_config)
    hidden_dim = sc["hidden_size"]
    
    # ARPredictor
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=predictor_depth,
        heads=predictor_heads,
        mlp_dim=predictor_mlp_dim,
        dim_head=64,
        dropout=predictor_dropout,
        emb_dropout=0.0,
    )
    
    # Action encoder
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=action_dim,
        emb_dim=embed_dim,
        mlp_scale=4,
    )
    
    # Projectors
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )
    
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    
    return model


# ──────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────

def train_epoch(
    model: JEPA,
    sigreg: SIGReg,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambd: float = 0.1,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
):
    """One training epoch."""
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        pixels = batch["pixels"].to(device)       # (B, ctx_len, 3, H, W)
        actions = batch["action"].to(device)       # (B, ctx_len, action_dim)
        tgt_pixels = batch["tgt_pixels"].to(device) # (B, 1, 3, H, W)
        
        B, T, C, H, W = pixels.shape
        
        # Flatten for ViT: (B*T, C, H, W)
        pixels_flat = pixels.reshape(B * T, C, H, W)
        actions_flat = actions.reshape(B * T, -1)
        
        # Encode
        vit_output = model.encoder(pixels_flat)
        cls_tokens = vit_output.last_hidden_state[:, 0]  # (B*T, hidden_dim)
        emb_flat = model.projector(cls_tokens)            # (B*T, embed_dim)
        emb = emb_flat.reshape(B, T, -1)                  # (B, T, embed_dim)
        
        # Encode target
        tgt_flat = tgt_pixels.reshape(B * 1, C, H, W)
        tgt_vit = model.encoder(tgt_flat)
        tgt_cls = tgt_vit.last_hidden_state[:, 0]
        tgt_emb = model.projector(tgt_cls).reshape(B, 1, -1)
        
        # Encode actions
        act_emb = model.action_encoder(actions)  # (B, T, embed_dim)
        
        # Predict next embedding
        pred_emb = model.predictor(emb, act_emb)  # (B, T, hidden_dim)
        pred_emb = model.pred_proj(pred_emb.reshape(B * T, -1))
        pred_emb = pred_emb.reshape(B, T, -1)
        
        # Loss: predict last position against target
        pred_loss = F.mse_loss(pred_emb[:, -1], tgt_emb[:, 0])
        
        # SIGReg on all embeddings
        emb_for_sigreg = emb.transpose(0, 1)  # (T, B, embed_dim)
        sigreg_loss = sigreg(emb_for_sigreg)
        
        loss = pred_loss + lambd * sigreg_loss
        loss = loss / gradient_accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_pred_loss += pred_loss.item()
        total_sigreg_loss += sigreg_loss.item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "pred_loss": total_pred_loss / n_batches,
        "sigreg_loss": total_sigreg_loss / n_batches,
    }


@torch.no_grad()
def validate_epoch(
    model: JEPA,
    sigreg: SIGReg,
    dataloader: DataLoader,
    device: torch.device,
    lambd: float = 0.1,
):
    """One validation epoch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        pixels = batch["pixels"].to(device)
        actions = batch["action"].to(device)
        tgt_pixels = batch["tgt_pixels"].to(device)
        
        B, T, C, H, W = pixels.shape
        
        pixels_flat = pixels.reshape(B * T, C, H, W)
        actions_flat = actions.reshape(B * T, -1)
        
        vit_output = model.encoder(pixels_flat)
        cls_tokens = vit_output.last_hidden_state[:, 0]
        emb_flat = model.projector(cls_tokens)
        emb = emb_flat.reshape(B, T, -1)
        
        tgt_flat = tgt_pixels.reshape(B * 1, C, H, W)
        tgt_vit = model.encoder(tgt_flat)
        tgt_cls = tgt_vit.last_hidden_state[:, 0]
        tgt_emb = model.projector(tgt_cls).reshape(B, 1, -1)
        
        act_emb = model.action_encoder(actions)
        pred_emb = model.predictor(emb, act_emb)
        pred_emb = model.pred_proj(pred_emb.reshape(B * T, -1))
        pred_emb = pred_emb.reshape(B, T, -1)
        
        pred_loss = F.mse_loss(pred_emb[:, -1], tgt_emb[:, 0])
        emb_for_sigreg = emb.transpose(0, 1)
        sigreg_loss = sigreg(emb_for_sigreg)
        loss = pred_loss + lambd * sigreg_loss
        
        total_loss += loss.item()
        n_batches += 1
    
    return {"val_loss": total_loss / n_batches}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LeWM for Computer Use")
    parser.add_argument("--data", type=str, default=None, help="Path to HDF5 file")
    parser.add_argument("--dummy-data", action="store_true", help="Use synthetic dummy data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--ctx-len", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=0, help="Action dim (0=auto-detect from data)")
    parser.add_argument("--lambd", type=float, default=0.09, help="SIGReg weight")
    parser.add_argument("--encoder-scale", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--dropout", type=float, default=0.1, help="Predictor dropout rate")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    if args.dummy_data or args.data is None:
        print("\nUsing DUMMY data")
        if args.action_dim == 0:
            args.action_dim = 5  # default for dummy
        train_ds = DummyGuiDataset(
            num_samples=500, context_len=args.ctx_len,
            img_size=args.img_size, action_dim=args.action_dim,
        )
        val_ds = DummyGuiDataset(
            num_samples=100, context_len=args.ctx_len,
            img_size=args.img_size, action_dim=args.action_dim,
        )
    else:
        print(f"\nLoading HDF5: {args.data}")
        # Auto-detect action_dim from HDF5
        import h5py
        with h5py.File(args.data, "r") as f:
            detected_dim = f.attrs["action_dim"]
            if args.action_dim == 0:
                args.action_dim = detected_dim
                print(f"  Auto-detected action_dim: {args.action_dim}")
            elif args.action_dim != detected_dim:
                print(f"  WARNING: action_dim mismatch: CLI={args.action_dim}, data={detected_dim}")
        full_ds = HDF5GuiDataset(
            Path(args.data),
            context_len=args.ctx_len,
            img_size=args.img_size,
        )
        # Split
        n_train = int(0.9 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    
    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_ds)} samples, {len(val_loader)} batches")
    
    # Effective batch size
    eff_bs = args.batch_size * args.grad_accum
    print(f"Effective batch size: {eff_bs}")
    
    # Build model
    print("\nBuilding LeWM model...")
    model = build_lewm(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        history_size=args.ctx_len,
        action_dim=args.action_dim,
        encoder_scale=args.encoder_scale,
        predictor_dropout=args.dropout,
    )
    
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params / 1e6:.1f}M")
    
    model = model.to(device)
    
    # SIGReg
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.grad_accum,
        pct_start=0.1,
    )
    
    # Training
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "pred_loss": [], "sigreg_loss": []}
    
    print(f"\n{'='*60}")
    print(f"Training LeWM for {args.epochs} epochs")
    print(f"  LR: {args.lr}, λ: {args.lambd}, img_size: {args.img_size}")
    print(f"  Batch: {args.batch_size} × {args.grad_accum} = {eff_bs}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, sigreg, train_loader, optimizer, device,
            lambd=args.lambd,
            gradient_accumulation_steps=args.grad_accum,
        )
        
        val_metrics = validate_epoch(
            model, sigreg, val_loader, device, lambd=args.lambd,
        )
        
        scheduler.step()
        
        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["pred_loss"].append(train_metrics["pred_loss"])
        history["sigreg_loss"].append(train_metrics["sigreg_loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        
        lr_now = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train_loss: {train_metrics['loss']:.4f} | "
                  f"pred_loss: {train_metrics['pred_loss']:.4f} | "
                  f"sigreg: {train_metrics['sigreg_loss']:.4f} | "
                  f"val_loss: {val_metrics['val_loss']:.4f} | "
                  f"lr: {lr_now:.2e}")
        
        # Save best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(args),
            }, output_dir / "best_model.pt")
    
    # Save final
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
    }, output_dir / "final_model.pt")
    
    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Final train_loss: {history['train_loss'][-1]:.4f}")
    print(f"  Model saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
