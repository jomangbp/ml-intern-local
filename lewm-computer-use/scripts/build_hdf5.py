#!/usr/bin/env python3
"""
Convert collected trajectory NPZ files to LeWM-compatible HDF5 format.

LeWM expects HDF5 with:
  - "pixels": (N, H, W, 3) uint8  — the observation images (screenshots)
  - "action": (N, action_dim) float32 — actions  
  - "state" or "pixels_next" not needed directly; LeWM uses temporal chunks

The trick: LeWM's HDF5Dataset loads sequential chunks. We need to provide
the data as (frameskip * steps) sequences. For computer-use, we structure each
trajectory as a single block with frame_skip=1 (no frame skipping since GUI
actions are discrete and immediate).

Format:
  pixels:  (total_steps, H, W, 3) — all screenshots flattened
  action:  (total_steps, action_dim) — corresponding actions  
  episode_idx: (total_steps,) — which episode each step belongs to

Usage:
    python scripts/build_hdf5.py --input-dir data/raw --output data/gui_trajectories.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def build_hdf5(input_dir: Path, output_path: Path):
    """Convert NPZ trajectories to HDF5 for LeWM."""
    
    npz_path = input_dir / "trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No trajectories.npz found in {input_dir}")
    
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    pixels_before = data["pixels_before"]  # (N, H, W, 3)
    actions = data["actions"]              # (N, action_dim)
    pixels_after = data["pixels_after"]    # (N, H, W, 3)
    episode_ids = data["episode_ids"]      # (N,)
    
    print(f"Loaded {len(pixels_before)} steps")
    print(f"  pixels_before shape: {pixels_before.shape}, dtype: {pixels_before.dtype}")
    print(f"  actions shape: {actions.shape}, dtype: {actions.dtype}")
    
    # LeWM expects single "pixels" column with all observations
    # We interleave: pixels_before[0], pixels_before[1], ..., pixels_after[-1]
    # This creates (2N samples) but for training we use temporal windows
    
    # For now, store both before and after as a single pixels array
    # LeWM's training loop takes sequences of length (history_size + num_preds)
    # We'll store pixels sequentially and use episode boundaries
    
    # Strategy: store all screenshots in order
    all_pixels = []
    all_actions = []
    episode_boundaries = []  # marks where episodes start
    
    for ep_id in sorted(set(episode_ids)):
        mask = episode_ids == ep_id
        p_before = pixels_before[mask]  # (steps, H, W, 3)
        p_after = pixels_after[mask]    # (steps, H, W, 3)
        a = actions[mask]               # (steps, action_dim)
        
        # Interleave: before_0, after_0(=before_1), before_1, after_1(=before_2), ...
        # Actually: before_0, before_1, before_2, ..., before_{n-1}, after_{n-1}
        episode_start = len(all_pixels)
        episode_boundaries.append(episode_start)
        
        for i in range(len(p_before)):
            all_pixels.append(p_before[i])
            all_actions.append(a[i])
        
        # Add final after state (no action for it)
        all_pixels.append(p_after[-1])
        all_actions.append(np.zeros_like(a[0]))  # zero action for terminal state
    
    all_pixels = np.stack(all_pixels).astype(np.uint8)
    all_actions = np.stack(all_actions).astype(np.float32)
    
    print(f"\nFinal dataset:")
    print(f"  pixels: {all_pixels.shape}, {all_pixels.dtype}")
    print(f"  action: {all_actions.shape}, {all_actions.dtype}")
    print(f"  episodes: {len(episode_boundaries)}")
    
    # Write HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        # Create datasets with compression
        f.create_dataset(
            "pixels", data=all_pixels,
            chunks=(1, *all_pixels.shape[1:]),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "action", data=all_actions,
            chunks=(1, all_actions.shape[1]),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "episode_boundaries", data=np.array(episode_boundaries, dtype=np.int32),
        )
        
        # Metadata
        f.attrs["action_dim"] = all_actions.shape[1]
        f.attrs["img_height"] = all_pixels.shape[1]
        f.attrs["img_width"] = all_pixels.shape[2]
        f.attrs["img_channels"] = all_pixels.shape[3]
        f.attrs["total_steps"] = len(all_pixels)
        f.attrs["total_action_steps"] = len(all_actions)
        f.attrs["frameskip"] = 1
    
    print(f"\nWrote HDF5 to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    
    # Print sample for verification
    print(f"\nSample verification:")
    print(f"  pixels[0]: shape={all_pixels[0].shape}, "
          f"min={all_pixels[0].min()}, max={all_pixels[0].max()}")
    print(f"  action[0]: {all_actions[0]}")
    print(f"  action[1]: {all_actions[1]}")


def main():
    parser = argparse.ArgumentParser(description="Build HDF5 from collected trajectories")
    parser.add_argument("--input-dir", default="data/raw", help="Directory with trajectories.npz")
    parser.add_argument("--output", default="data/gui_trajectories.h5", help="Output HDF5 path")
    args = parser.parse_args()
    
    build_hdf5(Path(args.input_dir), Path(args.output))


if __name__ == "__main__":
    main()
