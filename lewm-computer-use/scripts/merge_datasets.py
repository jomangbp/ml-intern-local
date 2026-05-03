#!/usr/bin/env python3
"""
Merge multiple HDF5 datasets into one unified training dataset.

Handles:
  - Same action_dim across all files (verifies compatibility)
  - Same image dimensions (auto-resizes if needed)
  - Concatenates episodes with correct boundary markers
  - Adds source metadata

Usage:
    python scripts/merge_datasets.py data/mind2web_full.h5 data/cli_large.h5 --output data/combined.h5
"""

import argparse
from pathlib import Path
from typing import List

import h5py
import numpy as np


def merge_hdf5_datasets(
    input_paths: List[Path],
    output_path: Path,
    target_img_size: int = 128,
):
    """Merge multiple HDF5 datasets into one."""
    
    all_pixels = []
    all_actions = []
    all_ep_starts = []
    all_ep_tasks = []
    all_ep_sources = []
    
    action_dim = None
    
    print(f"Merging {len(input_paths)} datasets...")
    
    for i, path in enumerate(input_paths):
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        
        print(f"\n  [{i+1}/{len(input_paths)}] {path}")
        
        with h5py.File(path, "r") as f:
            # Verify action dim
            ad = f.attrs["action_dim"]
            if action_dim is None:
                action_dim = ad
            elif ad != action_dim:
                print(f"    ERROR: action_dim mismatch: {ad} vs {action_dim}")
                print(f"    Can't merge. All datasets must have same action_dim.")
                continue
            
            ih = f.attrs.get("img_height", f["pixels"].shape[1])
            iw = f.attrs.get("img_width", f["pixels"].shape[2])
            source = f.attrs.get("source", path.stem)
            
            n_frames = f["pixels"].shape[0]
            n_actions = f["action"].shape[0]
            
            print(f"    Frames: {n_frames}, Actions: {n_actions}")
            print(f"    Size: {ih}×{iw}, Action dim: {action_dim}")
            print(f"    Source: {source}")
            
            # Load episode info
            if "ep_start" in f:
                ep_starts = list(f["ep_start"][:])
                offset = len(all_pixels) if all_pixels else 0
                all_ep_starts.extend([s + offset for s in ep_starts])
            else:
                all_ep_starts.append(len(all_pixels) if all_pixels else 0)
            
            if "ep_task" in f:
                tasks = f["ep_task"][:]
                all_ep_tasks.extend([
                    t.decode("utf-8") if isinstance(t, bytes) else str(t)
                    for t in tasks
                ])
            else:
                all_ep_tasks.append(f"Dataset from {source}")
            
            if "ep_website" in f:
                sites = f["ep_website"][:]
                all_ep_sources.extend([
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in sites
                ])
            else:
                all_ep_sources.extend([source] * len(all_ep_starts))
            
            # Load pixels
            pixels = f["pixels"][:]
            
            # Resize if needed
            if ih != target_img_size or iw != target_img_size:
                from PIL import Image
                print(f"    Resizing from {ih}×{iw} → {target_img_size}×{target_img_size}...")
                resized = np.zeros(
                    (n_frames, target_img_size, target_img_size, 3),
                    dtype=np.uint8
                )
                for j in range(n_frames):
                    img = Image.fromarray(pixels[j])
                    resized[j] = np.array(img.resize((target_img_size, target_img_size)))
                pixels = resized
            
            all_pixels.append(pixels)
            all_actions.append(f["action"][:])
    
    if not all_pixels:
        print("\nERROR: No data to merge!")
        return
    
    # Concatenate
    all_pixels = np.concatenate(all_pixels, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"\n{'='*50}")
    print(f"Merge Summary:")
    print(f"  Total frames: {len(all_pixels)}")
    print(f"  Total actions: {len(all_actions)}")
    print(f"  Episodes: {len(all_ep_starts)}")
    print(f"  Action dim: {action_dim}")
    print(f"  Image size: {target_img_size}")
    print(f"  Sources:")
    for src in set(all_ep_sources):
        count = all_ep_sources.count(src)
        print(f"    - {src}: {count} episodes")
    
    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "pixels", data=all_pixels,
            chunks=(1, target_img_size, target_img_size, 3),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "action", data=all_actions,
            chunks=(1, all_actions.shape[1]),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "ep_start", data=np.array(all_ep_starts, dtype=np.int64),
        )
        
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("ep_task", data=np.array(all_ep_tasks, dtype=object), dtype=dt)
        f.create_dataset("ep_website", data=np.array(all_ep_sources, dtype=object), dtype=dt)
        
        f.attrs["action_dim"] = action_dim
        f.attrs["img_height"] = target_img_size
        f.attrs["img_width"] = target_img_size
        f.attrs["img_channels"] = 3
        f.attrs["total_frames"] = len(all_pixels)
        f.attrs["num_episodes"] = len(all_ep_starts)
        f.attrs["source"] = "+".join(set(all_ep_sources))
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Merge LeWM HDF5 datasets")
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()
    
    merge_hdf5_datasets(
        [Path(p) for p in args.inputs],
        Path(args.output),
        args.img_size,
    )


if __name__ == "__main__":
    main()
