#!/usr/bin/env python3
"""
Convert Multimodal-Mind2Web trajectories to LeWM-compatible HDF5.

Mind2Web format: each row = one step, annotation_id groups rows into trajectories.
  - screenshot: PIL Image — the page BEFORE the action
  - operation: {"op": "CLICK", "original_op": "CLICK", "value": ""}
  - pos_candidates: list of element JSON with bounding_box_rect
  - target_action_index: which candidate was actually acted upon

We extract (s_t, a_t, s_{t+1}) by pairing consecutive steps in each trajectory.
Screenshots are embedded as bytes in parquet; we decode, resize, and store.

Output HDF5 structure (LeWM-compatible):
  /pixels     (N, H, W, 3) uint8  — all screenshots
  /action     (N, action_dim) float32 — action vectors
  /ep_start   (E,) int64 — start indices of each episode
  /ep_task    (E,) string — task descriptions

Usage:
    python scripts/convert_mind2web.py \
        --num-parquet 5 --max-trajs 200 \
        --img-size 128 --output data/mind2web_train.h5

    # Full dataset (27 parquet files, ~2.3K trajectories):
    python scripts/convert_mind2web.py --all --img-size 224 --output data/mind2web_full.h5
"""

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

# ──────────────────────────────────────────────
# Action encoding
# ──────────────────────────────────────────────

ACTION_DIM = 10

# Operation type mapping
OP_MAP = {
    "CLICK": 0,
    "TYPE": 1,
    "SELECT": 2,
    "HOVER": 3,
    "SCROLL": 4,
    "GO BACK": 5,
    "CLOSE TAB": 6,
    "NEW TAB": 7,
}


def parse_bounding_box(candidate_json: str) -> Optional[Dict]:
    """Extract bounding box from a pos_candidate JSON string.
    
    Bounding_box_rect in Mind2Web is a comma-separated string: "x,y,width,height"
    Example: "283.1875,220.390625,93.59375,33"
    """
    try:
        cand = json.loads(candidate_json)
        attrs_str = cand.get("attributes", "{}")
        if isinstance(attrs_str, str):
            attrs = json.loads(attrs_str)
        else:
            attrs = attrs_str
        bbox_str = attrs.get("bounding_box_rect", None)
        if bbox_str and isinstance(bbox_str, str):
            parts = bbox_str.split(",")
            if len(parts) == 4:
                return {
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "width": float(parts[2]),
                    "height": float(parts[3]),
                }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return None


def encode_action(
    operation: dict,
    pos_candidates: list,
    target_action_index: str,
    screenshot_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert a Mind2Web action step into a 10-dim continuous action vector.
    
    Args:
        operation: {"op": "CLICK", "original_op": "CLICK", "value": "Brooklyn"}
        pos_candidates: list of JSON strings with element info
        target_action_index: "0" or "1" etc — which candidate was acted on
        screenshot_size: (width, height) of the original screenshot
    
    Returns:
        np.ndarray of shape (ACTION_DIM,) float32
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    
    op_type = operation.get("op", "").upper()
    op_value = operation.get("value", "")
    img_w, img_h = screenshot_size
    
    # Try to get target element bbox
    bbox = None
    try:
        tar_idx = int(target_action_index) if target_action_index else 0
        if pos_candidates is not None and len(pos_candidates) > 0 and 0 <= tar_idx < len(pos_candidates):
            bbox = parse_bounding_box(pos_candidates[tar_idx])
    except (ValueError, IndexError, TypeError):
        pass
    
    # Fill action vector based on operation type
    if op_type in ("CLICK",):
        action[2] = 1.0  # left_click
        if bbox:
            center_x = (bbox["x"] + bbox["width"] / 2) / img_w
            center_y = (bbox["y"] + bbox["height"] / 2) / img_h
            action[0] = np.clip(center_x, 0.0, 1.0)
            action[1] = np.clip(center_y, 0.0, 1.0)
            action[8] = bbox["x"] / img_w
            action[9] = bbox["y"] / img_h
    
    elif op_type in ("TYPE",):
        action[3] = 1.0  # type_active
        text_len = len(str(op_value))
        action[4] = min(text_len / 50.0, 1.0)  # normalize text length
        if bbox:
            center_x = (bbox["x"] + bbox["width"] / 2) / img_w
            center_y = (bbox["y"] + bbox["height"] / 2) / img_h
            action[0] = np.clip(center_x, 0.0, 1.0)
            action[1] = np.clip(center_y, 0.0, 1.0)
    
    elif op_type in ("SELECT",):
        action[5] = 1.0  # select_active
        if bbox:
            action[0] = np.clip((bbox["x"] + bbox["width"] / 2) / img_w, 0.0, 1.0)
            action[1] = np.clip((bbox["y"] + bbox["height"] / 2) / img_h, 0.0, 1.0)
    
    elif op_type in ("HOVER",):
        action[2] = 2.0  # hover
        if bbox:
            action[0] = np.clip((bbox["x"] + bbox["width"] / 2) / img_w, 0.0, 1.0)
            action[1] = np.clip((bbox["y"] + bbox["height"] / 2) / img_h, 0.0, 1.0)
    
    elif op_type in ("SCROLL",):
        # Scroll value: positive = down
        try:
            action[6] = np.clip(float(op_value) / 500.0, -1.0, 1.0)
        except (ValueError, TypeError):
            action[6] = 0.1  # default scroll down
    
    # Confidence from action_reprs selection
    # pos_candidates may be numpy array or list; check safely
    try:
        n_candidates = len(pos_candidates) if pos_candidates is not None and len(pos_candidates) > 0 else 1
    except (ValueError, TypeError):
        n_candidates = 1
    action[7] = 1.0 / max(n_candidates, 1)  # inverse of number of candidates
    
    return action


# ──────────────────────────────────────────────
# Data loading & conversion
# ──────────────────────────────────────────────

def download_parquet_files(num_files: Optional[int] = None) -> List[Path]:
    """Download Mind2Web parquet files from Hugging Face."""
    print("Downloading Multimodal-Mind2Web parquet files...")
    
    # Try to download just the train split
    local_dir = Path.home() / ".cache" / "mind2web_data"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    for i in range(27):  # train has 27 files
        if num_files and i >= num_files:
            break
        try:
            path = hf_hub_download(
                repo_id="osunlp/Multimodal-Mind2Web",
                filename=f"data/train-{i:05d}-of-00027.parquet",
                repo_type="dataset",
                cache_dir=str(local_dir),
            )
            files.append(Path(path))
            print(f"  [{i+1}/{'?' if not num_files else num_files}] "
                  f"train-{i:05d}-of-00027.parquet")
        except Exception as e:
            # Try alternate filename pattern
            try:
                # List available files
                import requests
                resp = requests.get(
                    "https://huggingface.co/api/datasets/osunlp/Multimodal-Mind2Web"
                )
                siblings = resp.json().get("siblings", [])
                train_files = sorted([
                    s["rfilename"] for s in siblings
                    if "train" in s["rfilename"] and s["rfilename"].endswith(".parquet")
                ])
                if i < len(train_files):
                    path = hf_hub_download(
                        repo_id="osunlp/Multimodal-Mind2Web",
                        filename=train_files[i],
                        repo_type="dataset",
                        cache_dir=str(local_dir),
                    )
                    files.append(Path(path))
                    print(f"  [{i+1}] {train_files[i]}")
            except Exception as e2:
                print(f"  Skipping file {i}: {e2}")
                break
    
    if not files:
        raise RuntimeError(
            "Could not download any Mind2Web parquet files. "
            "Check your internet connection and HF_TOKEN."
        )
    
    print(f"  Downloaded {len(files)} files")
    return files


def load_parquet_to_trajectories(
    parquet_files: List[Path],
    max_trajs: Optional[int] = None,
) -> List[Dict]:
    """
    Load parquet files and group into trajectories.
    
    Returns:
        List of trajectories, each:
        {
            "annotation_id": str,
            "task": str,
            "website": str,
            "steps": [
                {"screenshot": PIL.Image, "action": np.ndarray, "op_label": str},
                ...
            ]
        }
    """
    print("Loading and grouping trajectories...")
    
    # First pass: collect all rows grouped by annotation_id
    ann_steps = defaultdict(list)
    ann_meta = {}
    total_rows = 0
    
    for pf_path in tqdm(parquet_files, desc="Reading parquet"):
        try:
            table = pq.read_table(pf_path)
            df = table.to_pandas()
        except Exception as e:
            print(f"  Error reading {pf_path.name}: {e}")
            continue
        
        for _, row in df.iterrows():
            aid = row["annotation_id"]
            total_rows += 1
            
            # Parse operation
            op = row["operation"]
            if isinstance(op, str):
                op = json.loads(op)
            
            # Get pos_candidates
            pos_cand = row.get("pos_candidates", [])
            if pos_cand is None:
                pos_cand = []
            
            # Screenshot
            ss = row["screenshot"]
            
            ann_steps[aid].append({
                "screenshot": ss,  # PIL Image
                "operation": op,
                "pos_candidates": pos_cand,
                "target_action_index": str(row.get("target_action_index", "0")),
            })
            
            if aid not in ann_meta:
                ann_meta[aid] = {
                    "task": row.get("confirmed_task", ""),
                    "website": row.get("website", ""),
                }
    
    print(f"  Total rows: {total_rows}")
    print(f"  Unique trajectories: {len(ann_steps)}")
    
    # Filter to trajectories with 2+ steps (need at least 2 for one transition pair)
    valid = {aid: steps for aid, steps in ann_steps.items() if len(steps) >= 2}
    print(f"  Valid trajectories (≥2 steps): {len(valid)}")
    
    # Sort by step count descending, take top max_trajs
    sorted_aids = sorted(valid.keys(), key=lambda a: len(valid[a]), reverse=True)
    if max_trajs:
        sorted_aids = sorted_aids[:max_trajs]
    
    trajectories = []
    for aid in sorted_aids:
        trajectories.append({
            "annotation_id": aid,
            "task": ann_meta[aid]["task"],
            "website": ann_meta[aid]["website"],
            "steps": valid[aid],
        })
    
    lengths = [len(t["steps"]) for t in trajectories]
    print(f"  Selected {len(trajectories)} trajectories")
    print(f"  Step counts: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")
    
    return trajectories


def trajectories_to_hdf5(
    trajectories: List[Dict],
    output_path: Path,
    img_size: int = 224,
) -> None:
    """
    Convert trajectories to LeWM-compatible HDF5.
    
    For each trajectory with N steps:
      - Pixels: s_0, s_1, s_2, ..., s_{N-1}  (N screenshots)
      - Actions: a_0, a_1, ..., a_{N-2}      (N-1 actions)
      - Pairs: (s_0, a_0) → s_1, (s_1, a_1) → s_2, ...
    """
    print(f"\nConverting to HDF5 (image size: {img_size}×{img_size})...")
    
    all_pixels = []
    all_actions = []
    ep_starts = []
    ep_tasks = []
    ep_websites = []
    
    skipped = 0
    
    for traj in tqdm(trajectories, desc="Converting"):
        steps = traj["steps"]
        ep_start = len(all_pixels)
        ep_starts.append(ep_start)
        ep_tasks.append(traj.get("task", ""))
        ep_websites.append(traj.get("website", ""))
        
        for i in range(len(steps)):
            step = steps[i]
            ss = step["screenshot"]
            
            # Decode screenshot
            try:
                if isinstance(ss, dict) and "bytes" in ss:
                    img = Image.open(io.BytesIO(ss["bytes"]))
                elif isinstance(ss, bytes):
                    img = Image.open(io.BytesIO(ss))
                elif isinstance(ss, Image.Image):
                    img = ss
                elif isinstance(ss, dict) and "path" in ss:
                    img = Image.open(ss["path"])
                else:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            
            # Resize
            orig_w, orig_h = img.size
            img_resized = img.resize((img_size, img_size), Image.LANCZOS)
            pixels = np.array(img_resized)
            
            # Ensure RGB
            if pixels.ndim == 2:
                pixels = np.stack([pixels] * 3, axis=-1)
            elif pixels.shape[-1] == 4:
                pixels = pixels[:, :, :3]
            
            all_pixels.append(pixels)
            
            # Encode action (except for last step which has no following action)
            if i < len(steps) - 1:
                action_vec = encode_action(
                    step["operation"],
                    step["pos_candidates"],
                    step["target_action_index"],
                    (orig_w, orig_h),
                )
                all_actions.append(action_vec)
    
    # Add zero-action for last screenshot of last trajectory (terminal state)
    if len(all_actions) < len(all_pixels):
        pad_count = len(all_pixels) - len(all_actions)
        for _ in range(pad_count):
            all_actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
    
    all_pixels = np.stack(all_pixels).astype(np.uint8)
    all_actions = np.stack(all_actions).astype(np.float32)
    
    print(f"\nFinal dataset:")
    print(f"  pixels: {all_pixels.shape}, dtype={all_pixels.dtype}")
    print(f"  action: {all_actions.shape}, dtype={all_actions.dtype}")
    print(f"  episodes: {len(ep_starts)}")
    print(f"  skipped: {skipped} images")
    
    # Write HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  Writing {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "pixels", data=all_pixels,
            chunks=(1, img_size, img_size, 3),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "action", data=all_actions,
            chunks=(1, ACTION_DIM),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "ep_start", data=np.array(ep_starts, dtype=np.int64),
        )
        
        # Store tasks as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("ep_task", data=np.array(ep_tasks, dtype=object), dtype=dt)
        f.create_dataset("ep_website", data=np.array(ep_websites, dtype=object), dtype=dt)
        
        # Metadata
        f.attrs["action_dim"] = ACTION_DIM
        f.attrs["img_height"] = img_size
        f.attrs["img_width"] = img_size
        f.attrs["img_channels"] = 3
        f.attrs["total_frames"] = len(all_pixels)
        f.attrs["total_action_steps"] = len(all_actions) - pad_count
        f.attrs["frameskip"] = 1
        f.attrs["num_episodes"] = len(ep_starts)
        f.attrs["source"] = "osunlp/Multimodal-Mind2Web"
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Done! {size_mb:.1f} MB")
    
    # Verification
    print(f"\nVerification sample:")
    print(f"  pixels[0]: shape={all_pixels[0].shape}, "
          f"min={all_pixels[0].min()}, max={all_pixels[0].max()}")
    print(f"  action[0]: {all_actions[0]}")
    print(f"  Episode 0: {ep_tasks[0][:100]} ({ep_websites[0]})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Mind2Web to LeWM-compatible HDF5"
    )
    parser.add_argument("--num-parquet", type=int, default=5,
                        help="Number of parquet files to download (max 27 for train)")
    parser.add_argument("--all", action="store_true",
                        help="Download all 27 train parquet files")
    parser.add_argument("--max-trajs", type=int, default=200,
                        help="Maximum number of trajectories to include")
    parser.add_argument("--img-size", type=int, default=128,
                        help="Output image size (128 or 224)")
    parser.add_argument("--output", default="data/mind2web_train.h5",
                        help="Output HDF5 path")
    parser.add_argument("--local-parquet", type=str, default=None,
                        help="Path to local parquet directory (skip download)")
    args = parser.parse_args()
    
    # Step 1: Get parquet files
    if args.local_parquet:
        local_dir = Path(args.local_parquet)
        parquet_files = sorted(local_dir.glob("*.parquet"))
        print(f"Using {len(parquet_files)} local parquet files from {local_dir}")
    else:
        n_files = None if args.all else args.num_parquet
        parquet_files = download_parquet_files(n_files)
    
    # Step 2: Load and group
    trajectories = load_parquet_to_trajectories(parquet_files, args.max_trajs)
    
    if not trajectories:
        print("ERROR: No valid trajectories found!")
        sys.exit(1)
    
    # Step 3: Convert to HDF5
    output_path = Path(args.output)
    trajectories_to_hdf5(trajectories, output_path, args.img_size)


if __name__ == "__main__":
    main()
