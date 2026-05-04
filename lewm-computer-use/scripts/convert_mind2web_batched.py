#!/usr/bin/env python3
"""
BATCHED Mind2Web → LeWM HDF5 converter — WSL/memory-safe.

Processes 47 parquet files in batches of 5:
  1. Download 5 parquet files
  2. Parse, group into trajectories, render screenshots
  3. Append to resizeable HDF5
  4. Delete the 5 parquet files (free disk)
  5. Repeat

Memory: never holds more than ~5 parquet files (~1.5GB disk, ~200MB RAM).
Safe for WSL with 11GB RAM / 6GB VRAM.

Usage:
    python scripts/convert_mind2web_batched.py --batch-size 5 --output data/mind2web_all.h5
"""

import argparse
import io
import json
import os
import sys
import tempfile
import gc
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ACTION_DIM = 10


def parse_bounding_box(candidate_json: str) -> Optional[Dict]:
    """Extract bbox from pos_candidate JSON. Format: 'x,y,w,h' string."""
    try:
        cand = json.loads(candidate_json)
        attrs_str = cand.get("attributes", "{}")
        attrs = json.loads(attrs_str) if isinstance(attrs_str, str) else attrs_str
        bbox_str = attrs.get("bounding_box_rect", None)
        if bbox_str and isinstance(bbox_str, str):
            parts = bbox_str.split(",")
            if len(parts) == 4:
                return {"x": float(parts[0]), "y": float(parts[1]),
                        "width": float(parts[2]), "height": float(parts[3])}
    except Exception:
        pass
    return None


def encode_action(operation: dict, pos_candidates: list,
                  target_action_index: str, img_w: int, img_h: int) -> np.ndarray:
    """Convert Mind2Web action → 10-dim vector."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    op_type = operation.get("op", "").upper()
    op_value = operation.get("value", "")

    bbox = None
    try:
        tar_idx = int(target_action_index) if target_action_index else 0
        if pos_candidates is not None and len(pos_candidates) > 0 and 0 <= tar_idx < len(pos_candidates):
            bbox = parse_bounding_box(pos_candidates[tar_idx])
    except (ValueError, IndexError, TypeError):
        pass

    if op_type == "CLICK":
        action[2] = 1.0
        if bbox:
            action[0] = np.clip((bbox["x"] + bbox["width"]/2) / img_w, 0, 1)
            action[1] = np.clip((bbox["y"] + bbox["height"]/2) / img_h, 0, 1)
    elif op_type == "TYPE":
        action[3] = 1.0
        action[4] = min(len(str(op_value)) / 50.0, 1.0)
        if bbox:
            action[0] = np.clip((bbox["x"] + bbox["width"]/2) / img_w, 0, 1)
            action[1] = np.clip((bbox["y"] + bbox["height"]/2) / img_h, 0, 1)
    elif op_type == "SELECT":
        action[5] = 1.0
        if bbox:
            action[0] = np.clip((bbox["x"] + bbox["width"]/2) / img_w, 0, 1)
            action[1] = np.clip((bbox["y"] + bbox["height"]/2) / img_h, 0, 1)
    elif op_type == "HOVER":
        action[2] = 2.0
        if bbox:
            action[0] = np.clip((bbox["x"] + bbox["width"]/2) / img_w, 0, 1)
            action[1] = np.clip((bbox["y"] + bbox["height"]/2) / img_h, 0, 1)

    try:
        n_c = len(pos_candidates) if pos_candidates is not None and len(pos_candidates) > 0 else 1
    except (ValueError, TypeError):
        n_c = 1
    action[7] = 1.0 / max(n_c, 1)

    if bbox:
        action[8] = bbox["x"] / img_w
        action[9] = bbox["y"] / img_h

    return action


def get_all_parquet_paths() -> List[str]:
    """Fetch all parquet filenames from the Multimodal-Mind2Web repo."""
    import requests
    resp = requests.get("https://huggingface.co/api/datasets/osunlp/Multimodal-Mind2Web")
    siblings = resp.json().get("siblings", [])
    paths = sorted([s["rfilename"] for s in siblings if s["rfilename"].endswith(".parquet")])
    return paths


def download_batch(filenames: List[str], cache_dir: Path) -> List[Path]:
    """Download a batch of parquet files. Returns local paths."""
    paths = []
    for fn in filenames:
        try:
            path = hf_hub_download(
                repo_id="osunlp/Multimodal-Mind2Web",
                filename=fn, repo_type="dataset",
                cache_dir=str(cache_dir),
            )
            paths.append(Path(path))
        except Exception as e:
            print(f"    SKIP {fn}: {e}")
    return paths


def process_batch(parquet_paths: List[Path], img_size: int) -> Dict:
    """
    Process a batch of parquet files into frames + actions.

    Returns:
        {"pixels": np.array (N, H, W, 3), "actions": np.array (N, 10),
         "ep_starts": list[int], "ep_tasks": list[str], "ep_sites": list[str]}
    """
    all_rows = []
    ann_steps = defaultdict(list)
    ann_meta = {}

    for pf_path in parquet_paths:
        try:
            table = pq.read_table(pf_path)
            df = table.to_pandas()
        except Exception as e:
            print(f"    Error reading {pf_path.name}: {e}")
            continue

        for _, row in df.iterrows():
            aid = row["annotation_id"]
            op = row["operation"]
            if isinstance(op, str):
                op = json.loads(op)
            pos_cand = row.get("pos_candidates", [])
            if pos_cand is None:
                pos_cand = []
            ss = row["screenshot"]

            ann_steps[aid].append({
                "screenshot": ss, "operation": op,
                "pos_candidates": pos_cand,
                "target_action_index": str(row.get("target_action_index", "0")),
            })
            if aid not in ann_meta:
                ann_meta[aid] = {
                    "task": row.get("confirmed_task", ""),
                    "website": row.get("website", ""),
                }

    # Filter to trajectories with 2+ steps
    valid = {aid: steps for aid, steps in ann_steps.items() if len(steps) >= 2}
    if not valid:
        return {"pixels": np.empty((0, img_size, img_size, 3), dtype=np.uint8),
                "actions": np.empty((0, ACTION_DIM), dtype=np.float32),
                "ep_starts": [], "ep_tasks": [], "ep_sites": []}

    # Render frames
    all_pixels = []
    all_actions = []
    ep_starts = []
    ep_tasks = []
    ep_sites = []

    for aid, steps in valid.items():
        ep_starts.append(len(all_pixels))
        ep_tasks.append(ann_meta[aid]["task"])
        ep_sites.append(ann_meta[aid]["website"])

        for i, step in enumerate(steps):
            ss = step["screenshot"]
            try:
                if isinstance(ss, dict) and "bytes" in ss:
                    img = Image.open(io.BytesIO(ss["bytes"]))
                elif isinstance(ss, bytes):
                    img = Image.open(io.BytesIO(ss))
                elif isinstance(ss, Image.Image):
                    img = ss
                else:
                    continue
            except Exception:
                continue

            orig_w, orig_h = img.size
            img = img.resize((img_size, img_size), Image.LANCZOS)
            pixels = np.array(img)
            if pixels.ndim == 2:
                pixels = np.stack([pixels]*3, axis=-1)
            elif pixels.shape[-1] == 4:
                pixels = pixels[:,:,:3]

            all_pixels.append(pixels)

            if i < len(steps) - 1:
                action_vec = encode_action(
                    step["operation"], step["pos_candidates"],
                    step["target_action_index"], orig_w, orig_h,
                )
                all_actions.append(action_vec)

    # Pad actions
    while len(all_actions) < len(all_pixels):
        all_actions.append(np.zeros(ACTION_DIM, dtype=np.float32))

    pixels = np.stack(all_pixels).astype(np.uint8) if all_pixels else \
             np.empty((0, img_size, img_size, 3), dtype=np.uint8)
    actions = np.stack(all_actions).astype(np.float32) if all_actions else \
              np.empty((0, ACTION_DIM), dtype=np.float32)

    return {"pixels": pixels, "actions": actions,
            "ep_starts": ep_starts, "ep_tasks": ep_tasks, "ep_sites": ep_sites}


def create_hdf5(output_path: Path, img_size: int):
    """Create a new HDF5 file with resizeable datasets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")

    f.create_dataset("pixels", shape=(0, img_size, img_size, 3),
                     maxshape=(None, img_size, img_size, 3),
                     dtype=np.uint8, chunks=(1, img_size, img_size, 3),
                     compression="gzip", compression_opts=4)
    f.create_dataset("action", shape=(0, ACTION_DIM),
                     maxshape=(None, ACTION_DIM),
                     dtype=np.float32, chunks=(64, ACTION_DIM),
                     compression="gzip", compression_opts=4)
    f.create_dataset("ep_start", shape=(0,),
                     maxshape=(None,), dtype=np.int64)
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset("ep_task", shape=(0,),
                     maxshape=(None,), dtype=dt)
    f.create_dataset("ep_website", shape=(0,),
                     maxshape=(None,), dtype=dt)

    f.attrs["action_dim"] = ACTION_DIM
    f.attrs["img_height"] = img_size
    f.attrs["img_width"] = img_size
    f.attrs["img_channels"] = 3
    f.attrs["frameskip"] = 1
    f.attrs["source"] = "osunlp/Multimodal-Mind2Web (all splits)"

    return f


def append_to_hdf5(f: h5py.File, batch: Dict, batch_num: int):
    """Append a batch's data to the HDF5 file."""
    pixels = batch["pixels"]
    actions = batch["actions"]
    ep_starts = batch["ep_starts"]
    ep_tasks = batch["ep_tasks"]
    ep_sites = batch["ep_sites"]

    if len(pixels) == 0:
        return

    # Resize and write
    old_n = f["pixels"].shape[0]
    new_n = old_n + len(pixels)

    f["pixels"].resize(new_n, axis=0)
    f["pixels"][old_n:new_n] = pixels

    f["action"].resize(new_n, axis=0)
    f["action"][old_n:new_n] = actions

    old_e = f["ep_start"].shape[0]
    new_e = old_e + len(ep_starts)
    f["ep_start"].resize(new_e, axis=0)
    f["ep_start"][old_e:new_e] = [s + old_n for s in ep_starts]

    f["ep_task"].resize(new_e, axis=0)
    for i, t in enumerate(ep_tasks):
        f["ep_task"][old_e + i] = t

    f["ep_website"].resize(new_e, axis=0)
    for i, s in enumerate(ep_sites):
        f["ep_website"][old_e + i] = s

    f.flush()
    print(f"    Batch {batch_num}: +{len(pixels)} frames, +{len(ep_starts)} episodes "
          f"(total: {new_n} frames, {new_e} episodes)")


def main():
    parser = argparse.ArgumentParser(description="Batched Mind2Web → HDF5 (WSL-safe)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Parquet files per batch")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--output", default="data/mind2web_all.h5")
    args = parser.parse_args()

    # Get all parquet filenames
    print("Fetching parquet file list...")
    all_files = get_all_parquet_paths()
    print(f"  Found {len(all_files)} parquet files")
    print(f"  Processing in batches of {args.batch_size}")

    # Temp cache dir
    cache_dir = Path(tempfile.mkdtemp(prefix="mw_cache_"))
    print(f"  Cache: {cache_dir}")

    output_path = Path(args.output)

    # Create HDF5
    f = create_hdf5(output_path, args.img_size)

    try:
        total_batches = (len(all_files) + args.batch_size - 1) // args.batch_size
        grand_frames = 0
        grand_eps = 0

        for batch_idx in range(total_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(all_files))
            batch_files = all_files[start:end]

            print(f"\n  Batch {batch_idx+1}/{total_batches}: "
                  f"files {start+1}-{end} ({len(batch_files)} files)")

            # Download
            print(f"    Downloading...", end=" ", flush=True)
            local_paths = download_batch(batch_files, cache_dir)
            print(f"{len(local_paths)} downloaded")

            if not local_paths:
                continue

            # Process
            print(f"    Processing...")
            batch_data = process_batch(local_paths, args.img_size)

            # Append to HDF5
            append_to_hdf5(f, batch_data, batch_idx + 1)
            grand_frames += len(batch_data["pixels"])
            grand_eps += len(batch_data["ep_starts"])

            # Delete parquet files to free disk
            for p in local_paths:
                try:
                    p.unlink()
                except Exception:
                    pass

            # Force garbage collection
            del batch_data
            gc.collect()

        # Final metadata
        f.attrs["total_frames"] = grand_frames
        f.attrs["num_episodes"] = grand_eps

    finally:
        f.close()

    # Clean cache dir
    import shutil
    try:
        shutil.rmtree(cache_dir)
    except Exception:
        pass

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nDone! {grand_frames} frames, {grand_eps} episodes, {size_mb:.1f} MB")
    print(f"Output: {output_path}")
    print(f"\nNext: bash run_improved.sh")


if __name__ == "__main__":
    main()
