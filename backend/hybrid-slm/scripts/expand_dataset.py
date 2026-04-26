#!/usr/bin/env python3
"""
Expand the training dataset beyond TinyStories by adding FineWeb-Edu data.

Strategy:
- Use FineWeb-Edu sample-10BT (10B tokens, pre-sampled high-quality subset)
- Tokenize with cl100k_base (same as TinyStories)
- Write to separate .bin files so we can mix datasets during training
- Support for multiple datasets with configurable mixing

Outputs:
  data/fineweb_train_tokens.bin   - FineWeb-Edu training tokens
  data/fineweb_val_tokens.bin     - FineWeb-Edu validation tokens
  data/fineweb_train_meta.json    - Metadata
  data/fineweb_val_meta.json      - Metadata

Usage:
  python scripts/expand_dataset.py --target-tokens 500000000  # 500M tokens (~2GB)
  python scripts/expand_dataset.py --target-tokens 2000000000 # 2B tokens (~8GB)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
from datasets import load_dataset


@dataclass
class ExpandConfig:
    """Configuration for dataset expansion."""
    target_tokens: int = 500_000_000  # 500M tokens default
    max_length: int = 1024
    stride: int = 256
    min_score: int = 3  # FineWeb-Edu educational quality score threshold
    flush_tokens: int = 500_000
    checkpoint_every_docs: int = 2_000
    chunk_docs: int = 2_000
    val_fraction: float = 0.005  # 0.5% for validation
    num_threads: int = 1


def _configure_env(num_threads: int) -> None:
    """Set memory-friendly runtime defaults."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))


def _flush_buffer(buffer: array, out_file) -> int:
    """Flush token buffer to disk."""
    if not buffer:
        return 0
    written = len(buffer)
    buffer.tofile(out_file)
    del buffer[:]
    return written


def _iter_windows(tokens: list[int], max_length: int, stride: int) -> Iterator[list[int]]:
    """Yield token windows for one document."""
    if len(tokens) <= max_length:
        yield tokens
        return
    last_start = len(tokens) - max_length
    for start in range(0, last_start + 1, stride):
        yield tokens[start : start + max_length]


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"splits": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def tokenize_fineweb(
    *,
    config: ExpandConfig,
    output_dir: Path,
    resume: bool = True,
) -> dict:
    """
    Tokenize FineWeb-Edu sample-10BT with streaming.
    Split into train/val on the fly.
    """
    checkpoint_path = output_dir / "fineweb_tokenize_checkpoint.json"
    train_path = output_dir / "fineweb_train_tokens.bin"
    val_path = output_dir / "fineweb_val_tokens.bin"

    ckpt = _load_checkpoint(checkpoint_path)
    state = ckpt.setdefault("fineweb", {})

    train_tokens_written = int(state.get("train_tokens_written", 0)) if resume else 0
    val_tokens_written = int(state.get("val_tokens_written", 0)) if resume else 0
    docs_processed = int(state.get("docs_processed", 0)) if resume else 0
    total_tokens = train_tokens_written + val_tokens_written

    if not resume:
        if train_path.exists():
            train_path.unlink()
        if val_path.exists():
            val_path.unlink()
        state.update({
            "train_tokens_written": 0,
            "val_tokens_written": 0,
            "docs_processed": 0,
        })
        _save_checkpoint(checkpoint_path, ckpt)

    if total_tokens >= config.target_tokens:
        print(f"Already have {total_tokens:,} tokens (target: {config.target_tokens:,}). Skipping.")
        return {
            "train_tokens": train_tokens_written,
            "val_tokens": val_tokens_written,
            "docs_processed": docs_processed,
        }

    print(f"\n{'='*70}")
    print(f"TOKENIZING FINEWEB-EDU (sample-10BT)")
    print(f"{'='*70}")
    print(f"Target tokens: {config.target_tokens:,}")
    print(f"Current tokens: {total_tokens:,}")
    print(f"Remaining: {config.target_tokens - total_tokens:,}")
    print(f"Min educational score: {config.min_score}")
    print(f"Max length: {config.max_length}, Stride: {config.stride}")

    tokenizer = tiktoken.get_encoding("cl100k_base")
    eot = int(tokenizer.eot_token)

    train_buffer = array("I")
    val_buffer = array("I")
    val_interval = max(1, int(1 / config.val_fraction))
    start_t = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    with train_path.open("ab") as train_file, val_path.open("ab") as val_file:
        # Stream from FineWeb-Edu sample-10BT
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        # Skip already-processed docs
        if docs_processed > 0:
            print(f"Skipping first {docs_processed:,} documents (resuming)...")
            ds = ds.skip(docs_processed)

        for idx, row in enumerate(ds):
            if total_tokens >= config.target_tokens:
                print(f"\nReached target of {config.target_tokens:,} tokens!")
                break

            # Filter by educational quality score
            int_score = row.get("int_score", 0)
            if int_score < config.min_score:
                continue

            text = row.get("text", "")
            if not text or len(text) < 50:
                continue

            # Tokenize
            doc_tokens = tokenizer.encode(text)
            added = 0

            for window in _iter_windows(doc_tokens, max_length=config.max_length, stride=config.stride):
                if (docs_processed + 1) % val_interval == 0:
                    # This goes to validation
                    val_buffer.extend(window)
                    val_buffer.append(eot)
                    added += len(window) + 1
                else:
                    # This goes to training
                    train_buffer.extend(window)
                    train_buffer.append(eot)
                    added += len(window) + 1

            docs_processed += 1
            total_tokens = train_tokens_written + val_tokens_written + len(train_buffer) + len(val_buffer)

            # Flush buffers periodically
            if len(train_buffer) >= config.flush_tokens:
                train_tokens_written += _flush_buffer(train_buffer, train_file)
            if len(val_buffer) >= config.flush_tokens:
                val_tokens_written += _flush_buffer(val_buffer, val_file)

            # Checkpoint
            if docs_processed % config.checkpoint_every_docs == 0:
                train_tokens_written += _flush_buffer(train_buffer, train_file)
                val_tokens_written += _flush_buffer(val_buffer, val_file)
                total_tokens = train_tokens_written + val_tokens_written

                state.update({
                    "train_tokens_written": train_tokens_written,
                    "val_tokens_written": val_tokens_written,
                    "docs_processed": docs_processed,
                })
                _save_checkpoint(checkpoint_path, ckpt)

                elapsed = time.time() - start_t
                rate = docs_processed / elapsed if elapsed > 0 else 0.0
                toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
                train_mb = train_tokens_written * 4 / 1024 / 1024
                print(
                    f"[fineweb] docs={docs_processed:,} total_tokens={total_tokens:,} "
                    f"train={train_tokens_written:,} val={val_tokens_written:,} "
                    f"rate={rate:.1f} docs/s | {toks_per_sec:,.0f} toks/s | "
                    f"train_file={train_mb:.1f}MB"
                )

        # Final flush
        train_tokens_written += _flush_buffer(train_buffer, train_file)
        val_tokens_written += _flush_buffer(val_buffer, val_file)
        total_tokens = train_tokens_written + val_tokens_written

    state.update({
        "train_tokens_written": train_tokens_written,
        "val_tokens_written": val_tokens_written,
        "docs_processed": docs_processed,
    })
    _save_checkpoint(checkpoint_path, ckpt)

    # Save metadata
    train_meta = {
        "split": "train",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "text_field": "text",
        "max_length": config.max_length,
        "stride": config.stride,
        "min_score": config.min_score,
        "tokenized_docs": docs_processed,
        "num_tokens": train_tokens_written,
        "dtype": "uint32",
        "tokenizer": "cl100k_base",
        "output_path": str(train_path),
    }
    with (output_dir / "fineweb_train_meta.json").open("w") as f:
        json.dump(train_meta, f, indent=2)

    val_meta = {
        "split": "validation",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "text_field": "text",
        "max_length": config.max_length,
        "stride": config.stride,
        "min_score": config.min_score,
        "tokenized_docs": docs_processed,
        "num_tokens": val_tokens_written,
        "dtype": "uint32",
        "tokenizer": "cl100k_base",
        "output_path": str(val_path),
    }
    with (output_dir / "fineweb_val_meta.json").open("w") as f:
        json.dump(val_meta, f, indent=2)

    elapsed = time.time() - start_t
    print(f"\n{'='*70}")
    print(f"FINEWEB-EDU TOKENIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total documents processed: {docs_processed:,}")
    print(f"Train tokens: {train_tokens_written:,} ({train_tokens_written * 4 / 1024 / 1024:.1f} MB)")
    print(f"Val tokens:   {val_tokens_written:,} ({val_tokens_written * 4 / 1024 / 1024:.1f} MB)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Train file: {train_path}")
    print(f"Val file:   {val_path}")

    return {
        "train_tokens": train_tokens_written,
        "val_tokens": val_tokens_written,
        "docs_processed": docs_processed,
    }


def merge_datasets(output_dir: Path) -> None:
    """
    Merge TinyStories + FineWeb-Edu token files into combined training data.
    Creates:
      data/combined_train_tokens.bin
      data/combined_val_tokens.bin
    """
    print(f"\n{'='*70}")
    print("MERGING DATASETS")
    print(f"{'='*70}")

    combined_train_path = output_dir / "combined_train_tokens.bin"
    combined_val_path = output_dir / "combined_val_tokens.bin"

    sources_train = []
    sources_val = []

    # TinyStories
    ts_train = output_dir / "train_tokens.bin"
    ts_val = output_dir / "val_tokens.bin"
    if ts_train.exists():
        sources_train.append(("TinyStories/train", ts_train))
    if ts_val.exists():
        sources_val.append(("TinyStories/val", ts_val))

    # FineWeb-Edu
    fw_train = output_dir / "fineweb_train_tokens.bin"
    fw_val = output_dir / "fineweb_val_tokens.bin"
    if fw_train.exists():
        sources_train.append(("FineWeb-Edu/train", fw_train))
    if fw_val.exists():
        sources_val.append(("FineWeb-Edu/val", fw_val))

    # Merge training data
    total_train = 0
    with combined_train_path.open("wb") as out:
        for name, path in sources_train:
            size = path.stat().st_size
            tokens = size // 4  # uint32 = 4 bytes
            print(f"  {name}: {tokens:,} tokens ({size / 1024 / 1024:.1f} MB)")
            with path.open("rb") as inp:
                # Copy in chunks
                while True:
                    chunk = inp.read(64 * 1024 * 1024)  # 64MB chunks
                    if not chunk:
                        break
                    out.write(chunk)
            total_train += tokens
    print(f"  Combined train: {total_train:,} tokens ({combined_train_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Merge validation data
    total_val = 0
    with combined_val_path.open("wb") as out:
        for name, path in sources_val:
            size = path.stat().st_size
            tokens = size // 4
            print(f"  {name}: {tokens:,} tokens ({size / 1024 / 1024:.1f} MB)")
            with path.open("rb") as inp:
                while True:
                    chunk = inp.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            total_val += tokens
    print(f"  Combined val: {total_val:,} tokens ({combined_val_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save combined metadata
    combined_meta = {
        "sources": [name for name, _ in sources_train],
        "train_tokens": total_train,
        "val_tokens": total_val,
        "total_tokens": total_train + total_val,
        "dtype": "uint32",
        "tokenizer": "cl100k_base",
    }
    with (output_dir / "combined_meta.json").open("w") as f:
        json.dump(combined_meta, f, indent=2)

    print(f"\nCombined metadata saved to {output_dir / 'combined_meta.json'}")
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_train + total_val:,} tokens")
    print(f"{'='*70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand training dataset with FineWeb-Edu")
    parser.add_argument("--target-tokens", type=int, default=500_000_000,
                        help="Target number of tokens from FineWeb-Edu (default: 500M)")
    parser.add_argument("--min-score", type=int, default=3,
                        help="Minimum educational quality score (1-5, default: 3)")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for tokenized data")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start from scratch instead of resuming")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing datasets, skip tokenization")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _configure_env(1)

    config = ExpandConfig(
        target_tokens=args.target_tokens,
        min_score=args.min_score,
    )

    if not args.merge_only:
        tokenize_fineweb(
            config=config,
            output_dir=output_dir,
            resume=not args.no_resume,
        )

    # Always merge at the end
    merge_datasets(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
