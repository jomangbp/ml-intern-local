#!/usr/bin/env python3
"""
Low-memory TinyStories download + tokenization pipeline.

Designed for constrained machines (e.g. WSL with ~6GB usable RAM):
- Uses Hugging Face datasets in streaming mode (no full in-memory dataset)
- Tokenizes document-by-document
- Writes tokens incrementally to binary files
- Supports crash-safe resume with checkpoints

Outputs:
  data/train_tokens.bin
  data/val_tokens.bin
  data/train_meta.json
  data/val_meta.json
  data/tokenizer_meta.json
  data/tokenize_checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
from datasets import load_dataset


@dataclass
class Profile:
    name: str
    max_length: int
    stride: int
    train_doc_limit: int | None
    val_doc_limit: int | None
    flush_tokens: int
    checkpoint_every_docs: int
    chunk_docs: int
    num_threads: int


PROFILES: dict[str, Profile] = {
    # Conservative default for low-memory WSL setups
    "6gb": Profile(
        name="6gb",
        max_length=1024,
        stride=256,
        train_doc_limit=120_000,
        val_doc_limit=10_000,
        flush_tokens=250_000,      # ~1MB buffer (uint32)
        checkpoint_every_docs=1_000,
        chunk_docs=2_000,
        num_threads=1,
    ),
    "12gb": Profile(
        name="12gb",
        max_length=1024,
        stride=256,
        train_doc_limit=300_000,
        val_doc_limit=20_000,
        flush_tokens=500_000,
        checkpoint_every_docs=2_000,
        chunk_docs=4_000,
        num_threads=2,
    ),
    "full": Profile(
        name="full",
        max_length=1024,
        stride=256,
        train_doc_limit=None,
        val_doc_limit=None,
        flush_tokens=1_000_000,
        checkpoint_every_docs=5_000,
        chunk_docs=10_000,
        num_threads=2,
    ),
}


def _configure_env(num_threads: int) -> None:
    """Set memory-friendly runtime defaults."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))


def _iter_texts_chunked(
    dataset: str,
    split: str,
    text_field: str,
    *,
    start_index: int,
    chunk_docs: int,
) -> Iterator[tuple[int, str]]:
    """Yield (dataset_index, text) by loading dataset slices in small chunks.

    We intentionally avoid `streaming=True` because in this environment the
    datasets streaming iterator aborts the Python process at interpreter exit.
    Chunked slicing keeps memory bounded while remaining stable.
    """
    cursor = start_index
    while True:
        shard = load_dataset(dataset, split=f"{split}[{cursor}:{cursor + chunk_docs}]")
        shard_len = len(shard)
        if shard_len == 0:
            break

        for local_idx, row in enumerate(shard):
            text = row.get(text_field, "")
            if text:
                yield cursor + local_idx, text

        cursor += shard_len


def _iter_windows(tokens: list[int], max_length: int, stride: int) -> Iterator[list[int]]:
    """Yield token windows for one document."""
    if len(tokens) <= max_length:
        yield tokens
        return

    # Sliding window for long docs
    last_start = len(tokens) - max_length
    for start in range(0, last_start + 1, stride):
        yield tokens[start : start + max_length]


def _flush_buffer(buffer: array, out_file) -> int:
    """Flush token buffer to disk. Returns number of tokens written."""
    if not buffer:
        return 0
    written = len(buffer)
    buffer.tofile(out_file)
    del buffer[:]
    return written


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


def tokenize_split(
    *,
    dataset: str,
    split: str,
    text_field: str,
    output_path: Path,
    checkpoint_path: Path,
    tokenizer: tiktoken.Encoding,
    max_length: int,
    stride: int,
    doc_limit: int | None,
    flush_tokens: int,
    checkpoint_every_docs: int,
    chunk_docs: int,
    resume: bool,
) -> dict:
    """Tokenize one split incrementally with checkpoint resume support."""
    ckpt = _load_checkpoint(checkpoint_path)
    split_state = ckpt.setdefault("splits", {}).setdefault(split, {})

    dataset_docs_seen = int(split_state.get("dataset_docs_seen", 0)) if resume else 0
    tokenized_docs = int(split_state.get("tokenized_docs", 0)) if resume else 0
    tokens_written = int(split_state.get("tokens_written", 0)) if resume else 0

    if not resume:
        if output_path.exists():
            output_path.unlink()
        split_state.update({
            "dataset_docs_seen": 0,
            "tokenized_docs": 0,
            "tokens_written": 0,
        })
        _save_checkpoint(checkpoint_path, ckpt)

    if doc_limit is not None and tokenized_docs >= doc_limit:
        return {
            "split": split,
            "dataset": dataset,
            "text_field": text_field,
            "max_length": max_length,
            "stride": stride,
            "dataset_docs_seen": dataset_docs_seen,
            "tokenized_docs": tokenized_docs,
            "tokens_written": tokens_written,
            "num_tokens": tokens_written,
            "dtype": "uint32",
            "tokenizer": "cl100k_base",
            "output_path": str(output_path),
            "resumed": resume,
            "skipped": True,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = array("I")
    eot = int(tokenizer.eot_token)
    start_t = time.time()

    with output_path.open("ab") as out_file:
        for dataset_idx, text in _iter_texts_chunked(
            dataset,
            split,
            text_field,
            start_index=dataset_docs_seen,
            chunk_docs=chunk_docs,
        ):
            # Respect total doc limit across resumes
            if doc_limit is not None and tokenized_docs >= doc_limit:
                break

            doc_tokens = tokenizer.encode(text)
            added = 0
            for window in _iter_windows(doc_tokens, max_length=max_length, stride=stride):
                buffer.extend(window)
                buffer.append(eot)
                added += len(window) + 1

                if len(buffer) >= flush_tokens:
                    tokens_written += _flush_buffer(buffer, out_file)

            tokenized_docs += 1
            dataset_docs_seen = dataset_idx + 1

            # Ensure tokens_written matches even if nothing flushed yet
            split_state["dataset_docs_seen"] = dataset_docs_seen
            split_state["tokenized_docs"] = tokenized_docs
            split_state["tokens_written"] = tokens_written + len(buffer)

            if tokenized_docs % checkpoint_every_docs == 0:
                tokens_written += _flush_buffer(buffer, out_file)
                split_state["tokens_written"] = tokens_written
                _save_checkpoint(checkpoint_path, ckpt)

                elapsed = time.time() - start_t
                rate = tokenized_docs / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{split}] docs={tokenized_docs:,} seen={dataset_docs_seen:,} "
                    f"tokens~={split_state['tokens_written']:,} rate={rate:.1f} docs/s"
                )

        tokens_written += _flush_buffer(buffer, out_file)

    split_state["dataset_docs_seen"] = dataset_docs_seen
    split_state["tokenized_docs"] = tokenized_docs
    split_state["tokens_written"] = tokens_written
    _save_checkpoint(checkpoint_path, ckpt)

    meta = {
        "split": split,
        "dataset": dataset,
        "text_field": text_field,
        "max_length": max_length,
        "stride": stride,
        "tokenized_docs": tokenized_docs,
        "dataset_docs_seen": dataset_docs_seen,
        "num_tokens": tokens_written,
        "dtype": "uint32",
        "tokenizer": "cl100k_base",
        "output_path": str(output_path),
        "resumed": resume,
    }
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-memory TinyStories tokenization")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--profile", choices=PROFILES.keys(), default="6gb")

    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train-doc-limit", type=int, default=None)
    parser.add_argument("--val-doc-limit", type=int, default=None)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    preset = PROFILES[args.profile]
    max_length = args.max_length or preset.max_length
    stride = args.stride or preset.stride
    train_doc_limit = args.train_doc_limit if args.train_doc_limit is not None else preset.train_doc_limit
    val_doc_limit = args.val_doc_limit if args.val_doc_limit is not None else preset.val_doc_limit

    _configure_env(preset.num_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "tokenize_checkpoint.json"
    train_path = output_dir / "train_tokens.bin"
    val_path = output_dir / "val_tokens.bin"

    print("=" * 70)
    print("LOW-MEMORY TOKENIZATION")
    print("=" * 70)
    print(f"profile={args.profile} max_length={max_length} stride={stride}")
    print(f"train_doc_limit={train_doc_limit} val_doc_limit={val_doc_limit}")
    print(f"resume={args.resume} output_dir={output_dir}")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    tokenizer_meta = {
        "name": "cl100k_base",
        "vocab_size": tokenizer.n_vocab,
        "eot_token": int(tokenizer.eot_token),
        "type": "tiktoken",
        "profile": args.profile,
        "max_length": max_length,
        "stride": stride,
    }
    with (output_dir / "tokenizer_meta.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_meta, f, indent=2)

    train_meta = tokenize_split(
        dataset=args.dataset,
        split=args.train_split,
        text_field=args.text_field,
        output_path=train_path,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        doc_limit=train_doc_limit,
        flush_tokens=preset.flush_tokens,
        checkpoint_every_docs=preset.checkpoint_every_docs,
        chunk_docs=preset.chunk_docs,
        resume=args.resume,
    )

    with (output_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    val_meta = tokenize_split(
        dataset=args.dataset,
        split=args.val_split,
        text_field=args.text_field,
        output_path=val_path,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        doc_limit=val_doc_limit,
        flush_tokens=preset.flush_tokens,
        checkpoint_every_docs=max(250, preset.checkpoint_every_docs // 2),
        chunk_docs=max(500, preset.chunk_docs // 2),
        resume=args.resume,
    )

    with (output_dir / "val_meta.json").open("w", encoding="utf-8") as f:
        json.dump(val_meta, f, indent=2)

    print("\n" + "=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)
    print(f"train tokens: {train_meta['num_tokens']:,}")
    print(f"val tokens:   {val_meta['num_tokens']:,}")
    print(f"train file:   {train_path}")
    print(f"val file:     {val_path}")
    print(f"checkpoint:   {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
