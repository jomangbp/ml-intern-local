#!/usr/bin/env python3
"""
Download & Tokenize Pipeline for SYNTH + Hermes Agent Reasoning Traces.

Resumable, crash-safe, low-memory. Follows the same patterns as
download_and_tokenize.py (streaming, periodic checkpoints, append-mode
binary output).

Datasets
--------
1. PleIAs/SYNTH (~80M samples, 75B tokens, multilingual)
   - Columns used: synthetic_reasoning + synthetic_answer (concatenated)
   - Filter: language == "en" (can be overridden)
   - Exercise types: memorization, rewriting, summary, analysis, etc.

2. lambda/hermes-agent-reasoning-traces (~15K samples)
   - Configs: kimi, glm-5.1
   - Conversations: multi-turn tool-calling traces
   - Format: ShareGPT-style (from/value pairs)

Outputs (into --output-dir):
  synth_train_tokens.bin / synth_val_tokens.bin
  hermes_train_tokens.bin / hermes_val_tokens.bin
  combined_synth_hermes_train_tokens.bin (concatenated)
  *_meta.json files with statistics
  download_checkpoint.json for crash resume

Usage:
  python scripts/download_tokenize_datasets.py --profile 6gb
  python scripts/download_tokenize_datasets.py --dataset synth --profile 12gb
  python scripts/download_tokenize_datasets.py --dataset hermes --resume

Resume:
  If the script is killed (OOM, Ctrl-C, power loss), just re-run with
  the same arguments. The checkpoint file tracks per-split progress.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
from datasets import load_dataset


# ── Profiles ────────────────────────────────────────────────────

@dataclass
class Profile:
    name: str
    max_length: int
    stride: int
    synth_train_limit: int | None   # None = all
    synth_val_limit: int | None
    hermes_train_limit: int | None
    hermes_val_limit: int | None
    synth_flush_tokens: int
    hermes_flush_tokens: int
    checkpoint_every_docs: int
    chunk_docs: int
    synth_val_ratio: float          # fraction of synth to hold out for val
    num_threads: int


PROFILES: dict[str, Profile] = {
    "6gb": Profile(
        name="6gb",
        max_length=1024,
        stride=256,
        synth_train_limit=None,       # process all
        synth_val_limit=None,
        hermes_train_limit=None,
        hermes_val_limit=None,
        synth_flush_tokens=500_000,
        hermes_flush_tokens=100_000,
        checkpoint_every_docs=2_000,
        chunk_docs=2_000,
        synth_val_ratio=0.001,        # 0.1% holdout (~80K samples)
        num_threads=1,
    ),
    "12gb": Profile(
        name="12gb",
        max_length=1024,
        stride=256,
        synth_train_limit=None,
        synth_val_limit=None,
        hermes_train_limit=None,
        hermes_val_limit=None,
        synth_flush_tokens=1_000_000,
        hermes_flush_tokens=200_000,
        checkpoint_every_docs=5_000,
        chunk_docs=5_000,
        synth_val_ratio=0.001,
        num_threads=2,
    ),
    "test": Profile(
        name="test",
        max_length=1024,
        stride=256,
        synth_train_limit=5_000,
        synth_val_limit=500,
        hermes_train_limit=500,
        hermes_val_limit=50,
        synth_flush_tokens=50_000,
        hermes_flush_tokens=10_000,
        checkpoint_every_docs=500,
        chunk_docs=500,
        synth_val_ratio=0.05,         # 5% for quick test
        num_threads=1,
    ),
}


# ── Helpers ─────────────────────────────────────────────────────

def _configure_env(num_threads: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))


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


def _flush_buffer(buffer: array, out_file) -> int:
    if not buffer:
        return 0
    written = len(buffer)
    buffer.tofile(out_file)
    del buffer[:]
    return written


def _iter_windows(tokens: list[int], max_length: int, stride: int) -> Iterator[list[int]]:
    if len(tokens) <= max_length:
        yield tokens
        return
    last_start = len(tokens) - max_length
    for start in range(0, last_start + 1, stride):
        yield tokens[start : start + max_length]


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = seconds / 60
    if mins < 60:
        return f"{mins:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


# ── SYNTH Processing ───────────────────────────────────────────

def _format_synth_text(row: dict) -> str:
    """Convert one SYNTH row into training text.

    Format:
        <reasoning>
        {synthetic_reasoning}
        </reasoning>
        <answer>
        {synthetic_answer}
        </answer>
    """
    reasoning = row.get("synthetic_reasoning", "") or ""
    answer = row.get("synthetic_answer", "") or ""

    parts = []
    if reasoning.strip():
        parts.append("<reasoning>\n" + reasoning.strip() + "\n</reasoning>")
    if answer.strip():
        parts.append("<answer>\n" + answer.strip() + "\n</answer>")

    return "\n\n".join(parts)


def tokenize_synth(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    tokenizer: tiktoken.Encoding,
    profile: Profile,
    language_filter: str | None,
    resume: bool,
) -> dict:
    """Tokenize PleIAs/SYNTH with resume support.

    SYNTH has only a 'train' split, so we create our own val split
    by holding out every Nth document (based on synth_val_ratio).
    """
    split_name = "synth_train"
    ckpt = _load_checkpoint(checkpoint_path)
    split_state = ckpt.setdefault("splits", {}).setdefault(split_name, {})

    # Skip entirely if already completed
    if resume and split_state.get("completed"):
        print(f"  [synth] Already completed, skipping.", flush=True)
        return {
            "dataset": "PleIAs/SYNTH",
            "language_filter": language_filter,
            "train_docs": int(split_state.get("tokenized_docs", 0)),
            "val_docs": int(split_state.get("val_docs", 0)),
            "skipped_docs": int(split_state.get("skipped_docs", 0)),
            "dataset_docs_seen": int(split_state.get("dataset_docs_seen", 0)),
            "train_tokens": int(split_state.get("tokens_written", 0)),
            "max_length": profile.max_length,
            "stride": profile.stride,
            "dtype": "uint32",
            "tokenizer": "cl100k_base",
        }

    dataset_docs_seen = int(split_state.get("dataset_docs_seen", 0)) if resume else 0
    tokenized_docs = int(split_state.get("tokenized_docs", 0)) if resume else 0
    tokens_written = int(split_state.get("tokens_written", 0)) if resume else 0
    skipped_docs = int(split_state.get("skipped_docs", 0)) if resume else 0
    val_docs = int(split_state.get("val_docs", 0)) if resume else 0

    train_path = output_dir / "synth_train_tokens.bin"
    val_path = output_dir / "synth_val_tokens.bin"

    if not resume:
        for p in [train_path, val_path]:
            if p.exists():
                p.unlink()
        split_state.update({
            "dataset_docs_seen": 0,
            "tokenized_docs": 0,
            "tokens_written": 0,
            "skipped_docs": 0,
            "val_docs": 0,
        })
        _save_checkpoint(checkpoint_path, ckpt)

    train_buffer = array("I")
    val_buffer = array("I")
    eot = int(tokenizer.eot_token)
    val_interval = max(1, int(1 / profile.synth_val_ratio))
    start_t = time.time()

    train_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("ab") as train_file, val_path.open("ab") as val_file:
        ds = load_dataset("PleIAs/SYNTH", split="train", streaming=True)

        for row in ds:
            idx = dataset_docs_seen
            dataset_docs_seen += 1

            if resume and idx < int(split_state.get("dataset_docs_seen", 0)):
                continue  # skip already-processed docs on resume

            # Language filter
            if language_filter and row.get("language") != language_filter:
                skipped_docs += 1
                if dataset_docs_seen % profile.checkpoint_every_docs == 0:
                    split_state.update({
                        "dataset_docs_seen": dataset_docs_seen,
                        "tokenized_docs": tokenized_docs,
                        "tokens_written": tokens_written + len(train_buffer),
                        "skipped_docs": skipped_docs,
                        "val_docs": val_docs,
                    })
                    _save_checkpoint(checkpoint_path, ckpt)
                continue

            # Format text
            text = _format_synth_text(row)
            if not text.strip():
                continue

            # Determine if this goes to val set
            is_val = (idx % val_interval == 0)

            # Tokenize
            doc_tokens = tokenizer.encode(text, disallowed_special=())

            # Write windows
            target_file = val_file if is_val else train_file
            target_buffer = val_buffer if is_val else train_buffer

            for window in _iter_windows(doc_tokens, profile.max_length, profile.stride):
                target_buffer.extend(window)
                target_buffer.append(eot)

                if len(target_buffer) >= profile.synth_flush_tokens:
                    if target_buffer is val_buffer:
                        val_docs += len(val_buffer) // (profile.max_length + 1)
                    tokens_written += _flush_buffer(target_buffer, target_file)

            if is_val:
                val_docs += 1
            else:
                tokenized_docs += 1

            # Checkpoint
            if dataset_docs_seen % profile.checkpoint_every_docs == 0:
                tokens_written += _flush_buffer(train_buffer, train_file)
                _flush_buffer(val_buffer, val_file)

                split_state.update({
                    "dataset_docs_seen": dataset_docs_seen,
                    "tokenized_docs": tokenized_docs,
                    "tokens_written": tokens_written + len(train_buffer),
                    "skipped_docs": skipped_docs,
                    "val_docs": val_docs,
                })
                _save_checkpoint(checkpoint_path, ckpt)

                elapsed = time.time() - start_t
                rate = dataset_docs_seen / elapsed if elapsed > 0 else 0.0
                print(
                    f"[synth] seen={dataset_docs_seen:,} train_docs={tokenized_docs:,} "
                    f"val_docs={val_docs:,} skipped={skipped_docs:,} "
                    f"tokens~={split_state['tokens_written']:,} "
                    f"rate={rate:.0f} docs/s elapsed={_format_time(elapsed)}",
                    flush=True,
                )

            # Doc limit
            if profile.synth_train_limit and tokenized_docs >= profile.synth_train_limit:
                break

        # Final flush
        tokens_written += _flush_buffer(train_buffer, train_file)
        _flush_buffer(val_buffer, val_file)

    split_state.update({
        "dataset_docs_seen": dataset_docs_seen,
        "tokenized_docs": tokenized_docs,
        "tokens_written": tokens_written,
        "skipped_docs": skipped_docs,
        "val_docs": val_docs,
        "completed": True,
    })
    _save_checkpoint(checkpoint_path, ckpt)

    return {
        "dataset": "PleIAs/SYNTH",
        "language_filter": language_filter,
        "train_docs": tokenized_docs,
        "val_docs": val_docs,
        "skipped_docs": skipped_docs,
        "dataset_docs_seen": dataset_docs_seen,
        "train_tokens": tokens_written,
        "max_length": profile.max_length,
        "stride": profile.stride,
        "dtype": "uint32",
        "tokenizer": "cl100k_base",
    }


# ── Hermes Processing ──────────────────────────────────────────

def _format_hermes_conversation(row: dict) -> str:
    """Convert Hermes multi-turn conversation to training text.

    Format (flat text for causal LM):
        [SYSTEM] {system message}
        [USER] {user message}
        [ASSISTANT] {assistant response with thinking}
        [TOOL] {tool output}
        ...
    """
    conversations = row.get("conversations", [])
    if not conversations:
        return ""

    parts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))

        if role == "system":
            parts.append("[SYSTEM]\n" + content)
        elif role in ("human", "user"):
            parts.append("[USER]\n" + content)
        elif role in ("gpt", "assistant"):
            parts.append("[ASSISTANT]\n" + content)
        elif role == "tool":
            parts.append("[TOOL]\n" + content)

    return "\n".join(parts)


def tokenize_hermes(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    tokenizer: tiktoken.Encoding,
    profile: Profile,
    configs: list[str],
    val_ratio: float,
    resume: bool,
) -> dict:
    """Tokenize Hermes Agent Reasoning Traces.

    Processes both configs (kimi, glm-5.1) and creates train/val splits.
    """
    train_path = output_dir / "hermes_train_tokens.bin"
    val_path = output_dir / "hermes_val_tokens.bin"

    ckpt = _load_checkpoint(checkpoint_path)

    total_train_docs = 0
    total_val_docs = 0
    total_train_tokens = 0
    total_val_tokens = 0

    for config in configs:
        split_name = "hermes_" + config
        split_state = ckpt.setdefault("splits", {}).setdefault(split_name, {})

        # Skip already completed configs
        if resume and split_state.get("completed"):
            print(f"  [hermes/{config}] Already completed, skipping.", flush=True)
            total_train_docs += int(split_state.get("train_docs", 0))
            total_val_docs += int(split_state.get("val_docs", 0))
            continue

        docs_seen = int(split_state.get("docs_seen", 0)) if resume else 0
        train_docs = int(split_state.get("train_docs", 0)) if resume else 0
        val_docs = int(split_state.get("val_docs", 0)) if resume else 0

        if not resume:
            split_state.update({"docs_seen": 0, "train_docs": 0, "val_docs": 0})
            _save_checkpoint(checkpoint_path, ckpt)

        train_buffer = array("I")
        val_buffer = array("I")
        eot = int(tokenizer.eot_token)
        val_interval = max(1, int(1 / val_ratio))
        start_t = time.time()

        with train_path.open("ab") as tf, val_path.open("ab") as vf:
            ds = load_dataset(
                "lambda/hermes-agent-reasoning-traces", config, split="train",
                streaming=True,
            )

            for i, row in enumerate(ds):
                if i < docs_seen:
                    continue

                text = _format_hermes_conversation(row)
                if not text.strip():
                    docs_seen += 1
                    continue

                is_val = (i % val_interval == 0)

                doc_tokens = tokenizer.encode(text, disallowed_special=())
                target_file = vf if is_val else tf
                target_buffer = val_buffer if is_val else train_buffer

                for window in _iter_windows(doc_tokens, profile.max_length, profile.stride):
                    target_buffer.extend(window)
                    target_buffer.append(eot)

                    if len(target_buffer) >= profile.hermes_flush_tokens:
                        written = _flush_buffer(target_buffer, target_file)
                        if is_val:
                            total_val_tokens += written
                        else:
                            total_train_tokens += written

                if is_val:
                    val_docs += 1
                    total_val_docs += 1
                else:
                    train_docs += 1
                    total_train_docs += 1

                docs_seen += 1

                if docs_seen % 500 == 0:
                    _flush_buffer(train_buffer, tf)
                    _flush_buffer(val_buffer, vf)
                    split_state.update({
                        "docs_seen": docs_seen,
                        "train_docs": train_docs,
                        "val_docs": val_docs,
                    })
                    _save_checkpoint(checkpoint_path, ckpt)

                    elapsed = time.time() - start_t
                    print(
                        f"[hermes/{config}] seen={docs_seen:,} train={train_docs} "
                        f"val={val_docs} elapsed={_format_time(elapsed)}",
                        flush=True,
                    )

                if profile.hermes_train_limit and train_docs >= profile.hermes_train_limit:
                    break

            _flush_buffer(train_buffer, tf)
            _flush_buffer(val_buffer, vf)

        split_state.update({
            "docs_seen": docs_seen,
            "train_docs": train_docs,
            "val_docs": val_docs,
            "completed": True,
        })
        _save_checkpoint(checkpoint_path, ckpt)

    return {
        "dataset": "lambda/hermes-agent-reasoning-traces",
        "configs": configs,
        "train_docs": total_train_docs,
        "val_docs": total_val_docs,
        "train_tokens": total_train_tokens,
        "val_tokens": total_val_tokens,
        "max_length": profile.max_length,
        "stride": profile.stride,
        "dtype": "uint32",
        "tokenizer": "cl100k_base",
    }


# ── Combine Files ──────────────────────────────────────────────

def combine_token_files(*input_paths: Path, output_path: Path) -> int:
    """Concatenate multiple .bin token files into one.

    Uses chunked copy to keep memory low. Returns total bytes written.
    """
    total_bytes = 0
    CHUNK = 64 * 1024 * 1024  # 64MB chunks

    with output_path.open("wb") as out:
        for inp in input_paths:
            if not inp.exists():
                print(f"  WARNING: {inp} not found, skipping", flush=True)
                continue
            size = inp.stat().st_size
            print(f"  Combining {inp.name} ({size / 1e9:.2f} GB)...", flush=True)
            with inp.open("rb") as f:
                while True:
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    total_bytes += len(chunk)

    return total_bytes


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download & tokenize SYNTH + Hermes datasets for hybrid5 SLM"
    )
    parser.add_argument(
        "--dataset",
        choices=["synth", "hermes", "both"],
        default="both",
        help="Which dataset to process",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES.keys(),
        default="6gb",
        help="Memory/speed profile",
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--language",
        default="en",
        help="SYNTH language filter (default: en). Use 'all' for no filter.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        default=True,
        help="Also create combined synth+hermes token files",
    )
    parser.add_argument(
        "--no-combine",
        dest="combine",
        action="store_false",
    )
    parser.add_argument(
        "--synth-limit",
        type=int,
        default=None,
        help="Max SYNTH train docs to process (overrides profile default)",
    )
    parser.add_argument(
        "--hermes-limit",
        type=int,
        default=None,
        help="Max Hermes train docs per config (overrides profile default)",
    )

    args = parser.parse_args()
    profile = PROFILES[args.profile]
    _configure_env(profile.num_threads)

    # Override profile limits with CLI args
    if args.synth_limit is not None:
        profile.synth_train_limit = args.synth_limit
    if args.hermes_limit is not None:
        profile.hermes_train_limit = args.hermes_limit

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "download_checkpoint.json"

    language_filter = args.language if args.language != "all" else None

    print("=" * 70)
    print("DOWNLOAD & TOKENIZE PIPELINE")
    print("=" * 70)
    print(f"  Datasets:    {args.dataset}")
    print(f"  Profile:     {args.profile}")
    print(f"  Output dir:  {output_dir.resolve()}")
    print(f"  Resume:      {args.resume}")
    print(f"  SYNTH lang:  {language_filter or 'all'}")
    print(f"  Max length:  {profile.max_length}")
    print(f"  Stride:      {profile.stride}")
    print()

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Save tokenizer meta
    tokenizer_meta = {
        "name": "cl100k_base",
        "vocab_size": tokenizer.n_vocab,
        "eot_token": int(tokenizer.eot_token),
        "type": "tiktoken",
    }
    with (output_dir / "tokenizer_meta.json").open("w") as f:
        json.dump(tokenizer_meta, f, indent=2)

    synth_meta = None
    hermes_meta = None

    # Graceful shutdown handler
    shutdown_requested = False

    def _signal_handler(sig, frame):
        nonlocal shutdown_requested
        print(f"\nSignal {sig} received. Finishing current doc and saving checkpoint...", flush=True)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── SYNTH ──
    if args.dataset in ("synth", "both") and not shutdown_requested:
        print("\n" + "-" * 70)
        print("Processing: PleIAs/SYNTH")
        print("-" * 70)
        try:
            synth_meta = tokenize_synth(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                tokenizer=tokenizer,
                profile=profile,
                language_filter=language_filter,
                resume=args.resume,
            )
            with (output_dir / "synth_meta.json").open("w") as f:
                json.dump(synth_meta, f, indent=2)
            print(f"\n  SYNTH done: {synth_meta['train_docs']:,} train docs, "
                  f"{synth_meta['val_docs']:,} val docs, "
                  f"~{synth_meta['train_tokens']:,} train tokens")
        except Exception as e:
            print(f"\n  ERROR processing SYNTH: {e}", flush=True)
            traceback.print_exc()
            print("  Checkpoint saved. Re-run with --resume to continue.", flush=True)

    # ── Hermes ──
    if args.dataset in ("hermes", "both") and not shutdown_requested:
        print("\n" + "-" * 70)
        print("Processing: lambda/hermes-agent-reasoning-traces")
        print("-" * 70)
        try:
            hermes_meta = tokenize_hermes(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                tokenizer=tokenizer,
                profile=profile,
                configs=["kimi", "glm-5.1"],
                val_ratio=0.1,  # 10% val (small dataset)
                resume=args.resume,
            )
            with (output_dir / "hermes_meta.json").open("w") as f:
                json.dump(hermes_meta, f, indent=2)
            print(f"\n  Hermes done: {hermes_meta['train_docs']:,} train docs, "
                  f"{hermes_meta['val_docs']:,} val docs")
        except Exception as e:
            print(f"\n  ERROR processing Hermes: {e}", flush=True)
            traceback.print_exc()
            print("  Checkpoint saved. Re-run with --resume to continue.", flush=True)

    # ── Combine ──
    if args.combine and not shutdown_requested:
        print("\n" + "-" * 70)
        print("Combining token files...")
        print("-" * 70)

        synth_train = output_dir / "synth_train_tokens.bin"
        hermes_train = output_dir / "hermes_train_tokens.bin"
        synth_val = output_dir / "synth_val_tokens.bin"
        hermes_val = output_dir / "hermes_val_tokens.bin"

        combined_train = output_dir / "combined_synth_hermes_train_tokens.bin"
        combined_val = output_dir / "combined_synth_hermes_val_tokens.bin"

        # Train: SYNTH first (vast majority), then Hermes
        train_bytes = combine_token_files(synth_train, hermes_train, output_path=combined_train)
        print(f"  Combined train: {train_bytes / 1e9:.2f} GB ({train_bytes // 4:,} tokens)")

        # Val: both
        val_bytes = combine_token_files(synth_val, hermes_val, output_path=combined_val)
        print(f"  Combined val:   {val_bytes / 1e9:.2f} GB ({val_bytes // 4:,} tokens)")

        combined_meta = {
            "train_files": [str(f) for f in [synth_train, hermes_train] if f.exists()],
            "val_files": [str(f) for f in [synth_val, hermes_val] if f.exists()],
            "train_bytes": train_bytes,
            "val_bytes": val_bytes,
            "train_tokens_est": train_bytes // 4,
            "val_tokens_est": val_bytes // 4,
            "dtype": "uint32",
            "tokenizer": "cl100k_base",
        }
        with (output_dir / "combined_synth_hermes_meta.json").open("w") as f:
            json.dump(combined_meta, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    if synth_meta:
        print(f"  SYNTH:  {synth_meta['train_tokens']:,} train tokens")
    if hermes_meta:
        print(f"  Hermes: {hermes_meta['train_docs']:,} train conversations")
    print(f"  Files in: {output_dir.resolve()}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
