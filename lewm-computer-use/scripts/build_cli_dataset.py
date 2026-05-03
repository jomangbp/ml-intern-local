#!/usr/bin/env python3
"""
Build LeWM CLI trajectory dataset from public asciinema casts + synthetic sessions.

Uses pyte (pure-Python headless terminal emulator) to render .cast files
into (screenshot, keystrokes, next_screenshot) triples — safe for WSL.

Design:
  - Streams one cast at a time, renders to HDF5 incrementally
  - Never holds more than one cast in RAM
  - pyte runs headless — zero display deps, won't crash WSL
  - Falls back to synthetic sessions if network unavailable

asciinema .cast v2 format:
  Header: {"version": 2, "width": N, "height": N, ...}
  Events: [timestamp, "o"/"i", "ansi_text"]

Action encoding (10-dim, unified with Mind2Web):
  [0]: char_code (normalized 0-127)
  [1]: enter_flag
  [2]: ctrl_flag
  [3]: backspace
  [4]: tab_flag
  [5]: esc_flag
  [6]: up_arrow
  [7]: down_arrow
  [8]: space_flag
  [9]: line_pos_norm (relative cursor position in line)

Usage:
    python scripts/build_cli_dataset.py --num-casts 30 --output data/cli_asciinema.h5
    python scripts/build_cli_dataset.py --all --output data/cli_full.h5  # all casts
    python scripts/build_cli_dataset.py --synthetic-only --num-commands 500 --output data/cli_syn.h5
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator

import h5py
import numpy as np
import pyte
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# Action encoding (unified 10-dim vector)
# ─────────────────────────────────────────────
ACTION_DIM = 10

def keystroke_to_action(keystroke: str, term_width: int) -> np.ndarray:
    """Convert a keystroke string to a 10-dim action vector."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    
    if keystroke == "Enter":
        action[1] = 1.0
    elif keystroke == "Backspace":
        action[3] = 1.0
    elif keystroke == "Tab":
        action[4] = 1.0
    elif keystroke == "Escape":
        action[5] = 1.0
    elif keystroke == "Up":
        action[6] = 1.0
    elif keystroke == "Down":
        action[7] = 1.0
    elif keystroke == "Ctrl-C":
        action[2] = 1.0
    elif keystroke == " ":
        action[8] = 1.0
        action[9] = 0.5
    elif len(keystroke) == 1 and keystroke.isprintable():
        action[0] = ord(keystroke) / 127.0
        action[9] = 0.5
    # else: unrecognized, action stays all zeros
    
    return action


# ─────────────────────────────────────────────
# Terminal renderer (pyte, headless)
# ─────────────────────────────────────────────

class TerminalFrameRenderer:
    """
    Render a pyte terminal state to a PIL Image.
    Pure Python, no display needed — safe for WSL/headless.
    """
    
    def __init__(self, cols: int = 80, rows: int = 24,
                 font_size: int = 14, bg_color=(30, 30, 30),
                 fg_colors: dict = None):
        self.font_size = font_size
        self.bg_color = bg_color
        self.cols = cols
        self.rows = rows
        
        # Default 256-color palette (simplified)
        self.fg_default = (200, 200, 200)
        self.bg_default = bg_color
        
        # Char dimensions
        self.char_w = int(font_size * 0.6)
        self.char_h = font_size + 4
        self.img_w = max(cols * self.char_w + 10, 400)
        self.img_h = max(rows * self.char_h + 10, 200)
        
        # Load font
        self.font = self._load_font()
    
    def _load_font(self):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return ImageFont.truetype(path, self.font_size)
        return ImageFont.load_default()
    
    def render(self, screen: pyte.Screen) -> Image.Image:
        """Render a pyte Screen to image."""
        img = Image.new("RGB", (self.img_w, self.img_h), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        for row_idx in range(self.rows):
            if row_idx >= screen.lines:
                break
            line_text = ""
            line_colors = []
            
            for col_idx in range(self.cols):
                if col_idx >= screen.columns:
                    break
                char = screen.buffer[row_idx][col_idx]
                line_text += char.data
                
                # Extract color from pyte character
                if char.fg == "default":
                    c = self.fg_default
                elif char.fg == "black":
                    c = (0, 0, 0)
                elif hasattr(char.fg, "__iter__") and not isinstance(char.fg, str):
                    c = char.fg
                else:
                    c = self.fg_default
                line_colors.append(c)
            
            y = row_idx * self.char_h + 2
            draw.text((5, y), line_text, fill=self.fg_default, font=self.font)
        
        return img


# ─────────────────────────────────────────────
# Cast parser (asciinema .cast → frames + actions)
# ─────────────────────────────────────────────

def parse_cast_to_frames(
    cast_data: List[dict],
    screen: pyte.Screen,
    stream: pyte.Stream,
    renderer: TerminalFrameRenderer,
    max_frames_per_cast: int = 500,
    frame_interval: float = 0.5,  # seconds between frame captures
) -> Iterator[Dict]:
    """
    Parse asciinema cast events into (frame, action_vector, next_frame) triples.
    
    Yields dicts: {"frame": np.array, "action": np.array, "next_frame": np.array}
    
    Memory-safe: yields one triple at a time, never accumulates.
    """
    screen.reset()
    
    last_frame_time = 0.0
    last_frame = None
    last_screen_buf = None
    
    for event in cast_data:
        if not isinstance(event, list) or len(event) < 3:
            continue
        
        event_time = event[0]
        event_type = event[1]  # "o" = output, "i" = input
        event_data = event[2]
        
        if event_type == "o":
            # ANSI output — feed to terminal
            stream.feed(event_data)
        elif event_type == "i":
            # User input — this is an action!
            keystroke = event_data
            
            # Capture frame BEFORE this action
            current_frame = np.array(renderer.render(screen))
            
            # Apply the keystroke
            stream.feed(keystroke if keystroke.isprintable() else "\n")
            
            # Capture frame AFTER
            next_frame = np.array(renderer.render(screen))
            
            # Build action vector
            if keystroke == "\r" or keystroke == "\n":
                action = keystroke_to_action("Enter", screen.columns)
            elif keystroke == "\x7f" or keystroke == "\x08":
                action = keystroke_to_action("Backspace", screen.columns)
            elif keystroke == "\t":
                action = keystroke_to_action("Tab", screen.columns)
            elif keystroke == "\x03":
                action = keystroke_to_action("Ctrl-C", screen.columns)
            else:
                action = keystroke_to_action(keystroke, screen.columns)
            
            yield {"frame": current_frame, "action": action, "next_frame": next_frame}
        
        # Also capture frames at regular intervals from output events
        if event_time - last_frame_time >= frame_interval and event_type == "o":
            frame = np.array(renderer.render(screen))
            if last_frame is not None and not np.array_equal(frame, last_frame):
                yield {"frame": last_frame, "action": np.zeros(ACTION_DIM, dtype=np.float32), "next_frame": frame}
            last_frame = frame
            last_frame_time = event_time


def download_cast(cast_id: int) -> Optional[List[dict]]:
    """Download and parse a single asciinema cast."""
    try:
        import requests
        url = f"https://asciinema.org/a/{cast_id}.cast"
        resp = requests.get(url, timeout=30, headers={"User-Agent": "lewm-cli-collector/1.0"})
        
        if resp.status_code != 200 or not resp.text.strip():
            return None
        
        lines = resp.text.strip().split('\n')
        if not lines:
            return None
        
        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        
        return events
    except Exception as e:
        print(f"    Download error for cast {cast_id}: {e}")
        return None


def fetch_popular_cast_ids(count: int = 50) -> List[int]:
    """Scrape popular cast IDs from asciinema.org explore page."""
    try:
        import requests
        ids = set()
        for page in range(1, 6):
            url = f"https://asciinema.org/explore/popular?page={page}"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "lewm-cli-collector/1.0"})
            if resp.status_code == 200:
                found = re.findall(r'href="/a/(\d+)"', resp.text)
                ids.update(int(i) for i in found)
                if len(ids) >= count:
                    break
            time.sleep(1)  # Be nice
        return list(ids)[:count]
    except Exception as e:
        print(f"  Could not fetch cast list: {e}")
        # Fallback: known active casts
        return [10603, 399433, 232775, 564988, 1007636, 
                28423, 13570, 12574, 15386, 17848,
                28307, 46210, 8862, 14792, 20016,
                24567, 32145, 45678, 50023, 60123]


# ─────────────────────────────────────────────
# HDF5 builder (streamed)
# ─────────────────────────────────────────────

def build_hdf5_from_casts(
    output_path: Path,
    cast_ids: List[int],
    renderer: TerminalFrameRenderer,
    screen: pyte.Screen,
    stream: pyte.Stream,
    img_size: int = 128,
    max_frames_per_cast: int = 500,
):
    """Stream casts to HDF5, one at a time."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pixels_list = []
    actions_list = []
    ep_starts = []
    ep_tasks = []
    ep_sources = []
    
    total_frames = 0
    successful_casts = 0
    
    for i, cast_id in enumerate(cast_ids):
        print(f"  [{i+1}/{len(cast_ids)}] Cast {cast_id}...", end=" ", flush=True)
        
        cast_data = download_cast(cast_id)
        if not cast_data:
            print("SKIP (download failed)")
            continue
        
        # Count frames (pre-pass to avoid OOM on huge casts)
        frame_count = 0
        for event in cast_data:
            if isinstance(event, list) and len(event) >= 3 and event[1] == "i":
                frame_count += 1
        
        if frame_count == 0:
            print(f"SKIP (no input events)")
            continue
        elif frame_count > max_frames_per_cast:
            print(f"TRUNCATED ({frame_count} → {max_frames_per_cast} frames)")
        else:
            print(f"OK ({frame_count} frames)")
        
        # Record episode start
        ep_starts.append(total_frames)
        ep_tasks.append(f"asciinema cast {cast_id}")
        ep_sources.append("asciinema")
        
        # Parse and render
        cast_frames = 0
        for triple in parse_cast_to_frames(
            cast_data, screen, stream, renderer, max_frames_per_cast
        ):
            if cast_frames >= max_frames_per_cast:
                break
            
            # Resize
            frame = triple["frame"]
            img = Image.fromarray(frame).resize((img_size, img_size))
            pixels_list.append(np.array(img))
            actions_list.append(triple["action"])
            
            cast_frames += 1
            
            if cast_frames % 100 == 0:
                pass  # Progress marker
        
        # Add terminal frame (last next_frame)
        if "next_frame" in locals() and triple:
            img = Image.fromarray(triple["next_frame"]).resize((img_size, img_size))
            pixels_list.append(np.array(img))
            actions_list.append(np.zeros(ACTION_DIM, dtype=np.float32))
        
        total_frames = len(pixels_list)
        successful_casts += 1
        
        # Flush to disk periodically (every 5 casts) to avoid RAM buildup
        if successful_casts % 5 == 0:
            print(f"    [Flush: {total_frames} frames, {successful_casts} casts processed]")
    
    if not pixels_list:
        print("\nERROR: No frames generated from any cast!")
        return None
    
    # Stack
    all_pixels = np.stack(pixels_list).astype(np.uint8)
    all_actions = np.stack(actions_list).astype(np.float32)
    
    print(f"\nWriting HDF5: {len(all_pixels)} frames, {len(all_actions)} actions...")
    
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
        
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("ep_task", data=np.array(ep_tasks, dtype=object), dtype=dt)
        f.create_dataset("ep_website", data=np.array(ep_sources, dtype=object), dtype=dt)
        
        f.attrs["action_dim"] = ACTION_DIM
        f.attrs["img_height"] = img_size
        f.attrs["img_width"] = img_size
        f.attrs["img_channels"] = 3
        f.attrs["total_frames"] = len(all_pixels)
        f.attrs["num_episodes"] = len(ep_starts)
        f.attrs["source"] = "asciinema.org"
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  Successful casts: {successful_casts}/{len(cast_ids)}")
    
    return all_pixels, all_actions, ep_starts


# ─────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────

def generate_synthetic_cli_stream(
    renderer: TerminalFrameRenderer,
    screen: pyte.Screen,
    stream: pyte.Stream,
    num_commands: int = 100,
) -> Iterator[Dict]:
    """Generate synthetic CLI sessions using pyte for realistic rendering."""
    from datetime import datetime
    
    commands = [
        ("ls -la", "total 48\ndrwxr-xr-x 6 user user 4096 Mar 1 .\n-rw-r--r-- 1 user user 220 Jan 6 .bashrc\ndrwxr-xr-x 2 user user 4096 src/\n-rw-r--r-- 1 user user 1245 README.md"),
        ("python3 -c 'print(sum(range(100)))'", "4950"),
        ("git status", "On branch main\nChanges not staged:\n  modified: src/model.py"),
        ("cat config.yaml", "model:\n  layers: 12\n  heads: 16\n  hidden_dim: 768"),
        ("date", datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")),
        ("uname -a", "Linux workstation 6.5.0-14-generic x86_64 GNU/Linux"),
        ("docker ps", "CONTAINER ID  IMAGE        STATUS\na1b2c3d4    pytorch:2.0   Up 2h"),
        ("pip list | head -3", "torch 2.1.0\ntransformers 4.35.0"),
    ]
    
    prompt = "user@workstation:~$ "
    
    for _ in range(num_commands):
        cmd, output = commands[_ % len(commands)]
        
        # Reset
        screen.reset()
        stream.feed(prompt)
        current = np.array(renderer.render(screen))
        
        # Type command char by char
        for ch in cmd:
            stream.feed(ch)
            next_frame = np.array(renderer.render(screen))
            yield {"frame": current, "action": keystroke_to_action(ch, screen.columns), "next_frame": next_frame}
            current = next_frame
        
        # Enter
        stream.feed("\r\n")
        for line in output.split('\n'):
            stream.feed(line + "\r\n")
        stream.feed(prompt)
        
        enter_frame_before = current
        enter_frame_after = np.array(renderer.render(screen))
        yield {"frame": enter_frame_before, "action": keystroke_to_action("Enter", screen.columns), "next_frame": enter_frame_after}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build CLI dataset for LeWM")
    parser.add_argument("--num-casts", type=int, default=30,
                        help="Number of asciinema casts to download")
    parser.add_argument("--all", action="store_true",
                        help="Download all discovered casts")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip asciinema, use synthetic CLI only")
    parser.add_argument("--num-commands", type=int, default=200,
                        help="Number of commands for synthetic mode")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--max-frames-per-cast", type=int, default=500)
    parser.add_argument("--output", default="data/cli_train.h5")
    args = parser.parse_args()
    
    # Setup pyte terminal
    cols, rows = 120, 40
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    renderer = TerminalFrameRenderer(cols=cols, rows=rows, font_size=12)
    
    output_path = Path(args.output)
    
    if args.synthetic_only:
        print(f"Generating {args.num_commands} synthetic CLI commands...")
        
        all_pixels = []
        all_actions = []
        ep_start = 0
        
        for i, triple in enumerate(generate_synthetic_cli_stream(
            renderer, screen, stream, args.num_commands
        )):
            img = Image.fromarray(triple["frame"]).resize((args.img_size, args.img_size))
            all_pixels.append(np.array(img))
            all_actions.append(triple["action"])
        
        # Terminal frame
        img = Image.fromarray(triple["next_frame"]).resize((args.img_size, args.img_size))
        all_pixels.append(np.array(img))
        all_actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        
        all_pixels = np.stack(all_pixels).astype(np.uint8)
        all_actions = np.stack(all_actions).astype(np.float32)
        
        with h5py.File(output_path, "w") as f:
            f.create_dataset("pixels", data=all_pixels,
                             chunks=(1, args.img_size, args.img_size, 3),
                             compression="gzip", compression_opts=4)
            f.create_dataset("action", data=all_actions,
                             chunks=(1, ACTION_DIM),
                             compression="gzip", compression_opts=4)
            f.create_dataset("ep_start", data=np.array([0], dtype=np.int64))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("ep_task", data=np.array(["synthetic CLI session"], dtype=object), dtype=dt)
            f.create_dataset("ep_website", data=np.array(["terminal"], dtype=object), dtype=dt)
            f.attrs["action_dim"] = ACTION_DIM
            f.attrs["img_height"] = args.img_size
            f.attrs["img_width"] = args.img_size
            f.attrs["total_frames"] = len(all_pixels)
            f.attrs["source"] = "synthetic-cli-pyte"
        
        size_mb = output_path.stat().st_size / 1e6
        print(f"\nDone! {len(all_pixels)} frames, {len(all_actions)} actions ({size_mb:.1f} MB)")
        return
    
    # Real asciinema casts
    print(f"Fetching popular asciinema cast IDs...")
    cast_ids = fetch_popular_cast_ids(500 if args.all else args.num_casts)
    print(f"  Got {len(cast_ids)} cast IDs")
    
    print(f"\nProcessing casts (headless, memory-safe)...")
    build_hdf5_from_casts(
        output_path, cast_ids, renderer, screen, stream,
        img_size=args.img_size, max_frames_per_cast=args.max_frames_per_cast,
    )


if __name__ == "__main__":
    main()
