#!/usr/bin/env python3
"""
Generate synthetic CLI (terminal) interaction trajectories for LeWM training.

Simulates a terminal emulator: renders a terminal session as images,
produces (screenshot, keystrokes, next_screenshot) triples.

Unlike Mind2Web's web actions, CLI actions are keypresses.
Action vector (10-dim, same as Mind2Web for unified training):
  [0]: char_code  (normalized ASCII/Unicode of typed char, 0 for special keys)
  [1]: enter_flag (1.0 = Enter pressed)
  [2]: ctrl_flag  (1.0 = Ctrl held)
  [3]: backspace  (1.0)
  [4]: tab_flag   (1.0)
  [5]: esc_flag   (1.0)
  [6]: up_arrow   (1.0)
  [7]: down_arrow (1.0)
  [8]: space_flag (1.0 for space specifically, helps with word boundaries)
  [9]: cursor_pos_norm (0-1, horizontal cursor position in line)

Output HDF5 (same format as Mind2Web converter):
  /pixels     (N, H, W, 3) uint8
  /action     (N, 10) float32
  /ep_start   (E,) int64
  /ep_task    (E,) string
  /ep_website (E,) string (for CLI: "terminal")

Usage:
    python scripts/generate_cli.py --num-commands 200 --output data/cli_train.h5
    python scripts/generate_cli.py --num-commands 2000 --img-size 128 --output data/cli_large.h5
"""

import argparse
import itertools
import os
import random
import string
import subprocess
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────
# Terminal Emulator
# ──────────────────────────────────────────────

class TerminalSimulator:
    """
    Simulates a terminal and renders it to image frames.
    
    Pure Python — no external dependencies beyond PIL.
    Mimics a standard bash session: prompt, typed commands, outputs.
    """
    
    # Common shell commands that produce interesting output
    COMMANDS = [
        # File listing
        ("ls -la", "total 48\ndrwxr-xr-x 6 user user 4096 Mar  1 10:30 .\ndrwxr-xr-x 4 root root 4096 Feb 28 09:15 ..\n-rw-r--r-- 1 user user  220 Jan  6 14:22 .bashrc\n-rw-r--r-- 1 user user  807 Jan  6 14:22 .profile\ndrwxr-xr-x 2 user user 4096 Feb 15 11:00 src\ndrwxr-xr-x 3 user user 4096 Feb 20 16:45 data\n-rw-r--r-- 1 user user 1245 Mar  1 10:28 README.md\n-rwxr-xr-x 1 user user 3456 Feb 25 08:30 train.py"),
        ("ls src/", "__init__.py  model.py  utils.py  config.yaml"),
        
        # File content
        ("cat README.md", "# Project\n\nA machine learning pipeline for training models.\n\n## Usage\n\n```bash\npython train.py --config config.yaml\n```\n\n## Results\n\n| Model | Accuracy |\n|-------|----------|\n| baseline | 0.82 |\n| improved | 0.91 |"),
        ("head -5 data/results.csv", "epoch,train_loss,val_loss,accuracy\n1,1.234,1.456,0.45\n2,0.987,1.234,0.52\n3,0.876,1.012,0.61\n4,0.765,0.890,0.68"),
        
        # System info
        ("pwd", "/home/user/project"),
        ("whoami", "user"),
        ("date", datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")),
        ("uname -a", "Linux workstation 6.5.0-14-generic #14-Ubuntu SMP PREEMPT_DYNAMIC x86_64 GNU/Linux"),
        ("free -h", "               total        used        free      shared  buff/cache   available\nMem:           16Gi       4.2Gi       8.1Gi       234Mi       3.7Gi        11Gi\nSwap:         8.0Gi       1.2Gi       6.8Gi"),
        ("df -h", "Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  256G   89G  154G  37% /\n/dev/nvme0n1p1  511M  6.1M  505M   2% /boot/efi\n/dev/sda1       1.8T  456G  1.3T  26% /data"),
        
        # Python
        ("python3 -c \"print('Hello, World!')\"", "Hello, World!"),
        ("python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\" 2>/dev/null || echo 'torch not found'", "torch not found"),
        ("python3 << 'EOF'\nfor i in range(5):\n    print(f'Iteration {i}: loss={1.0/(i+1):.4f}')\nEOF", "Iteration 0: loss=1.0000\nIteration 1: loss=0.5000\nIteration 2: loss=0.3333\nIteration 3: loss=0.2500\nIteration 4: loss=0.2000"),
        
        # Docker
        ("docker ps", "CONTAINER ID   IMAGE          COMMAND       CREATED        STATUS       PORTS     NAMES\na1b2c3d4e5f6   pytorch:2.0    \"jupyter\"     2 hours ago    Up 2 hours   8888/tcp  ml-notebook\nb2c3d4e5f6a1   nginx:latest   \"nginx -g\"    3 days ago     Up 3 days    80/tcp    web-server"),
        
        # Git
        ("git status", "On branch main\nYour branch is up to date with 'origin/main'.\n\nChanges not staged for commit:\n  modified:   src/model.py\n  modified:   config/train.yaml\n\nUntracked files:\n  scripts/new_experiment.py"),
        ("git log --oneline -5", "a1b2c3d Fix training loop bug\ne4f5g6h Add evaluation metrics\ni7j8k9l Implement data loader\nm0n1o2p Initial commit\nq3r4s5t Setup project structure"),
        ("git diff --stat", " src/model.py | 15 +++++++++------\n 1 file changed, 9 insertions(+), 6 deletions(-)"),
        
        # Grep/find
        ("grep -r 'learning_rate' config/", "config/train.yaml:learning_rate: 5e-5\nconfig/default.yaml:learning_rate: 1e-4"),
        ("find . -name '*.py' | head -5", "./src/__init__.py\n./src/model.py\n./src/utils.py\n./scripts/train.py\n./scripts/eval.py"),
        
        # Pip/npm
        ("pip list | head -5", "Package            Version\n------------------ ---------\ntorch              2.1.0\ntransformers       4.35.0\nnumpy              1.24.3"),
        
        # Network
        ("ping -c 2 8.8.8.8 2>/dev/null || echo 'network: simulating'", "network: simulating"),
        ("curl -sI https://api.github.com 2>/dev/null | head -3 || echo 'HTTP/1.1 200 OK'", "HTTP/1.1 200 OK"),
        
        # Htop-like
        ("ps aux | head -5", "USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nroot           1  0.0  0.1 168940 12340 ?        Ss   09:00   0:03 /sbin/init\nuser       1234  2.5  3.2 2345678 524288 ?      Sl   09:15   1:23 python train.py\nuser       5678  0.1  0.3 876540 49152 pts/0    Ss   10:00   0:00 /bin/bash"),
        
        # Wget / curl
        ("wget -qO- https://example.com 2>/dev/null | head -5 || echo '<!doctype html><html><head><title>Example</title>'", "<!doctype html><html><head><title>Example</title>"),
        
        # Config files
        ("cat ~/.bashrc | head -5", "# ~/.bashrc\n# Source global definitions\nif [ -f /etc/bashrc ]; then\n    . /etc/bashrc\nfi"),
        ("cat config/train.yaml", "model:\n  type: transformer\n  layers: 12\n  heads: 16\n  hidden_dim: 768\n\ntraining:\n  epochs: 100\n  batch_size: 32\n  learning_rate: 5e-5\n  weight_decay: 0.01"),
        
        # SSH
        ("ssh -T git@github.com 2>&1 | head -1", "Hi user! You've successfully authenticated."),
        
        # Tar / zip
        ("tar -czf model_checkpoint.tar.gz outputs/", "(compressing...)"),
        ("ls -lh model_checkpoint.tar.gz", "-rw-r--r-- 1 user user 234M Mar  1 10:45 model_checkpoint.tar.gz"),
    ]
    
    # Prompt variations
    PROMPTS = [
        "user@workstation:~/project$ ",
        "user@workstation:~/project/src$ ",
        "user@workstation:~/data$ ",
        "user@workstation:~$ ",
        "(venv) user@workstation:~/project$ ",
        "root@server:/opt/app# ",
    ]
    
    # Common typos and corrections (realistic human behavior)
    TYPOS = {
        "ls": ["lss", "ks"],
        "cat": ["car", "cst"],
        "git": ["got", "gir"],
        "python": ["pyhton", "pythin"],
        "docker": ["dicker", "dockr"],
    }
    
    def __init__(
        self,
        width: int = 800,
        height: int = 480,
        font_size: int = 14,
        bg_color: Tuple[int, int, int] = (30, 30, 30),
        fg_color: Tuple[int, int, int] = (200, 200, 200),
        prompt_color: Tuple[int, int, int] = (100, 255, 100),
        error_color: Tuple[int, int, int] = (255, 100, 100),
    ):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.prompt_color = prompt_color
        self.error_color = error_color
        
        # Try to load a monospace font
        self.font = self._load_font(font_size)
        self.font_bold = self._load_font(font_size, bold=True)
        
        # Terminal state
        self.buffer: List[Tuple[str, Tuple[int, int, int]]] = []  # (text, color)
        self.cursor_col = 0
        self.line_height = font_size + 4
        self.max_lines = (height - 20) // self.line_height
        self.chars_per_line = int((width - 20) // (font_size * 0.6))
    
    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Try to load a monospace font, fall back to default."""
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "C:\\Windows\\Fonts\\consola.ttf",
        ]
        
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        
        # Fallback
        try:
            return ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            return ImageFont.load_default()
    
    def write(self, text: str, color: Optional[Tuple[int, int, int]] = None):
        """Write text to the terminal buffer."""
        color = color or self.fg_color
        self.buffer.append((text, color))
        self.cursor_col += len(text)
    
    def writeln(self, text: str = "", color: Optional[Tuple[int, int, int]] = None):
        """Write a line of text."""
        color = color or self.fg_color
        self.buffer.append((text, color))
        self.cursor_col = 0
    
    def clear(self):
        """Clear the terminal buffer."""
        self.buffer = []
        self.cursor_col = 0
    
    def render(self) -> Image.Image:
        """Render the terminal buffer to a PIL Image."""
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Calculate visible lines (scroll from bottom)
        total_lines = sum(
            (len(text) // self.chars_per_line + 1) if text else 1
            for text, _ in self.buffer
        )
        
        # Keep last max_lines visible
        y_offset = self.height - 10 - self.line_height
        visible = []
        remaining_lines = self.max_lines
        
        for text, color in reversed(self.buffer):
            if remaining_lines <= 0:
                break
            num_lines = (len(text) // self.chars_per_line + 1) if text else 1
            visible.append((text, color, num_lines))
            remaining_lines -= num_lines
        
        visible.reverse()
        
        # Draw from top
        y = 5
        for text, color, num_lines in visible:
            if not text:
                y += self.line_height
                continue
            
            # Handle long lines by wrapping
            for i in range(0, len(text), self.chars_per_line):
                chunk = text[i:i + self.chars_per_line]
                if y + self.line_height > self.height:
                    break
                draw.text((10, y), chunk, fill=color, font=self.font)
                y += self.line_height
            
            # If text ends exactly at line boundary, don't add extra line
            if text and len(text) % self.chars_per_line == 0:
                pass  # already advanced
        
        return img
    
    def run_command(self, command: str, output: str, prompt: str) -> List[dict]:
        """
        Run a command and generate trajectory steps.
        
        Returns list of steps, each:
        {
            "screenshot": np.ndarray (H, W, 3) uint8,
            "action": np.ndarray (10,) float32,
            "next_screenshot": np.ndarray (H, W, 3) uint8,
        }
        """
        steps = []
        self.clear()
        
        # Step 0: Show prompt (empty terminal for first command, else has previous output)
        self.write(prompt, self.prompt_color)
        current_frame = np.array(self.render())
        
        # Type the command character by character
        typed_so_far = ""
        for i, char in enumerate(command):
            # Encode the action for this character
            action = np.zeros(10, dtype=np.float32)
            
            if char == '\n':
                pass  # handled below
            elif char == ' ':
                action[8] = 1.0  # space
            elif char.isprintable():
                action[0] = ord(char) / 127.0  # normalized char code
            
            action[9] = len(typed_so_far) / max(self.chars_per_line, 1)  # cursor pos
            
            # Render with this character typed
            self.clear()
            self.write(prompt + typed_so_far + char, self.prompt_color)
            next_frame = np.array(self.render())
            
            if i > 0 or True:  # include all keystrokes including first
                steps.append({
                    "screenshot": current_frame.copy(),
                    "action": action.copy(),
                    "next_screenshot": next_frame.copy(),
                })
            
            typed_so_far += char
            current_frame = next_frame
        
        # Step N: Press Enter
        enter_action = np.zeros(10, dtype=np.float32)
        enter_action[1] = 1.0  # Enter
        
        self.clear()
        # Show command + output
        self.write(prompt + command, self.prompt_color)
        
        # Output lines
        for line in output.split('\n'):
            if line.startswith("(error)") or line.startswith("Error"):
                self.writeln(line, self.error_color)
            else:
                self.writeln(line, self.fg_color)
        
        # Next prompt
        next_prompt = random.choice(self.PROMPTS)
        self.write(next_prompt, self.prompt_color)
        
        output_frame = np.array(self.render())
        
        steps.append({
            "screenshot": current_frame.copy(),
            "action": enter_action.copy(),
            "next_screenshot": output_frame.copy(),
        })
        
        # Store the final state for next command
        self._last_frame = output_frame.copy()
        self._last_buffer = list(self.buffer)
        
        return steps
    
    def run_session(self, num_commands: int, include_typos: bool = True) -> List[dict]:
        """
        Run a full terminal session with multiple commands.
        
        Returns all trajectory steps.
        """
        all_steps = []
        prompt = random.choice(self.PROMPTS)
        
        # Start with an initial command to set up state
        for cmd_idx in range(num_commands):
            cmd, output = random.choice(self.COMMANDS)
            
            # Add occasional typos
            if include_typos and random.random() < 0.1:
                # Get first word to find typo
                first_word = cmd.split()[0] if ' ' in cmd else cmd
                if first_word in self.TYPOS:
                    typo = random.choice(self.TYPOS[first_word])
                    typo_cmd = typo + cmd[len(first_word):]
                    
                    # Typo → error → backspace → correct
                    typo_output = f"bash: {typo}: command not found"
                    typo_steps = self.run_command(typo_cmd, typo_output, prompt)
                    all_steps.extend(typo_steps)
                    
                    # Backspace action: clear and retype
                    self.clear()
                    for line in self._last_buffer[:-1]:  # keep all but last prompt
                        self.writeln(line[0], line[1])
                    self.write(prompt, self.prompt_color)
                    
                    # Brief pause
                    prompt = random.choice(self.PROMPTS)
            
            # Run the actual command
            cmd_steps = self.run_command(cmd, output, prompt)
            all_steps.extend(cmd_steps)
            
            # Vary prompt occasionally
            if random.random() < 0.3:
                prompt = random.choice(self.PROMPTS)
        
        return all_steps


# ──────────────────────────────────────────────
# Alternative: real asciinema casts (if available)
# ──────────────────────────────────────────────

def download_asciinema_casts(num_casts: int = 20) -> List[Path]:
    """
    Try downloading public asciinema casts from asciinema.org API.
    Falls back to None if network unavailable.
    """
    try:
        import requests
        import tempfile
        
        casts = []
        # Browse popular casts
        resp = requests.get(
            "https://asciinema.org/api/asciicasts",
            params={"order": "popularity", "count": min(num_casts, 50)},
            timeout=10,
        )
        
        if resp.status_code == 200:
            items = resp.json()
            for item in items[:num_casts]:
                cast_url = item.get("url", "")
                if cast_url.endswith(".cast"):
                    cast_resp = requests.get(cast_url, timeout=10)
                    if cast_resp.status_code == 200:
                        tmp = tempfile.NamedTemporaryFile(suffix=".cast", delete=False)
                        tmp.write(cast_resp.content)
                        tmp.close()
                        casts.append(Path(tmp.name))
        
        return casts
    except Exception:
        return []


# ──────────────────────────────────────────────
# HDF5 builder (reuses format from convert_mind2web)
# ──────────────────────────────────────────────

def generate_cli_dataset(
    num_commands: int,
    img_size: int = 128,
    output_path: Optional[Path] = None,
    sessions: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Generate CLI trajectory data.
    
    Returns:
        all_pixels: (N, H, W, 3) uint8
        all_actions: (N, 10) float32
        ep_starts: list of start indices
        ep_tasks: list of task descriptions
    """
    sim = TerminalSimulator()
    all_steps = []
    all_pixels = []
    all_actions = []
    ep_starts = []
    ep_tasks = []
    
    commands_per_session = num_commands // sessions
    
    for sess in range(sessions):
        ep_start = len(all_steps)
        
        n_cmds = min(commands_per_session, num_commands - sess * commands_per_session)
        ep_steps = sim.run_session(n_cmds)
        
        ep_starts.append(len(all_steps))
        ep_tasks.append(f"CLI session {sess + 1}: {n_cmds} commands")
        
        all_steps.extend(ep_steps)
    
    # Resize and stack
    print(f"\nGenerated {len(all_steps)} CLI steps across {sessions} sessions")
    
    for step in all_steps:
        # Before screenshot
        img_before = Image.fromarray(step["screenshot"]).resize((img_size, img_size))
        all_pixels.append(np.array(img_before))
        all_actions.append(step["action"])
        
        # Also add the after as next frame
        # (we handle pairing in the dataset loader)
    
    # Also add the final "after" frame
    if all_steps:
        img_final = Image.fromarray(all_steps[-1]["next_screenshot"]).resize((img_size, img_size))
        all_pixels.append(np.array(img_final))
        all_actions.append(np.zeros(10, dtype=np.float32))
    
    all_pixels = np.stack(all_pixels).astype(np.uint8)
    all_actions = np.stack(all_actions).astype(np.float32)
    
    return all_pixels, all_actions, ep_starts, ep_tasks


def save_to_hdf5(
    pixels: np.ndarray,
    actions: np.ndarray,
    ep_starts: List[int],
    ep_tasks: List[str],
    output_path: Path,
    img_size: int,
):
    """Save generated data to HDF5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "pixels", data=pixels,
            chunks=(1, img_size, img_size, 3),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "action", data=actions,
            chunks=(1, actions.shape[1]),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "ep_start", data=np.array(ep_starts, dtype=np.int64),
        )
        
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("ep_task", data=np.array(ep_tasks, dtype=object), dtype=dt)
        f.create_dataset("ep_website", data=np.array(["terminal"] * len(ep_tasks), dtype=object), dtype=dt)
        
        f.attrs["action_dim"] = actions.shape[1]
        f.attrs["img_height"] = img_size
        f.attrs["img_width"] = img_size
        f.attrs["img_channels"] = 3
        f.attrs["total_frames"] = len(pixels)
        f.attrs["total_action_steps"] = len(actions) - 1
        f.attrs["frameskip"] = 1
        f.attrs["num_episodes"] = len(ep_starts)
        f.attrs["source"] = "synthetic-cli"
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CLI trajectory data for LeWM")
    parser.add_argument("--num-commands", type=int, default=200,
                        help="Total number of commands to simulate")
    parser.add_argument("--sessions", type=int, default=5,
                        help="Number of terminal sessions")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--output", default="data/cli_train.h5")
    parser.add_argument("--no-typos", action="store_true", help="Disable realistic typos")
    args = parser.parse_args()
    
    print(f"Generating CLI data: {args.num_commands} commands across {args.sessions} sessions")
    print(f"  Image size: {args.img_size}×{args.img_size}")
    print(f"  Typos: {'no' if args.no_typos else 'yes'}")
    
    pixels, actions, ep_starts, ep_tasks = generate_cli_dataset(
        num_commands=args.num_commands,
        img_size=args.img_size,
        sessions=args.sessions,
    )
    
    save_to_hdf5(
        pixels, actions, ep_starts, ep_tasks,
        Path(args.output), args.img_size,
    )
    
    print(f"\nDone! Total: {len(pixels)} frames, {len(actions)} actions, {len(ep_starts)} episodes")


if __name__ == "__main__":
    main()
