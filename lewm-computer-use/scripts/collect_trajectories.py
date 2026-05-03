#!/usr/bin/env python3
"""
Collect GUI interaction trajectories for LeWorldModel training.

Uses a VLM (via HuggingFace MCP / Ollama / local) to navigate the desktop,
collecting (screenshot, action, next_screenshot) triples as HDF5.

Actions: [mouse_x_norm, mouse_y_norm, click_type, scroll_delta, keycode_1..keycode_N]
  - mouse_x_norm, mouse_y_norm: normalized [0,1] coordinates
  - click_type: 0=no_click, 1=left_click, 2=right_click, 3=double_click
  - scroll_delta: normalized scroll amount
  - keycode slots: reserved for keyboard input (not used in MVP)

Usage:
    python scripts/collect_trajectories.py --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-episodes 50 --max-steps 10 --task "open browser, search for cats"
    python scripts/collect_trajectories.py --model ollama/llava --num-episodes 20
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyautogui
from mss import mss
from PIL import Image

# Action constants
ACTION_DIM = 5  # [x, y, click, scroll, unused]
CLICK_NONE = 0
CLICK_LEFT = 1
CLICK_RIGHT = 2
CLICK_DOUBLE = 3

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


@dataclass
class TrajectoryStep:
    """A single (observation, action, next_observation) step."""
    screenshot_before: np.ndarray  # (H, W, 3) uint8
    action: np.ndarray             # (ACTION_DIM,) float32
    screenshot_after: np.ndarray   # (H, W, 3) uint8
    action_label: str              # human-readable action description

@dataclass
class Trajectory:
    """A full episode of steps."""
    episode_id: int
    task: str
    steps: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ScreenshotCapture:
    """Captures screenshots using mss (fast)."""
    
    def __init__(self, monitor: int = 1, resize: Optional[tuple] = None):
        self.sct = mss()
        self.monitor = self.sct.monitors[monitor]
        self.resize = resize
    
    def capture(self) -> np.ndarray:
        img = self.sct.grab(self.monitor)
        arr = np.array(img)[:, :, :3]  # BGRA -> RGB
        arr = arr[:, :, ::-1]  # BGR -> RGB
        if self.resize:
            arr = np.array(Image.fromarray(arr).resize(self.resize))
        return arr


class VLMAgent:
    """
    Abstract VLM agent that decides what action to take given a screenshot + task.
    Implementations: HFApiAgent, OllamaAgent, LocalAgent
    """
    
    def decide_action(self, screenshot: np.ndarray, task: str, step: int) -> dict:
        """Returns {'action': np.ndarray, 'label': str, 'done': bool}"""
        raise NotImplementedError


class HFApiAgent(VLMAgent):
    """Uses HuggingFace Inference API (free tier) for action prediction."""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.token = os.environ.get("HF_TOKEN", "")
        if not self.token:
            raise ValueError("HF_TOKEN not set. Export HF_TOKEN or use --model ollama/llava")
    
    def decide_action(self, screenshot: np.ndarray, task: str, step: int) -> dict:
        import requests
        
        # Resize for API efficiency
        img = Image.fromarray(screenshot)
        img = img.resize((640, 360))
        
        # Convert to bytes
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        img_bytes = buf.getvalue()
        
        prompt = (
            f"You are controlling a desktop computer. Task: {task}\n"
            f"Step {step}. Look at the screenshot and decide the NEXT action.\n"
            f"Respond with JSON only: {{\"action\": \"click x y\" or \"type text\" or \"scroll N\" or \"done\", "
            f"\"x\": 0.0-1.0, \"y\": 0.0-1.0, \"text\": \"\"}}"
            f"Where x,y are relative screen positions (0.0=left/top, 1.0=right/bottom)."
        )
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Send image + prompt
        response = requests.post(
            self.api_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 200, "temperature": 0.1}
            },
            timeout=30
        )
        
        # The API may not support images directly; fall back to text-only
        # For proper VLM, we'd need to use messages format with image_url
        # This is a simplified version
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        try:
            result = response.json()
            if isinstance(result, list):
                text = result[0].get("generated_text", "")
            else:
                text = str(result)
            
            # Parse action from text
            return self._parse_action(text, action)
        except Exception as e:
            print(f"  API error: {e}")
            return {"action": action, "label": "noop (api error)", "done": True}
    
    def _parse_action(self, text: str, action: np.ndarray) -> dict:
        done = False
        label = text[:100]
        try:
            # Try to extract JSON
            import re
            match = re.search(r'\{[^}]+\}', text)
            if match:
                parsed = json.loads(match.group())
                act = parsed.get("action", "").lower()
                if "done" in act:
                    done = True
                elif "click" in act:
                    x = float(parsed.get("x", 0.5))
                    y = float(parsed.get("y", 0.5))
                    action[0] = np.clip(x, 0, 1)
                    action[1] = np.clip(y, 0, 1)
                    action[2] = CLICK_LEFT
                elif "scroll" in act:
                    action[3] = 5.0  # scroll down
        except (json.JSONDecodeError, ValueError):
            pass
        return {"action": action, "label": label, "done": done}


class DummyAgent(VLMAgent):
    """Random agent for testing the pipeline."""
    
    def decide_action(self, screenshot: np.ndarray, task: str, step: int) -> dict:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0] = np.random.uniform(0.1, 0.9)  # x
        action[1] = np.random.uniform(0.1, 0.9)  # y
        action[2] = np.random.choice([CLICK_NONE, CLICK_LEFT])  # click
        done = step >= np.random.randint(3, 8)
        label = f"move_to ({action[0]:.2f},{action[1]:.2f}) click={action[2]}"
        return {"action": action, "label": label, "done": done}


def execute_action(action: np.ndarray, screen_w: int, screen_h: int):
    """Execute an action vector on the real desktop."""
    x_px = int(action[0] * screen_w)
    y_px = int(action[1] * screen_h)
    click_type = int(action[2])
    scroll = int(action[3])
    
    if click_type == CLICK_LEFT:
        pyautogui.click(x_px, y_px)
    elif click_type == CLICK_RIGHT:
        pyautogui.rightClick(x_px, y_px)
    elif click_type == CLICK_DOUBLE:
        pyautogui.doubleClick(x_px, y_px)
    elif action[0] > 0 or action[1] > 0:
        pyautogui.moveTo(x_px, y_px)
    
    if scroll != 0:
        pyautogui.scroll(scroll)
    
    time.sleep(0.5)  # Wait for UI to update


def collect_episode(
    agent: VLMAgent,
    capture: ScreenshotCapture,
    episode_id: int,
    task: str,
    max_steps: int = 10,
) -> Trajectory:
    """Collect one episode of GUI interaction."""
    traj = Trajectory(episode_id=episode_id, task=task)
    
    # Get screen dimensions
    monitor = capture.sct.monitors[1]
    screen_w = monitor["width"]
    screen_h = monitor["height"]
    
    print(f"\n  Episode {episode_id}: '{task}'")
    
    for step_idx in range(max_steps):
        # Capture BEFORE state
        screenshot_before = capture.capture()
        
        # Agent decides action
        decision = agent.decide_action(screenshot_before, task, step_idx)
        action_vec = decision["action"]
        
        if decision["done"]:
            print(f"    Step {step_idx}: DONE ({decision['label'][:80]})")
            break
        
        # Execute action
        execute_action(action_vec, screen_w, screen_h)
        
        # Capture AFTER state
        screenshot_after = capture.capture()
        
        # Store step
        step = TrajectoryStep(
            screenshot_before=screenshot_before,
            action=action_vec,
            screenshot_after=screenshot_after,
            action_label=decision["label"],
        )
        traj.steps.append(step)
        
        print(f"    Step {step_idx}: {decision['label'][:80]}")
    
    return traj


def save_trajectories(trajectories: list, output_dir: Path):
    """Save trajectories as individual NPZ files (converted to HDF5 later)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_steps = []
    for traj in trajectories:
        for step in traj.steps:
            # Resize screenshots to target size
            img_before = np.array(Image.fromarray(step.screenshot_before).resize((224, 224)))
            img_after = np.array(Image.fromarray(step.screenshot_after).resize((224, 224)))
            
            all_steps.append({
                "pixels_before": img_before,
                "action": step.action,
                "pixels_after": img_after,
                "episode_id": traj.episode_id,
                "task": traj.task,
            })
    
    # Save as compressed NPZ
    npz_path = output_dir / "trajectories.npz"
    np.savez_compressed(
        npz_path,
        pixels_before=np.stack([s["pixels_before"] for s in all_steps]),
        actions=np.stack([s["action"] for s in all_steps]),
        pixels_after=np.stack([s["pixels_after"] for s in all_steps]),
        episode_ids=np.array([s["episode_id"] for s in all_steps]),
        tasks=np.array([s["task"] for s in all_steps], dtype=object),
    )
    
    print(f"\nSaved {len(all_steps)} steps to {npz_path}")
    
    # Also save metadata
    meta = {
        "num_episodes": len(trajectories),
        "total_steps": len(all_steps),
        "action_dim": ACTION_DIM,
        "img_size": 224,
        "tasks": list(set(t.task for t in trajectories)),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect GUI trajectories for LeWM")
    parser.add_argument("--model", default="dummy", help="VLM model: dummy, Qwen/Qwen2.5-VL-3B-Instruct, ollama/llava")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--task", default="navigate the desktop and open applications")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Create agent
    if args.model == "dummy":
        agent = DummyAgent()
        print("Using DUMMY agent (random actions) — for pipeline testing only")
    elif args.model.startswith("ollama/"):
        # TODO: implement OllamaAgent
        print("Ollama agent not yet implemented, falling back to dummy")
        agent = DummyAgent()
    else:
        agent = HFApiAgent(args.model)
        print(f"Using HF API agent: {args.model}")
    
    # Create capture
    capture = ScreenshotCapture(resize=(args.img_size, args.img_size))
    
    # Collect episodes
    trajectories = []
    for ep in range(args.num_episodes):
        task = args.task if args.num_episodes == 1 else f"{args.task} (variation {ep})"
        traj = collect_episode(agent, capture, ep, task, args.max_steps)
        trajectories.append(traj)
    
    # Save
    save_trajectories(trajectories, output_dir)
    
    print(f"\nDone! Collected {len(trajectories)} episodes, "
          f"{sum(len(t.steps) for t in trajectories)} total steps.")
    print(f"Next: python scripts/build_hdf5.py --input-dir {output_dir}")


if __name__ == "__main__":
    main()
