#!/usr/bin/env python3
"""
EgRoll Training Health Monitor

Runs as a cron job every 10 minutes to check:
1. Process is alive
2. Loss is not NaN/Inf
3. Loss is decreasing over time (no divergence)
4. GPU memory is stable
5. Throughput hasn't dropped significantly
6. Log file is growing (not stuck)

Outputs health report to stdout (captured by cron) and appends to health log.
"""

import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

WORK_DIR = os.environ.get("ML_INTERN_HYBRID_SLM_DIR", os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = os.path.join(WORK_DIR, "outputs/egroll-exclusive/train.log")
HEALTH_LOG = os.path.join(WORK_DIR, "outputs/egroll-exclusive/health.log")
TRAINING_LOG_TXT = os.path.join(WORK_DIR, "outputs/egroll-exclusive/training_log.txt")
PID_FILE = os.path.join(WORK_DIR, "outputs/egroll-exclusive/train.pid")

# Anomaly thresholds
MAX_LOSS_SPIKE = 5.0          # Loss increase > 5.0 from recent average = anomaly
MAX_STALE_SECONDS = 600       # No new log line for 10 min = stale
MIN_THROUGHPUT_RATIO = 0.3    # Throughput < 30% of peak = anomaly
MAX_GPU_MEMORY_MIB = 6100     # Near OOM on 6144 MiB GPU
MIN_REWARD_IMPROVEMENT_WINDOW = 200  # Steps to look back for reward improvement


def parse_log_lines(log_path: str) -> List[dict]:
    """Parse training log lines into structured data"""
    entries = []
    pattern = re.compile(
        r"step=(\d[\d,]*)\s*\|\s*loss=([-\d.]+)\s*\|\s*"
        r"reward=([-\d.]+)\s*\(±([-\d.]+)\)\s*\|\s*"
        r"best_reward=([-\d.]+)\s*\|\s*"
        r"tokens/s=([\d,]+)\s*\|\s*epoch=(\d+)\s*\|\s*"
        r"mem=([\d.]+)GB"
        r"(?:\s*\|\s*⚠️ ANOMALY)?"
    )
    
    paths_to_check = [log_path, TRAINING_LOG_TXT]
    
    for path in paths_to_check:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r') as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        entries.append({
                            'step': int(m.group(1).replace(',', '')),
                            'loss': float(m.group(2)),
                            'reward_mean': float(m.group(3)),
                            'reward_std': float(m.group(4)),
                            'best_reward': float(m.group(5)),
                            'tokens_per_sec': float(m.group(6).replace(',', '')),
                            'epoch': int(m.group(7)),
                            'mem_gb': float(m.group(8)),
                            'anomaly': '⚠️ ANOMALY' in line,
                            'source': path,
                        })
        except Exception as e:
            pass
    
    # Deduplicate by step (prefer train.log entries)
    seen = {}
    for e in entries:
        s = e['step']
        if s not in seen or e['source'] == log_path:
            seen[s] = e
    
    return sorted(seen.values(), key=lambda x: x['step'])


def get_gpu_stats() -> dict:
    """Get GPU utilization and memory"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(parts[0]),
                'mem_used_mib': float(parts[1]),
                'mem_total_mib': float(parts[2]),
                'temp_c': float(parts[3]),
                'power_w': float(parts[4]),
            }
    except Exception:
        pass
    return {}


def is_process_alive() -> bool:
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'train_egroll.py'],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_log_mtime(path: str) -> float:
    """Get last modification time of log file"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def check_health() -> dict:
    """Run all health checks and return report"""
    now = datetime.now()
    report = {
        'timestamp': now.isoformat(),
        'status': 'HEALTHY',
        'checks': {},
        'alerts': [],
        'metrics': {},
    }

    # ── Check 1: Process alive ───────────────────────────
    alive = is_process_alive()
    report['checks']['process_alive'] = alive
    if not alive:
        report['status'] = 'CRITICAL'
        report['alerts'].append('TRAINING PROCESS IS NOT RUNNING')
        return report

    # ── Check 2: GPU status ──────────────────────────────
    gpu = get_gpu_stats()
    report['metrics']['gpu'] = gpu
    
    if gpu:
        gpu_ok = gpu.get('gpu_util', 0) > 10
        report['checks']['gpu_active'] = gpu_ok
        if not gpu_ok:
            report['alerts'].append(f"GPU utilization low: {gpu.get('gpu_util', 0)}%")
            report['status'] = 'WARNING'
        
        if gpu.get('mem_used_mib', 0) > MAX_GPU_MEMORY_MIB:
            report['alerts'].append(f"GPU memory near OOM: {gpu['mem_used_mib']:.0f}/{gpu['mem_total_mib']:.0f} MiB")
            report['status'] = 'WARNING'
        
        if gpu.get('temp_c', 0) > 85:
            report['alerts'].append(f"GPU temperature high: {gpu['temp_c']}°C")
            report['status'] = 'WARNING'

    # ── Check 3: Log freshness ───────────────────────────
    log_mtime = max(get_log_mtime(LOG_FILE), get_log_mtime(TRAINING_LOG_TXT))
    if log_mtime > 0:
        stale_seconds = time.time() - log_mtime
        report['checks']['log_fresh'] = stale_seconds < MAX_STALE_SECONDS
        report['metrics']['log_stale_seconds'] = stale_seconds
        
        if stale_seconds > MAX_STALE_SECONDS:
            report['alerts'].append(f"Log stale for {stale_seconds:.0f}s (>{MAX_STALE_SECONDS}s)")
            report['status'] = 'WARNING'
    else:
        report['checks']['log_fresh'] = False
        report['alerts'].append('No log file found')
        report['status'] = 'UNKNOWN'

    # ── Check 4: Parse training metrics ──────────────────
    entries = parse_log_lines(LOG_FILE)
    
    if len(entries) == 0:
        report['alerts'].append('No training entries parsed from log')
        return report

    latest = entries[-1]
    report['metrics']['latest_step'] = latest['step']
    report['metrics']['latest_loss'] = latest['loss']
    report['metrics']['latest_reward'] = latest['reward_mean']
    report['metrics']['best_reward'] = latest['best_reward']
    report['metrics']['tokens_per_sec'] = latest['tokens_per_sec']

    # ── Check 5: NaN / Inf loss ──────────────────────────
    import math
    has_nan = math.isnan(latest['loss']) or math.isinf(latest['loss'])
    report['checks']['loss_valid'] = not has_nan
    if has_nan:
        report['status'] = 'CRITICAL'
        report['alerts'].append(f"Loss is NaN/Inf: {latest['loss']}")

    # ── Check 6: Loss spike detection ────────────────────
    if len(entries) >= 20:
        recent = [e['loss'] for e in entries[-5:]]
        older = [e['loss'] for e in entries[-20:-5]]
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        
        if older_mean > 0 and (recent_mean - older_mean) > MAX_LOSS_SPIKE:
            report['checks']['no_spike'] = False
            report['alerts'].append(
                f"Loss spike detected: recent_avg={recent_mean:.4f} vs older_avg={older_mean:.4f} "
                f"(delta={recent_mean - older_mean:.4f})"
            )
            report['status'] = 'WARNING'
        else:
            report['checks']['no_spike'] = True

    # ── Check 7: Loss convergence (is it improving?) ─────
    if len(entries) >= 10:
        first_10_avg = sum(e['loss'] for e in entries[:10]) / 10
        last_10_avg = sum(e['loss'] for e in entries[-10:]) / 10
        report['metrics']['loss_improvement'] = first_10_avg - last_10_avg
        
        if last_10_avg >= first_10_avg and latest['step'] > 100:
            report['alerts'].append(
                f"Loss not improving: start_avg={first_10_avg:.4f}, current_avg={last_10_avg:.4f}"
            )

    # ── Check 8: Throughput stability ────────────────────
    if len(entries) >= 10:
        throughputs = [e['tokens_per_sec'] for e in entries[-50:]]
        peak_tps = max(throughputs)
        current_tps = latest['tokens_per_sec']
        
        if peak_tps > 0 and current_tps < peak_tps * MIN_THROUGHPUT_RATIO:
            report['checks']['throughput_ok'] = False
            report['alerts'].append(
                f"Throughput dropped: {current_tps:.0f} vs peak {peak_tps:.0f} tokens/s "
                f"({current_tps/peak_tps*100:.0f}%)"
            )
        else:
            report['checks']['throughput_ok'] = True

    # ── Check 9: Reward improvement over window ──────────
    if len(entries) >= 2:
        # Look for improvement in best_reward over the last N entries
        window = [e for e in entries if e['step'] >= latest['step'] - MIN_REWARD_IMPROVEMENT_WINDOW]
        if len(window) >= 2:
            earliest_in_window = window[0]['best_reward']
            latest_best = latest['best_reward']
            reward_improved = latest_best > earliest_in_window
            report['checks']['reward_improving'] = reward_improved
            if not reward_improved and latest['step'] > 500:
                report['alerts'].append(
                    f"Reward not improving in last {MIN_REWARD_IMPROVEMENT_WINDOW} steps"
                )

    # ── Summary ──────────────────────────────────────────
    report['total_entries_parsed'] = len(entries)
    
    return report


def main():
    report = check_health()
    
    # Format output
    status = report['status']
    ts = report['timestamp']
    step = report['metrics'].get('latest_step', '?')
    loss = report['metrics'].get('latest_loss', '?')
    reward = report['metrics'].get('latest_reward', '?')
    tps = report['metrics'].get('tokens_per_sec', '?')
    
    status_emoji = {
        'HEALTHY': '✅',
        'WARNING': '⚠️',
        'CRITICAL': '🔴',
        'UNKNOWN': '❓',
    }.get(status, '❓')
    
    line = (f"[{ts}] {status_emoji} {status} | step={step} | loss={loss} | "
            f"reward={reward} | tokens/s={tps}")
    
    if report['alerts']:
        line += f" | alerts={len(report['alerts'])}"
    
    print(line)
    
    for alert in report['alerts']:
        print(f"  → {alert}")
    
    # Append to health log
    os.makedirs(os.path.dirname(HEALTH_LOG), exist_ok=True)
    with open(HEALTH_LOG, 'a') as f:
        f.write(json.dumps(report) + "\n")
    
    sys.exit(0 if status != 'CRITICAL' else 1)


if __name__ == "__main__":
    main()
