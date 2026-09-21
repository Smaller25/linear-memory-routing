"""In-training speed monitor: kill training if iter time > threshold for N consecutive iters.

Watches the pretrain log for "^iter" lines, extracts iter time, tracks consecutive
slow-iters. If threshold exceeded for N consecutive logged iters (after warmup),
kills the training process.

Usage (called from pretrain_mc_370m_30bt_v3.sh):
    python speed_watchdog.py \\
        --log-file path/to/training.log \\
        --train-pid 12345 \\
        --iter-threshold-ms 550 \\
        --warmup-iters 100 \\
        --consecutive-slow-limit 10 \\
        --check-interval-s 30

Default thresholds (vanilla 30B baseline = 393ms):
  iter_threshold_ms = 550  (1.4× vanilla, target was 1.3× = 510ms but allow 10% slack)
  warmup_iters = 100       (skip first 100 logged iters; transient due to CUDA caches)
  consecutive_slow_limit = 10  (10 consecutive slow iters = 100 iters at log_step_interval=10)
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time


# iter 10 step 5: loss 6.42, iter time: 1295.24ms (optimizer.step) remaining time: ...
_ITER_RE = re.compile(r"^iter\s+(\d+)\s+step\s+\d+:\s+loss\s+[\d.]+,\s+iter time:\s+([\d.]+)ms")


def parse_iter_line(line: str):
    """Return (iter_num, iter_ms) or None."""
    m = _ITER_RE.match(line.strip())
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def tail_follow(path: str, train_pid: int, stop_event):
    """Yield new lines from log file as they appear. Stops when train_pid dies or stop_event set."""
    while not os.path.exists(path):
        if stop_event.is_set():
            return
        try:
            os.kill(train_pid, 0)
        except OSError:
            return  # train died
        time.sleep(2)
    with open(path, "r") as f:
        f.seek(0, 2)  # end of file
        while not stop_event.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.5)
                # Check if training process still alive
                try:
                    os.kill(train_pid, 0)
                except OSError:
                    return  # train died
                continue
            yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--iter-threshold-ms", type=float, default=550.0)
    ap.add_argument("--warmup-iters", type=int, default=100, help="skip first N logged iters")
    ap.add_argument("--consecutive-slow-limit", type=int, default=10)
    ap.add_argument("--check-interval-s", type=float, default=30.0, help="legacy; tail_follow is event-driven")
    ap.add_argument("--report-file", default=None, help="write slow-iter report here")
    args = ap.parse_args()

    print(f"[watchdog] start: pid={os.getpid()} train_pid={args.train_pid}", flush=True)
    print(f"[watchdog] threshold: iter > {args.iter_threshold_ms:.0f}ms", flush=True)
    print(f"[watchdog] warmup: {args.warmup_iters} iters | limit: {args.consecutive_slow_limit} consecutive slow", flush=True)
    print(f"[watchdog] log: {args.log_file}", flush=True)

    seen_iters = 0
    consecutive_slow = 0
    slow_log = []
    killed = False

    import threading
    stop_event = threading.Event()

    try:
        for line in tail_follow(args.log_file, args.train_pid, stop_event):
            parsed = parse_iter_line(line)
            if parsed is None:
                continue
            iter_num, iter_ms = parsed
            seen_iters += 1
            if seen_iters <= args.warmup_iters:
                continue  # skip warmup
            if iter_ms > args.iter_threshold_ms:
                consecutive_slow += 1
                slow_log.append((iter_num, iter_ms))
                print(f"[watchdog] SLOW_ITER iter={iter_num} time={iter_ms:.0f}ms ({consecutive_slow}/{args.consecutive_slow_limit})", flush=True)
                if consecutive_slow >= args.consecutive_slow_limit:
                    print(f"[watchdog] KILL: {consecutive_slow} consecutive slow iters >= limit {args.consecutive_slow_limit}", flush=True)
                    print(f"[watchdog] last 5 slow: {slow_log[-5:]}", flush=True)
                    try:
                        os.kill(args.train_pid, signal.SIGTERM)
                        print(f"[watchdog] SIGTERM sent to {args.train_pid}", flush=True)
                    except OSError as e:
                        print(f"[watchdog] kill failed: {e}", flush=True)
                    killed = True
                    # Give training 30s to flush, then SIGKILL if still alive
                    time.sleep(30)
                    try:
                        os.kill(args.train_pid, signal.SIGKILL)
                        print(f"[watchdog] SIGKILL sent to {args.train_pid}", flush=True)
                    except OSError:
                        pass
                    break
            else:
                if consecutive_slow > 0:
                    print(f"[watchdog] recovered after {consecutive_slow} slow iters (iter={iter_num}, time={iter_ms:.0f}ms)", flush=True)
                consecutive_slow = 0
    except KeyboardInterrupt:
        print("[watchdog] interrupted", flush=True)

    if args.report_file:
        with open(args.report_file, "w") as f:
            f.write(f"# speed_watchdog report\n")
            f.write(f"killed: {killed}\n")
            f.write(f"threshold_ms: {args.iter_threshold_ms}\n")
            f.write(f"warmup_iters: {args.warmup_iters}\n")
            f.write(f"consecutive_slow_limit: {args.consecutive_slow_limit}\n")
            f.write(f"seen_iters: {seen_iters}\n")
            f.write(f"slow_iter_count: {len(slow_log)}\n")
            f.write(f"\nslow_iters (iter, ms):\n")
            for it, ms in slow_log:
                f.write(f"  {it}: {ms:.0f}\n")
        print(f"[watchdog] report: {args.report_file}", flush=True)

    sys.exit(1 if killed else 0)


if __name__ == "__main__":
    main()
