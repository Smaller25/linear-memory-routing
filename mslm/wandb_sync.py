"""Upload a run's train.jsonl to Weights & Biases (one wandb run per run folder, id = run name).

Reads only train.jsonl and config.env, so it needs no change to the training loop and also works
for runs that are already in progress or finished. Lines wandb already has (step below the resumed
run's step) are skipped, so running it again or from two places does not duplicate points.

    python -m mslm.wandb_sync RUN_DIR [RUN_DIR ...]   # upload what is there now, then exit
    python -m mslm.wandb_sync --follow RUN_DIR        # keep uploading until the "end" event or SIGTERM

Project/entity come from WANDB_PROJECT / WANDB_ENTITY. The API key comes from `wandb login`
(~/.netrc) or WANDB_API_KEY in the environment; it is never read from or written to a config file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

TRAIN_KEYS = ("loss", "aux_loss", "lr", "grad_norm", "tok_per_s", "mem_gb")


def read_config(run_dir: str) -> dict:
    cfg = {}
    try:
        with open(os.path.join(run_dir, "config.env")) as f:
            for line in f:
                k, _, v = line.rstrip("\n").partition("=")
                try:
                    cfg[k] = json.loads(v)
                except ValueError:
                    cfg[k] = v
    except FileNotFoundError:
        pass
    return cfg


def record_to_metrics(rec: dict) -> dict:
    """train.jsonl line -> wandb metrics (empty for events)."""
    if "loss" in rec:
        out = {f"train/{k}": rec[k] for k in TRAIN_KEYS if k in rec}
        out["tokens"] = rec.get("tokens")
        return out
    if "val_loss" in rec:
        out = {"val/loss": rec["val_loss"], "tokens": rec.get("tokens")}
        for k, v in rec.items():                     # routing hit rates (mslm/routing_monitor.py)
            if k in ("step", "tokens", "val_loss"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"val/{k}"] = v
            elif isinstance(v, list) and k.endswith("_by_layer"):
                out.update({f"val/{k[:-len('_by_layer')]}/L{i:02d}": x for i, x in enumerate(v)})
        return out
    return {}


class JsonlTail:
    """Yields complete new lines of a growing file (a half-written last line waits for the next read)."""

    def __init__(self, path: str):
        self.path, self.pos, self.buf = path, 0, ""

    def read(self) -> list[dict]:
        try:
            with open(self.path) as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except FileNotFoundError:
            return []
        self.buf += chunk
        *lines, self.buf = self.buf.split("\n")
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def sync(run_dir: str, follow: bool = False, poll: float = 30.0) -> None:
    import wandb

    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)
    stop = {"flag": False}
    if follow:
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    tail = JsonlTail(os.path.join(run_dir, "train.jsonl"))
    while follow and not os.path.exists(tail.path) and not stop["flag"]:
        time.sleep(poll)

    for k in ("WANDB_PROJECT", "WANDB_ENTITY"):  # config 에서 비운 값이 wandb 기본값을 가리지 않게
        if not os.environ.get(k):
            os.environ.pop(k, None)
    run = wandb.init(project=os.environ.get("WANDB_PROJECT") or "linear-memory-routing",
                     entity=os.environ.get("WANDB_ENTITY") or None, id=name, name=name, resume="allow",
                     config=read_config(run_dir), dir=run_dir)
    first = run.step  # resumed run: wandb already holds every step below this
    ended, n = False, 0
    try:
        while True:
            for rec in tail.read():
                if rec.get("event") == "start":
                    run.config.update({k: rec[k] for k in ("total_steps", "warmup", "tokens_per_step", "world",
                                                           "params", "params_no_embed") if k in rec},
                                      allow_val_change=True)
                if rec.get("event") == "end":
                    ended = True
                m = record_to_metrics(rec)
                if m and rec.get("step", -1) >= first:
                    run.log(m, step=rec["step"])
                    n += 1
            if not follow or ended or stop["flag"]:
                break
            time.sleep(poll)
    finally:
        print(f"[wandb_sync] {name}: {n} records (from step {first}){' end' if ended else ''}", flush=True)
        run.finish()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--follow", action="store_true", help="keep uploading until the run ends (one run only)")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between reads in --follow")
    a = ap.parse_args(argv)
    if a.follow and len(a.run_dirs) != 1:
        ap.error("--follow takes exactly one RUN_DIR (start one process per run)")
    for d in a.run_dirs:
        sync(d, follow=a.follow, poll=a.poll)


if __name__ == "__main__":
    sys.exit(main())
