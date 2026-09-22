"""One place to decide whether a run reports to wandb, and what it reports.

These experiments are probes and evaluations rather than training runs, so the
useful unit is a *cell* (seed x length x needle count) and an *arm*, not a
step. A routing number that cannot be traced back to the arm, the seeds and
the protocol it came from is the thing that cost this project a month: the
same untouched baseline was recorded as 4.0, 2.3 and 3.0 under three different
seed sets, and none of the three could be compared with the others.

So `start()` refuses to open a run without the protocol fields, and they land
in the run config rather than in a log line, where a later sweep can group by
them.

Offline by default. A missing key degrades to `WANDB_MODE=disabled` and the
experiment still runs; it never takes down a job that has been on the GPU for
hours. The key is read from a file so it never reaches a process listing, a
shell history or a log.
"""
from __future__ import annotations

import os

_RUN = None

REQUIRED = ("arm", "seeds", "lengths", "needles", "max_samples", "topk")


def _key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    path = os.environ.get("WANDB_KEY_FILE", "/root/.wandb_key")
    try:
        with open(path) as fh:
            k = fh.read().strip()
        return k or None
    except OSError:
        return None


def start(name: str, config: dict, project: str | None = None,
          group: str | None = None, tags: list[str] | None = None):
    """Open a run, or return None when wandb is unavailable or keyless.

    `config` must carry the protocol: which arm, which seeds, which cells, how
    many samples, which k. Anything missing raises here rather than producing
    a run whose numbers cannot be placed later.
    """
    global _RUN
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise ValueError(
            f"refusing to open a run without {missing} in config — a routing "
            "number that does not record its protocol cannot be compared "
            "with any other run")
    k = _key()
    os.environ.setdefault("WANDB_MODE", "online" if k else "disabled")
    if k:
        os.environ.setdefault("WANDB_API_KEY", k)
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; continuing without logging", flush=True)
        return None
    try:
        _RUN = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "mc-routing"),
            name=name, group=group, tags=tags or [], config=config,
            dir=os.environ.get("WANDB_DIR", "/root/cache/wandb"),
            reinit=True)
    except Exception as e:                      # noqa: BLE001
        print(f"[wandb] init failed ({type(e).__name__}); continuing without "
              "logging", flush=True)
        return None
    print(f"[wandb] {os.environ['WANDB_MODE']} run {name} "
          f"({_RUN.url if os.environ['WANDB_MODE'] == 'online' else 'local'})",
          flush=True)
    return _RUN


def log(metrics: dict, step: int | None = None) -> None:
    if _RUN is None:
        return
    try:
        _RUN.log(metrics, step=step)
    except Exception:                           # noqa: BLE001
        pass


def log_cells(rows: list[dict], key: str) -> None:
    """Log one scalar per cell plus the pooled value, under stable names.

    `rows` are dicts with at least `cell` and `key`. Cell names carry the
    seed and the needle count, so a sweep can group by them without parsing
    a run name.
    """
    if _RUN is None or not rows:
        return
    flat = {f"{key}/{r['cell']}": r[key] for r in rows if key in r}
    vals = [r[key] for r in rows if key in r]
    if vals:
        flat[f"{key}/pooled"] = sum(vals) / len(vals)
    log(flat)


def summary(**kv) -> None:
    if _RUN is None:
        return
    try:
        for k, v in kv.items():
            _RUN.summary[k] = v
    except Exception:                           # noqa: BLE001
        pass


def finish() -> None:
    global _RUN
    if _RUN is None:
        return
    try:
        _RUN.finish()
    except Exception:                           # noqa: BLE001
        pass
    _RUN = None
