"""From-scratch pretraining loop (single GPU, multi-GPU DDP via torchrun, or CPU).

All settings come from the resolved config exported by scripts/train.sh.
Writes into RUNS_ROOT/<RUN_NAME>/:
    config.env          resolved variables of this run (exact reproduction)
    model_config.json   architecture
    train.jsonl         one json line per LOG_EVERY steps (+ val lines)
    ckpt/step_XXXXXXX.pt, ckpt/latest   (atomic writes, keeps KEEP_CKPTS)
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mslm.data.loader import TokenWindows, sample_ids
from mslm.models import LM, ModelConfig
from mslm.settings import Settings


def lr_at(step: int, total: int, warmup: int, lr: float, min_ratio: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def param_groups(model, wd: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def ckpt_dir(run_dir):
    return os.path.join(run_dir, "ckpt")


def find_resume(s: Settings, run_dir: str) -> str | None:
    mode = s.str("RESUME")
    if mode == "none":
        return None
    if mode != "auto":
        return mode
    latest = os.path.join(ckpt_dir(run_dir), "latest")
    if os.path.exists(latest):
        with open(latest) as f:
            return os.path.join(ckpt_dir(run_dir), f.read().strip())
    return None


def save_ckpt(path_dir, step, model, opt, state, keep, backup_root=None, run_name=""):
    os.makedirs(path_dir, exist_ok=True)
    name = f"step_{step:07d}.pt"
    tmp = os.path.join(path_dir, name + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                "model_config": json.loads(model.cfg.to_json())}, tmp)
    os.replace(tmp, os.path.join(path_dir, name))
    with open(os.path.join(path_dir, "latest.tmp"), "w") as f:
        f.write(name)
    os.replace(os.path.join(path_dir, "latest.tmp"), os.path.join(path_dir, "latest"))
    for old in sorted(glob.glob(os.path.join(path_dir, "step_*.pt")))[:-keep]:
        os.remove(old)
    if backup_root:
        dst = os.path.join(backup_root, run_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(path_dir, name), os.path.join(dst, name))
        for old in sorted(glob.glob(os.path.join(dst, "step_*.pt")))[:-1]:
            os.remove(old)
    return name


def main():
    s = Settings.from_env()
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0

    seq, mb = s.int("SEQ_LEN"), s.int("MICRO_BATCH")
    per_micro = seq * mb * world
    accum = max(1, round(s.int("GLOBAL_BATCH_TOKENS") / per_micro))
    tokens_per_step = per_micro * accum
    total_steps = math.ceil(s.float("MAX_TOKENS") / tokens_per_step)
    warmup = max(1, round(s.float("WARMUP_FRAC") * total_steps))
    stop_step = min(total_steps, s.int("MAX_STEPS")) if s.int("MAX_STEPS") > 0 else total_steps

    run_dir = s.run_dir()
    cfg = ModelConfig.from_settings(s)
    if master:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.env"), "w") as f:
            f.writelines(f"{k}={json.dumps(v)}\n" for k, v in s.values.items())
        with open(os.path.join(run_dir, "model_config.json"), "w") as f:
            f.write(cfg.to_json())
        if tokens_per_step != s.int("GLOBAL_BATCH_TOKENS"):
            print(f"[train] WARNING GLOBAL_BATCH_TOKENS rounded to {tokens_per_step}", flush=True)

    torch.manual_seed(s.int("SEED"))
    model = LM(cfg).to(device)
    train_data = TokenWindows(s.str("TRAIN_DATA_DIR"), "train", seq, s.int("SEED"))
    val_data = TokenWindows(s.str("TRAIN_DATA_DIR"), "val", seq, 0)
    if train_data.meta["tokenizer"] != s.str("TOKENIZER"):
        raise ValueError(f"data was tokenised with {train_data.meta['tokenizer']}, config says {s.str('TOKENIZER')}")
    opt = torch.optim.AdamW(param_groups(model, s.float("WEIGHT_DECAY")), lr=s.float("LR"),
                            betas=(s.float("BETA1"), s.float("BETA2")), fused=device.type == "cuda")

    state = {"step": 0, "tokens": 0, "tokenizer": s.str("TOKENIZER"), "seed": s.int("SEED")}
    resume = find_resume(s, run_dir)
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        if master:
            print(f"[train] resumed from {resume} at step {state['step']}", flush=True)
    dmodel = DDP(model, device_ids=[device.index]) if ddp else model

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (device.type == "cuda" and s.str("PRECISION") == "bf16") \
        else torch.autocast("cpu", enabled=False)
    log_path = os.path.join(run_dir, "train.jsonl")

    def log(rec):
        if master:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    if master:
        log({"event": "start", "step": state["step"], "total_steps": total_steps, "stop_step": stop_step,
             "warmup": warmup, "accum": accum, "world": world, "tokens_per_step": tokens_per_step,
             "params": model.num_params(), "params_no_embed": model.num_params(True),
             "resumed_from": resume, "device": str(device)})

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for i in range(s.int("VAL_BATCHES")):
            x, y = val_data.batch(range((i * world + rank) * mb, (i * world + rank + 1) * mb))
            with amp:
                losses.append(model(x.to(device), y.to(device))["lm_loss"].float())
        model.train()
        v = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
        return v.item()

    model.train()
    t_last, tok_last = time.time(), state["tokens"]
    while state["step"] < stop_step:
        step = state["step"]
        lr = lr_at(step, total_steps, warmup, s.float("LR"), s.float("MIN_LR_RATIO"))
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        aux_sum = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = train_data.batch(sample_ids(step, micro, accum, mb, rank, world))
            sync = (not ddp) or micro == accum - 1
            ctx = dmodel.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx, amp:
                out = dmodel(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                (out["loss"] / accum).backward()
            loss_sum += out["lm_loss"].detach() / accum
            aux_sum += out["aux_loss"].detach() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), s.float("GRAD_CLIP"))
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["tokens"] += tokens_per_step
        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"non-finite loss at step {state['step']}")

        if state["step"] % s.int("LOG_EVERY") == 0 or state["step"] == stop_step:
            now = time.time()
            rec = {"step": state["step"], "tokens": state["tokens"], "loss": round(loss_sum.item(), 5),
                   "aux_loss": round(aux_sum.item(), 5), "lr": lr, "grad_norm": round(float(gnorm), 4),
                   "tok_per_s": round((state["tokens"] - tok_last) / max(now - t_last, 1e-9)),
                   "mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else 0}
            log(rec)
            t_last, tok_last = now, state["tokens"]
        if s.int("VAL_EVERY") > 0 and (state["step"] % s.int("VAL_EVERY") == 0 or state["step"] == stop_step):
            log({"step": state["step"], "tokens": state["tokens"], "val_loss": round(val_loss(), 5)})
        if master and (state["step"] % s.int("SAVE_EVERY") == 0 or state["step"] == stop_step):
            name = save_ckpt(ckpt_dir(run_dir), state["step"], model, opt, state, s.int("KEEP_CKPTS"),
                             s.str("BACKUP_ROOT") or None, s.run_name())
            log({"event": "save", "step": state["step"], "ckpt": name})
        if ddp:
            dist.barrier()

    log({"event": "end", "step": state["step"], "tokens": state["tokens"]})
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
