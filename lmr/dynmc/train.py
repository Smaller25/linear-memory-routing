# -*- coding: utf-8 -*-
"""DynMC from-scratch pretraining loop (plan §4).

torchrun --nproc_per_node=N -m lmr.dynmc.train --config lmr/dynmc/configs/0024.json

Single-file DDP loop (no flame/torchtitan dependency): bf16 autocast, fused
AdamW, cosine schedule with linear warmup, grad clip 1.0, checkpoint every
`ckpt_interval_tokens`, jsonl logging (loss/lr/tok/s/MFU), resumable
(model+optim+sched+data position). Run from repo root with PYTHONPATH=. .
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from lmr.dynmc.data import TokenizedCorpus, PackedDocIterator, make_batch, uniform_plan
from lmr.dynmc.model import build_model, param_count
from lmr.dynmc.segmenting import build_batch_segments

DEFAULTS = dict(
    size="340m", dynmc=True, seg_mode="random", fixed_seg_len=256, cache_budget=32,
    ctx=16384, rows_per_micro=2, grad_accum=4,
    lr=4e-4, min_lr_ratio=0.0, weight_decay=0.01, betas=(0.9, 0.95),
    warmup_steps=1000, grad_clip=1.0,
    total_tokens=15_000_000_000, ckpt_interval_tokens=1_000_000_000,
    data_dir="", data_prefix="", plan_path="", uniform_plan_tokens=0,
    merge_docs_per_row=False,  # G1b: row 전체를 한 문서로 (budget 포화 regime)
    out_dir="", run_name="dynmc", seed=1234,
    log_interval=20, gradient_checkpointing=False,
    peak_flops=312e12,  # A100 bf16 dense
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="", help="checkpoint dir to resume from")
    return p.parse_args()


def lr_at(step: int, total_steps: int, cfg) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    t = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
    return cfg["lr"] * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    args = get_args()
    cfg = dict(DEFAULTS)
    with open(args.config) as f:
        cfg.update(json.load(f))

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(cfg["seed"] + rank)

    out_dir = cfg["out_dir"] or f"_workspace/dynmc/{cfg['run_name']}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"train_rank{rank}.jsonl")

    # --- model ---
    model = build_model(cfg["size"], dynmc=cfg["dynmc"], cache_budget=cfg["cache_budget"])
    model = model.to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    n_total, n_nonemb = param_count(model)
    if rank == 0:
        print(f"[dynmc] params total={n_total/1e6:.1f}M non-emb={n_nonemb/1e6:.1f}M", flush=True)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=bool(cfg["dynmc"]))  # read 미발동 micro-batch 안전장치
    raw_model = model.module if world > 1 else model

    # --- optimizer (no-wd for norms/A_log/dt_bias/1d params) ---
    decay, no_decay = [], []
    for n, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or getattr(p, "_no_weight_decay", False)) else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=tuple(cfg["betas"]), fused=True)

    # --- data ---
    corpus = TokenizedCorpus(cfg["data_dir"], cfg["data_prefix"])
    if cfg["plan_path"]:
        plan = np.load(cfg["plan_path"], mmap_mode="r")
    else:
        assert cfg["uniform_plan_tokens"] > 0
        plan = uniform_plan(corpus, cfg["uniform_plan_tokens"], seed=cfg["seed"])
    plan_rank = np.asarray(plan[rank::world])
    rng = np.random.default_rng(cfg["seed"] * 1000 + rank)

    micro_tokens = cfg["rows_per_micro"] * cfg["ctx"]
    step_tokens = micro_tokens * cfg["grad_accum"] * world
    total_steps = cfg["total_tokens"] // step_tokens
    if rank == 0:
        print(f"[dynmc] step_tokens={step_tokens} total_steps={total_steps}", flush=True)

    start_step, rows_done = 0, 0
    if args.resume:
        ck = torch.load(os.path.join(args.resume, f"state_rank0.pt"), map_location="cpu")
        raw_model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"]
        rows_done = ck["rows_done"]
        if rank == 0:
            print(f"[dynmc] resumed from {args.resume} @ step {start_step}", flush=True)
    it = PackedDocIterator(corpus, plan_rank, cfg["ctx"], start_row=rows_done)

    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    next_ckpt = ((start_step * step_tokens) // cfg["ckpt_interval_tokens"] + 1) * cfg["ckpt_interval_tokens"]
    t0, tok0 = time.time(), start_step * step_tokens
    model.train()

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, cfg)
        for g in optim.param_groups:
            g["lr"] = lr
        losses = []
        for micro in range(cfg["grad_accum"]):
            batch = make_batch(it, cfg["rows_per_micro"])
            if batch is None:
                if rank == 0:
                    print("[dynmc] plan exhausted — stopping", flush=True)
                batch = None
                break
            rows_done += cfg["rows_per_micro"]
            if cfg["merge_docs_per_row"]:
                batch["doc_lens_per_row"] = [[cfg["ctx"]] for _ in batch["doc_lens_per_row"]]
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            kwargs = {}
            if cfg["dynmc"]:
                segs = build_batch_segments(batch["doc_lens_per_row"], rng,
                                            mode=cfg["seg_mode"], fixed_len=cfg["fixed_seg_len"])
                doc_ids = segs["seg_doc_ids"]
                change = torch.ones(len(doc_ids), dtype=torch.bool)
                change[1:] = doc_ids[1:] != doc_ids[:-1]
                first_idx = torch.nonzero(change).flatten()
                first = first_idx[torch.cumsum(change.long(), 0) - 1]
                kwargs["dynmc_segs"] = {
                    "cu_seqlens": segs["cu_seqlens"].to(device),
                    "seg_doc_start": first.to(device),
                }
                ids = ids.reshape(1, -1)
                labels = labels.reshape(1, -1)
            sync = (micro == cfg["grad_accum"] - 1) or world == 1
            ctxm = torch.enable_grad() if sync else model.no_sync()
            with ctxm:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, **kwargs).logits
                loss = ce(logits.float().view(-1, logits.shape[-1]), labels.view(-1))
                (loss / cfg["grad_accum"]).backward()
            losses.append(loss.detach())
        if batch is None:
            break
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
        optim.zero_grad(set_to_none=True)

        if rank == 0 and (step % cfg["log_interval"] == 0 or step == total_steps - 1):
            tokens_done = (step + 1) * step_tokens
            dt = time.time() - t0
            tps = (tokens_done - tok0) / max(dt, 1e-9)
            mfu = tps / world * 6 * n_nonemb / cfg["peak_flops"]
            rec = dict(step=step, loss=float(torch.stack(losses).mean()), lr=lr,
                       gnorm=float(gnorm), tokens=tokens_done, tps=tps, mfu=mfu,
                       time=time.strftime("%H:%M:%S"))
            print("[dynmc] " + json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t0, tok0 = time.time(), tokens_done

        tokens_done = (step + 1) * step_tokens
        if tokens_done >= next_ckpt or step == total_steps - 1:
            if rank == 0:
                ck_dir = os.path.join(out_dir, f"ckpt_{tokens_done//1_000_000}M")
                os.makedirs(ck_dir, exist_ok=True)
                torch.save({"model": raw_model.state_dict(), "optim": optim.state_dict(),
                            "step": step + 1, "rows_done": rows_done, "cfg": cfg},
                           os.path.join(ck_dir, "state_rank0.pt"))
                print(f"[dynmc] checkpoint -> {ck_dir}", flush=True)
            if world > 1:
                dist.barrier()
            next_ckpt += cfg["ckpt_interval_tokens"]

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
