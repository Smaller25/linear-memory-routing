# -*- coding: utf-8 -*-
"""Gate G1 판정 — 46M random vs fixed, held-out ppl 2×2 matrix.

Held-out = uniform_plan(600M)에 안 뽑힌 문서들 (plan은 seed 고정이라 재현됨).
각 checkpoint를 random / fixed-256 segmentation 양쪽에서 평가:
  - 대각(native mode) 비교가 G1 기준: random 모델 열화 ≤ 3% (vs fixed 모델)
  - 비대각은 policy-swap zero-shot의 예비 신호

usage: PYTHONPATH=. python lmr/dynmc/g1_eval.py \
    --ckpt-random /data2/.../0024pre_random/ckpt_500M \
    --ckpt-fixed  /data2/.../0024pre_fixed/ckpt_500M \
    --data /data2/sohyung/dynmc/tokenized --n-rows 64
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from lmr.dynmc.data import TokenizedCorpus, PackedDocIterator, make_batch, uniform_plan
from lmr.dynmc.model import build_model
from lmr.dynmc.segmenting import build_batch_segments


def load_ckpt(path, device):
    ck = torch.load(f"{path}/state_rank0.pt", map_location="cpu")
    cfg = ck["cfg"]
    model = build_model(cfg["size"], dynmc=cfg["dynmc"], cache_budget=cfg["cache_budget"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def eval_ppl(model, corpus, holdout_plan, ctx, n_rows, seg_mode, device, seed=7):
    it = PackedDocIterator(corpus, holdout_plan, ctx)
    rng = np.random.default_rng(seed)
    ce_sum, n_tok = 0.0, 0
    rows_left = n_rows
    while rows_left > 0:
        take = min(8, rows_left)
        batch = make_batch(it, take)
        if batch is None:
            break
        rows_left -= take
        ids = batch["input_ids"].to(device).reshape(1, -1)
        labels = batch["labels"].to(device).reshape(1, -1)
        segs = build_batch_segments(batch["doc_lens_per_row"], rng, mode=seg_mode, fixed_len=256)
        doc_ids = segs["seg_doc_ids"]
        change = torch.ones(len(doc_ids), dtype=torch.bool)
        change[1:] = doc_ids[1:] != doc_ids[:-1]
        first_idx = torch.nonzero(change).flatten()
        first = first_idx[torch.cumsum(change.long(), 0) - 1]
        dynmc_segs = {"cu_seqlens": segs["cu_seqlens"].to(device),
                      "seg_doc_start": first.to(device)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, dynmc_segs=dynmc_segs).logits
        mask = labels.view(-1) != -100
        ce = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1])[mask], labels.view(-1)[mask],
            reduction="sum")
        ce_sum += float(ce)
        n_tok += int(mask.sum())
    return math.exp(ce_sum / max(n_tok, 1)), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-random", required=True)
    ap.add_argument("--ckpt-fixed", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prefix", default="fineweb-")
    ap.add_argument("--n-rows", type=int, default=64)
    a = ap.parse_args()
    device = "cuda"

    corpus = TokenizedCorpus(a.data, a.prefix)
    m_rand, cfg = load_ckpt(a.ckpt_random, device)
    m_fix, _ = load_ckpt(a.ckpt_fixed, device)
    ctx = cfg["ctx"]

    # holdout = 학습 plan(seed 고정)에 없는 문서
    train_plan = set(uniform_plan(corpus, cfg["uniform_plan_tokens"], seed=cfg["seed"]).tolist())
    holdout = np.array([d for d in range(corpus.num_docs) if d not in train_plan],
                       dtype=np.uint64)
    np.random.default_rng(7).shuffle(holdout)
    print(f"holdout docs: {len(holdout)} ({int(corpus.doc_len[holdout.astype(np.int64)].sum())/1e6:.0f}M tokens)")

    results = {}
    for mname, model in [("random", m_rand), ("fixed", m_fix)]:
        for emode in ["random", "fixed"]:
            ppl, n = eval_ppl(model, corpus, holdout, ctx, a.n_rows, emode, device)
            results[f"{mname}_model/{emode}_eval"] = ppl
            print(f"model={mname:6s} eval-seg={emode:6s} ppl={ppl:.3f} ({n/1e6:.1f}M tokens)")

    native_r = results["random_model/random_eval"]
    native_f = results["fixed_model/fixed_eval"]
    delta = (native_r - native_f) / native_f
    verdict = "PASS" if delta <= 0.03 else "FAIL_R3_CURRICULUM"
    print(f"\nG1: random {native_r:.3f} vs fixed {native_f:.3f} → delta {delta:+.2%} → {verdict}")
    print(json.dumps({"results": results, "delta": delta, "verdict": verdict}))


if __name__ == "__main__":
    main()
