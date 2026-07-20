# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Descriptor mechanism probe (from-scratch GDN-2 + multi-key MQAR, ground-truth).

Tests the multi-key descriptor hypotheses (notes/descriptor-hypotheses.md) that the frozen-1.3B infra
can't run here. Uses irregular MQAR where we KNOW each value's position and token, so per-key
retrievability is measured exactly. Trains a GDN-2 to recall, then probes its per-position hiddens vs
chunk-pooled descriptors (the SSC-style summary):

  H1 (pooling collapse): value-decode from the value-position hidden vs the pooled chunk descriptor.
  H2 (query-conditioning): localize the QUERIED value among the k values by per-position MaxSim
      (query-conditioned) vs by pooled-chunk dot-product routing.
  E0/E1 pieces: chunk-hit (routing) vs within-chunk position resolution (extraction).

All metrics are retrieval/decoding rates against ground truth; no read-out training needed.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from lmr.mosc.backbone import GDN2LM
from lmr.tasks.mqar import make_mqar_gapped

IGNORE = -100


def train(model, device, kvs, ctx_filler, vocab, steps, batch, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    model.train()
    for step in range(1, steps + 1):
        k = kvs[step % len(kvs)]
        b = make_mqar_gapped(num_examples=batch, vocab_size=vocab, num_kv_pairs=k,
                             ctx_filler=ctx_filler, seed=step)
        ids, labels = b["input_ids"].to(device), b["labels"].to(device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 1:
            print(f"[train] step {step:5d}  loss {loss.item():.4f}", flush=True)
    return loss.item()


@torch.no_grad()
def probe(model, device, k, ctx_filler, vocab, chunk, n=128):
    """Return retrieval/decode rates for the queried value, per-position vs pooled-chunk."""
    model.eval()
    b = make_mqar_gapped(num_examples=n, vocab_size=vocab, num_kv_pairs=k, ctx_filler=ctx_filler,
                         seed=777 + k)
    ids, labels, vpos = b["input_ids"].to(device), b["labels"].to(device), b["value_pos"].to(device)
    B, T = ids.shape
    ctx_len = 2 * k + ctx_filler                      # context region length (values live here)
    h = model(ids, return_hidden=True)                # [B, T, d]
    logits = model.lm_head(h)                          # for value-decode probe
    d = h.shape[-1]

    # pooled chunk descriptors over the context region (SSC-style mean pooling)
    n_chunks = (ctx_len + chunk - 1) // chunk
    desc = h.new_zeros(B, n_chunks, d)
    for c in range(n_chunks):
        s, e = c * chunk, min((c + 1) * chunk, ctx_len)
        desc[:, c] = h[:, s:e].mean(1)

    pp_loc = pool_loc = pp_dec = pool_dec = qcount = 0
    for nrow in range(B):
        keys_ctx = ids[nrow, vpos[nrow] - 1]                       # [k] the k context keys
        val_pos = vpos[nrow]                                       # [k] value positions
        val_tok = ids[nrow, val_pos]                               # [k] value tokens
        # query positions = supervised positions in the query region
        qpos = (labels[nrow] != IGNORE).nonzero(as_tuple=True)[0]
        for t in qpos.tolist():
            qkey = ids[nrow, t]
            match = (keys_ctx == qkey).nonzero(as_tuple=True)[0]
            if len(match) == 0:
                continue
            i = int(match[0])                                       # index of the queried kv
            q = h[nrow, t]                                          # query representation
            qcount += 1
            # --- H2/H1: localize the QUERIED value among the k value positions ---
            # per-position (query-conditioned): score each value-position hidden by q . h[pos]
            pp_scores = h[nrow, val_pos] @ q                        # [k]
            if int(pp_scores.argmax()) == i:
                pp_loc += 1
            # pooled routing: score each CHUNK by q . desc, then the value's chunk must win
            chunk_scores = desc[nrow] @ q                          # [n_chunks]
            picked = int(chunk_scores.argmax())
            true_chunk = int(val_pos[i].item() // chunk)
            if picked == true_chunk:
                pool_loc += 1
            # --- H1 extraction: decode the value token from per-position vs pooled ---
            if int(logits[nrow, val_pos[i]].argmax()) == int(val_tok[i]):
                pp_dec += 1
            if int(model.lm_head(desc[nrow, true_chunk]).argmax()) == int(val_tok[i]):
                pool_dec += 1
    q = max(qcount, 1)
    return {"n_q": qcount, "n_chunks": n_chunks,
            "per_pos_localize": pp_loc / q, "pooled_chunk_hit": pool_loc / q,
            "per_pos_decode": pp_dec / q, "pooled_decode": pool_dec / q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-kv", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--eval-kv", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--ctx-filler", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GDN2LM(args.vocab, d_model=args.d_model, n_layers=args.n_layers,
                   head_dim=args.head_dim, num_heads=args.num_heads).to(device)
    print(f"probe: train-kv={args.train_kv} eval-kv={args.eval_kv} ctx-filler={args.ctx_filler} "
          f"chunk={args.chunk} | device={device}", flush=True)
    final = train(model, device, args.train_kv, args.ctx_filler, args.vocab, args.steps, args.batch, args.lr)
    print(f"=== final train loss {final:.4f} (random ~= {torch.log(torch.tensor(args.vocab/2.)):.2f}) ===")
    print("=== descriptor mechanism probe (per-position vs pooled-chunk) ===")
    print(f"{'kv':>5} {'#q':>5} {'#chk':>5} | {'pp_localize':>11} {'pool_chunkhit':>13} | {'pp_decode':>9} {'pool_decode':>11}")
    for k in args.eval_kv:
        r = probe(model, device, k, args.ctx_filler, args.vocab, args.chunk)
        print(f"{k:>5} {r['n_q']:>5} {r['n_chunks']:>5} | {r['per_pos_localize']:>11.3f} "
              f"{r['pooled_chunk_hit']:>13.3f} | {r['per_pos_decode']:>9.3f} {r['pooled_decode']:>11.3f}", flush=True)


if __name__ == "__main__":
    main()
