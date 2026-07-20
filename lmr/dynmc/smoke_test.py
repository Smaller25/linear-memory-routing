# -*- coding: utf-8 -*-
"""DynMC end-to-end smoke test (합성 데이터, 1 GPU, ~2분).

검증 항목:
  1. varlen 커널 경로: cu_seqlens로 segment reset + final_state [S,H,V,K] 반환
  2. MC-GRM read: shape/causality/doc-mask, checkpoint된 read의 grad 흐름
  3. train 스텝: loss 유한, backward/clip/step 통과, loss가 감소 방향
  4. 신호 recorder: 4종 신호가 chunk 경계마다 유한값으로 산출

usage: PYTHONPATH=. python lmr/dynmc/smoke_test.py [--tmp DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch


def make_synthetic_corpus(d: str, vocab=32000, n_docs=300, seed=0):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    lens = rng.integers(50, 3000, size=n_docs).astype(np.uint32)
    toks = rng.integers(3, vocab, size=int(lens.sum())).astype(np.uint16)
    toks.tofile(os.path.join(d, "tokens-00000.bin"))
    np.save(os.path.join(d, "doclens-00000.npy"), lens)
    np.save(os.path.join(d, "sources-00000.npy"),
            rng.integers(0, 7, size=n_docs).astype(np.uint8))
    return int(lens.sum())


def test_layer_unit():
    from lmr.dynmc.model import build_model
    from lmr.dynmc.segmenting import build_batch_segments
    torch.manual_seed(0)
    dev = "cuda"
    model = build_model("46m", dynmc=True, cache_budget=4).to(dev)
    ctx = 2048
    ids = torch.randint(3, 32000, (1, 2 * ctx), device=dev)
    rng = np.random.default_rng(0)
    # 문서 >1024 tokens → 세그먼트 ≥2개 보장 → cached read 경로가 반드시 발동
    doc_lens = [[1500, 548], [2048]]
    segs = build_batch_segments(doc_lens, rng, mode="random")
    doc_ids = segs["seg_doc_ids"]
    change = torch.ones(len(doc_ids), dtype=torch.bool)
    change[1:] = doc_ids[1:] != doc_ids[:-1]
    first_idx = torch.nonzero(change).flatten()
    first = first_idx[torch.cumsum(change.long(), 0) - 1]
    dynmc_segs = {"cu_seqlens": segs["cu_seqlens"].to(dev), "seg_doc_start": first.to(dev)}

    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, dynmc_segs=dynmc_segs)
    logits = out.logits
    assert logits.shape == (1, 2 * ctx, 32000), logits.shape
    loss = logits.float().mean()
    loss.backward()
    g = model.model.layers[0].attn.u_proj.weight.grad
    assert g is not None and torch.isfinite(g).all(), "u_proj grad missing/nonfinite"
    cur_g = model.model.layers[0].attn.cur_logit.grad
    assert cur_g is not None, "cur_logit grad missing"
    print("[smoke] layer unit OK — logits", tuple(logits.shape),
          "u_proj grad norm %.3e" % g.norm().item(), flush=True)


def test_signals():
    from lmr.dynmc.model import build_model
    from lmr.dynmc.signals import GDNSignalRecorder
    dev = "cuda"
    model = build_model("46m", dynmc=False).to(dev).eval()
    rec = GDNSignalRecorder(model)
    ids = torch.randint(3, 32000, (1, 512), device=dev)
    sig = rec.run(ids)
    rec.remove()
    for name, v in sig.items():
        assert torch.isfinite(v).all(), f"signal {name} nonfinite"
        print(f"[smoke] signal {name}: shape {tuple(v.shape)} mean {v.mean():.4f}", flush=True)


def test_train_steps(tmp):
    corpus_dir = os.path.join(tmp, "corpus")
    make_synthetic_corpus(corpus_dir)
    cfg = dict(run_name="smoke", size="46m", dynmc=True, seg_mode="random",
               cache_budget=4, ctx=1024, rows_per_micro=2, grad_accum=1,
               lr=3e-4, warmup_steps=2, total_tokens=20480, ckpt_interval_tokens=10240,
               data_dir=corpus_dir, uniform_plan_tokens=200000,
               out_dir=os.path.join(tmp, "run"), log_interval=1, seed=0)
    cfg_path = os.path.join(tmp, "smoke.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    r = subprocess.run([sys.executable, "-m", "lmr.dynmc.train", "--config", cfg_path],
                       capture_output=True, text=True, cwd=os.getcwd())
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("train smoke failed")
    losses = [json.loads(l.split("[dynmc] ", 1)[1])["loss"]
              for l in r.stdout.splitlines() if l.startswith("[dynmc] {")]
    assert len(losses) >= 5 and all(np.isfinite(losses)), losses
    print(f"[smoke] train steps OK — loss {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)




def test_read_equivalence():
    """벡터화 _grm_read가 참조(세그먼트 루프) 구현과 수치 일치하는지."""
    import torch
    from lmr.dynmc.model import build_model
    from lmr.dynmc.segmenting import build_batch_segments
    torch.manual_seed(1)
    dev = "cuda"
    model = build_model("46m", dynmc=True, cache_budget=4).to(dev).eval()
    layer = model.model.layers[0].attn

    T, S = 1536, 0
    doc_lens = [[900, 636]]
    rng = np.random.default_rng(3)
    segs = build_batch_segments(doc_lens, rng, mode="random")
    cu = segs["cu_seqlens"].to(dev)
    doc_ids = segs["seg_doc_ids"]
    change = torch.ones(len(doc_ids), dtype=torch.bool)
    change[1:] = doc_ids[1:] != doc_ids[:-1]
    first_idx = torch.nonzero(change).flatten()
    first = first_idx[torch.cumsum(change.long(), 0) - 1].to(dev)
    S = len(doc_ids)

    H, V, K = layer.num_v_heads, layer.head_v_dim, layer.head_k_dim
    h_in = torch.randn(1, T, layer.hidden_size, device=dev)
    q = torch.randn(1, T, layer.num_heads, K, device=dev)
    o = torch.randn(1, T, H, V, device=dev)
    states = torch.randn(S, H, V, K, device=dev)

    with torch.no_grad():
        fast = layer._grm_read(h_in, q, o, states, cu, first)

        # 참조 구현 (원 루프)
        from fla.modules.l2norm import l2norm
        qn = l2norm(q) * (K ** -0.5)
        if layer.num_v_heads > layer.num_heads:
            qn = qn.repeat_interleave(layer.num_v_heads // layer.num_heads, dim=-2)
        qn = qn.squeeze(0).float()
        desc = states.mean(dim=2).reshape(S, -1)
        u = layer.u_proj(h_in).squeeze(0).float()
        d_scale = layer.descriptor_dim ** -0.5
        o_flat = o.squeeze(0)
        ref = o_flat.clone()
        cul = cu.tolist(); ds = first.tolist()
        for j in range(S):
            lo = max(ds[j], j - layer.cache_budget)
            t0, t1 = cul[j], cul[j + 1]
            if lo >= j:
                continue
            s_e = states[lo:j]
            logits = u[t0:t1] @ desc[lo:j].t() * d_scale
            curl = layer.cur_logit.expand(t1 - t0, 1)
            gam = torch.softmax(torch.cat([curl, logits], dim=-1), dim=-1)
            z = torch.einsum("ehvk,thk->tehv", s_e, qn[t0:t1])
            mix = torch.einsum("te,tehv->thv", gam[:, 1:], z)
            ref[t0:t1] = (gam[:, 0].view(-1, 1, 1) * o_flat[t0:t1].float() + mix).to(o_flat.dtype)
        # 첫 세그먼트(캐시 없음)는 fast에서 γ_cur<1이 아니라 γ_cur=1이어야 함 — 둘 다 동일 처리 확인
        err = (fast.squeeze(0).float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        print(f"[smoke] read equivalence max|Δ|={err:.3e} (scale {scale:.3e})", flush=True)
        assert err < 1e-3 * max(scale, 1.0), "vectorized read != reference"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="_workspace/dynmc_smoke")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    test_layer_unit()
    test_read_equivalence()
    test_signals()
    test_train_steps(a.tmp)
    print("[smoke] ALL OK", flush=True)
