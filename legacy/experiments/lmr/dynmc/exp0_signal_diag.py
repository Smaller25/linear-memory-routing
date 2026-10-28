# -*- coding: utf-8 -*-
"""실험 0 — 신호 진단 (plan §5.5, Gate G0).

신호 4종(drift/erasure/saturation/surprise)을 실제 텍스트 위에서 chunk 경계마다
측정하고, (i) pairwise Pearson/Spearman correlation, (ii) matched-density
boundary 집합의 Jaccard overlap을 산출한다.

G0 판정 (plan Day 1): 전 쌍 Jaccard > 0.9 → STOP (신호들이 사실상 동일 —
"dynamic trigger 간 차이" 서사 불성립). 0.5–0.9 → 진행 + regime-conditional
분석 추가. < 0.5 → 청신호.

대상 checkpoint:
  0-i : --model hf:m-a-p/1.3B-100B-GatedDeltaNet-pure (기존 mosc 학습 ckpt가
        디스크에 없어 pretrained GDN으로 대체)
  0-ii: --model dynmc:<ckpt_dir> (0024의 2–3B 시점 checkpoint)

usage (sbatch):
  PYTHONPATH=. python lmr/dynmc/exp0_signal_diag.py \
      --model hf:m-a-p/1.3B-100B-GatedDeltaNet-pure \
      --val /path/to/val-tokens.bin --ctx 8192 --n-seq 32 \
      --target-seg 256 --out _workspace/dynmc/exp0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

SIGNALS = ["drift", "erasure", "saturation", "surprise"]


def load_model(spec: str, device):
    kind, _, name = spec.partition(":")
    if kind == "hf":
        import fla  # noqa: F401  (register archs)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        return model.to(device).eval()
    elif kind == "dynmc":
        from lmr.dynmc.model import build_model
        ck = torch.load(os.path.join(name, "state_rank0.pt"), map_location="cpu")
        model = build_model(ck["cfg"]["size"], dynmc=ck["cfg"]["dynmc"],
                            cache_budget=ck["cfg"]["cache_budget"])
        model.load_state_dict(ck["model"])
        return model.to(device).to(torch.bfloat16).eval()
    raise ValueError(spec)


def matched_density_boundaries(series: np.ndarray, target_density: float,
                               level: bool = False) -> np.ndarray:
    """series: [C] chunk-level signal → bool [C] boundary mask with matched density.

    누적형: 마지막 boundary 이후 누적합 > τ 시 fire (τ는 bisection으로 밀도 맞춤).
    레벨형(saturation): series ≥ τ_sat, τ_sat = (1-d) quantile → 밀도 근사 일치.
    """
    C = len(series)
    n_target = max(1, int(round(C * target_density)))
    if level:
        tau = np.quantile(series, 1.0 - target_density)
        return series >= tau

    def fire_count(tau: float) -> tuple[int, np.ndarray]:
        acc, mask = 0.0, np.zeros(C, dtype=bool)
        for i, s in enumerate(series):
            acc += s
            if acc > tau:
                mask[i] = True
                acc = 0.0
        return int(mask.sum()), mask

    lo, hi = 0.0, float(series.sum())
    mask = None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n, mask = fire_count(mid)
        if n > n_target:
            lo = mid
        elif n < n_target:
            hi = mid
        else:
            break
    return mask if mask is not None else np.zeros(C, dtype=bool)


def jaccard(a: np.ndarray, b: np.ndarray, tol: int = 0) -> float:
    ia, ib = set(np.nonzero(a)[0]), set(np.nonzero(b)[0])
    if not ia and not ib:
        return 1.0
    if tol == 0:
        return len(ia & ib) / max(1, len(ia | ib))
    hit = sum(1 for x in ia if any(abs(x - y) <= tol for y in ib))
    return hit / max(1, len(ia | ib) - (len(ia) - hit) // 2) if (ia or ib) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True, help="uint16 token bin (val-tokens.bin)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--target-seg", type=int, default=256, help="평균 segment 길이(tokens)")
    ap.add_argument("--out", default="_workspace/dynmc/exp0")
    a = ap.parse_args()

    from scipy import stats as sps
    from lmr.dynmc.signals import GDNSignalRecorder

    device = "cuda"
    model = load_model(a.model, device)
    rec = GDNSignalRecorder(model)

    toks = np.memmap(a.val, dtype=np.uint16, mode="r")
    seqs = [torch.from_numpy(np.asarray(toks[i * a.ctx:(i + 1) * a.ctx], dtype=np.int64))
            for i in range(a.n_seq)]

    per_sig_series = {s: [] for s in SIGNALS}   # 시퀀스별 chunk-series (layer-mean, head-mean)
    per_sig_layer = {s: [] for s in SIGNALS}    # [C, L] (head-mean)
    for si, ids in enumerate(seqs):
        sig = rec.run(ids.unsqueeze(0).to(device))
        for s in SIGNALS:
            v = sig[s].float().mean(-1)         # [C, L] head-mean
            per_sig_layer[s].append(v.numpy())
            per_sig_series[s].append(v.mean(-1).numpy())  # layer-mean → [C]
        print(f"[exp0] seq {si+1}/{a.n_seq} done", flush=True)
    rec.remove()

    os.makedirs(a.out, exist_ok=True)
    density = 64.0 / a.target_seg  # boundary per chunk

    # correlation (시퀀스 concat, layer-mean 시리즈 기준 + per-layer 부록)
    cat = {s: np.concatenate(per_sig_series[s]) for s in SIGNALS}
    pear = np.zeros((4, 4)); spear = np.zeros((4, 4)); jac = np.zeros((4, 4))
    bmask = {}
    for s in SIGNALS:
        masks = [matched_density_boundaries(x, density, level=(s == "saturation"))
                 for x in per_sig_series[s]]
        bmask[s] = masks
    for i, s1 in enumerate(SIGNALS):
        for j, s2 in enumerate(SIGNALS):
            pear[i, j] = np.corrcoef(cat[s1], cat[s2])[0, 1]
            spear[i, j] = sps.spearmanr(cat[s1], cat[s2]).statistic
            jac[i, j] = float(np.mean([jaccard(x, y) for x, y in zip(bmask[s1], bmask[s2])]))

    np.savez(os.path.join(a.out, "exp0_signals.npz"),
             **{f"series_{s}": cat[s] for s in SIGNALS},
             **{f"layer_{s}": np.concatenate(per_sig_layer[s]) for s in SIGNALS},
             pearson=pear, spearman=spear, jaccard=jac)

    def fmt(m):
        hdr = "            " + " ".join(f"{s[:8]:>10s}" for s in SIGNALS)
        rows = [f"{SIGNALS[i]:>10s}  " + " ".join(f"{m[i, j]:10.3f}" for j in range(4))
                for i in range(4)]
        return "\n".join([hdr] + rows)

    print("\n=== Pearson ===\n" + fmt(pear))
    print("\n=== Spearman ===\n" + fmt(spear))
    print("\n=== Jaccard (matched density %.3f/chunk) ===\n" % density + fmt(jac))

    off = [jac[i, j] for i in range(4) for j in range(4) if i < j]
    verdict = ("STOP_R1" if min(off) > 0.9 else
               "PROCEED_WITH_REGIME_ANALYSIS" if max(off) > 0.5 else "GREEN")
    print(f"\nG0_VERDICT={verdict} (off-diag Jaccard min {min(off):.3f} max {max(off):.3f})")
    with open(os.path.join(a.out, "exp0_verdict.json"), "w") as f:
        json.dump(dict(verdict=verdict, pearson=pear.tolist(), spearman=spear.tolist(),
                       jaccard=jac.tolist(), density=density), f, indent=2)


if __name__ == "__main__":
    main()
