### Task 8: E3 — write/read fidelity 프로파일 (`fidelity.py`)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/fidelity.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/e3.sbatch`

**Interfaces:**
- Consumes: `load_mc`, `routing_stats.capture_hidden`, Dataset B jsonl (single+multi 전부), `data.annotate`
- Produces: `results/e3_fidelity.json` + `results/e3_fidelity.png` — layer(16) × {b1_after, b1_final, b2_qk_align, b2_read_cos_vs_single, b2_interf_ratio} × {S,D} × {single,multi}

- [ ] **Step 1: fidelity.py 작성**

```python
"""E3: gold segment state의 write/read fidelity, layer별·조건별.

r = q·M ≈ (q·k*)v* + interference 분해의 각 인자를 직접 측정.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, data as mcdata
from routing_stats import capture_hidden

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _scan(attn, q, k, v, g, b, w, sl):
    """학습과 동일한 chunk_gdn2 경로로 [sl] 구간을 zero-state 스캔 → 최종 state [H,K,V]."""
    load_mc.bootstrap()
    from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
    _, state = chunk_gdn2(q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl],
                          b=b[:, sl], w=w[:, sl], initial_state=None,
                          output_final_state=True, use_qk_l2norm_in_kernel=True,
                          use_gate_in_kernel=False, cu_seqlens=None)
    return state[0].float()                                    # [H,K,V]


def _cos(a, b, dim=-1):
    return float(F.cosine_similarity(a.flatten(0, -2), b.flatten(0, -2), dim=dim).mean())


@torch.no_grad()
def sample_metrics(model, tok, row):
    ann = mcdata.annotate(row["input"], tok)
    ids = torch.tensor([tok(row["input"], add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    hiddens = capture_hidden(model, ids)
    gold = ann["gold_seg"]
    tgt = next(n for n in ann["needles"] if n["key"] == ann["query_key"])
    v_pos = list(range(tgt["tok_start"], tgt["tok_end"] + 1))   # value 토큰들
    seg_sl = slice(gold * 256, min((gold + 1) * 256, T))
    after_sl = slice(gold * 256, tgt["tok_end"] + 1)
    out = []
    for i, attn in load_mc.mc_layers(model):
        h = hiddens[i].unsqueeze(0)
        q, k, v, g, b, w = attn._project(h)
        kn = F.normalize(k.float(), p=2, dim=-1)
        m_after = _scan(attn, q, k, v, g, b, w, after_sl)       # [H,K,V]
        m_final = _scan(attn, q, k, v, g, b, w, seg_sl)
        # b-1: value 토큰별 k로 재독출 → 실제 v와 cosine (토큰 평균)
        b1a, b1f = [], []
        for t in v_pos:
            kt, vt = kn[0, t], v[0, t].float()                  # [H,K],[H,V]
            b1a.append(_cos(torch.einsum("hk,hkv->hv", kt, m_after), vt))
            b1f.append(_cos(torch.einsum("hk,hkv->hv", kt, m_final), vt))
        # b-2: answer position query
        qn = F.normalize(q[0, T - 1].float(), p=2, dim=-1)      # [H,K]
        t_last = v_pos[-1]
        qk = float((qn * kn[0, t_last]).sum(-1).mean())         # head 평균 정렬도
        r = torch.einsum("hk,hkv->hv", qn, m_final)             # [H,V]
        sig = (qn * kn[0, t_last]).sum(-1, keepdim=True) * v[0, t_last].float()
        interf = float((r - sig).norm() / (sig.norm() + 1e-8))
        out.append({"layer": i, "b1_after": sum(b1a) / len(b1a),
                    "b1_final": sum(b1f) / len(b1f), "b2_qk_align": qk,
                    "b2_interf_ratio": interf,
                    "_r": r.cpu()})                             # single-vs-multi 비교용
    return out
```

드라이버(`__main__`): `--model {mc-5B,mc-30B}` × `{S,D}.jsonl` 전 행(single+multi). pair_id로 single/multi를 짝지어 `b2_read_cos_vs_single = cos(r_multi, r_single)`을 layer별 계산(`_r` 사용 후 폐기, json에는 저장 안 함). 집계: 조건(S/D)·variant별 layer 곡선 평균 → json + png (2×2 subplot: b1_final, b1_after, b2_qk_align, b2_read_cos_vs_single).

- [ ] **Step 2: sbatch/e3.sbatch 작성·제출** (`-t 03:00:00`)

Expected sanity: single의 b1_after가 모든 layer에서 높음(≥0.5 수준; 완전 붕괴면 코드 버그 의심 — E0에서 single 60을 맞히는 모델임). S/multi의 b1·b2가 상대적으로 낮은지가 관측 대상.

- [ ] **Step 3: 커밋** (`mc-niah: E3 write/read fidelity profiling`)

---

