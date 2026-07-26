### Task 6: E1 — routing 정확도 (`routing_stats.py`)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/routing_stats.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/e1.sbatch`

**Interfaces:**
- Consumes: `load_mc`, `data.annotate`, Dataset A/B jsonl (correctness join은 E2 산출물 `results/e2_oracle.json`에서 — 없으면 생략하고 나중에 채움)
- Produces: `capture_hidden(model, ids) -> list[Tensor[T,D]]` (layer별 attn 입력; E3도 재사용); `routing_scores_at(attn, h, t) -> Tensor[n_seg]` ; 결과 `results/e1_routing.json` + `results/e1_routing.png`

- [ ] **Step 1: routing_stats.py 작성**

```python
"""E1: answer position에서 layer별 gold-chunk routing 정확도.

생성 불필요 — 프롬프트 1-pass. 점수는 SSC와 동일 수식으로 재계산:
u = ssc.connector(h); summaries = segment_key_sums(normalize(k)); score = <u, c_i> (head 합).
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, data as mcdata

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def capture_hidden(model, ids):
    """각 layer의 attn 입력(norm_1 이후) [T,D]를 hook으로 수집."""
    store = {}
    hooks = []
    for i, blk in enumerate(model.transformer.h):
        def mk(i):
            def pre(mod, args, kwargs):
                store[i] = args[0].detach()
            return pre
        hooks.append(blk.attn.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(ids)
    for h in hooks:
        h.remove()
    return [store[i][0] for i in range(len(model.transformer.h))]


@torch.no_grad()
def routing_scores_at(attn, h, t):
    """h [T,D] (bf16 cuda). t 위치의 과거 segment별 routing score [n_seg] (미래는 -inf)."""
    load_mc.bootstrap()
    from dsc.mc_baseline.mc_ssc import segment_key_sums
    hb = h.unsqueeze(0)
    q, k, v, g, b, w = attn._project(hb)
    rk = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    summaries = segment_key_sums(rk, attn.ssc.chunk_size)          # [1,N,H,K]
    u = attn.ssc.connector(hb[:, t:t + 1]).view(1, 1, attn.ssc.num_heads,
                                                attn.ssc.head_qk_dim)
    scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())[0, 0]  # [N]
    cur_seg = t // attn.ssc.chunk_size
    scores[cur_seg:] = float("-inf")
    return scores


def analyze_sample(model, tok, input_text, topk=2):
    ann = mcdata.annotate(input_text, tok)
    ids = torch.tensor([tok(input_text, add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    hiddens = capture_hidden(model, ids)
    per_layer = []
    key_segs = sorted({n["seg"] for n in ann["needles"]})
    for li, (i, attn) in enumerate(load_mc.mc_layers(model)):
        s = routing_scores_at(attn, hiddens[i], T - 1)
        order = torch.argsort(s, descending=True)
        gold_rank = int((order == ann["gold_seg"]).nonzero()[0, 0])
        hit = ann["gold_seg"] in order[:topk].tolist()
        amongkeys = None
        if len(key_segs) > 1:
            best_key_seg = max(key_segs, key=lambda ks: float(s[ks]))
            amongkeys = (best_key_seg == ann["gold_seg"])
        per_layer.append({"layer": i, "gold_rank": gold_rank, "hit": hit,
                          "amongkeys": amongkeys})
    return {"gold_seg": ann["gold_seg"], "n_seg": ann["n_seg"], "per_layer": per_layer}
```

드라이버(같은 파일 `__main__`): `--model {mc-5B,mc-30B}` 별로 (i) Dataset A 두 태스크 50샘플, (ii) Dataset B S/D×multi 32샘플을 `analyze_sample`로 처리. 집계:
- layer별 `hit@2`율, `gold_rank` 평균, `amongkeys` 정답률(multi만) — 태스크·조건(S/D)별
- Dataset B multi에 한해, E2(Task 7) baseline 생성 결과(`results/e2_oracle.json`의 baseline rows)의 `correct`와 pair_id로 join해 "best layer 기준 hit인 샘플의 정답률 vs miss" 교차표 (E0 생략에 따라 Dataset A 상관 분석은 제외; E2보다 먼저 실행되는 경우 이 교차표만 나중에 채움)
- matplotlib로 layer(x) × hit@2(y) 곡선을 태스크·조건별 오버레이 → `results/e1_routing.png`; 수치 전체 `results/e1_routing.json`

- [ ] **Step 2: sbatch/e1.sbatch 작성·제출** (`-t 02:00:00`; 실행줄 `$PY "$ANA/routing_stats.py" --model mc-5B && $PY "$ANA/routing_stats.py" --model mc-30B`)

Expected: json에 single hit@2 高(≥0.8 예상, report의 S-NIAH 우세와 부합) vs multikey/D 조건 저하 여부가 드러남. 판정 기준이 아니라 **측정**이므로 수치 자체가 산출물.

- [ ] **Step 3: 커밋** (`mc-niah: E1 per-layer routing accuracy (A + S/D)`)

---

