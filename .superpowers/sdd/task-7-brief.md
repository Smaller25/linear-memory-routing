### Task 7: E2 — oracle routing 개입 (`oracle.py`)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/oracle.py`
- Test: `tests/lmr/test_mc_niah_oracle.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/e2.sbatch`

**Interfaces:**
- Consumes: `load_mc`, `gen_eval.run_file`, Dataset B jsonl
- Produces: `inject_gold(top_indices[B,T,k], top_scores[B,T,k], online_score[B,T], gold:int, segment_ids[T]) -> (idx, scores)` (순수 함수); `patch_oracle(model) -> list[OracleGDN2SSC]`; 각 oracle 인스턴스의 `gold_segment: int|None` 속성; 결과 `results/e2_oracle.json`

- [ ] **Step 1: 실패하는 inject 테스트 작성**

```python
# tests/lmr/test_mc_niah_oracle.py
import os, sys
import torch

ANA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "lmr", "analysis", "260725_mc_niah_analysis"))
sys.path.insert(0, ANA)
from oracle import inject_gold  # noqa: E402


def test_inject_gold_forces_missing_gold():
    # T=3 tokens, k=2. segment_ids: 토큰이 속한 segment
    top_idx = torch.tensor([[[0, 1], [0, 1], [2, 0]]])          # [1,3,2]
    top_sc = torch.tensor([[[5.0, 3.0], [5.0, 3.0], [4.0, 2.0]]])
    online = torch.tensor([[1.0, 6.0, 1.0]])
    seg_ids = torch.tensor([3, 3, 4])
    idx, sc = inject_gold(top_idx, top_sc, online, gold=2, segment_ids=seg_ids)
    # t=0: gold(2) 미포함 → 마지막 슬롯 교체, score = max(top.max, online) = 5
    assert idx[0, 0].tolist() == [0, 2] and sc[0, 0, 1] == 5.0
    # t=1: online이 최고(6) → 강제 슬롯 score도 6 (gate에서 열세 방지)
    assert idx[0, 1].tolist() == [0, 2] and sc[0, 1, 1] == 6.0
    # t=2: gold 이미 선택됨 → 불변
    assert idx[0, 2].tolist() == [2, 0] and torch.equal(sc[0, 2], top_sc[0, 2])


def test_inject_gold_respects_eligibility():
    top_idx = torch.tensor([[[0, 1]]]); top_sc = torch.tensor([[[5.0, 3.0]]])
    online = torch.tensor([[1.0]])
    idx, sc = inject_gold(top_idx, top_sc, online, gold=2,
                          segment_ids=torch.tensor([2]))   # gold==현재 seg → 미래/현재는 불가
    assert idx[0, 0].tolist() == [0, 1] and torch.equal(sc, top_sc)
```

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/lmr/test_mc_niah_oracle.py -x -q` → FAIL (oracle 없음)

- [ ] **Step 3: oracle.py 작성**

```python
"""E2: gold chunk를 top-k에 강제 주입한 oracle routing으로 재생성."""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def inject_gold(top_indices, top_scores, online_score, gold, segment_ids):
    """gold가 eligible(과거 segment)인데 미선택인 토큰의 마지막 슬롯을 gold로 교체.
    교체 슬롯 score = max(선택 score 최대, online score) → gate에서 공정한 무게."""
    eligible = (segment_ids > gold).unsqueeze(0).expand(top_indices.shape[:2])  # [B,T]
    has = (top_indices == gold).any(-1)
    force = eligible & ~has
    idx, sc = top_indices.clone(), top_scores.clone()
    idx[force, -1] = gold
    best = torch.maximum(top_scores.max(-1).values, online_score)
    sc[force, -1] = best[force]
    return idx, sc


def _make_oracle_class():
    import load_mc
    load_mc.bootstrap()
    from torch.nn import functional as F
    from dsc.mc_gdn2.ssc import GDN2SSC
    from dsc.mc_baseline.mc_ssc import (SSCOutput, segment_key_sums,
                                        causal_online_key_sums)
    from dsc.mc_baseline.cached_memory_read import ssc_gather_read

    class OracleGDN2SSC(GDN2SSC):
        gold_segment = None  # 샘플마다 설정

        def forward(self, hidden_states, queries, keys, online_output, memories):
            # mc_ssc.SparseSelectiveCaching.forward를 복사, topk 뒤 inject만 추가
            batch, length, heads, key_dim = queries.shape
            num_segments = memories.shape[1]
            u = self.connector(hidden_states).view(batch, length, heads, key_dim)
            summaries = segment_key_sums(keys, self.chunk_size)
            all_scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())
            segment_ids = torch.arange(length, device=queries.device) // self.chunk_size
            eligible = (torch.arange(num_segments, device=queries.device)[None, :]
                        < segment_ids[:, None])
            past_scores = all_scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)
            route_count = min(self.topk, num_segments)
            top_scores, top_indices = torch.topk(past_scores, k=route_count, dim=-1)
            online_summary = causal_online_key_sums(keys, self.chunk_size)
            online_score = torch.einsum("bthk,bthk->bt", u.float(), online_summary.float())
            if self.gold_segment is not None:                       # === oracle 개입 ===
                top_indices, top_scores = inject_gold(
                    top_indices, top_scores, online_score, self.gold_segment, segment_ids)
            valid = torch.isfinite(top_scores)
            safe_indices = top_indices.masked_fill(~valid, 0)
            gate_logits = torch.cat([online_score.unsqueeze(-1), top_scores], dim=-1)
            gate_valid = torch.cat([torch.ones(batch, length, 1, device=queries.device,
                                               dtype=torch.bool), valid], dim=-1)
            gate_logits = gate_logits.masked_fill(~gate_valid, -torch.inf)
            gates = torch.softmax(gate_logits, dim=-1).to(online_output.dtype)
            online_weight, route_weights = gates[..., :1], gates[..., 1:]
            cached_output = ssc_gather_read(
                queries, memories, safe_indices, route_weights,
                scale=self.read_scale, normalize_queries=self.normalize_queries,
            ).to(online_output.dtype)
            output = online_weight.unsqueeze(-1) * online_output + cached_output
            return SSCOutput(output=output, online_output=online_output,
                             cached_output=cached_output,
                             route_indices=top_indices.masked_fill(~valid, -1),
                             route_weights=route_weights, online_weight=online_weight,
                             route_scores=top_scores.masked_fill(~valid, -torch.inf))

    return OracleGDN2SSC


def patch_oracle(model):
    """모든 MC layer의 ssc를 Oracle 버전으로 교체(가중치 복사). 교체본 리스트 반환."""
    import load_mc
    cls = _make_oracle_class()
    oracles = []
    for _, attn in load_mc.mc_layers(model):
        old = attn.ssc
        new = cls(old.hidden_size, old.num_heads, old.head_qk_dim,
                  topk=old.topk, chunk_size=old.chunk_size)
        new.load_state_dict(old.state_dict())
        new = new.to(next(old.parameters()).device, next(old.parameters()).dtype)
        attn.ssc = new
        oracles.append(new)
    return oracles
```

주의: `GDN2SSC.__init__`은 `normalize_queries=True`를 강제하므로 추가 인자 불필요. `_make_oracle_class`가 import 시점이 아니라 호출 시점에 dsc를 import하므로 CPU 테스트(`inject_gold`만)는 worktree 없이도 통과.

드라이버(`__main__`): `--model {mc-5B,mc-30B}` × Dataset B `{S,D}.jsonl`의 **multi** 행만: (1) oracle 없이(baseline, `gold_segment=None`) run_file, (2) 각 샘플 forward 전 모든 oracle의 `gold_segment = row["gold_seg"]` 설정 후 생성. gen_eval.run_file은 샘플별 콜백이 없으므로 oracle 드라이버는 run_file을 쓰지 말고 jsonl 루프를 직접 돌며 `greedy_generate` 호출(≈15줄). S/D × {baseline, oracle} 4셀 score + per-sample을 `results/e2_oracle.json`에 저장.

- [ ] **Step 4: 테스트 통과 확인** — `$PY -m pytest tests/lmr/test_mc_niah_oracle.py -x -q` → 2 passed

- [ ] **Step 5: sbatch/e2.sbatch 작성·제출** (`-t 03:00:00`) → `results/e2_oracle.json` 생성 확인. sanity: oracle이 baseline보다 **낮으면** 버그 의심(개입은 gold를 추가할 뿐).

- [ ] **Step 6: 커밋** (`mc-niah: E2 oracle routing intervention (S/D)`)

---

