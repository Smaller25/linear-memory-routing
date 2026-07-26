### Task 5: 생성 유틸 `gen_eval.py` (E0 앵커 재현은 **생략** — 2026-07-25 사용자 지시)

> E0 성능 재현 평가는 하지 않는다. anchor 수치는 collaborator 종합 성적표(spec §1 표:
> @2K에서 vanilla-5B 90/20, MC-5B 60/2, MC-30B 92/32)를 그대로 사용한다.
> `eval_niah.py`/`e0.sbatch`는 만들지 않는다. 이 task는 E2(Task 7)가 소비하는
> 생성·채점 유틸 `gen_eval.py`만 작성한다 (GPU 불필요, import 문법 확인만).

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/gen_eval.py`

**Interfaces:**
- Consumes: (없음 — 순수 유틸)
- Produces: `greedy_generate(model, ids, n_gen=128) -> list[int]`; `string_match_all(preds, refs) -> float`; `run_file(model, tok, jsonl_path, n_gen=128, variant_filter=None, tag="") -> dict(score, rows=[{index, pred, outputs, correct}])`

- [ ] **Step 1: gen_eval.py 작성**

```python
"""Greedy free generation (full re-forward; MC는 cache 미지원) + RULER string match."""
import importlib.util, json, os, sys
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_metrics():
    p = os.path.join(REPO, "src", "ruler", "eval_metrics.py")
    spec = importlib.util.spec_from_file_location("ruler_metrics", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

string_match_all = _load_metrics().string_match_all


@torch.no_grad()
def greedy_generate(model, ids, n_gen=128, eos_id=2):
    """ids [1,T] on cuda. 매 step 전체 re-forward (세그먼트 캐시 없음 = 학습 경로와 동일)."""
    out = []
    for _ in range(n_gen):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
        nxt = int(logits[0, -1].float().argmax())
        if nxt == eos_id:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return out


def run_file(model, tok, jsonl_path, n_gen=128, variant_filter=None, tag=""):
    rows = []
    for line in open(jsonl_path):
        r = json.loads(line)
        if variant_filter and r.get("variant") != variant_filter:
            continue
        ids = torch.tensor([tok(r["input"], add_special_tokens=False).input_ids],
                           device="cuda")
        gen = greedy_generate(model, ids, n_gen=n_gen)
        pred = tok.decode(gen)
        correct = all(o.lower() in pred.lower() for o in r["outputs"])
        rows.append({"index": r.get("index", r.get("pair_id")), "pred": pred,
                     "outputs": r["outputs"], "correct": correct,
                     **{k: r[k] for k in ("condition", "gold_seg", "variant") if k in r}})
        print(f"[{tag}] {len(rows)}: correct={correct}", flush=True)
    score = string_match_all([r["pred"] for r in rows], [r["outputs"] for r in rows])
    return {"score": score, "n": len(rows), "rows": rows}
```

- [ ] **Step 2: 문법 확인 및 커밋**

Run: `$PY -c "import sys; sys.path.insert(0,'lmr/analysis/260725_mc_niah_analysis'); import gen_eval; print(gen_eval.string_match_all(['x 123'],[['123']]))"`
Expected: `100.0`

- [ ] **Step 5: 커밋**

```bash
git add lmr/analysis/260725_mc_niah_analysis/
git commit -m "mc-niah: gen_eval util (greedy free-gen + RULER string match; E0 skipped per user)"
```

---

