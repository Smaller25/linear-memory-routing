# MC-SSC GDN2 Multi-NIAH 실패 원인 분석 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MC-SSC(mean-pool) GDN2 370M이 multi-NIAH에서 실패하는 원인을 write→read→route→생성 4단계로 분해해 지목하는 분석 파이프라인 구축·실행.

**Architecture:** `linear-memory-routing` 브랜치 `sh/mc-niah-analysis`에 분석 코드(`lmr/analysis/260725_mc_niah_analysis/`)를 두고, 모델 정의는 long-gdn **pinned worktree(`e71713e`)**를 PYTHONPATH import. 데이터는 vendored RULER(Dataset A) + 자체 paired-controlled 생성기(Dataset B). GPU 작업은 전부 SLURM sbatch.

**Tech Stack:** PyTorch(bf16), litgpt-style GPT (dsc/lit_gpt), dsc/mc_gdn2 SSC, vendored NVIDIA RULER, TinyLlama tokenizer, matplotlib, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-25-mc-niah-analysis-design.md` (판정표·조건 S/D 정의 포함)
- GPU 실행은 **반드시** `sbatch -p main --gres=gpu:rtx6000:1` (로그인/노드 직접 CUDA 금지)
- python = `/data2/sohyung/conda-envs/sh_infocap/bin/python` (1.3B gdn2 평가 검증된 env; smoke 실패 시에만 `~/.conda/envs/sh_routing` 시도)
- `HF_HOME=/data2/sohyung/hf_home`, 대용량 산출물은 `/data2/sohyung/mc_niah/` (root 디스크 만원)
- worktree: `/data2/sohyung/worktrees/long-gdn-e71713e` — long-gdn `origin/main@e71713e` 고정, **worktree 내 파일 수정 금지**
- 모델: `mc-30B`/`mc-5B` = `LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool`의 `checkpoint-{30B,5B}-model-ckpt.pth` (config `mc_370M`: topk=2, chunk 256); `vanilla-5B` = `LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla`의 `checkpoint-5B-model-ckpt.pth` (config `gdn2_370M`)
- 길이 2048 고정 (=8 segments), Dataset A 50 샘플/태스크, Dataset B 32쌍(S 16 + D 16)
- tokenizer `TinyLlama/TinyLlama_v1.1`; RULER 프로토콜 = greedy free-gen(n_gen=128) + `string_match_all`; `answer_prefix`는 입력에 넣지 않음(`scripts/ruler.py` 기본과 동일)
- anti-oracle 없음. 폴더명은 반드시 `260725_mc_niah_analysis` (숫자 시작 → `-m` 불가, 스크립트는 파일 경로 실행; 폴더 내 상호 import는 각 스크립트가 자기 dir을 `sys.path`에 추가)
- 커밋 말미: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**핵심 코드 사실 (구현자는 그대로 신뢰할 것, 전부 검증됨):**
- `dsc/lit_gpt/model.py` Block은 `h, _, new_kv_cache = self.attn(n_1, attention_mask=None)`로 attn을 호출 → attn 대체물은 **3-tuple 반환** 필수
- `MemoryCachingGDN2Layer`(= `blk.attn`)는 `.base`(GatedDeltaNet2), `.ssc`(GDN2SSC), `._project(h)→(q,k,v,g,b,w)`, `.forward_with_diagnostics(h)→(out, SSCOutput)` 제공
- `SSCOutput` 필드: `output, online_output, cached_output, route_indices[B,T,k], route_weights, online_weight, route_scores`
- `dsc/mc_baseline/mc_ssc.py`: `segment_key_sums(keys,chunk)`(mean-pool), `causal_online_key_sums`, `linear_memory_read`; SSC.forward 내부: `u=connector(h)`, `all_scores=einsum("bthk,bnhk->btn", u, summaries)`, eligible = past-only, `topk`, online+topk joint softmax, `ssc_gather_read`(triton)
- routing keys = `F.normalize(k.float(),p=2,dim=-1)` (`dsc/mc_gdn2/ssc.py`)
- ckpt 포맷: `torch.load(...)["model"]` → 키가 GPT의 `transformer.h.N.attn.{base,ssc}...`와 1:1 (371 tensors)
- sys.path는 worktree **root**(`dsc.` 패키지용)와 **root/dsc**(`lit_gpt.`, `mc_gdn2.`용) 둘 다 필요
- RULER niah: `tokens_to_generate=128`; `niah_single_1`(noise haystack), `niah_multikey_1`(essay haystack, num_needle_k=4) — multikey_1은 `data/PaulGrahamEssays.json` 필요(`python scripts/ruler.py download-essays`)
- needle 문장: `"One of the special magic numbers for {key} is: {value}."`, value=7자리 숫자; 질문(단수형 변환 후): `"...What is the special magic number for {query} mentioned in the provided text?"`

---

### Task 1: Pinned worktree + 실험 환경 스캐폴드

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/env_common.sh`
- Create: `lmr/analysis/260725_mc_niah_analysis/__init__.py` (빈 파일)

**Interfaces:**
- Produces: worktree `/data2/sohyung/worktrees/long-gdn-e71713e`; 이후 모든 sbatch가 source하는 `env_common.sh` (`$PY`, `$LMR`, `$MC_OUT`, `$MC_LONGGDN_WORKTREE` 정의)

- [ ] **Step 1: worktree 생성 및 검증**

```bash
mkdir -p /data2/sohyung/worktrees
git -C /home/sohyung/long-gdn worktree add /data2/sohyung/worktrees/long-gdn-e71713e e71713e
ls /data2/sohyung/worktrees/long-gdn-e71713e/dsc/mc_gdn2/ssc.py \
   /data2/sohyung/worktrees/long-gdn-e71713e/dsc/mc_baseline/mc_ssc.py
grep -n "mean of L2-normalized keys" /data2/sohyung/worktrees/long-gdn-e71713e/dsc/mc_baseline/mc_ssc.py
grep -n '"mc_370M"\|name="mc_370M"' /data2/sohyung/worktrees/long-gdn-e71713e/dsc/lit_gpt/config.py
```
Expected: 파일 존재, mean-pool docstring 매치, `mc_370M` config 존재. (하나라도 실패 → 사용자 보고 후 중단)

- [ ] **Step 2: env_common.sh 작성**

```bash
#!/usr/bin/env bash
# MC NIAH 분석 공통 환경 — 모든 sbatch가 source
export HF_HOME=/data2/sohyung/hf_home TMPDIR=/data2/sohyung/tmp XDG_CACHE_HOME=/data2/sohyung/cache
export TRITON_CACHE_DIR=/data2/sohyung/tmp/.triton
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_LONGGDN_WORKTREE=/data2/sohyung/worktrees/long-gdn-e71713e
export MC_OUT=/data2/sohyung/mc_niah
PY=/data2/sohyung/conda-envs/sh_infocap/bin/python
LMR=/home/sohyung/linear-memory-routing
ANA=$LMR/lmr/analysis/260725_mc_niah_analysis
mkdir -p "$MC_OUT"/{data,logs,results} /data2/sohyung/tmp/.triton
```

- [ ] **Step 3: 커밋**

```bash
cd /home/sohyung/linear-memory-routing
git add lmr/analysis/260725_mc_niah_analysis/
git commit -m "mc-niah: env scaffold + pinned long-gdn worktree recipe"
```

---

### Task 2: 모델 로더 `load_mc.py` + GPU smoke test

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/load_mc.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/smoke.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/smoke.sbatch`

**Interfaces:**
- Produces: `bootstrap()` (sys.path 주입); `load_model(kind, device="cuda", dtype=torch.bfloat16) -> GPT` (kind ∈ `"mc-30B","mc-5B","vanilla-5B"`); `load_tokenizer()`; 상수 `CKPTS: dict[str, tuple[repo, fname, config_name]]`, `CHUNK=256`, `TOPK=2`

- [ ] **Step 1: load_mc.py 작성**

```python
"""Pinned long-gdn worktree에서 MC/vanilla GDN2 370M litgpt 모델 로드."""
import os, sys
import torch

WORKTREE = os.environ.get("MC_LONGGDN_WORKTREE", "/data2/sohyung/worktrees/long-gdn-e71713e")
CHUNK, TOPK = 256, 2
CKPTS = {
    "mc-30B": ("LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool",
               "checkpoint-30B-model-ckpt.pth", "mc_370M"),
    "mc-5B": ("LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool",
              "checkpoint-5B-model-ckpt.pth", "mc_370M"),
    "vanilla-5B": ("LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla",
                   "checkpoint-5B-model-ckpt.pth", "gdn2_370M"),
}


def bootstrap():
    for p in (WORKTREE, os.path.join(WORKTREE, "dsc")):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_model(kind, device="cuda", dtype=torch.bfloat16):
    bootstrap()
    from huggingface_hub import hf_hub_download
    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    repo, fname, cfg_name = CKPTS[kind]
    path = hf_hub_download(repo, fname)
    model = GPT(Config.from_name(cfg_name))
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


def mc_layers(model):
    """[(layer_idx, MemoryCachingGDN2Layer)] — vanilla 모델이면 빈 리스트."""
    out = []
    for i, blk in enumerate(model.transformer.h):
        if type(blk.attn).__name__ == "MemoryCachingGDN2Layer":
            out.append((i, blk.attn))
    return out
```

- [ ] **Step 2: smoke.py 작성** — 3모델 로드 + 512-token forward + MC 진단 확인

```python
"""GPU smoke: 로드 → forward → diagnostics 확인. sbatch로 실행."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc

tok = load_mc.load_tokenizer()
ids = torch.tensor([tok("The grass is green. " * 120).input_ids[:512]], device="cuda")
for kind in ("vanilla-5B", "mc-5B", "mc-30B"):
    model = load_mc.load_model(kind)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    assert logits.shape[:2] == (1, 512), logits.shape
    layers = load_mc.mc_layers(model)
    print(f"[smoke] {kind}: logits ok, mc_layers={len(layers)}")
    if layers:
        _, attn = layers[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.transformer.wte(ids)
            out, diag = attn.forward_with_diagnostics(model.transformer.h[0].norm_1(h))
        print(f"[smoke] diag route_indices {tuple(diag.route_indices.shape)} "
              f"n_seg={(512 + 255)//256}")
        assert diag.route_indices.shape == (1, 512, 2)
    del model; torch.cuda.empty_cache()
print("[smoke] ALL OK")
```

- [ ] **Step 3: sbatch/smoke.sbatch 작성**

```bash
#!/usr/bin/env bash
#SBATCH -J mc_smoke
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/smoke_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/smoke_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh
$PY "$ANA/smoke.py"
```

- [ ] **Step 4: 제출 및 확인**

Run: `sbatch lmr/analysis/260725_mc_niah_analysis/sbatch/smoke.sbatch` 후 로그 tail
Expected: `[smoke] ALL OK`. 실패 시(예: fused import, triton sm_120) 로그 첫 traceback을 보고 — flash-attn 계열 import 실패면 `sh_routing` env로 PY 교체 재시도, 그래도 실패면 사용자 보고.

- [ ] **Step 5: 커밋**

```bash
git add lmr/analysis/260725_mc_niah_analysis/
git commit -m "mc-niah: model loader + GPU smoke (3 ckpts load & forward)"
```

---

### Task 3: Dataset A 준비 + gold-chunk annotation (CPU 테스트)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/data.py`
- Test: `tests/lmr/test_mc_niah_data.py`

**Interfaces:**
- Produces: `annotate(input_text, tokenizer, chunk=256) -> dict(query_key, needles=[{key,value,tok_start,tok_end,seg}], gold_seg:int, n_seg:int)`; CLI `data.py prepare-a` → `$MC_OUT/data/2048/{niah_single_1,niah_multikey_1}/validation.jsonl` (50샘플, TinyLlama 토크나이저)

- [ ] **Step 1: 실패하는 annotation 테스트 작성**

```python
# tests/lmr/test_mc_niah_data.py
import importlib.util, os, sys
import pytest

ANA = os.path.join(os.path.dirname(__file__), "..", "..",
                   "lmr", "analysis", "260725_mc_niah_analysis")
sys.path.insert(0, os.path.abspath(ANA))
import data as mcdata  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


def test_annotate_finds_gold_needle(tok):
    filler = "The grass is green. The sky is blue. " * 120
    needle = "One of the special magic numbers for apple-pie is: 7301562."
    text = (filler[:2000] + " " + needle + " " + filler[2000:4000]
            + "\nWhat is the special magic number for apple-pie mentioned in the provided text?")
    ann = mcdata.annotate(text, tok)
    assert ann["query_key"] == "apple-pie"
    assert len(ann["needles"]) == 1
    n = ann["needles"][0]
    assert n["value"] == "7301562"
    # 토큰 위치가 실제 needle 문자 위치와 일치하는 segment를 가리킴
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    char_pos = text.find("7301562")
    tok_idx = next(i for i, (s, e) in enumerate(enc.offset_mapping) if s <= char_pos < e)
    assert n["seg"] == tok_idx // 256 == ann["gold_seg"]


def test_annotate_multikey_picks_queried(tok):
    needles = [f"One of the special magic numbers for key-{i} is: 100000{i}." for i in range(4)]
    filler = "The grass is green. " * 60
    text = (" ".join([filler, needles[0], filler, needles[1], filler, needles[2],
                      filler, needles[3], filler])
            + "\nWhat is the special magic number for key-2 mentioned in the provided text?")
    ann = mcdata.annotate(text, tok)
    assert ann["query_key"] == "key-2"
    assert len(ann["needles"]) == 4
    assert ann["gold_seg"] == next(n["seg"] for n in ann["needles"] if n["key"] == "key-2")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `PY=/data2/sohyung/conda-envs/sh_infocap/bin/python; HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_data.py -x -q`
Expected: FAIL (`data` 모듈/`annotate` 없음)

- [ ] **Step 3: data.py 작성 (annotate + prepare-a)**

```python
"""Dataset A(RULER 표준) 준비 + needle annotation. GPU 불필요."""
import argparse, json, os, re, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
TOKENIZER = "TinyLlama/TinyLlama_v1.1"
CHUNK = 256

NEEDLE_RE = re.compile(r"One of the special magic numbers? for ([\w-]+) is:? (\d+)")
QUERY_RE = re.compile(r"What (?:is|are all) the special magic numbers? for ([\w-]+)")


def annotate(input_text, tokenizer, chunk=CHUNK):
    """needle들의 토큰 위치·segment와 질의 key를 식별."""
    q_matches = QUERY_RE.findall(input_text)
    if not q_matches:
        raise ValueError("no query found")
    query_key = q_matches[-1]
    enc = tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.offset_mapping

    def tok_at(char_pos):
        return next(i for i, (s, e) in enumerate(offsets) if s <= char_pos < e)

    needles = []
    for m in NEEDLE_RE.finditer(input_text):
        key, value = m.group(1), m.group(2)
        ts, te = tok_at(m.start(2)), tok_at(m.end(2) - 1)
        needles.append({"key": key, "value": value,
                        "tok_start": ts, "tok_end": te, "seg": te // chunk})
    gold = [n for n in needles if n["key"] == query_key]
    if not gold:
        raise ValueError(f"queried key {query_key} not among needles")
    return {"query_key": query_key, "needles": needles, "gold_seg": gold[0]["seg"],
            "n_seg": (len(offsets) + chunk - 1) // chunk, "n_tok": len(offsets)}


def prepare_a(num_samples=50, length=2048, tasks=("niah_single_1", "niah_multikey_1")):
    """vendored RULER prepare.py를 TinyLlama 토크나이저로 호출."""
    gen_dir = os.path.join(REPO, "src", "ruler", "gen")
    essays = os.path.join(REPO, "data", "PaulGrahamEssays.json")
    if not os.path.exists(essays):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "ruler.py"),
                        "download-essays"], check=True)
    save_dir = os.path.join(MC_OUT, "data", str(length))
    for task in tasks:
        r = subprocess.run(
            [sys.executable, "prepare.py", "--save_dir", save_dir,
             "--benchmark", "synthetic", "--task", task,
             "--tokenizer_path", TOKENIZER, "--tokenizer_type", "hf",
             "--max_seq_length", str(length), "--model_template_type", "base",
             "--num_samples", str(num_samples)],
            cwd=gen_dir, capture_output=True, text=True)
        out = os.path.join(save_dir, task, "validation.jsonl")
        assert os.path.exists(out), f"{task} prepare failed:\n{r.stderr[-2000:]}"
        n = sum(1 for _ in open(out))
        print(f"[prepare-a] {task}: {n} samples -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare-a", "prepare-b"])
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--n-pairs", type=int, default=16, help="쌍 수 per condition (prepare-b)")
    a = ap.parse_args()
    if a.cmd == "prepare-a":
        prepare_a(num_samples=a.num_samples)
    else:
        from paired_gen import prepare_b  # Task 4에서 추가
        prepare_b(n_pairs_per_cond=a.n_pairs)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_data.py -x -q`
Expected: 2 passed

- [ ] **Step 5: prepare-a 실행 (CPU, 로컬 실행 가능)**

Run: `source lmr/analysis/260725_mc_niah_analysis/env_common.sh && $PY lmr/analysis/260725_mc_niah_analysis/data.py prepare-a`
Expected: 두 태스크 각 `50 samples` 출력. `head -c 500 $MC_OUT/data/2048/niah_multikey_1/validation.jsonl`로 needle 4개 형식 눈검사. 추가 검증: annotate가 50샘플 전부에서 예외 없이 gold_seg를 찾는지 —

```bash
$PY - <<'EOF'
import json, os, sys
sys.path.insert(0, "lmr/analysis/260725_mc_niah_analysis")
import data as d
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d.TOKENIZER)
for task in ("niah_single_1", "niah_multikey_1"):
    p = os.path.join(d.MC_OUT, "data", "2048", task, "validation.jsonl")
    anns = [d.annotate(json.loads(l)["input"], tok) for l in open(p)]
    segs = [a["gold_seg"] for a in anns]
    print(task, "n=", len(anns), "gold_seg range", min(segs), max(segs),
          "n_tok max", max(a["n_tok"] for a in anns))
EOF
```
Expected: 예외 없음, n=50, n_tok ≤ 1920 부근, gold_seg 0~7 분포

- [ ] **Step 6: 커밋**

```bash
git add lmr/analysis/260725_mc_niah_analysis/data.py tests/lmr/test_mc_niah_data.py
git commit -m "mc-niah: Dataset A prepare (RULER, TinyLlama tok) + needle annotation"
```

---

### Task 4: Dataset B paired-controlled 생성기 (CPU 테스트)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/paired_gen.py`
- Modify: `tests/lmr/test_mc_niah_data.py` (테스트 추가)

**Interfaces:**
- Consumes: `data.annotate`, `data.MC_OUT`, `data.TOKENIZER`, `data.CHUNK`
- Produces: `prepare_b(n_pairs_per_cond=16, seq_len=2048, n_gen=128, seed=42)` → `$MC_OUT/data/paired/{S,D}.jsonl`. 각 줄 = `{"pair_id", "condition": "S"|"D", "variant": "single"|"multi", "input", "outputs": [value], "gold_seg", "distractor_segs": [int], "needle_key", "codist_tok_dist": int|null}` (같은 pair_id의 single/multi 연속 2줄)

- [ ] **Step 1: 실패하는 테스트 추가** (tests/lmr/test_mc_niah_data.py에 append)

```python
def test_paired_gen_invariants(tok):
    import paired_gen
    rows = paired_gen.build_pairs(tok, n_pairs=3, condition="S", seed=7) \
         + paired_gen.build_pairs(tok, n_pairs=3, condition="D", seed=7)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["condition"], r["pair_id"]), {})[r["variant"]] = r
    assert len(by_pair) == 6
    for (cond, _), pair in by_pair.items():
        s, m = pair["single"], pair["multi"]
        # 쌍 불변식: 같은 needle/answer/gold segment
        assert s["needle_key"] == m["needle_key"] and s["outputs"] == m["outputs"]
        assert s["gold_seg"] == m["gold_seg"]
        # annotate로 실측 재검증
        ann_m = mcdata.annotate(m["input"], tok)
        assert ann_m["gold_seg"] == m["gold_seg"]
        assert ann_m["query_key"] == m["needle_key"]
        assert len(ann_m["needles"]) == 4          # gold + distractor 3
        dsegs = sorted(n["seg"] for n in ann_m["needles"] if n["key"] != m["needle_key"])
        assert dsegs == sorted(m["distractor_segs"])
        if cond == "S":
            assert m["gold_seg"] in m["distractor_segs"]      # 1개는 같은 segment
            assert m["codist_tok_dist"] is not None
        else:
            assert m["gold_seg"] not in m["distractor_segs"]  # 전부 다른 segment
        ann_s = mcdata.annotate(s["input"], tok)
        assert len(ann_s["needles"]) == 1
        # 길이 제약: 전체 ≤ 2048-128
        assert ann_s["n_tok"] <= 1920 and ann_m["n_tok"] <= 1920
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_data.py::test_paired_gen_invariants -x -q`
Expected: FAIL (`paired_gen` 없음)

- [ ] **Step 3: paired_gen.py 작성**

```python
"""Dataset B: single/multi paired NIAH, distractor 배치 통제(S=같은 segment, D=다른 segment).

토큰 정밀 배치: 문장 단위로 토큰 수를 누적하며 목표 offset에 needle 삽입 후
annotate()로 실측 segment를 검증, 어긋나면 삽입점을 한 문장씩 밀며 재시도.
"""
import json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as mcdata

WORDS = ("apple bridge candle dragon engine forest guitar harbor island jungle kettle "
         "ladder magnet needle orchid pillow quartz rocket saddle temple umbrella violin "
         "walnut yonder zephyr anchor bamboo canyon dolphin ember falcon glacier").split()
NEEDLE_FMT = "One of the special magic numbers for {key} is: {value}."
TEMPLATE = ("Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards.\n"
            "{context}\n"
            "What is the special magic number for {query} mentioned in the provided text?")
CHUNK = mcdata.CHUNK


def _essay_sentences(tokenizer, budget_toks):
    essays = json.load(open(os.path.join(mcdata.REPO, "data", "PaulGrahamEssays.json")))["text"]
    sents, total = [], 0
    for s in re.split(r"(?<=[.!?]) +", essays):
        s = s.strip()
        if not s:
            continue
        n = len(tokenizer(s, add_special_tokens=False).input_ids)
        sents.append((s, n)); total += n
        if total > budget_toks * 3:
            break
    return sents


def _compose(sents, inserts, tokenizer, body_budget):
    """inserts: [(target_tok_offset, text)] 오름차순. 문장 누적으로 배치."""
    inserts = sorted(inserts)
    out, cum, ii = [], 0, 0
    for s, n in sents:
        while ii < len(inserts) and cum >= inserts[ii][0]:
            out.append(inserts[ii][1]); ii += 1
            cum += len(tokenizer(inserts[ii - 1][1], add_special_tokens=False).input_ids)
        if cum + n > body_budget:
            break
        out.append(s); cum += n
    while ii < len(inserts):          # budget 끝에 못 넣었으면 마지막에
        out.append(inserts[ii][1]); ii += 1
    return " ".join(out)


def _make_one(tokenizer, sents, rng, condition, seq_len=2048, n_gen=128, num_keys=4):
    overhead = 96                      # 템플릿+질문 여유
    body = seq_len - n_gen - overhead  # ≈1824 tokens
    n_seg_body = body // CHUNK         # 7 — gold는 1..n_seg_body-2에서 선택
    keys = rng.sample(WORDS, num_keys)
    vals = [str(rng.randint(1000000, 9999999)) for _ in range(num_keys)]
    gold_key, gold_val = keys[0], vals[0]

    for attempt in range(8):
        gold_seg = rng.randint(1, n_seg_body - 2)
        others = [s for s in range(1, n_seg_body - 1) if s != gold_seg]
        if condition == "S":
            d_segs = [gold_seg] + rng.sample(others, num_keys - 2)
        else:
            d_segs = rng.sample(others, num_keys - 1)
        gold_off = gold_seg * CHUNK + rng.randint(16, CHUNK - 96)
        ins = [(gold_off, NEEDLE_FMT.format(key=gold_key, value=gold_val))]
        codist_dist = None
        for k, v, s in zip(keys[1:], vals[1:], d_segs):
            if s == gold_seg:
                delta = rng.choice([-64, 64])
                off = min(max(s * CHUNK + 8, gold_off + delta), (s + 1) * CHUNK - 40)
                codist_dist = abs(off - gold_off)
            else:
                off = s * CHUNK + rng.randint(16, CHUNK - 96)
            ins.append((off, NEEDLE_FMT.format(key=k, value=v)))

        ctx_multi = _compose(sents, ins, tokenizer, body)
        ctx_single = _compose(sents, ins[:1], tokenizer, body)
        row_m = {"input": TEMPLATE.format(context=ctx_multi, query=gold_key),
                 "outputs": [gold_val]}
        row_s = {"input": TEMPLATE.format(context=ctx_single, query=gold_key),
                 "outputs": [gold_val]}
        try:
            am = mcdata.annotate(row_m["input"], tokenizer)
            asg = mcdata.annotate(row_s["input"], tokenizer)
        except ValueError:
            continue
        d_actual = sorted(n["seg"] for n in am["needles"] if n["key"] != gold_key)
        ok_cond = ((gold_seg in d_actual) if condition == "S"
                   else (am["gold_seg"] not in d_actual))
        # single/multi에서 gold가 같은 segment에 실측 배치됐는지까지 확인
        if (len(am["needles"]) == num_keys and am["gold_seg"] == asg["gold_seg"]
                and ok_cond and am["n_tok"] <= seq_len - n_gen
                and asg["n_tok"] <= seq_len - n_gen):
            meta = {"gold_seg": am["gold_seg"], "distractor_segs": d_actual,
                    "needle_key": gold_key, "condition": condition,
                    "codist_tok_dist": codist_dist}
            return {**row_s, **meta, "variant": "single"}, {**row_m, **meta, "variant": "multi"}
    raise RuntimeError(f"placement failed after 8 attempts (condition={condition})")


def build_pairs(tokenizer, n_pairs, condition, seed=42, seq_len=2048, n_gen=128):
    rng = random.Random(seed + (0 if condition == "S" else 1000))
    sents = _essay_sentences(tokenizer, seq_len)
    rows = []
    for pid in range(n_pairs):
        s, m = _make_one(tokenizer, sents, rng, condition, seq_len, n_gen)
        s["pair_id"] = m["pair_id"] = pid
        rows += [s, m]
    return rows


def prepare_b(n_pairs_per_cond=16, seq_len=2048, n_gen=128, seed=42):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mcdata.TOKENIZER)
    out_dir = os.path.join(mcdata.MC_OUT, "data", "paired")
    os.makedirs(out_dir, exist_ok=True)
    for cond in ("S", "D"):
        rows = build_pairs(tok, n_pairs_per_cond, cond, seed, seq_len, n_gen)
        p = os.path.join(out_dir, f"{cond}.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[prepare-b] {cond}: {len(rows)} rows ({n_pairs_per_cond} pairs) -> {p}")
```

essays가 dict가 아니라 다른 구조면 (`json.load(...)["text"]` 실패) 실제 파일 구조를 보고 맞출 것 — `python -c "import json; d=json.load(open('data/PaulGrahamEssays.json')); print(type(d), list(d)[:3] if isinstance(d,dict) else d[0].keys())"`.

- [ ] **Step 4: 테스트 통과 확인**

Run: `HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_data.py -x -q`
Expected: 3 passed (essays 파일 필요 — Task 3 Step 5에서 download 완료 상태)

- [ ] **Step 5: prepare-b 실행 (CPU)**

Run: `source .../env_common.sh && $PY lmr/analysis/260725_mc_niah_analysis/data.py prepare-b`
Expected: `S: 32 rows`, `D: 32 rows`

- [ ] **Step 6: 커밋**

```bash
git add lmr/analysis/260725_mc_niah_analysis/paired_gen.py tests/lmr/test_mc_niah_data.py
git commit -m "mc-niah: Dataset B paired-controlled generator (S/D placement)"
```

---

### Task 5: E0 — free-gen 평가 (`gen_eval.py`, `eval_niah.py`)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/gen_eval.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/eval_niah.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/e0.sbatch`

**Interfaces:**
- Consumes: `load_mc.load_model/load_tokenizer`
- Produces: `greedy_generate(model, ids, n_gen=128) -> list[int]`; `string_match_all(preds, refs) -> float`; `run_file(model, tok, jsonl_path, n_gen=128, variant_filter=None, tag="") -> dict(score, rows=[{index, pred, outputs, correct}])`; 결과 `$MC_OUT/results/e0_scores.json` + repo `lmr/analysis/260725_mc_niah_analysis/results/e0_scores.json`

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

- [ ] **Step 2: eval_niah.py 작성**

```python
"""E0: 3 모델 × {niah_single_1, niah_multikey_1} @2048 앵커 재현."""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, gen_eval

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
REPO_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["vanilla-5B", "mc-5B", "mc-30B"])
    ap.add_argument("--tasks", nargs="+",
                    default=["niah_single_1", "niah_multikey_1"])
    a = ap.parse_args()
    tok = load_mc.load_tokenizer()
    results = {}
    out_path = os.path.join(MC_OUT, "results", "e0_scores.json")
    if os.path.exists(out_path):
        results = json.load(open(out_path))          # 재실행 시 skip-existing
    for kind in a.models:
        model = load_mc.load_model(kind)
        for task in a.tasks:
            key = f"{kind}/{task}"
            if key in results:
                print(f"[skip] {key}"); continue
            p = os.path.join(MC_OUT, "data", "2048", task, "validation.jsonl")
            results[key] = gen_eval.run_file(model, tok, p, tag=key)
            print(f"[E0] {key}: {results[key]['score']}")
            json.dump(results, open(out_path, "w"), indent=1)
        del model; torch.cuda.empty_cache()
    os.makedirs(REPO_RES, exist_ok=True)
    slim = {k: {"score": v["score"], "n": v["n"]} for k, v in results.items()}
    json.dump(slim, open(os.path.join(REPO_RES, "e0_scores.json"), "w"), indent=1)
    print(json.dumps(slim, indent=1))
```

- [ ] **Step 3: sbatch/e0.sbatch 작성** (smoke.sbatch 복제, `-J mc_e0`, `-t 05:50:00`, 실행줄만 `$PY "$ANA/eval_niah.py"`)

- [ ] **Step 4: 제출·확인**

Run: `sbatch .../sbatch/e0.sbatch`; 로그와 `results/e0_scores.json` 확인
Expected: 6개 셀 완주. **앵커 판정**: vanilla single≈88/multikey≈20, mc-5B single≈60/multikey≈2 (±15pt 허용 — 생성 경로 차이 감안). mc-30B는 신규 수치. 크게 어긋나면(예: vanilla multikey 0 또는 mc single 20) **E1 진행 전 사용자 보고** (spec §7).

- [ ] **Step 5: 커밋**

```bash
git add lmr/analysis/260725_mc_niah_analysis/
git commit -m "mc-niah: E0 free-gen anchor eval (3 models x 2 tasks @2k)"
```

---

### Task 6: E1 — routing 정확도 (`routing_stats.py`)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/routing_stats.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/e1.sbatch`

**Interfaces:**
- Consumes: `load_mc`, `data.annotate`, Dataset A/B jsonl, `$MC_OUT/results/e0_scores.json` (per-sample correctness)
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
- E0 `rows`에서 같은 index의 `correct`를 붙여, "best layer 기준 hit인 샘플의 정답률 vs miss인 샘플의 정답률" 교차표
- matplotlib로 layer(x) × hit@2(y) 곡선을 태스크·조건별 오버레이 → `results/e1_routing.png`; 수치 전체 `results/e1_routing.json`

- [ ] **Step 2: sbatch/e1.sbatch 작성·제출** (`-t 02:00:00`; 실행줄 `$PY "$ANA/routing_stats.py" --model mc-5B && $PY "$ANA/routing_stats.py" --model mc-30B`)

Expected: json에 single hit@2 高(≥0.8 예상, report의 S-NIAH 우세와 부합) vs multikey/D 조건 저하 여부가 드러남. 판정 기준이 아니라 **측정**이므로 수치 자체가 산출물.

- [ ] **Step 3: 커밋** (`mc-niah: E1 per-layer routing accuracy (A + S/D)`)

---

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

### Task 9: 종합 보고서 + 메모리 갱신

**Files:**
- Create: `report/00XX.md` (작성 시점 `report/`의 최대 번호+1; 현재 0023이 최신이므로 충돌 없으면 0024)
- Create: `report/00XX_figs/` (e1/e3 png 복사)
- Modify: `/home/sohyung/.claude/projects/-home-sohyung/memory/` (프로젝트 메모리 1건)

- [ ] **Step 1: 결과 종합** — e0/e1/e2/e3 json을 모두 읽고 spec §4 판정표의 각 행에 실측값 대입. 보고서 구성: (1) 앵커 재현표, (2) E1 layer별 routing 곡선 + S/D 분해, (3) E2 oracle 회복표 (S/D × baseline/oracle 4셀), (4) E3 fidelity 프로파일, (5) 판정표 결론 + 후속 memory-routing 알고리즘 시사점, (6) 재현 커맨드 (sbatch 4개 + 데이터 준비 2개, pinned commit 해시 명기)

- [ ] **Step 2: report/README.md 인덱스에 한 줄 추가, 커밋**

```bash
git add report/
git commit -m "report: MC-SSC multi-NIAH failure decomposition (write/read/route/gen)"
```

- [ ] **Step 3: 메모리 파일 작성** — `mc-niah-analysis.md` (type: project): 브랜치·pinned commit·판정 결론·미완 항목. `MEMORY.md`에 한 줄 추가.

- [ ] **Step 4: 최종 검증** — `superpowers:verification-before-completion` 체크: 전 테스트 재실행, sbatch 로그 에러 스캔, 판정표의 모든 셀에 근거 수치 존재 확인.

---

## Self-Review 결과

- **Spec coverage**: E0→Task5, E1→Task6, E2→Task7(anti-oracle 제외 확인), E3→Task8(S/D 통제는 Task4), 보고서·판정표→Task9. 리스크 4건 모두 반영(smoke=Task2, 앵커 gate=Task5 Step4, config 검증=Task1 Step1, 재프리필 비용=n_gen 조정 여지).
- **Placeholder scan**: 실코드 블록 제공; 드라이버 `__main__` 2곳(Task6/8)은 집계 로직을 산문으로 명세(입출력·차원 명시)했고 계산 코어는 코드로 제공 — 구현자가 조립 가능.
- **Type consistency**: `capture_hidden`(Task6 정의→Task8 소비), `mc_layers`/`bootstrap`(Task2→6,7,8), `annotate` 반환 스키마(Task3→4,6,8), `inject_gold` 시그니처(테스트↔구현) 일치 확인.
