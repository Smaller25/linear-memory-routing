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

