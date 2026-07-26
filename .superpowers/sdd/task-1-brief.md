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

