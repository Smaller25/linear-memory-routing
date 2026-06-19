# SESSION HANDOFF — linear-memory-routing (as of 2026-06-18)

Everything needed to resume. **Read `report/README.md` for the narrative; this file is the
operational layer** (state, how to run, hard-won gotchas, next steps).

## 0. TL;DR
Add a small **trained read-out router over cached recurrent-state checkpoints** to a *frozen*
pretrained linear-recurrent LM (Mamba2 / Gated DeltaNet). A hard top-k router (**SSC**) lets the
frozen model **exceed its fixed-state recall where the state saturates** (long-context single-needle)
— ~30M router, no backbone retrain. **Validated; PR #2 open.**

## 1. Where things live
- **Repo:** `Smaller25/linear-memory-routing` (private; a fork of flash-linear-attention + `lmr/`).
- **Working branch:** `gdn-base-and-mechanisms` (ALL work). **PR #2 → main is open** (PR #1 closed,
  superseded). **GitHub Actions DISABLED** on the repo (FLA's inherited CI was failing — keep it off;
  pushing to the branch triggers no CI).
- **Auth:** token at `/root/smaller/.gh_token_new` (⚠️ was pasted in chat — revoke/reissue). git
  credential helper already points at it for this repo.
- Reports: `report/0001`…`0011` + `report/README.md` (index). Design: `notes/method-mocm-design.md`.

## 2. Key results
| metric | result |
|--------|--------|
| passkey @8k net win (SSC vs vanilla) | mamba2-1.3b 0.738→0.986 (+0.25); **2.7b 0.500→1.000 (+0.50)** — grows w/ size |
| RULER `niah_single_1` (SSC zero-shot, passkey-trained) | 4k 0.71→0.89, 8k 0.75→0.82 — transfers to the standard benchmark |
| mechanism cmp @8k (only SSC generalizes) | SSC 0.986 vs AoM 0.014, MoM 0.053, GRM 0.0, RM 0.0 |
| top-k sweep @8k | k=1 collapses (0.05); k=2/4/8 strong (k=4/8→1.0); sweet spot k∈[2,8] |
| GDN-1.3b vanilla passkey | long-context-robust (8k≈0.93 vs mamba2 0.74); collapses only 16k(0.77)/32k(0.59) |

## 3. Honest scope / negative results (important — don't re-discover)
- **Multi-key / MQAR = out of scope.** SSC does NOT solve it; training the router on multikey (aux
  1e-2 & 1e-4) did NOT converge. Segment-level temporal caching can't disambiguate keys clustered in
  a segment — it's the *interference* regime (parallel axis / MoM), not *saturation* (report 0010).
- **Not constant-memory.** Win needs ~all O(N) segment snapshots; capping the cache to B degrades
  recall ∝ B/N (an evicted needle is unrecoverable). top-k makes the *read* O(N·k) but the *cache*
  stays O(N) → SSC is a compressed-cache point on the RNN↔attention spectrum (report 0011).
- **Dense/merge read-outs fail** (AoM, MoM-slot-merge, hierarchical) — only hard top-k generalizes.

## 4. Environment & how to run (resume checklist)
**One-shot, GPU-aware setup: `bash scripts/setup_env.sh`** — auto-detects the GPU (sm_80 A100 /
sm_90 H100 / sm_120 Blackwell = RTX PRO 6000) and installs the right deps, branching only where the
hardware actually differs: `TORCH_CUDA_ARCH_LIST` for source builds, and tilelang on/off (A100/H100
yes; Blackwell+py3.13 skipped — tilelang crashes on import there). `vessl_run.sh` now calls it.
Knobs: `GPU=a100|h100|blackwell` (force target), `BACKBONE=mamba2` (also source-build the mamba
CUDA kernels), `WITH_TILELANG=1` (force-attempt). The cu128 torch wheel covers all three arches.

**Pro 6000 server (Slurm + conda).** This server is Slurm-managed and requires a dedicated conda
env; by owner convention the env is **`sh_routing`** and all custom vars are `SH_*`. Two scripts:
- `bash scripts/sh_env_pro6000.sh` — run ONCE on the login node: creates conda env `sh_routing`
  (`SH_PY`, default 3.11; lands in `~/.conda/envs`) and populates it via `setup_env.sh` (GPU=blackwell).
- `sbatch scripts/sh_slurm_run.sh [cmd...]` — all GPU work goes through Slurm (partition `main`,
  `--gres=gpu:rtx6000:N`, 6h cap). No args → runs the GDN correctness gate; else runs the given
  command inside `sh_routing`. e.g. `sbatch scripts/sh_slurm_run.sh python -m lmr.scripts.eval_long ...`.

Underlying env (what the script installs): torch 2.9.1+cu128, **transformers 5.12**, **fla = the
in-repo `fla/`** (use `PYTHONPATH=.`; do NOT pip-install a different fla), `mamba_ssm`/`causal_conv1d`
`.post1` source-built (only for a mamba2 backbone), `tilelang 0.1.9` (A100/H100 only — see gotchas),
`datasets`, `tokenizers`, `sentencepiece`. The validated runs were on A100 (py3.10); the current box
is 2× RTX PRO 6000 Blackwell (96GB, sm_120, py3.13) — GDN here is forward/eval-only (tilelang gap).
Always run from repo root with `PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Core commands (mamba2 = gpt-neox tok; GDN = Mistral tok, auto):
```
# convert/verify pretrained mamba2 -> FLA (logit-match)
python -m lmr.scripts.convert_mamba2 --repo state-spaces/mamba2-1.3b
# train an SSC router (frozen backbone), then eval vanilla/RM/SSC
python -m lmr.scripts.train_grm_passkey --arch mamba2 --variant ssc --topk 4 --low-rank-dim 64 \
    --train-len 2048 --batch 2 --steps 250 --eval-lengths 512 2048 4096
# long-context eval (memory-light: lm_head only at labelled positions; avoids OOM at >=8k)
python -m lmr.scripts.eval_long --arch mamba2 --heads ckpt/ssc_k4_mamba13b.pt --variant ssc \
    --topk 4 --low-rank-dim 64 --lengths 512 2048 4096 8192
# RULER (standard): prepare data then eval (teacher-forced, single-answer tasks)
python scripts/ruler.py prepare --lengths 4096 8192 --tasks niah_single_1,niah_multikey_2
python -m lmr.scripts.eval_ruler --heads ckpt/ssc_k4_mamba13b.pt --variant ssc --topk 4 \
    --low-rank-dim 64 --tasks niah_single_1 --lengths 4096 8192
```
Checkpoints (`ckpt/*.pt`) and data are gitignored → retrain/regenerate (SSC train ~25 min; eval at
8k is slow — use fast n=16 via score_hidden for long lengths).

## 5. Gotchas (hard-won)
- **GDN router TRAINING is blocked** on our HW. GDN-1.3b has head_dim=256 → chunk-backward needs
  ~225KB shared mem > A100's 167KB (OOM); on Hopper+Triton≥3.4 the kernel is miscomputed (fla #640)
  and needs tilelang, which **crashes on import in the py3.13 container** (TVM-FFI). `fused_recurrent`
  has no backward; naive is too slow. → GDN is **forward/RM-only**; trained-GDN awaits a working
  tilelang/H100 or a differentiable chunked scan. (FROM-SCRATCH GDN with head_dim=64 trains fine.)
- **MQAR has a delayed phase transition (~2000 steps)** — loss sits at random (ln(vocab/2)≈8.3) then
  drops sharply. Earlier ≤1500-step runs looked "stuck" but weren't. Run ≥3000 steps. `lmr/tasks/
  mqar.py` is now **Zoology-faithful** (the old format was unlearnable).
- **transformers 5.12 fix**: `_tied_weights_keys` must be a *dict* (we patched fla mamba2 + GDN).
- **mamba2 vocab pad**: checkpoint embedding is padded to a mult. of 16 (50277→50288); converter
  derives vocab from the embedding tensor.
- **GDN-1.3b load**: `lmr/loaders.load_gdn` splits the old fused `mlp.gate_proj`→gate+up, drops
  legacy `attn.D`, and uses the **Mistral** tokenizer (repo ships none).
- **Eval OOM at ≥8k**: don't materialise full-vocab logits over all segments — use
  `run_segmented_lm(return_hidden=True)` + `score_hidden` (lm_head only at labelled positions).
- **SSC/MoM aux**: the Switch load-balance aux is summed over layers and can dominate the loss
  (blocks selectivity). Keep `--aux-scale` small (~1e-4) for selective tasks.
- **Fine chunk OOMs**: the read-out stacks all cached contributions; small chunk → many segments →
  OOM. Don't go below chunk 256 without a streaming read-out.

## 6. Future work (next session)
1. **MoCM (parallel × temporal, from-scratch)** for the multi-key/interference regime SSC can't reach
   — design `notes/method-mocm-design.md`; WIP `lmr/layers/mocm.py` + `lmr/scripts/train_mocm_mqar.py`
   (MoCM mixer runs; from-scratch MQAR validated learnable, needs ≥3000 steps). ⚠️ flagged concerns:
   cost (parallel M× + temporal O(N) cache) and incremental novelty — sharpen before investing
   (e.g., bounded cache w/ *consolidation*, not naive merge).
2. **Bounded-memory via consolidation/landmark write-back** (naive cap & hierarchical merge both
   fail) — the only way to a constant-memory SSC.
3. **GDN router training** once tilelang/H100 works (`scripts/vessl_run.sh` is ready) or a
   differentiable chunked GDN scan.
4. More **RULER single-needle** tasks + length sweep (16k/32k); full RULER free-gen metric for
   publication-grade numbers.
