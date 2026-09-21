# `dsc/` — MC-GDN2 backbone and the diverse-key routing harness

Ported from `gyunggyung/long-gdn` (branch `sh_exp/diverse-key-niah-relu`, commit
`fd8762b`) so this repository stands on its own. The previous arrangement had
`lmr/analysis/260725_mc_niah_analysis/load_mc.py` push an external long-gdn
worktree onto `sys.path`; that path broke once already when the evaluation pod
was lost, taking its checkpoints with it.

Only the import closure of the routing experiments was copied, not the whole
`dsc/` tree: 89 modules against the original 400-odd, with training runs,
posters, papers and unrelated tracks left behind.

## Layout

| Path | What it is |
|---|---|
| `lit_gpt/` | model, config presets (`mc_370M`, `mc_50M`), GDN-2 layer, norms |
| `mc_baseline/` | `mc_ssc.py` (SSC equations 16-17, **never edited**), Triton read kernel, eval-only router extensions |
| `mc_gdn2/` | the `MemoryCachingGDN2Layer` wrapper around GDN-2 |
| `mc_v3/`, `mc_v4/`, `mc_v5/` | alternative read kernels, selected by `MC_KERNEL_VERSION` (default `v2` lives in `mc_baseline`) |
| `log_linear_gdn2/` | Fenwick-hierarchy variant, lazily imported by config |
| `scripts/` | evaluation, capture, router fitting, probes, verification gates, pod runners |
| `pretrain.py`, `data.py` | training entry point and the streaming FineWeb loader |

## Two rules carried over

**`mc_baseline/mc_ssc.py` is never edited.** Every routing variant is an
eval-only subclass in `mc_ssc_mlp_router.py`, attached by class name at load
time. That is what keeps "the deployed path" and "the thing under test"
distinguishable.

**Every capture script embeds a null control.** `train_mlp_router.py` refits
the original linear connector on the captured tensors and dies if the majority
of layers land outside gold-AUC 0.35-0.60. This exists because a capture hook
that returned zeros once deleted sixteen attention sublayers and produced a
plausible AUC of 0.78 that had to be withdrawn.

## Running

```bash
export PYTHONPATH=$PWD:$PWD/dsc
export MC_KERNEL_VERSION=v2

# CPU only, no checkpoint needed: 10 check groups over the routing extensions
python dsc/scripts/smoke_test_mlp_router.py
```

Checkpoints live on HuggingFace, not in this repository:

| Tag | Repo | File |
|---|---|---|
| MC-SSC 30B | `LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool` | `checkpoint-30B-model-ckpt.pth` |
| vanilla 5B | `LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla` | `checkpoint-5B-model-ckpt.pth` |
