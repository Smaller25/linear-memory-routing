# Running GDN-2 on a single A100

Setup to a first training step is about an hour, most of it downloading. One
A100 80GB is enough; nothing here assumes more than one GPU.

## The box

The runs this was written from were on a VESSL pod, A100-SXM4 80GB, one GPU.
**The image tag was never recorded.** What matters is the fingerprint:

| | |
|---|---|
| python | 3.13, at `/opt/conda/bin/python` |
| torch | 2.9.1+cu128 |
| triton | 3.5.1 |
| transformers | 5.17.0 |
| GPU | A100-SXM4-80GB |

Any CUDA 12.x image with torch >= 2.7 should work; nothing in the code pins
2.9. `setup_a100.sh` prints what it ends up with, so compare before trusting a
divergent result.

Two directories matter and they are not interchangeable. The repo and every
cache go on **container-local disk** — network mounts are slow for caches and
unreliable for git. Checkpoints and results go on a mount that **outlives the
container**. A pod that kept checkpoints locally took 22 GPU-hours of training
with it when it stopped.

## Setup

```bash
git clone <repo> && cd <repo>
export PERSIST=/root/smaller     # must survive the container
export CACHE=/root/cache         # container-local is fine

bash dsc/scripts/setup_a100.sh
bash dsc/scripts/fetch_a100.sh   # 13 GB of FineWeb-Edu, no GPU needed
```

`setup_a100.sh` exits non-zero if `chunk_gla_fwd_o_gk` is missing `use_exp2`
or `transpose_state_layout`. That check is the whole point of the script; see
the fla note below.

Every run needs:

```bash
export PYTHONPATH=$REPO:$REPO/dsc
```

## flash-linear-attention must be pinned

`lit_gpt/gdn2_ops/chunk_gdn2.py` calls

```python
chunk_gla_fwd_o_gk(..., use_exp2=True, transpose_state_layout=...)
```

Neither pypi `flash-linear-attention==0.5.1` nor current upstream has those
keyword arguments. Commit **`4b02d15d`** does. Without it, every forward pass
dies about 15 seconds in with

```
TypeError: chunk_gla_fwd_o_gk() got an unexpected keyword argument 'use_exp2'
```

which reads like a code bug and is not one. `setup_a100.sh` installs the pin
and asserts the signature. If you reinstall fla by hand, delete
`site-packages/fla` first — a stale `fla/utils.py` left beside `fla/utils/`
produces a circular import that looks unrelated to the version.

## Data layout

`pretrain.py` globs training shards as `{train_dir}/*/*.parquet` and
validation as `{val_dir}/*.parquet`. Files must land at exactly those depths.
`hf_hub_download` preserves the repo-relative path and puts them two levels
too deep, which silently yields an empty glob; `fetch_a100.sh` writes them by
hand instead.

Train and val shards must be disjoint. Pointing val at a subdirectory of train
makes the reported perplexity a training-set number.

Validation reports four cumulative losses, over tokens `0..4096`, `0..8192`,
`0..12288` and `0..16384`, printed as `1 x` through `4 x`
(`pretrain.py:557`). With a 4096 training length, `2x` and beyond are
extrapolation. They are cumulative averages, so to read a single window you
have to difference them.

## Training

```bash
CKPT_MODE=... EXP_NAME=myrun bash dsc/scripts/pretrain_mc_50m_chinchilla.sh
```

The script is a thin wrapper over `pretrain.py`; read it and change what you
need. The knobs that matter, all overridable by environment variable:

| variable | default | note |
|---|---|---|
| `MODEL` | `mc_50M` | a preset in `lit_gpt/config.py` |
| `MAX_TOKENS` | `1530000000` | budget; global batch is 128 x 4096 = 524,288 |
| `LR` | `4e-4` | warmup 1% |
| `MICRO_BATCH_SIZE` | `2` | grad accum absorbs it; global batch unchanged |
| `SAVE_STEP_INTERVAL` | `500` | |
| `OUTPUT_ROOT` | `/root/smaller/mc/ladder` | put this on the persistent mount |

`--config_overrides "key=value,key=value"` sets model config fields without
adding a preset. Values cast to int or float when they parse as one, so a
string-valued field stays a string.

`micro_batch_size 2` peaks near 12 GB against 19.6 GB at 4, which leaves room
for a co-tenant on the same GPU. Throughput on an otherwise idle A100 is about
15K tokens/s for a 50M model; a co-tenant drops it to roughly 12K.

wandb is optional. Write the key to `/root/.wandb_key` (chmod 600); a missing
key degrades to `WANDB_MODE=disabled` rather than killing a day-long job. The
project name is set in `pretrain.py`, not the launcher.

Model presets live in `lit_gpt/config.py`. `gdn2_370M` is the plain GDN-2
370M; `mc_50M` is a 52M-non-embedding shrink holding `ffn = 2d`,
`H * d_head = 2d`, `d_head = 128`, depth 16.

### Sanity numbers

A 50M run on FineWeb-Edu starts at loss 10.54 and reaches about 3.3 by 1.5B
tokens, with validation 3.287 / 3.254 / 3.333 / 3.410 at the four lengths. If
the first hundred iterations are not falling off 10.5, something is wrong
before you have spent a day finding out.

## Things that cost us time

**Never `pgrep -f` or `pkill -f` a script name over ssh.** The pattern matches
the ssh command line carrying it, so a check finds itself and a kill finds its
own session. This bit us three times, once killing the monitoring session and
once stalling a run for 27 minutes. Write an explicit completion marker to the
log and grep for that.

**`pretrain.py` keeps only `latest-model-ckpt.pth`**, replaced at every save,
plus `final-model-ckpt.pth` at the end. To evaluate mid-training, copy it
first — reading the live file can catch a half-written one. There are no
intermediate checkpoints unless you add them.

**The checkpoint is a dict, not a state dict.** `torch.load` on
`latest-model-ckpt.pth` gives `{model, optimizer, hparams, iter_num,
step_count}`, with the weights under `"model"`. `final-model-ckpt.pth` is a
third of the size, consistent with weights only. Load with
`sd.get("model", sd)` so either shape works.

**`| tail` swallows the pipeline's exit status.** A runner that pipes a
training command into `tail` and then checks `$?` reports success for a job
that crashed.

**Pull results off the box as they appear.** Ephemeral containers take
everything. FineWeb-Edu and published checkpoints are re-downloadable; a
training run is not.
