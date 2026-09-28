# Running GDN-2 on a single A100

Everything needed is in this file. Setup to a first training step is about an
hour, most of it downloading.

The code is [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2),
which is public. Everything below was verified against a fork of it, so if your
copy is a different vintage, check the two call sites this file names rather
than trusting the advice blind.

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

Any CUDA 12.x image with torch >= 2.7 should work; nothing is pinned to 2.9.
The setup script below prints what it ends up with, so compare before trusting
a divergent result.

Two directories matter and they are not interchangeable. The code and every
cache go on **container-local disk** — network mounts are slow for caches and
unreliable for git. Checkpoints and results go on a mount that **outlives the
container**. A pod that kept checkpoints locally took 22 GPU-hours of training
with it when it stopped.

## flash-linear-attention must be pinned to 4b02d15d

This is the one thing that will stop you before anything else. GDN-2 chunk
kernels of this lineage call

```python
chunk_gla_fwd_o_gk(..., use_exp2=True, transpose_state_layout=...)
```

Neither pypi `flash-linear-attention==0.5.1` nor current upstream has those
keyword arguments. Commit **`4b02d15d`** does. Without it every forward pass
dies about 15 seconds in with

```
TypeError: chunk_gla_fwd_o_gk() got an unexpected keyword argument 'use_exp2'
```

which reads like a bug in your code and is not one. The call site is
`lit_gpt/gdn2_ops/chunk_gdn2.py`, in `chunk_gdn2_fwd`. Check it first: if it
does not pass `use_exp2`, you do not need the pin and a current release is
fine.

If you reinstall fla by hand, delete `site-packages/fla` first. A stale
`fla/utils.py` left beside the `fla/utils/` package produces a circular import
whose traceback says nothing about versions.

## Setup

Save as `setup_a100.sh` and run it.

```bash
#!/usr/bin/env bash
set -uo pipefail

PY="${PY:-$(command -v python3)}"
PERSIST="${PERSIST:-/root/smaller}"   # must survive the container
CACHE="${CACHE:-/root/cache}"         # container-local is fine
export PIP_ROOT_USER_ACTION=ignore

echo "=== [1] deps ($(date +%T)) ==="
$PY -m pip install -q --no-input \
  transformers datasets torchdata lightning einops numpy \
  huggingface_hub wandb 2>&1 | tail -3

FLA_PIN="${FLA_PIN:-4b02d15d}"
echo "=== [2] flash-linear-attention @ ${FLA_PIN} ==="
$PY -m pip uninstall -q -y flash-linear-attention 2>/dev/null
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE}/fla"          # stale fla/utils.py -> circular import
$PY -m pip install -q --no-input --no-deps \
  "git+https://github.com/fla-org/flash-linear-attention@${FLA_PIN}" 2>&1 | tail -3

echo "=== [3] layout ==="
mkdir -p "${PERSIST}" "${CACHE}"/{hf,triton,wandb}
export HF_HOME="${CACHE}/hf" TRITON_CACHE_DIR="${CACHE}/triton"
echo "  persistent: ${PERSIST}   caches: ${CACHE}"

echo "=== [4] versions ==="
$PY - <<'PY'
import importlib, torch
print("  torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
for m in ("triton","fla","transformers","lightning","datasets","torchdata","wandb"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m} MISSING {type(e).__name__}")
PY

echo "=== [5] the kwargs that decide whether anything runs ==="
$PY - <<'PY'
import inspect
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
p = inspect.signature(chunk_gla_fwd_o_gk).parameters
bad = False
for k in ("use_exp2", "transpose_state_layout"):
    ok = k in p
    bad |= not ok
    print(f"  chunk_gla_fwd_o_gk.{k}: {'OK' if ok else 'MISSING'}")
if bad:
    raise SystemExit("wrong flash-linear-attention -- nothing will run")
PY
echo "=== SETUP DONE ($(date +%T)) ==="
```

Add to your environment before any run:

```bash
export HF_HOME=$CACHE/hf
export TRITON_CACHE_DIR=$CACHE/triton
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
```

## FineWeb-Edu

Save as `fetch_fwedu.sh`. No GPU needed; run it while the box is idle. Six
training shards is 13 GB and enough for a couple of billion tokens.

```bash
#!/usr/bin/env bash
set -uo pipefail
PY="${PY:-$(command -v python3)}"
PERSIST="${PERSIST:-/root/smaller}"
export HF_HOME="${HF_HOME:-/root/cache/hf}"
export TOKENIZERS_PARALLELISM=false
TRAIN_SHARDS="${TRAIN_SHARDS:-6}"     # 2.15 GB each

PERSIST="$PERSIST" TRAIN_SHARDS="$TRAIN_SHARDS" $PY - <<'PY'
import os, shutil, glob
from huggingface_hub import hf_hub_download
ROOT = os.path.join(os.environ["PERSIST"], "fwedu")
n = int(os.environ["TRAIN_SHARDS"])
TRAIN = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(n)]
VAL   = ["sample/10BT/013_00000.parquet"]      # disjoint from train
for files, sub in ((TRAIN, "train/10BT"), (VAL, "val")):
    d = os.path.join(ROOT, sub); os.makedirs(d, exist_ok=True)
    for f in files:
        dest = os.path.join(d, os.path.basename(f))
        if os.path.exists(dest) and os.path.getsize(dest) > 1e8:
            print(f"  skip {dest}", flush=True); continue
        p = hf_hub_download("HuggingFaceFW/fineweb-edu", f, repo_type="dataset")
        shutil.copyfile(p, dest + ".tmp"); os.replace(dest + ".tmp", dest)
        print(f"  got {dest} ({os.path.getsize(dest)/1e9:.2f} GB)", flush=True)
print("  train glob:", len(glob.glob(os.path.join(ROOT, "train", "*", "*.parquet"))), f"(want {n})")
print("  val   glob:", len(glob.glob(os.path.join(ROOT, "val", "*.parquet"))), "(want 1)")
PY
echo "=== FETCH DONE ($(date +%T)) ==="
```

Two things about the layout. Check your loader's glob before trusting the
download: `data.py` upstream matches `["*.parquet", "**/*.parquet"]` and takes
any depth, but forks of it hardcode `{dir}/*.parquet` and `{dir}/*/*.parquet`.
Against a hardcoded pattern, letting `hf_hub_download` keep the repo-relative
path puts the shards two levels too deep and the glob silently comes back
empty, so the script above writes them at a fixed depth and prints the counts.
Separately, train and val must be disjoint shards; pointing val at a
subdirectory of train makes the reported perplexity a training-set number.

Published GDN-2 370M checkpoints, if you want a baseline to compare against:

```python
from huggingface_hub import hf_hub_download
hf_hub_download("LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched", "model.pth")
hf_hub_download("LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla", "checkpoint-5B-model-ckpt.pth")
```

## Training

The recipe the 370M and 50M runs used, if you want to match it: global batch
128 sequences x 4096 tokens = 524,288 per optimizer step, LR 4e-4, warmup 1%,
TinyLlama tokenizer, FineWeb-Edu, Chinchilla budget of 20 tokens per total
parameter.

`micro_batch_size 2` peaks near 12 GB against 19.6 GB at 4, which leaves room
for a co-tenant on the same GPU without either run dying. Global batch is
unchanged — gradient accumulation absorbs it. Throughput on an otherwise idle
A100 is about 15K tokens/s for a 50M model; a co-tenant drops it to 12K.

A 50M model on FineWeb-Edu starts at loss 10.54 and reaches about 3.3 by 1.5B
tokens. If the first hundred iterations are not falling off 10.5, stop and
find out why rather than spending a day on it.

wandb is optional and worth wiring before a long run, not after. Write the key
to a file, `chmod 600`, and read it in the launcher rather than passing it on
the command line where it reaches process listings and shell history:

```bash
if [[ -z "${WANDB_API_KEY:-}" && -r /root/.wandb_key ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < /root/.wandb_key)"; export WANDB_API_KEY
fi
export WANDB_MODE="${WANDB_API_KEY:+online}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
```

A missing key then degrades to an offline run instead of killing a day-long
job at startup.

## Things that cost us time

**Never `pgrep -f` or `pkill -f` a script name over ssh.** The pattern matches
the ssh command line carrying it, so a check finds itself and a kill finds its
own session. This bit us three times, once killing the monitoring session and
once stalling a run for 27 minutes. Write an explicit completion marker to the
log and grep for that:

```bash
# in the runner
echo "=== ARM ${MODE} DONE ($(date +%T)) rc=$? ==="
# waiting for it, from outside
until ssh "$POD" 'grep -q "ARM .* DONE" /root/train.log'; do sleep 600; done
```

**`| tail` swallows the pipeline's exit status.** A runner that pipes training
into `tail` and then tests `$?` will report success for a job that crashed.
Use `set -o pipefail`, or capture the status before the pipe.

**Save more than the latest checkpoint.** Trainers of this lineage write one
`latest-model-ckpt.pth` and replace it at every save, so there are no
intermediates unless you add them. To evaluate mid-training, copy the file
first — reading the live one can catch a half-written save.

**The checkpoint is a dict, not a state dict.** `torch.load` gives
`{model, optimizer, hparams, iter_num, step_count}` with the weights under
`"model"`. Load with `sd.get("model", sd)` so both shapes work. A wrong
checkpoint-to-config pairing should hard-fail rather than load partially: a
partially loaded model still produces plausible numbers.

**Validation over several lengths is cumulative.** If your trainer reports
`1x` through `4x`, those are average losses over tokens `0..4096`, `0..8192`,
`0..12288`, `0..16384`, not four independent windows. To read a single window
you have to difference them: the `8192..12288` window is
`3*L(0..12288) - 2*L(0..8192)`.

**Pull results off the box as they appear.** Ephemeral containers take
everything. FineWeb-Edu and published checkpoints are re-downloadable; a
training run is not.
