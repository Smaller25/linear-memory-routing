#!/usr/bin/env bash
# Bring a fresh A100 box up for GDN-2 training in this repo.
#
# Layout: the git repo and every cache go on container-local disk, while
# checkpoints and results go to $PERSIST, which must survive the container.
# A pod that kept checkpoints on container-local disk lost 22 GPU-hours of
# training when it stopped.
set -uo pipefail

PY="${PY:-$(command -v python3)}"
PERSIST="${PERSIST:-/root/smaller}"
CACHE="${CACHE:-/root/cache}"
export PIP_ROOT_USER_ACTION=ignore

echo "=== [1] deps ($(date +%T)) ==="
# What pretrain.py and lit_gpt actually import, plus wandb.
$PY -m pip install -q --no-input \
  transformers datasets torchdata lightning einops numpy \
  huggingface_hub wandb 2>&1 | tail -3

# flash-linear-attention MUST be this commit. lit_gpt/gdn2_ops/chunk_gdn2.py
# calls chunk_gla_fwd_o_gk(use_exp2=..., transpose_state_layout=...). Neither
# pypi 0.5.1 nor current upstream has those kwargs; every forward pass dies
# with "unexpected keyword argument 'use_exp2'" about 15 seconds in.
FLA_PIN="${FLA_PIN:-4b02d15d}"
echo "=== [2] flash-linear-attention @ ${FLA_PIN} ==="
$PY -m pip uninstall -q -y flash-linear-attention 2>/dev/null
# A stale fla/utils.py left beside fla/utils/ causes a circular import that
# looks unrelated, so clear the directory rather than trusting pip to.
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE}/fla"
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
