#!/usr/bin/env bash
# FineWeb-Edu shards for pretraining, and optionally the published GDN-2
# 370M checkpoints. No GPU needed; run it while the box is idle.
set -uo pipefail

PY="${PY:-$(command -v python3)}"
PERSIST="${PERSIST:-/root/smaller}"
export HF_HOME="${HF_HOME:-/root/cache/hf}"
export TOKENIZERS_PARALLELISM=false

WANT_FWEDU="${WANT_FWEDU:-1}"
WANT_CKPTS="${WANT_CKPTS:-0}"     # set 1 for the published GDN-2 370M runs
TRAIN_SHARDS="${TRAIN_SHARDS:-6}" # 2.15 GB each

if [[ "$WANT_FWEDU" == "1" ]]; then
echo "=== [1] FineWeb-Edu ($(date +%T)) ==="
# pretrain.py globs train as {dir}/*/*.parquet and val as {dir}/*.parquet, so
# the files must land at exactly those depths. hf_hub_download preserves the
# repo-relative path, which puts them two levels too deep.
PERSIST="$PERSIST" TRAIN_SHARDS="$TRAIN_SHARDS" $PY - <<'PY'
import os, shutil, glob
from huggingface_hub import hf_hub_download
ROOT = os.path.join(os.environ["PERSIST"], "fwedu")
n = int(os.environ["TRAIN_SHARDS"])
TRAIN = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(n)]
VAL   = ["sample/10BT/013_00000.parquet"]   # disjoint from train
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
fi

if [[ "$WANT_CKPTS" == "1" ]]; then
echo "=== [2] published GDN-2 370M checkpoints ($(date +%T)) ==="
PERSIST="$PERSIST" $PY - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
D = os.path.join(os.environ["PERSIST"], "ckpts")
WANT = [
    ("gdn2-30B", "LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched", "model.pth"),
    ("gdn2-5B",  "LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla", "checkpoint-5B-model-ckpt.pth"),
]
for tag, repo, fname in WANT:
    dest = os.path.join(D, repo.replace("/", "_"), fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1e8:
        print(f"  skip {tag}", flush=True); continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        p = hf_hub_download(repo, fname)
        shutil.copyfile(p, dest + ".tmp"); os.replace(dest + ".tmp", dest)
        print(f"  got {tag:<10} {os.path.getsize(dest)/1e9:.2f} GB", flush=True)
    except Exception as e:
        print(f"  FAIL {tag}: {type(e).__name__}: {e}", flush=True)
PY
fi

echo "=== FETCH DONE ($(date +%T)) ==="
du -sh "${PERSIST}"/* 2>/dev/null
