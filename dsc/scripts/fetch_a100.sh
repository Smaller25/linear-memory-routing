#!/usr/bin/env bash
# Everything the experiments read: pretrained checkpoints, FineWeb-Edu shards,
# and the diverse-key NIAH evaluation data.
#
# Nothing here needs a GPU. Run it while the box is idle.
set -uo pipefail

PY="${PY:-$(command -v python3)}"
REPO="${REPO:-$PWD}"
PERSIST="${PERSIST:-/root/smaller/mc}"
export HF_HOME="${HF_HOME:-/root/cache/hf}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO}:${REPO}/dsc:${REPO}/src:${PYTHONPATH:-}"

WANT_CKPTS="${WANT_CKPTS:-1}"
WANT_FWEDU="${WANT_FWEDU:-1}"
WANT_NIAH="${WANT_NIAH:-1}"

if [[ "$WANT_CKPTS" == "1" ]]; then
echo "=== [1] 370M checkpoints (7.3 GB) ($(date +%T)) ==="
PERSIST="$PERSIST" $PY - <<'PY'
import os
from huggingface_hub import hf_hub_download
D = os.path.join(os.environ["PERSIST"], "ckpts")
WANT = [
    ("mc-30B",      "LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool",
     "checkpoint-30B-model-ckpt.pth"),
    ("vanilla-30B", "LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched",
     "model.pth"),
    ("vanilla-5B",  "LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla",
     "checkpoint-5B-model-ckpt.pth"),
]
for tag, repo, fname in WANT:
    dest = os.path.join(D, repo.replace("/", "_"), fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1e8:
        print(f"  skip {tag}", flush=True); continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        p = hf_hub_download(repo, fname)
        # os.replace across filesystems raises; copy then unlink.
        import shutil; shutil.copyfile(p, dest + ".tmp"); os.replace(dest + ".tmp", dest)
        print(f"  got {tag:<12} {os.path.getsize(dest)/1e9:.2f} GB", flush=True)
    except Exception as e:
        print(f"  FAIL {tag}: {type(e).__name__}: {e}", flush=True)
PY
fi

if [[ "$WANT_FWEDU" == "1" ]]; then
echo "=== [2] FineWeb-Edu shards (13 GB) ($(date +%T)) ==="
# pretrain.py globs train as {dir}/*/*.parquet and val as {dir}/*.parquet, so
# the files must land at exactly those depths. hf_hub_download preserves the
# repo-relative path, which puts them two levels too deep; write them by hand.
PERSIST="$PERSIST" $PY - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
ROOT = os.path.join(os.environ["PERSIST"], "fwedu")
TRAIN = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(6)]   # 000-005
VAL   = ["sample/10BT/013_00000.parquet"]                          # disjoint
for files, sub in ((TRAIN, "train/10BT"), (VAL, "val")):
    d = os.path.join(ROOT, sub); os.makedirs(d, exist_ok=True)
    for f in files:
        dest = os.path.join(d, os.path.basename(f))
        if os.path.exists(dest) and os.path.getsize(dest) > 1e8:
            print(f"  skip {dest}", flush=True); continue
        p = hf_hub_download("HuggingFaceFW/fineweb-edu", f, repo_type="dataset")
        shutil.copyfile(p, dest + ".tmp"); os.replace(dest + ".tmp", dest)
        print(f"  got {dest} ({os.path.getsize(dest)/1e9:.2f} GB)", flush=True)
import glob
print("  train glob:", len(glob.glob(os.path.join(ROOT, "train", "*", "*.parquet"))), "(want 6)")
print("  val   glob:", len(glob.glob(os.path.join(ROOT, "val", "*.parquet"))), "(want 1)")
PY
fi

if [[ "$WANT_NIAH" == "1" ]]; then
echo "=== [3] diverse-key NIAH data ($(date +%T)) ==="
# Seeded, so this reproduces the same cells byte for byte. ~20 s per cell.
# 42/43/44 is the evaluation protocol; 45 is the router-training seed and
# must never appear in an evaluation.
$PY -u "${REPO}/dsc/scripts/gen_diverse_key_niah.py" \
    --data-root "${PERSIST}/dk_data" \
    --lengths 8192 --needles 4 16 --seeds 42 43 44 \
    --haystack essay --num-samples 50 \
    --tokenizer TinyLlama/TinyLlama_v1.1
fi

echo "=== FETCH DONE ($(date +%T)) ==="
du -sh "${PERSIST}"/* 2>/dev/null
