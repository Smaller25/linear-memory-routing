"""Generate RULER NIAH items with the vendored official generator (third_party/ruler).

Layout (separate folder and format from the training shards):
    EVAL_DATA_DIR/<tokenizer tag>/<length>/<task>/validation.jsonl
    EVAL_DATA_DIR/<tokenizer tag>/meta.json

Items are generated with the SAME tokenizer as training (TOKENIZER), so the
length budget is counted in the model's own tokens. A fixed EVAL_DATA_SEED
means every model is scored on identical items (paired comparison).
Haystacks are Paul Graham essays / synthetic noise, never FineWeb-Edu.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from mslm.envfile import REPO_ROOT
from mslm.settings import Settings

GEN_DIR = REPO_ROOT / "third_party" / "ruler" / "gen"


def tokenizer_tag(name: str) -> str:
    return name.strip("/").replace("/", "__")


def task_file(eval_dir: str, tokenizer: str, length: int, task: str) -> str:
    return os.path.join(eval_dir, tokenizer_tag(tokenizer), str(length), task, "validation.jsonl")


def main():
    s = Settings.from_env()
    root = os.path.join(s.str("EVAL_DATA_DIR"), tokenizer_tag(s.str("TOKENIZER")))
    train_dir = os.path.realpath(s.str("TRAIN_DATA_DIR"))
    if os.path.realpath(root).startswith(train_dir + os.sep):
        raise ValueError("EVAL_DATA_DIR must not live inside TRAIN_DATA_DIR")
    n = s.int("EVAL_NUM_SAMPLES")
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    for length in [int(x) for x in s.list("EVAL_LENGTHS")]:
        for task in s.list("EVAL_TASKS"):
            out = task_file(s.str("EVAL_DATA_DIR"), s.str("TOKENIZER"), length, task)
            if os.path.exists(out) and sum(1 for _ in open(out)) == n:
                print(f"[ruler] exists {out}")
                continue
            cmd = [sys.executable, str(GEN_DIR / "prepare.py"), "--save_dir", os.path.join(root, str(length)),
                   "--benchmark", "synthetic", "--task", task, "--tokenizer_path", s.str("TOKENIZER"),
                   "--tokenizer_type", "hf", "--max_seq_length", str(length), "--model_template_type", "base",
                   "--num_samples", str(n), "--random_seed", str(s.int("EVAL_DATA_SEED"))]
            r = subprocess.run(cmd, cwd=GEN_DIR, env=env, capture_output=True, text=True)
            # upstream prepare.py prints child errors instead of failing; check the file ourselves
            if not os.path.exists(out) or sum(1 for _ in open(out)) != n:
                sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
                raise RuntimeError(f"RULER generation failed for {task} @ {length}")
            print(f"[ruler] wrote {out}")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"tokenizer": s.str("TOKENIZER"), "seed": s.int("EVAL_DATA_SEED"), "num_samples": n,
                   "lengths": s.list("EVAL_LENGTHS"), "tasks": s.list("EVAL_TASKS"),
                   "template": "base", "generator": "third_party/ruler (NVIDIA RULER, vendored)"}, f, indent=2)


if __name__ == "__main__":
    main()
