"""RULER long-context benchmark harness for Mamba2 / state-space models.

Wraps NVIDIA RULER's synthetic data generators + string-match metrics (vendored
under src/ruler) with a mamba-ssm generation backend. Three stages:

    prepare  generate synthetic jsonl per (task, seq_length) via vendored RULER
    predict  run mamba-ssm generation over the prepared inputs
    eval     score predictions with RULER's string-match metrics

Usage:
    python scripts/ruler.py prepare --lengths 4096 --tasks niah_single_1,vt --num-samples 50
    python scripts/ruler.py predict --lengths 4096 --tasks niah_single_1,vt
    python scripts/ruler.py eval    --lengths 4096 --tasks niah_single_1,vt
    python scripts/ruler.py all     --lengths 4096 --tasks niah_single_1,vt --num-samples 50

Notes:
  - Tokenizer for length control = EleutherAI/gpt-neox-20b (what mamba2 uses).
  - Model template = `base` (mamba2-1.3b is a base model).
  - Faithful to RULER's call_api: only the `input` field is fed to the model
    (the `answer_prefix` is metadata, NOT prepended) unless --use-answer-prefix.
  - Essay-haystack tasks need data/PaulGrahamEssays.json first
    (python scripts/ruler.py download-essays). Tasks runnable WITHOUT any
    download: niah_single_1, niah_multikey_2, niah_multikey_3, vt, cwe, fwe.
  - mamba2-1.3b was pretrained at ~2k context → long-length scores will be
    near-floor. RULER is set up here as infrastructure, not for headline numbers.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULER_DIR = os.path.join(REPO_ROOT, "src", "ruler")
GEN_DIR = os.path.join(RULER_DIR, "gen")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "ruler")

TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
MODEL_NAME = "state-spaces/mamba2-1.3b"
MODEL_TEMPLATE = "base"

# tasks that need no external download (noise/needle haystack or word lists)
NO_DOWNLOAD_TASKS = ["niah_single_1", "niah_multikey_2", "niah_multikey_3",
                     "vt", "cwe", "fwe"]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_configs():
    """Return (yaml_tasks, base_tasks, metric_tasks)."""
    import yaml
    with open(os.path.join(RULER_DIR, "synthetic.yaml")) as f:
        yaml_tasks = yaml.safe_load(f)
    base_tasks = _load_module(
        "ruler_data_constants",
        os.path.join(GEN_DIR, "synthetic", "constants.py")).TASKS
    metric_tasks = _load_module(
        "ruler_eval_metrics", os.path.join(RULER_DIR, "eval_metrics.py")).TASKS
    return yaml_tasks, base_tasks, metric_tasks


def task_dir(length, task):
    return os.path.join(DATA_ROOT, str(length), task)


# ----------------------------------------------------------------------------- prepare
def cmd_prepare(args):
    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",")
    for length in lengths:
        save_dir = os.path.join(DATA_ROOT, str(length))
        for task in tasks:
            cmd = [
                sys.executable, "prepare.py",
                "--save_dir", save_dir,
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", TOKENIZER_NAME,
                "--tokenizer_type", "hf",
                "--max_seq_length", str(length),
                "--model_template_type", MODEL_TEMPLATE,
                "--num_samples", str(args.num_samples),
            ]
            print(f"[prepare] L={length} task={task} n={args.num_samples}")
            r = subprocess.run(cmd, cwd=GEN_DIR, capture_output=True, text=True)
            out = task_dir(length, task) + "/validation.jsonl"
            if not os.path.exists(out):
                print(f"  [FAIL] {out} not created\n{r.stderr[-800:]}")
            else:
                n = sum(1 for _ in open(out))
                print(f"  [ok] {n} samples -> {out}")


# ----------------------------------------------------------------------------- predict
def cmd_predict(args):
    import torch
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    _, base_tasks, _ = load_configs()
    import yaml
    with open(os.path.join(RULER_DIR, "synthetic.yaml")) as f:
        yaml_tasks = yaml.safe_load(f)

    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",")
    device = "cuda"
    dtype = getattr(torch, args.dtype)

    print(f"[load] tokenizer={TOKENIZER_NAME}  model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    model.eval()

    for length in lengths:
        for task in tasks:
            inp_path = os.path.join(task_dir(length, task), "validation.jsonl")
            if not os.path.exists(inp_path):
                print(f"[skip] no data: {inp_path}")
                continue
            base = yaml_tasks[task]["task"]
            n_gen = base_tasks[base]["tokens_to_generate"]
            samples = [json.loads(l) for l in open(inp_path)]
            out_path = os.path.join(task_dir(length, task), "pred.jsonl")
            t0 = time.time()
            with open(out_path, "w") as fout:
                for i, s in enumerate(samples):
                    prompt = s["input"]
                    if args.use_answer_prefix and s.get("answer_prefix"):
                        prompt = prompt + s["answer_prefix"]
                    pred = generate(model, tokenizer, prompt, n_gen, device,
                                    cg=not args.no_cg)
                    fout.write(json.dumps({
                        "index": s["index"], "pred": pred,
                        "outputs": s["outputs"], "others": {},
                        "length": s.get("length", length),
                    }) + "\n")
            dt = time.time() - t0
            print(f"[predict] L={length} {task}: {len(samples)} samples, "
                  f"{n_gen} gen-toks each, {dt:.0f}s -> {out_path}")


def generate(model, tokenizer, prompt, n_gen, device, cg):
    import torch
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + n_gen,
            cg=cg, return_dict_in_generate=True, output_scores=False,
            enable_timing=False, temperature=1.0, top_k=1,  # greedy
            eos_token_id=tokenizer.eos_token_id,
        )
    seqs = out.sequences if hasattr(out, "sequences") else out
    return tokenizer.decode(seqs[0, input_ids.shape[1]:], skip_special_tokens=True)


# ----------------------------------------------------------------------------- eval
def cmd_eval(args):
    import re
    yaml_tasks, _, metric_tasks = load_configs()
    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",")

    def postprocess(s):
        return re.sub(r"[\x00-\x1f]", "\n", s.strip()).strip()

    rows = []
    for length in lengths:
        for task in tasks:
            pred_path = os.path.join(task_dir(length, task), "pred.jsonl")
            if not os.path.exists(pred_path):
                print(f"[skip] no preds: {pred_path}")
                continue
            base = yaml_tasks[task]["task"]
            metric_fn = metric_tasks[base]["metric_fn"]
            preds, refs = [], []
            for l in open(pred_path):
                d = json.loads(l)
                preds.append(postprocess(d["pred"]))
                refs.append(d["outputs"])
            score = metric_fn(preds, refs)
            nulls = sum(1 for p in preds if len(p) == 0)
            rows.append((length, task, score, f"{nulls}/{len(preds)}"))

    print("\n" + "=" * 56)
    print(f"{'length':>8}  {'task':<18} {'score':>8}  {'nulls':>8}")
    print("-" * 56)
    for length, task, score, nulls in rows:
        print(f"{length:>8}  {task:<18} {score:>8.2f}  {nulls:>8}")
    print("=" * 56)
    summary = os.path.join(DATA_ROOT, "summary.json")
    json.dump([{"length": l, "task": t, "score": s, "nulls": n}
               for l, t, s, n in rows], open(summary, "w"), indent=2)
    print(f"[saved] {summary}")


# ----------------------------------------------------------------------------- essays
def cmd_download_essays(args):
    """Build data/PaulGrahamEssays.json (needed for essay-haystack tasks)."""
    json_dir = os.path.join(GEN_DIR, "synthetic", "json")
    print("[essays] downloading Paul Graham essays (this can take a few min)...")
    r = subprocess.run([sys.executable, "download_paulgraham_essay.py"],
                       cwd=json_dir, capture_output=True, text=True)
    built = os.path.join(json_dir, "PaulGrahamEssays.json")
    if os.path.exists(built):
        n = len(json.load(open(built)).get("text", ""))
        print(f"[essays] ok -> {built} ({n} chars)")
    else:
        print(f"[essays] FAILED\n{r.stdout[-500:]}\n{r.stderr[-800:]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lengths", default="4096")
    common.add_argument("--tasks", default=",".join(NO_DOWNLOAD_TASKS))

    p = sub.add_parser("prepare", parents=[common])
    p.add_argument("--num-samples", type=int, default=50)
    p.set_defaults(func=cmd_prepare)

    for name, fn in [("predict", cmd_predict)]:
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--model", default=MODEL_NAME)
        p.add_argument("--dtype", default="bfloat16",
                       choices=["bfloat16", "float16", "float32"])
        p.add_argument("--no-cg", action="store_true")
        p.add_argument("--use-answer-prefix", action="store_true",
                       help="prepend the answer prefix (diverges from RULER call_api)")
        p.set_defaults(func=fn)

    p = sub.add_parser("eval", parents=[common])
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("all", parents=[common])
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-cg", action="store_true")
    p.add_argument("--use-answer-prefix", action="store_true")
    p.set_defaults(func=lambda a: (cmd_prepare(a), cmd_predict(a), cmd_eval(a)))

    p = sub.add_parser("download-essays")
    p.set_defaults(func=cmd_download_essays)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
