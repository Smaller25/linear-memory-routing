"""RULER OFFICIAL standard eval: free generation + string-match metric.

This is the unmodified RULER protocol (NVIDIA's `ruler.py eval`). The previous
scripts `eval_gdn2_vanilla_ruler.py` / `eval_gdn1_ruler.py` used per-token
teacher-forced accuracy, which is NOT RULER standard and biased toward models
that had been SFT'd on NIAH-style teacher-forced targets. This script replaces
them.

Protocol:
  1. Prompt = sample["input"] + sample["answer_prefix"]   (the exact RULER call)
  2. Greedy free generation of `tokens_to_generate` new tokens
  3. Score = NVIDIA's official string-match metric from `lmr.ruler.eval_metrics`

Backends:
  --backend lit_gpt   DSC + vanilla GDN-2 checkpoints (lit_gpt GPT class)
  --backend hf         HF CausalLM (GDN-1 from linear-moe-hub)

Usage:
    python dsc/scripts/ruler_standard_eval.py \
        --backend lit_gpt --ckpt path/to/checkpoint-XXX-model-ckpt.pth \
        --config-name gdn2_370M --tokenizer meta-llama/Llama-2-7b-hf \
        --tasks S-NIAH-1 MK-NIAH-1 --lengths 4096 16384 --max-examples 50 \
        --output-json dsc/runs/eval/<name>/results.json

    python dsc/scripts/ruler_standard_eval.py \
        --backend hf --model-path models/linear-moe-hub/Gated-Deltanet-1.3B \
        --tasks S-NIAH-1 MK-NIAH-1 --lengths 4096 16384 --max-examples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DSC = os.path.join(REPO, "dsc")
LMR_SRC = os.path.join(REPO, "linear-memory-routing", "src")
SCRIPTS = os.path.join(REPO, "scripts")
for p in (REPO, DSC, LMR_SRC, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

RULER_DATA_ROOT = os.path.join(REPO, "linear-memory-routing", "data", "ruler")

OFFICIAL_NAME_TO_DIR = {
    "S-NIAH-1": "niah_single_1",
    "S-NIAH-2": "niah_single_2",
    "S-NIAH-3": "niah_single_3",
    "MK-NIAH-1": "niah_multikey_1",
    "MK-NIAH-2": "niah_multikey_2",
    "MK-NIAH-3": "niah_multikey_3",
    "MQ-NIAH": "niah_multiquery",
    "MV-NIAH": "niah_multivalue",
    "PASSKEY": "passkey",
    "MARKER-UUID": "marker_uuid",
    "VT": "vt",
    "CWE": "cwe",
    "FWE": "fwe",
    "QA-1": "qa_1",
    "QA-2": "qa_2",
}

TASK_BASE = {
    "S-NIAH-1": "niah", "S-NIAH-2": "niah", "S-NIAH-3": "niah",
    "MK-NIAH-1": "niah", "MK-NIAH-2": "niah", "MK-NIAH-3": "niah",
    "MQ-NIAH": "niah", "MV-NIAH": "niah",
    "PASSKEY": "niah", "MARKER-UUID": "niah",
    "VT": "variable_tracking",
    "CWE": "common_words_extraction",
    "FWE": "freq_words_extraction",
    "QA-1": "qa", "QA-2": "qa",
}

# Aligned with lm-eval-harness RULER originals:
#   lm_eval/tasks/ruler/niah_single_1.yaml:        max_gen_toks=128
#   lm_eval/tasks/ruler/{cwe,fwe,vt}.yaml:        max_gen_toks=128/128/128
#   lm_eval/tasks/ruler/{qa_hotpot,qa_squad}.yaml: max_gen_toks=32
# Previous default niah=12 was wrong; S-NIAH-3 (multi-value) requires 128.
TOKENS_TO_GENERATE = {
    "niah": 128,
    "variable_tracking": 128,
    "common_words_extraction": 128,
    "freq_words_extraction": 128,
    "qa": 32,
}


def get_metric_fn(base_task: str):
    from ruler.eval_metrics import TASKS
    return TASKS[base_task]["metric_fn"]


def load_ruler_samples(task: str, length: int, max_examples: int,
                       data_root: str | None = None):
    data_root = data_root or RULER_DATA_ROOT
    if task not in OFFICIAL_NAME_TO_DIR:
        raise ValueError(f"Unknown task: {task}")
    dirname = OFFICIAL_NAME_TO_DIR[task]
    path = os.path.join(data_root, str(length), dirname, "validation.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No RULER data at {path}")
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= max_examples:
                break
    return samples


# ---------------------------------------------------------------------------
# Greedy free generation per backend
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_lit_gpt(model, prompt_ids: torch.Tensor, n_gen: int,
                     eos_id: int | None, device: str) -> list[int]:
    """Greedy decode n_gen tokens for a lit_gpt GPT model. Full-forward each step
    (correct for GDN-2 recurrent blocks which have no KV cache)."""
    ids = prompt_ids.to(device)
    gen = []
    for _ in range(n_gen):
        logits = model(ids)
        nxt = int(logits[:, -1].argmax(dim=-1).item())
        if eos_id is not None and nxt == eos_id:
            break
        gen.append(nxt)
        ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)
    return gen


@torch.no_grad()
def generate_hf(model, prompt_ids: torch.Tensor, n_gen: int,
                eos_id: int | None, device: str) -> list[int]:
    """Greedy decode for HF CausalLM (uses HF generate with KV cache)."""
    out = model.generate(
        input_ids=prompt_ids.to(device),
        max_new_tokens=n_gen,
        do_sample=False, num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        use_cache=True,
    )
    return out[0, prompt_ids.shape[1]:].tolist()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_task(backend: str, model, tok, task: str, length: int,
               max_examples: int, device: str,
               generation_override: int | None = None) -> tuple[float, int, float, list]:
    base = TASK_BASE[task]
    n_gen = generation_override or TOKENS_TO_GENERATE[base]
    samples = load_ruler_samples(task, length, max_examples)
    eos_id = tok.eos_token_id

    preds, refs, debug = [], [], []
    t0 = time.time()
    for s in samples:
        prompt = s["input"] + s.get("answer_prefix", "")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if backend == "lit_gpt":
            g = generate_lit_gpt(model, ids, n_gen, eos_id, device)
        elif backend in ("hf", "gdn1"):
            g = generate_hf(model, ids, n_gen, eos_id, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        pred = tok.decode(g, skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(s["outputs"])
        debug.append({"index": s["index"], "pred": pred, "ref": s["outputs"]})
    dt = time.time() - t0

    metric_fn = get_metric_fn(base)
    score = metric_fn(preds, refs)
    return score, len(samples), dt, debug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["lit_gpt", "hf", "gdn1"], required=True)
    # lit_gpt args
    ap.add_argument("--ckpt", default=None, help="lit_gpt checkpoint .pth")
    ap.add_argument("--config-name", default="gdn2_370M", help="lit_gpt Config name")
    # hf args
    ap.add_argument("--model-path", default=None, help="HF model dir")
    # shared
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--tasks", nargs="+",
                    default=["S-NIAH-1", "S-NIAH-2", "S-NIAH-3",
                             "MK-NIAH-1", "MK-NIAH-2", "MK-NIAH-3",
                             "MQ-NIAH", "MV-NIAH"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096])
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--n-gen-override", type=int, default=None,
                    help="override tokens_to_generate (default: per-task standard)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--save-predictions", default=None,
                    help="optional path to dump per-sample predictions as JSONL")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    # ---- load model ----
    print(f"[load] backend={args.backend} dtype={args.dtype}", flush=True)
    if args.backend == "lit_gpt":
        if not args.ckpt:
            ap.error("--ckpt required for --backend lit_gpt")
        from lit_gpt.config import Config
        from lit_gpt.model import GPT
        cfg = Config.from_name(args.config_name)
        model = GPT(cfg).to(args.device).to(dtype)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[warn] missing keys ({len(missing)}):", missing[:5], flush=True)
        if unexpected:
            print(f"[warn] unexpected keys ({len(unexpected)}):", unexpected[:5], flush=True)
    elif args.backend == "hf":
        if not args.model_path:
            ap.error("--model-path required for --backend hf")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device)
    elif args.backend == "gdn1":
        if not args.model_path:
            ap.error("--model-path required for --backend gdn1")
        from pathlib import Path
        from gdn1_common import load_gdn1_causal_lm, setup_cpu_safe_env
        setup_cpu_safe_env()
        model = load_gdn1_causal_lm(Path(args.model_path), torch_dtype=dtype).to(args.device)
    model.eval()

    # ---- tokenizer ----
    from transformers import AutoTokenizer
    tok_path = args.model_path if args.backend in ("hf", "gdn1") and args.model_path else args.tokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[load] params={n_params:.2f}B", flush=True)

    print(f"\n{'task':>14} {'len':>7} | {'score':>7} | N   | dt     | tokens/gen", flush=True)
    print("-" * 70, flush=True)
    results = []
    all_predictions = []
    for length in args.lengths:
        for task in args.tasks:
            try:
                score, n, dt, debug = score_task(
                    args.backend, model, tok, task, length,
                    max_examples=args.max_examples,
                    device=args.device,
                    generation_override=args.n_gen_override,
                )
            except FileNotFoundError as e:
                print(f"{task:>14} {length:>7} | SKIP   | -   | -      | {e}", flush=True)
                continue
            base = TASK_BASE[task]
            n_gen = args.n_gen_override or TOKENS_TO_GENERATE[base]
            print(f"{task:>14} {length:>7} | {score:>7.2f} | {n:>3} | {dt:>5.1f}s | {n_gen}",
                  flush=True)
            results.append({"task": task, "length": length, "score": score,
                            "n_examples": n, "dt_sec": dt, "tokens_generated": n_gen,
                            "metric_base": base})
            if args.save_predictions:
                for d in debug:
                    d["task"] = task
                    d["length"] = length
                    all_predictions.append(d)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "backend": args.backend,
                "ckpt": args.ckpt,
                "model_path": args.model_path,
                "config_name": args.config_name,
                "n_params_B": n_params,
                "metric": "RULER official string-match (free generation, greedy decode)",
                "results": results,
            }, f, indent=2)
        print(f"[saved] {args.output_json}", flush=True)

    if args.save_predictions and all_predictions:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        with open(args.save_predictions, "w") as f:
            for d in all_predictions:
                f.write(json.dumps(d) + "\n")
        print(f"[saved predictions] {args.save_predictions}", flush=True)


if __name__ == "__main__":
    main()
