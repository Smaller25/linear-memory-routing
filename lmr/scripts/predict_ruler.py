# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Publication-grade RULER prediction: FREE-GENERATION with the Memory-Caching read-out in the loop.

The teacher-forced scorer (``eval_ruler.py``) is a cheap proxy; the standard RULER protocol is
greedy free generation + a string-match metric over the *generated* text. This script does exactly
that for our methods (vanilla / RM / trained SSC etc.), writing ``pred.jsonl`` in the SAME format
NVIDIA RULER's evaluator consumes, so scoring is the official, unmodified metric:

    # 1. official data (vendored RULER generators)
    python scripts/ruler.py prepare --lengths 4096 8192 --tasks niah_single_1,niah_multikey_2
    # 2. free-gen prediction WITH the read-out (this script)
    python -m lmr.scripts.predict_ruler --arch mamba2 --variant ssc --heads ckpt/ssc.pt --topk 4 \
        --lengths 4096 8192 --tasks niah_single_1 niah_multikey_2
    # 3. official string-match scoring (unchanged RULER metric)
    python scripts/ruler.py eval --lengths 4096 8192 --tasks niah_single_1,niah_multikey_2

How the read-out is run incrementally during decoding: the segment-state cache is built ONCE over
the prompt (the needle lives there and the prompt is fixed during generation); each decode step
re-runs only the *current* (growing) partial segment from a zero state and reads over that fixed
cache — O(segment) per token, not O(prompt). When the current segment crosses a ``chunk_size``
boundary its final state is frozen into the cache. ``--variant vanilla`` uses the model's native
generate (full recurrent state, no segmentation) — the honest single-state baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import torch
import torch.nn as nn

from lmr.adapters import descriptor_dim_for, get_adapter
from lmr.loaders import load_backbone
from lmr.readout import ResidualMemory, build_readout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_ruler_cli():
    """Reuse scripts/ruler.py's config loader + path helper (no duplication)."""
    spec = importlib.util.spec_from_file_location(
        "ruler_cli", os.path.join(REPO_ROOT, "scripts", "ruler.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def generate_segmented_readout(model, adapter, readouts, prompt_ids, *, chunk_size, n_gen,
                               backend="cuda", eos_id=None):
    """Greedy free generation with the per-layer Memory-Caching read-out.

    prompt_ids: [1, T]. Returns the list of generated token ids (≤ n_gen, stops at eos_id).
    Mirrors ``run_segmented_lm`` semantics: fixed [0:chunk),[chunk:2chunk),… segments; the segment
    holding the live position is re-run from zero each step, completed segments are frozen.
    """
    blocks = adapter.blocks(model)
    lm_head = adapter.lm_head(model)
    fine: list[list[torch.Tensor]] = [[] for _ in blocks]

    all_ids = prompt_ids
    chunk = chunk_size

    def freeze_one():
        """Freeze the next not-yet-cached segment: run it, append each layer's final state."""
        i = len(fine[0])
        seg = all_ids[:, i * chunk:(i + 1) * chunk]
        hidden = adapter.embed(model, seg)
        finals = []
        for li, blk in enumerate(blocks):
            hidden, fs, _ = adapter.run_block(blk, hidden, fine[li], readouts[li], backend)
            finals.append(fs.detach())
        for li, fs in enumerate(finals):
            fine[li].append(fs)

    gen = []
    for _ in range(n_gen):
        cur_len = all_ids.shape[1]
        seg_start = ((cur_len - 1) // chunk) * chunk          # segment holding the live token
        while len(fine[0]) < seg_start // chunk:               # freeze any fully-completed segments
            freeze_one()
        cur = all_ids[:, seg_start:cur_len]                    # current partial segment (≥1 token)

        hidden = adapter.embed(model, cur)
        for li, blk in enumerate(blocks):
            hidden, _, _ = adapter.run_block(blk, hidden, fine[li], readouts[li], backend)
        hidden = adapter.final_norm(model, hidden)
        logits = lm_head(hidden[:, -1])                        # [1, vocab]
        nxt = int(logits.argmax(dim=-1).item())
        if eos_id is not None and nxt == eos_id:
            break
        gen.append(nxt)
        all_ids = torch.cat([all_ids, logits.new_tensor([[nxt]], dtype=torch.long)], dim=1)
    return gen


@torch.no_grad()
def generate_vanilla(model, prompt_ids, n_gen, eos_id=None):
    out = model.generate(input_ids=prompt_ids, max_new_tokens=n_gen, do_sample=False,
                         num_beams=1, eos_token_id=eos_id)
    return out[0, prompt_ids.shape[1]:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mamba2", "gdn"], default="mamba2")
    ap.add_argument("--model", "--repo", dest="repo", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--variant", choices=["vanilla", "rm", "grm", "ssc", "mom", "aom"], default="ssc")
    ap.add_argument("--heads", default=None, help="trained head .pt (for grm/ssc/mom/aom)")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--num-slots", type=int, default=4)
    ap.add_argument("--low-rank-dim", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_2"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--use-answer-prefix", action="store_true",
                    help="prepend answer_prefix to the prompt (RULER call_api feeds only `input` by default)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    cli = _load_ruler_cli()
    yaml_tasks, base_tasks, metric_tasks = cli.load_configs()

    model, tok = load_backbone(args.arch, repo=args.repo, tokenizer=args.tokenizer,
                               device=args.device, dtype=dtype)
    model.eval()
    adapter = get_adapter(args.arch)
    eos_id = tok.eos_token_id

    readouts = None
    if args.variant != "vanilla":
        if args.variant == "rm":
            readouts = [ResidualMemory() for _ in adapter.blocks(model)]
        else:
            dd = descriptor_dim_for(model, args.arch)
            kw = {"low_rank_dim": None if not args.low_rank_dim else args.low_rank_dim}
            if args.variant == "ssc":
                kw["topk"] = args.topk
            if args.variant == "mom":
                kw["num_slots"] = args.num_slots
            readouts = nn.ModuleList([build_readout(args.variant, model.config.hidden_size, dd, **kw)
                                      for _ in adapter.blocks(model)]).to(args.device, dtype=dtype)
            assert args.heads, f"--heads required for variant={args.variant}"
            readouts.load_state_dict(torch.load(args.heads, map_location=args.device))
            readouts.eval()

    for length in args.lengths:
        for task in args.tasks:
            tdir = cli.task_dir(length, task)
            inp = os.path.join(tdir, "validation.jsonl")
            if not os.path.exists(inp):
                print(f"[skip] no data: {inp}"); continue
            base = yaml_tasks[task]["task"]
            n_gen = base_tasks[base]["tokens_to_generate"]
            samples = [json.loads(l) for l in open(inp)][:args.max_examples]
            preds, refs = [], []
            out_path = os.path.join(tdir, "pred.jsonl")
            with open(out_path, "w") as fout:
                for s in samples:
                    prompt = s["input"] + (s.get("answer_prefix", "") if args.use_answer_prefix else "")
                    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
                    if args.variant == "vanilla":
                        g = generate_vanilla(model, ids, n_gen, eos_id)
                    else:
                        g = generate_segmented_readout(model, adapter, readouts, ids,
                                                       chunk_size=args.chunk_size, n_gen=n_gen, eos_id=eos_id)
                    pred = tok.decode(g, skip_special_tokens=True)
                    preds.append(pred); refs.append(s["outputs"])
                    fout.write(json.dumps({"index": s["index"], "pred": pred,
                                           "outputs": s["outputs"], "length": s.get("length", length)}) + "\n")
            metric_fn = metric_tasks[base]["metric_fn"]
            score = metric_fn([p.strip() for p in preds], refs)
            print(f"{task:>18} L={length:>6} | {args.variant:>8} score={score:6.2f}  "
                  f"(n={len(preds)}, gen_toks={n_gen}) -> {out_path}")
    print("\nofficial scoring (unchanged metric):  "
          f"python scripts/ruler.py eval --lengths {' '.join(map(str, args.lengths))} "
          f"--tasks {','.join(args.tasks)}")


if __name__ == "__main__":
    main()
