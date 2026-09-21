#!/usr/bin/env python3
"""Diverse-key NIAH evaluation for one model arm (lengths x needles x seeds).

RULER-official protocol, identical to ruler_standard_eval.py:
    prompt = sample["input"] + sample["answer_prefix"]
    greedy free generation, then NVIDIA string_match scoring.

Pinned experiment constants honored here:
    * forward in bf16 autocast, score computations fp32 (the MC layers cast
      routing scores to fp32 internally; final logits are cast to fp32 before
      argmax)                                                       (CONSTANT 5)
    * per-sample raw records ALWAYS written as JSONL; aggregation is offline
      (plot_diverse_key_niah.py re-reads the raw files)             (CONSTANT 7)
    * every result row carries the composite key
      (model, gate, context_len, num_needles, seed) — never a bare model key,
      which historically caused silent arm-overwrite data loss     (CONSTANT 8)
    * MC_KERNEL_VERSION is pinned by the caller (campaign script exports v2)

Auxiliary DV (ReLU arm only, --log-active-states): per-token active cached
state count and gold-segment hit at the query position, read from the
ReLU aggregator's logging hook. This is the direct evidence that the active
state count is input-dependent.

Usage (one arm; the campaign script loops over arms):
    python dsc/scripts/diverse_key_niah_eval.py \
        --backend lit_gpt --ckpt <ckpt.pth> --config-name mc_370M \
        --config-overrides mc_topk=2 \
        --model-label mc-hard-top2-30B --gate-label hard_top2 \
        --data-root linear-memory-routing/data/ruler_diverse_key \
        --lengths 2048 4096 8192 --needles 1 4 8 16 32 --seeds 42 43 44 \
        --out-dir dsc/runs/eval/diverse_key_niah/<campaign>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DSC = os.path.join(REPO, "dsc")
LMR_SRC = os.path.join(REPO, "linear-memory-routing", "src")
for p in (REPO, DSC, LMR_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

TOKENS_TO_GENERATE = 128  # RULER niah standard (lm-eval-harness max_gen_toks)


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--config-overrides expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        out[key] = value
    return out


def load_samples(data_root: str, seed: int, length: int, task: str,
                 max_examples: int) -> list[dict]:
    path = os.path.join(data_root, f"seed{seed}", str(length), task,
                        "validation.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No diverse-key NIAH data at {path}")
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= max_examples:
                break
    return samples


def find_relu_aggregators(model) -> list:
    aggs = []
    for module in model.modules():
        if module.__class__.__name__ == "DenseMemoryCachingGDN2Layer" \
                and getattr(module, "variant", None) in ("relu", "relu_raw"):
            aggs.append(module.aggregator)
    return aggs


def attach_ssc_routing_hooks(model) -> list:
    """Expose SSC route_indices without touching the protected layer file.

    MemoryCachingGDN2Layer.forward drops the SSCOutput diagnostics; its
    forward_with_diagnostics returns them. We shadow forward on each INSTANCE
    (nn.Module.__call__ resolves instance attributes first) so every call
    stashes the routing picks as `last_route_indices` [B, T, k] (invalid
    slots are -1, per SSCOutput). The layer file itself stays byte-identical
    (no-change covenant: extensions by wrapping only).
    """
    layers = [m for m in model.modules()
              if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    for lyr in layers:
        def _fwd(hidden_states, attention_mask=None, past_key_values=None,
                 use_cache=False, output_attentions=False, _lyr=lyr, **kwargs):
            output, result = _lyr.forward_with_diagnostics(hidden_states)
            _lyr.last_route_indices = result.route_indices.detach()
            return output, None, past_key_values
        lyr.forward = _fwd
    return layers


def collect_gold_hits(ssc_layers: list, row_slot: int, last_pos: int,
                      gold_seg: int) -> float | None:
    """Fraction of SSC layers whose top-k picks at `last_pos` include the
    gold segment. This separates routing-precision failures (gold chunk never
    selected) from read/compression failures (selected but not decoded)."""
    hits = []
    for lyr in ssc_layers:
        idx = getattr(lyr, "last_route_indices", None)
        if idx is None or last_pos >= idx.shape[1]:
            continue
        hits.append(bool((idx[row_slot, last_pos] == gold_seg).any().item()))
    return sum(hits) / len(hits) if hits else None


@torch.no_grad()
def generate_greedy_batched(model, backend: str, prompts: list[torch.Tensor],
                            n_gen: int, eos_id: int | None, device: str,
                            autocast_ctx, tok,
                            stop_refs_per_row: list[list[str] | None],
                            pad_id: int = 0,
                            ssc_layers: list | None = None,
                            gold_positions: list[int | None] | None = None,
                            chunk_size: int = 256,
                            relu_aggs: list | None = None,
                            oracle_aggs: list | None = None):
    """Batched greedy decode with RIGHT padding — protocol-identical to bs=1.

    Every step re-forwards the full batch (no KV cache, same as bs=1) and
    reads each row's logits at ITS OWN last active position. Pads sit after
    the active tokens, so causality guarantees they never influence the read
    position; the result per row equals the bs=1 loop (modulo batched-kernel
    float reduction order). Rows stop independently on EOS or the
    score-invariant answer-found check; finished rows are dropped from the
    forward (compaction) so late stragglers don't pay for early finishers.
    """
    order = list(range(len(prompts)))
    gens: list[list[int]] = [[] for _ in prompts]
    seqs = [p[0].tolist() for p in prompts]
    live = list(order)
    routing: list[dict | None] = [None] * len(prompts)
    for step in range(n_gen):
        if not live:
            break
        maxlen = max(len(seqs[i]) for i in live)
        ids = torch.full((len(live), maxlen), pad_id, dtype=torch.long)
        for b, i in enumerate(live):
            ids[b, :len(seqs[i])] = torch.tensor(seqs[i], dtype=torch.long)
        ids = ids.to(device)
        if oracle_aggs and gold_positions is not None:
            # live-set composition changes on compaction: re-align the per-row
            # gold segments before EVERY forward.
            gold_t = torch.tensor(
                [(gold_positions[i] // chunk_size)
                 if gold_positions[i] is not None else -1 for i in live],
                dtype=torch.long)
            for agg in oracle_aggs:
                agg.oracle_gold = gold_t
        with autocast_ctx():
            if backend == "lit_gpt":
                logits = model(ids)
            else:
                logits = model(input_ids=ids).logits
        if step == 0 and ssc_layers and gold_positions is not None:
            # 프롬프트 패스에서만 계측: 질의를 읽는 마지막 프롬프트 위치의
            # 라우팅 선택에 gold 세그먼트가 포함됐는가.
            for b, i in enumerate(live):
                pos = gold_positions[i]
                if pos is None:
                    continue
                routing[i] = {
                    "gold_segment": pos // chunk_size,
                    "gold_hit_topk_layer_frac": collect_gold_hits(
                        ssc_layers, b, len(seqs[i]) - 1, pos // chunk_size),
                }
        if step == 0 and relu_aggs:
            # ReLU aux DV, 행별 자기 위치에서 (right-padding이므로 [:, -1]
            # 대신 각 행의 실제 마지막 프롬프트 위치를 읽어야 한다).
            for b, i in enumerate(live):
                last = len(seqs[i]) - 1
                per_mean, per_final, gold_hits, wsums = [], [], [], []
                for agg in relu_aggs:
                    counts = agg.last_active_counts  # [B, T]
                    if counts is None or last >= counts.shape[1]:
                        continue
                    per_mean.append(counts[b, :last + 1].float().mean().item())
                    per_final.append(counts[b, last].item())
                    pos = gold_positions[i] if gold_positions else None
                    weights = getattr(agg, "last_route_weights", None)
                    if weights is not None:
                        # Total cached read mass at the query position. For
                        # the raw (unnormalized) gate this disentangles
                        # "more segments" from "larger reads"; for the
                        # normalized gate it stays <= 1 by construction.
                        wsums.append(
                            weights[b, last].float().sum().item())
                        if pos is not None:
                            gseg = pos // chunk_size
                            if gseg < weights.shape[-1]:
                                gold_hits.append(
                                    bool(weights[b, last, gseg].float() > 0))
                if per_mean:
                    routing[i] = dict(routing[i] or {})
                    routing[i].update({
                        "mean_active_states": sum(per_mean) / len(per_mean),
                        "final_token_active_states":
                            sum(per_final) / len(per_final),
                        "route_weight_sum": (
                            sum(wsums) / len(wsums) if wsums else None),
                        "gold_hit_layer_frac": (
                            sum(gold_hits) / len(gold_hits)
                            if gold_hits else None),
                    })
        still_live = []
        for b, i in enumerate(live):
            nxt = int(logits[b, len(seqs[i]) - 1].float().argmax().item())
            if eos_id is not None and nxt == eos_id:
                continue
            gens[i].append(nxt)
            seqs[i].append(nxt)
            refs = stop_refs_per_row[i]
            if refs:
                pred = tok.decode(gens[i], skip_special_tokens=True).lower()
                if all(r.lower() in pred for r in refs):
                    continue
            still_live.append(i)
        live = still_live
    return gens, routing


@torch.no_grad()
def generate_greedy(model, backend: str, prompt_ids: torch.Tensor, n_gen: int,
                    eos_id: int | None, device: str, autocast_ctx,
                    relu_aggs: list, chunk_size: int,
                    gold_position: int | None,
                    tok=None, stop_refs: list[str] | None = None):
    """Greedy decode; returns (generated_ids, active_state_record | None).

    lit_gpt GDN-2 blocks have no KV cache: full forward each step (matches
    ruler_standard_eval.py). Active-state stats are read after the FIRST
    forward (the prompt-only pass) so the auxiliary DV reflects the routing
    the model performs while reading the context + query.
    """
    ids = prompt_ids.to(device)
    gen: list[int] = []
    active_record = None
    for step in range(n_gen):
        with autocast_ctx():
            if backend == "lit_gpt":
                logits = model(ids)
            else:
                logits = model(input_ids=ids).logits
        nxt = int(logits[:, -1].float().argmax(dim=-1).item())  # fp32 scores
        if step == 0 and relu_aggs:
            per_layer_mean, per_layer_final, gold_hits = [], [], []
            for agg in relu_aggs:
                counts = agg.last_active_counts  # [B, T]
                if counts is None:
                    continue
                per_layer_mean.append(counts.float().mean().item())
                per_layer_final.append(counts[0, -1].item())
                if gold_position is not None:
                    weights = agg.last_final_route_weights  # [B, N]
                    gold_seg = gold_position // chunk_size
                    if weights is not None and gold_seg < weights.shape[-1]:
                        gold_hits.append(bool(weights[0, gold_seg] > 0))
            if per_layer_mean:
                active_record = {
                    "mean_active_states": sum(per_layer_mean) / len(per_layer_mean),
                    "final_token_active_states": (
                        sum(per_layer_final) / len(per_layer_final)
                    ),
                    "per_layer_final_active": per_layer_final,
                    "gold_segment": (
                        gold_position // chunk_size
                        if gold_position is not None else None
                    ),
                    "gold_hit_layer_frac": (
                        sum(gold_hits) / len(gold_hits) if gold_hits else None
                    ),
                }
        if eos_id is not None and nxt == eos_id:
            break
        gen.append(nxt)
        ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)
        # Score-invariant early stop: string_match only checks whether every
        # ref substring appears in the prediction, and generating further can
        # never un-match it. Full-length free generation (the official 128
        # steps) is kept for samples that never produce the answer.
        if stop_refs and tok is not None:
            pred_so_far = tok.decode(gen, skip_special_tokens=True).lower()
            if all(r.lower() in pred_so_far for r in stop_refs):
                break
    return gen, active_record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["lit_gpt", "hf"], required=True)
    ap.add_argument("--ckpt", default=None, help="lit_gpt checkpoint .pth")
    ap.add_argument("--config-name", default="gdn2_370M")
    ap.add_argument("--config-overrides", nargs="*", default=[],
                    help="lit_gpt Config kwargs, e.g. mc_topk=2")
    ap.add_argument("--model-path", default=None, help="HF model dir (backend hf)")
    ap.add_argument("--model-label", required=True,
                    help="arm id for the result key, e.g. mc-hard-top2-30B")
    ap.add_argument("--gate-label", required=True,
                    help="gate id for the result key, e.g. hard_top2 | hard_top4 "
                         "| relu | single_state | softmax_attn")
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--haystack", choices=["essay", "noise"], default="essay")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--needles", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--n-gen", type=int, default=TOKENS_TO_GENERATE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-active-states", action="store_true",
                    help="record per-token active cached-state counts "
                         "(ReLU variant layers only)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false")
    ap.add_argument("--allow-partial-load", action="store_true",
                    help="tolerate state_dict missing/unexpected keys "
                         "(default: hard fail — wrong ckpt/config pairings "
                         "must never produce plausible-looking scores)")
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                    default=True,
                    help="disable the score-invariant answer-found early stop")
    ap.add_argument("--summary-name", default="results.json",
                    help="summary filename (shard launchers pass a distinct "
                         "name per process to avoid overwrites)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="samples decoded together (right-padded, "
                         "protocol-identical to bs=1); forced to 1 when "
                         "--log-active-states finds ReLU layers")
    ap.add_argument("--oracle-routing", action="store_true",
                    help="force each sample's gold segment into the top-k "
                         "selection (upper-bound arm: separates selection "
                         "failure from read failure; eval-only)")
    ap.add_argument("--mlp-router", default=None, metavar="DIR",
                    help="inject offline-trained MLP routers (router_L*.pt) "
                         "so the a-1 routing gain can be read as a SCORE; "
                         "eval-only, backbone weights untouched")
    ap.add_argument("--mlp-router-mode", default="select",
                    choices=["observe", "select", "full"],
                    help="select: MLP picks the top-k, native scores gate the "
                         "read (keeps the trained gate scale). full: MLP "
                         "scores also become gate logits (flattens the gate "
                         "toward uniform — dilution arm, for the record). "
                         "observe: score only, native selection — must "
                         "reproduce the baseline score exactly")
    ap.add_argument("--mlp-router-layers", type=int, nargs="*", default=None,
                    help="MC layers to inject (default: every layer with "
                         "weights). A partial injection leaves the rest on "
                         "the chance-level linear router")
    ap.add_argument("--broadcast-routing", action="store_true",
                    help="every MC layer routes on layer 0's top-k picks "
                         "(the rung between a-1's ~3-of-16 layers and the "
                         "oracle's 16-of-16); eval-only")
    ap.add_argument("--broadcast-source-layer", type=int, default=0,
                    help="which layer publishes the picks. Layers before it "
                         "run their own trained router, so keep it small")
    ap.add_argument("--broadcast-source", default="mlp",
                    choices=["mlp", "native"],
                    help="mlp: layer 0's trained router picks (needs "
                         "--mlp-router). native: the untouched linear router "
                         "picks — coherence control at chance quality")
    ap.add_argument("--gate-mode", default="native",
                    choices=["native", "order", "top1", "boost"],
                    help="how the shared picks split the read. native: each "
                         "layer's own scores. order: same magnitudes, largest "
                         "to the router's first pick (no hyperparameter). "
                         "top1: first pick raised to row max + margin, the "
                         "oracle's move on the router's guess. boost: every "
                         "selected logit gains the margin, shifting weight "
                         "off the online branch")
    ap.add_argument("--gate-margin", type=float, default=1.0,
                    help="margin for --gate-mode top1/boost (oracle uses 1.0)")
    ap.add_argument("--gate-scope", default="all",
                    choices=["all", "last_segment"],
                    help="where the boost applies. all rewrites the whole "
                         "prefill and collapsed the score even with an "
                         "oracle-like weight profile at the query position; "
                         "last_segment touches only where the answer is "
                         "produced")
    ap.add_argument("--log-routing", action="store_true",
                    help="record whether the gold segment is inside each SSC "
                         "layer's top-k picks at the query position (hard "
                         "top-k arms; separates routing-precision failures "
                         "from read failures)")
    args = ap.parse_args()

    overrides = parse_overrides(args.config_overrides)

    # ---- load model (bf16 weights + bf16 autocast forward; scores fp32) ----
    print(f"[load] backend={args.backend} label={args.model_label} "
          f"gate={args.gate_label}", flush=True)
    chunk_size = 256
    if args.backend == "lit_gpt":
        if not args.ckpt:
            ap.error("--ckpt required for --backend lit_gpt")
        from lit_gpt.config import Config
        from lit_gpt.model import GPT
        cfg = Config.from_name(args.config_name, **overrides)
        chunk_size = getattr(cfg, "mc_chunk_size", 256)
        model = GPT(cfg).to(args.device).to(torch.bfloat16)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            msg = (f"state_dict mismatch loading {args.ckpt} into "
                   f"{args.config_name}: {len(missing)} missing "
                   f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                   f"(e.g. {unexpected[:3]})")
            if args.allow_partial_load:
                print(f"[warn] {msg} — continuing (--allow-partial-load)",
                      flush=True)
            else:
                raise RuntimeError(
                    msg + " — wrong ckpt/config pairing? A partially loaded "
                    "model still produces plausible scores; pass "
                    "--allow-partial-load only if this is intentional.")
    else:
        if not args.model_path:
            ap.error("--model-path required for --backend hf")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        ).to(args.device)
    model.eval()

    relu_aggs: list = []
    if args.log_active_states:
        relu_aggs = find_relu_aggregators(model)
        for agg in relu_aggs:
            agg.log_active_states = True
        print(f"[hook] active-state logging on {len(relu_aggs)} ReLU layers",
              flush=True)
        if not relu_aggs:
            print("[warn] --log-active-states set but no ReLU layers found; "
                  "aux DV will be empty", flush=True)

    oracle_aggs: list = []
    if args.oracle_routing:
        from dsc.mc_baseline.mc_ssc_oracle import enable_oracle_routing
        oracle_aggs = enable_oracle_routing(model)
        print(f"[hook] ORACLE routing on {len(oracle_aggs)} SSC aggregators",
              flush=True)
        if not oracle_aggs:
            raise RuntimeError(
                "--oracle-routing attached to 0 aggregators — the arm would "
                "silently run as a plain baseline (this exact failure "
                "invalidated the first oracle run)")

    bcast_layers: list = []
    if args.broadcast_routing:
        if args.oracle_routing:
            raise RuntimeError("--broadcast-routing with --oracle-routing: "
                               "both swap the same forward")
        from dsc.mc_baseline.mc_ssc_mlp_router import enable_broadcast_routing
        bcast_layers = enable_broadcast_routing(
            model, args.broadcast_source_layer, args.broadcast_source,
            args.mlp_router, args.device, args.gate_mode, args.gate_margin,
            gate_scope=args.gate_scope)
        print(f"[hook] BROADCAST routing (source=L"
              f"{args.broadcast_source_layer}/{args.broadcast_source}, "
              f"gate={args.gate_mode}/{args.gate_scope}"
              + (f" margin={args.gate_margin}"
                 if args.gate_mode in ("top1", "boost") else "")
              + f") on {len(bcast_layers)} layers", flush=True)
        if len(bcast_layers) < 2:
            raise RuntimeError(
                f"--broadcast-routing attached to {len(bcast_layers)} layers — "
                "broadcasting to fewer than 2 layers is not the arm")

    mlp_layers: list = []
    if args.mlp_router and not args.broadcast_routing:
        if args.oracle_routing:
            raise RuntimeError(
                "--mlp-router with --oracle-routing: the oracle swaps the same "
                "forward, so one arm would silently overwrite the other")
        from dsc.mc_baseline.mc_ssc_mlp_router import enable_mlp_router
        mlp_layers = enable_mlp_router(model, args.mlp_router,
                                       args.mlp_router_layers,
                                       args.mlp_router_mode, args.device)
        print(f"[hook] MLP router mode={args.mlp_router_mode} on layers "
              f"{mlp_layers} from {args.mlp_router}", flush=True)
        if not mlp_layers:
            raise RuntimeError(
                "--mlp-router attached to 0 layers — the arm would silently "
                "run as a plain baseline")

    ssc_layers: list = []
    if args.log_routing:
        ssc_layers = attach_ssc_routing_hooks(model)
        print(f"[hook] routing logging on {len(ssc_layers)} SSC layers",
              flush=True)
        if not ssc_layers:
            print("[warn] --log-routing set but no SSC (hard top-k) layers "
                  "found; gold-hit DV will be empty", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    from ruler.eval_metrics import TASKS as METRIC_TASKS
    metric_fn = METRIC_TASKS["niah"]["metric_fn"]  # official string_match

    if args.device.startswith("cuda"):
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext

    arm_dir = os.path.join(args.out_dir, args.model_label)
    per_sample_dir = os.path.join(arm_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    rows = []
    for seed in args.seeds:
        for length in args.lengths:
            for needles in args.needles:
                task = f"niah_diversekey_{args.haystack}_{needles}"
                cell_id = f"seed{seed}_len{length}_n{needles}"
                per_sample_path = os.path.join(per_sample_dir,
                                               f"{cell_id}.jsonl")
                cell_json = os.path.join(arm_dir, f"cell_{cell_id}.json")
                if args.skip_existing and os.path.exists(cell_json):
                    with open(cell_json) as f:
                        rows.append(json.load(f))
                    print(f"[skip] {cell_id} (exists)", flush=True)
                    continue
                try:
                    samples = load_samples(args.data_root, seed, length, task,
                                           args.max_examples)
                except FileNotFoundError as e:
                    print(f"[skip] {cell_id}: {e}", flush=True)
                    continue

                preds, refs = [], []
                active_stats = []
                routing_stats = []
                t0 = time.time()
                # ReLU aux stats are captured per-row inside the batched
                # decode (right-padding-aware), so hooks no longer force bs=1.
                batch_size = max(1, args.batch_size)
                with open(per_sample_path, "w") as psf:
                    for lo in range(0, len(samples), batch_size):
                        chunk = samples[lo:lo + batch_size]
                        prompt_ids = [
                            tok(s["input"] + s.get("answer_prefix", ""),
                                return_tensors="pt",
                                add_special_tokens=False).input_ids
                            for s in chunk
                        ]
                        if batch_size == 1:
                            s = chunk[0]
                            gen, active = generate_greedy(
                                model, args.backend, prompt_ids[0], args.n_gen,
                                eos_id, args.device, autocast_ctx, relu_aggs,
                                chunk_size, s.get("token_position_answer"),
                                tok=tok,
                                stop_refs=(s["outputs"] if args.early_stop
                                           else None),
                            )
                            gens, actives = [gen], [active]
                            routings = [None]
                        else:
                            gens, routings = generate_greedy_batched(
                                model, args.backend, prompt_ids, args.n_gen,
                                eos_id, args.device, autocast_ctx, tok,
                                [s["outputs"] if args.early_stop else None
                                 for s in chunk],
                                pad_id=(eos_id or 0),
                                ssc_layers=ssc_layers,
                                gold_positions=[s.get("token_position_answer")
                                                for s in chunk],
                                chunk_size=chunk_size,
                                relu_aggs=relu_aggs,
                                oracle_aggs=oracle_aggs,
                            )
                            actives = [None] * len(chunk)
                        for j, (s, gen, active) in enumerate(
                                zip(chunk, gens, actives)):
                            pred = tok.decode(
                                gen, skip_special_tokens=True).strip()
                            preds.append(pred)
                            refs.append(s["outputs"])
                            if active is not None:
                                active_stats.append(active)
                            routing = routings[j]
                            if routing is not None:
                                if "gold_hit_topk_layer_frac" in routing:
                                    routing_stats.append(routing)
                                if "mean_active_states" in routing:
                                    active_stats.append(routing)
                            record = {
                                "model": args.model_label,
                                "gate": args.gate_label,
                                "context_len": length,
                                "num_needles": needles,
                                "seed": seed,
                                "sample_index": lo + j,
                                "pred": pred,
                                "ref": s["outputs"],
                                "prompt_tokens": int(prompt_ids[j].shape[1]),
                                "token_position_answer":
                                    s.get("token_position_answer"),
                                "active_states": active,
                                "routing": routing,
                            }
                            psf.write(json.dumps(record) + "\n")
                dt = time.time() - t0

                score = float(metric_fn(preds, refs))
                row = {
                    # composite result key — CONSTANT 8, never a bare model key
                    "model": args.model_label,
                    "gate": args.gate_label,
                    "context_len": length,
                    "num_needles": needles,
                    "seed": seed,
                    "score": score,
                    "n_examples": len(samples),
                    "dt_sec": dt,
                    "task": task,
                    "metric": "RULER official string_match "
                              "(greedy free generation)",
                    "per_sample_file": os.path.relpath(per_sample_path,
                                                       args.out_dir),
                }
                if active_stats:
                    n = len(active_stats)
                    row["mean_active_states"] = (
                        sum(a["mean_active_states"] for a in active_stats) / n
                    )
                    row["final_token_active_states"] = (
                        sum(a["final_token_active_states"]
                            for a in active_stats) / n
                    )
                    gold = [a["gold_hit_layer_frac"] for a in active_stats
                            if a["gold_hit_layer_frac"] is not None]
                    row["gold_hit_layer_frac"] = (
                        sum(gold) / len(gold) if gold else None
                    )
                    wsum = [a["route_weight_sum"] for a in active_stats
                            if a.get("route_weight_sum") is not None]
                    row["route_weight_sum"] = (
                        sum(wsum) / len(wsum) if wsum else None
                    )
                if routing_stats:
                    gold_tk = [r["gold_hit_topk_layer_frac"]
                               for r in routing_stats
                               if r["gold_hit_topk_layer_frac"] is not None]
                    row["gold_hit_topk_layer_frac"] = (
                        sum(gold_tk) / len(gold_tk) if gold_tk else None
                    )
                rows.append(row)
                with open(cell_json, "w") as f:
                    json.dump(row, f, indent=2)
                extra = ""
                if "mean_active_states" in row:
                    extra = (f"  active={row['mean_active_states']:.2f}"
                             f" gold_hit={row['gold_hit_layer_frac']}")
                print(f"[cell] {cell_id}  score={score:.2f}  "
                      f"n={len(samples)}  {dt:.0f}s{extra}", flush=True)

    def _env_record() -> dict:
        import hashlib
        import subprocess as sp
        rec = {"torch": torch.__version__}
        try:
            import fla
            rec["fla"] = getattr(fla, "__version__", "unknown")
        except Exception:
            rec["fla"] = None
        try:
            rec["git_commit"] = sp.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                text=True, timeout=10).stdout.strip()
            rec["git_dirty"] = bool(sp.run(
                ["git", "status", "--porcelain"], cwd=REPO,
                capture_output=True, text=True, timeout=10).stdout.strip())
        except Exception:
            rec["git_commit"] = None
        if args.device.startswith("cuda") and torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
        if args.ckpt and os.path.exists(args.ckpt):
            h = hashlib.sha256()
            with open(args.ckpt, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            rec["ckpt_sha256"] = h.hexdigest()
        return rec

    summary = {
        "model": args.model_label,
        "gate": args.gate_label,
        "backend": args.backend,
        "ckpt": args.ckpt,
        "model_path": args.model_path,
        "config_name": args.config_name,
        "config_overrides": overrides,
        "tokenizer": args.tokenizer,
        "haystack": args.haystack,
        "dtype": "bf16 weights + bf16 autocast forward, fp32 scores",
        "mc_kernel_version": os.environ.get("MC_KERNEL_VERSION", "v2"),
        "n_gen": args.n_gen,
        "early_stop": args.early_stop,
        "env": _env_record(),
        "rows": rows,
    }
    summary_path = os.path.join(arm_dir, args.summary_name)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {summary_path} ({len(rows)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
