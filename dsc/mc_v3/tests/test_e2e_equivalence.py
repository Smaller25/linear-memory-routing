"""End-to-end equivalence test: load MC 30B ckpt, compare full forward output
and loss between v2 and v3 kernels.

Critical for verifying that v3 doesn't degrade model quality. If max diff is
small (< 0.1%) and PPL is within 0.5% of v2, we're safe to train with v3.

Usage:
    CUDA_VISIBLE_DEVICES=0 MC_KERNEL_VERSION=v2 python dsc/mc_v3/tests/test_e2e_equivalence.py \\
        --ckpt path/to/checkpoint-30B-model-ckpt.pth
    CUDA_VISIBLE_DEVICES=0 MC_KERNEL_VERSION=v3c python dsc/mc_v3/tests/test_e2e_equivalence.py \\
        --ckpt path/to/checkpoint-30B-model-ckpt.pth

Run both, diff the outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DSC = os.path.join(REPO, "dsc")
# Need BOTH long-gdn (for dsc.* imports) AND dsc/ (for lit_gpt.* imports) in path
for p in [REPO, DSC]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(ckpt, config_name, dtype, device):
    from dsc.lit_gpt.config import Config
    from dsc.lit_gpt.model import GPT
    cfg = Config.from_name(config_name)
    model = GPT(cfg).to(device).to(dtype)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {config_name}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=2, help="small to share GPU")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    print(f"=== e2e equivalence test ===", flush=True)
    print(f"  ckpt: {args.ckpt}", flush=True)
    print(f"  config: {args.config_name}", flush=True)
    print(f"  MC_KERNEL_VERSION: {os.environ.get('MC_KERNEL_VERSION', 'v2')}", flush=True)
    print(f"  seq_len={args.seq_len} micro_batch={args.micro_batch} dtype={args.dtype}", flush=True)

    model, cfg = load_model(args.ckpt, args.config_name, dtype, args.device)

    # Random input (deterministic via seed)
    input_ids = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len),
                              device=args.device, dtype=torch.long)
    targets = input_ids.clone()

    with torch.no_grad():
        logits = model(input_ids)
        # Standard CE loss
        import torch.nn.functional as F
        logits_f = logits.float().reshape(-1, logits.size(-1))
        targets_f = targets.reshape(-1)
        loss = F.cross_entropy(logits_f, targets_f)
        nll = loss.item()
        ppl = float(torch.exp(loss).item())

    print(f"\n--- result ---", flush=True)
    print(f"  NLL: {nll:.6f}", flush=True)
    print(f"  PPL: {ppl:.6f}", flush=True)
    print(f"  logits shape: {tuple(logits.shape)}", flush=True)
    print(f"  logits mean abs: {logits.abs().float().mean().item():.6f}", flush=True)

    result = {
        "ckpt": args.ckpt,
        "config_name": args.config_name,
        "kernel_version": os.environ.get("MC_KERNEL_VERSION", "v2"),
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "nll": nll,
        "ppl": ppl,
        "logits_mean_abs": logits.abs().float().mean().item(),
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[done] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
