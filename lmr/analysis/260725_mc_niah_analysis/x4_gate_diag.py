"""One-off diagnostic (not part of the pipeline): quantify the logit drift at
the first token-argmax divergence between full re-forward and the
incremental engine, for a specific sample, to distinguish "real bug" from
"bf16 reassociation noise" (project's own documented ~2.3pp bf16 noise
floor -- see spec §1 point 6)."""
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, gen_eval, x4_gen

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")


def main():
    model_kind = sys.argv[1] if len(sys.argv) > 1 else "mc-5B"
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    path = os.path.join(MC_OUT, "data", "2048", "niah_single_1", "validation.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r = rows[sample_idx]

    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(model_kind)
    stock_pairs = [(attn, attn.ssc) for _, attn in load_mc.mc_layers(model)]
    overrides = x4_gen.patch_override(model, mode="stock", seed=0)
    engine = x4_gen.IncrementalEngine(model, overrides)

    def use_stock():
        for attn, ssc in stock_pairs:
            attn.ssc = ssc

    def use_override():
        for (attn, _), ov in zip(stock_pairs, overrides):
            attn.ssc = ov

    ids_list = tok(r["input"], add_special_tokens=False).input_ids

    # Full re-forward, greedy, capturing per-step top-2 logits/margin.
    use_stock()
    ids = torch.tensor([ids_list], device="cuda")
    full_tokens, full_logits = [], []
    with torch.no_grad():
        for _ in range(n_gen):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)[0, -1].float()
            full_logits.append(logits.clone())
            nxt = int(logits.argmax())
            if nxt == 2:
                break
            full_tokens.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)

    # Incremental engine, capturing the SAME per-step logits.
    use_override()
    x4_gen.set_mode(overrides, "stock", seed=0)
    engine.reset()
    device = next(model.parameters()).device
    cur = list(ids_list)
    inc_tokens, inc_logits = [], []
    chunk_size = x4_gen.CHUNK_DEFAULT
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pos = 0
        logits = None
        while pos < len(cur):
            seg_end = min(pos + chunk_size, len(cur))
            complete = (seg_end - pos) == chunk_size
            x = x4_gen._embed(model, cur[pos:seg_end], device)
            logits = engine._run_segment(x, commit=complete)
            if complete:
                pos = seg_end
            else:
                break
        cur_ids = list(cur[pos:])
        for _ in range(n_gen):
            if cur_ids:
                x = x4_gen._embed(model, cur_ids, device)
                logits = engine._run_segment(x, commit=(len(cur_ids) == chunk_size))
                if len(cur_ids) == chunk_size:
                    cur_ids = []
            lf = logits.float()
            inc_logits.append(lf.clone())
            nxt = int(lf.argmax())
            if nxt == 2:
                break
            inc_tokens.append(nxt)
            cur_ids.append(nxt)

    print(f"full_tokens={full_tokens}")
    print(f"inc_tokens ={inc_tokens}")
    n = min(len(full_logits), len(inc_logits))
    for i in range(n):
        fl, il = full_logits[i], inc_logits[i]
        diff = (fl - il).abs()
        max_diff = float(diff.max())
        argmax_match = int(fl.argmax()) == int(il.argmax())
        top2_full = torch.topk(fl, 2).values
        margin_full = float(top2_full[0] - top2_full[1])
        print(f"step {i:2d}: argmax_match={argmax_match} max_abs_logit_diff={max_diff:.4f} "
              f"stock_top1_margin={margin_full:.4f} full_tok={int(fl.argmax())} inc_tok={int(il.argmax())}")
        if not argmax_match:
            break


if __name__ == "__main__":
    main()
