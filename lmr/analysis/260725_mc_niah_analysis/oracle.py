"""E2: gold chunk를 top-k에 강제 주입한 oracle routing으로 재생성."""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_GEN_DEFAULT = 128


def inject_gold(top_indices, top_scores, online_score, gold, segment_ids):
    """gold가 eligible(과거 segment)인데 미선택인 토큰의 마지막 슬롯을 gold로 교체.
    교체 슬롯 score = max(선택 score 최대, online score) → gate에서 공정한 무게."""
    eligible = (segment_ids > gold).unsqueeze(0).expand(top_indices.shape[:2])  # [B,T]
    has = (top_indices == gold).any(-1)
    force = eligible & ~has
    idx, sc = top_indices.clone(), top_scores.clone()
    idx[force, -1] = gold
    best = torch.maximum(top_scores.max(-1).values, online_score)
    sc[force, -1] = best[force]
    return idx, sc


def _make_oracle_class():
    import load_mc
    load_mc.bootstrap()
    from torch.nn import functional as F
    from dsc.mc_gdn2.ssc import GDN2SSC
    from dsc.mc_baseline.mc_ssc import (SSCOutput, segment_key_sums,
                                        causal_online_key_sums)
    from dsc.mc_baseline.cached_memory_read import ssc_gather_read

    class OracleGDN2SSC(GDN2SSC):
        gold_segment = None  # 샘플마다 설정

        def forward(self, hidden_states, queries, keys, online_output, memories):
            # mc_ssc.SparseSelectiveCaching.forward를 복사, topk 뒤 inject만 추가
            batch, length, heads, key_dim = queries.shape
            num_segments = memories.shape[1]
            u = self.connector(hidden_states).view(batch, length, heads, key_dim)
            summaries = segment_key_sums(keys, self.chunk_size)
            all_scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())
            segment_ids = torch.arange(length, device=queries.device) // self.chunk_size
            eligible = (torch.arange(num_segments, device=queries.device)[None, :]
                        < segment_ids[:, None])
            past_scores = all_scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)
            route_count = min(self.topk, num_segments)
            top_scores, top_indices = torch.topk(past_scores, k=route_count, dim=-1)
            online_summary = causal_online_key_sums(keys, self.chunk_size)
            online_score = torch.einsum("bthk,bthk->bt", u.float(), online_summary.float())
            if self.gold_segment is not None:                       # === oracle 개입 ===
                top_indices, top_scores = inject_gold(
                    top_indices, top_scores, online_score, self.gold_segment, segment_ids)
            valid = torch.isfinite(top_scores)
            safe_indices = top_indices.masked_fill(~valid, 0)
            gate_logits = torch.cat([online_score.unsqueeze(-1), top_scores], dim=-1)
            gate_valid = torch.cat([torch.ones(batch, length, 1, device=queries.device,
                                               dtype=torch.bool), valid], dim=-1)
            gate_logits = gate_logits.masked_fill(~gate_valid, -torch.inf)
            gates = torch.softmax(gate_logits, dim=-1).to(online_output.dtype)
            online_weight, route_weights = gates[..., :1], gates[..., 1:]
            cached_output = ssc_gather_read(
                queries, memories, safe_indices, route_weights,
                scale=self.read_scale, normalize_queries=self.normalize_queries,
            ).to(online_output.dtype)
            output = online_weight.unsqueeze(-1) * online_output + cached_output
            return SSCOutput(output=output, online_output=online_output,
                             cached_output=cached_output,
                             route_indices=top_indices.masked_fill(~valid, -1),
                             route_weights=route_weights, online_weight=online_weight,
                             route_scores=top_scores.masked_fill(~valid, -torch.inf))

    return OracleGDN2SSC


def patch_oracle(model):
    """모든 MC layer의 ssc를 Oracle 버전으로 교체(가중치 복사). 교체본 리스트 반환."""
    import load_mc
    cls = _make_oracle_class()
    oracles = []
    for _, attn in load_mc.mc_layers(model):
        old = attn.ssc
        new = cls(old.hidden_size, old.num_heads, old.head_qk_dim,
                  topk=old.topk, chunk_size=old.chunk_size)
        new.load_state_dict(old.state_dict())
        new = new.to(next(old.parameters()).device, next(old.parameters()).dtype)
        attn.ssc = new
        oracles.append(new)
    return oracles


def _set_gold(oracles, gold):
    for o in oracles:
        o.gold_segment = gold


def _run_condition(model, tok, oracles, jsonl_path, n_gen, tag, limit=None):
    """jsonl(variant=='multi'만) 루프: baseline(gold_segment=None 고정) + oracle
    (샘플마다 gold_seg 설정) 두 결과를 함께 반환."""
    import gen_eval

    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    rows = [r for r in rows if r.get("variant") == "multi"]
    if limit is not None:
        rows = rows[:limit]

    baseline_rows, oracle_rows = [], []
    for i, r in enumerate(rows):
        ids = torch.tensor([tok(r["input"], add_special_tokens=False).input_ids],
                           device="cuda")
        gold = int(r["gold_seg"])

        _set_gold(oracles, None)
        gen_b = gen_eval.greedy_generate(model, ids, n_gen=n_gen)
        pred_b = tok.decode(gen_b)
        correct_b = all(o.lower() in pred_b.lower() for o in r["outputs"])
        baseline_rows.append({"pair_id": r["pair_id"], "gold_seg": gold,
                              "pred": pred_b, "correct": correct_b})
        print(f"[{tag}][baseline] {i+1}/{len(rows)} correct={correct_b}", flush=True)

        _set_gold(oracles, gold)
        gen_o = gen_eval.greedy_generate(model, ids, n_gen=n_gen)
        pred_o = tok.decode(gen_o)
        correct_o = all(o.lower() in pred_o.lower() for o in r["outputs"])
        oracle_rows.append({"pair_id": r["pair_id"], "gold_seg": gold,
                            "pred": pred_o, "correct": correct_o})
        print(f"[{tag}][oracle]   {i+1}/{len(rows)} correct={correct_o}", flush=True)
        _set_gold(oracles, None)

    def summarize(rows_):
        n = len(rows_)
        score = sum(1 for r in rows_ if r["correct"]) / n if n else 0.0
        return {"score": score, "n": n, "rows": rows_}

    return {"baseline": summarize(baseline_rows), "oracle": summarize(oracle_rows)}


def _sanity_check_baseline_matches_unpatched(model, tok, oracles, sample_row, n_gen):
    """gold_segment=None인 상태에서 oracle patch가 stock forward와 동일한 텍스트를
    생성하는지 ONE 샘플로 검증 (brief 요구사항)."""
    import gen_eval

    ids = torch.tensor([tok(sample_row["input"], add_special_tokens=False).input_ids],
                       device="cuda")
    _set_gold(oracles, None)
    patched_gen = gen_eval.greedy_generate(model, ids, n_gen=n_gen)

    # unpatch: swap oracle ssc back out temporarily by reloading model kind not
    # feasible cheaply here, so instead re-derive from stock GDN2SSC using the
    # same weights already loaded into the oracle (state_dict identical) — the
    # only difference is the forward path, so we compare against a **fresh
    # stock instance** wired into the same attn temporarily.
    import load_mc
    stocks = []
    for _, attn in load_mc.mc_layers(model):
        stocks.append((attn, attn.ssc))
    from dsc.mc_gdn2.ssc import GDN2SSC
    for attn, oracle_ssc in stocks:
        stock = GDN2SSC(oracle_ssc.hidden_size, oracle_ssc.num_heads, oracle_ssc.head_qk_dim,
                        topk=oracle_ssc.topk, chunk_size=oracle_ssc.chunk_size)
        stock.load_state_dict(oracle_ssc.state_dict())
        stock = stock.to(next(oracle_ssc.parameters()).device,
                         next(oracle_ssc.parameters()).dtype)
        attn.ssc = stock
    stock_gen = gen_eval.greedy_generate(model, ids, n_gen=n_gen)
    # restore oracle ssc instances
    for attn, oracle_ssc in stocks:
        attn.ssc = oracle_ssc

    assert patched_gen == stock_gen, (
        "oracle patch with gold_segment=None diverged from stock forward: "
        f"patched={patched_gen[:10]} stock={stock_gen[:10]}"
    )
    print(f"[sanity] baseline patch == stock output ({len(stock_gen)} tokens)", flush=True)


def main():
    import load_mc

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    ap.add_argument("--n-gen", type=int, default=N_GEN_DEFAULT)
    ap.add_argument("--limit", type=int, default=None,
                    help="dev/smoke only: cap rows per condition (default: all 16)")
    ap.add_argument("--dry-run", action="store_true",
                    help="dev/smoke only: skip writing results/e2_oracle.json")
    a = ap.parse_args()

    MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
    HERE = os.path.dirname(os.path.abspath(__file__))
    RES = os.path.join(HERE, "results")
    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)

    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(a.model)
    oracles = patch_oracle(model)
    print(f"[e2] {a.model}: n_oracle_layers={len(oracles)} n_gen={a.n_gen}", flush=True)

    results = {}
    for cond in ("S", "D"):
        path = os.path.join(MC_OUT, "data", "paired", f"{cond}.jsonl")
        rows = [json.loads(l) for l in open(path) if l.strip()]
        rows = [r for r in rows if r.get("variant") == "multi"]
        _sanity_check_baseline_matches_unpatched(model, tok, oracles, rows[0], n_gen=8)
        results[cond] = _run_condition(model, tok, oracles, path, n_gen=a.n_gen,
                                       tag=f"{a.model}/{cond}", limit=a.limit)
        b, o = results[cond]["baseline"]["score"], results[cond]["oracle"]["score"]
        print(f"[e2][sanity] {a.model}/{cond}: baseline={b:.3f} oracle={o:.3f} "
              f"(n={results[cond]['baseline']['n']})", flush=True)
        if o < b:
            print(f"[e2][WARN] {a.model}/{cond}: oracle < baseline — investigate!",
                  flush=True)

    del model
    torch.cuda.empty_cache()

    if a.dry_run:
        print("[e2] --dry-run: skipping results/e2_oracle.json write", flush=True)
        return

    out_repo = os.path.join(RES, "e2_oracle.json")
    out_mcout = os.path.join(MC_OUT, "results", "e2_oracle.json")

    merged = {"meta": {"models": [], "n_gen": a.n_gen,
                       "injection_score_mode": "max(top,online)"}, "results": {}}
    for p in (out_repo, out_mcout):
        if os.path.exists(p):
            try:
                prev = json.load(open(p))
                merged["results"].update(prev.get("results", {}))
            except Exception:
                pass
    merged["results"][a.model] = results
    merged["meta"]["models"] = sorted(merged["results"].keys())

    # Flat "rows" (baseline correctness only, condition+pair_id keyed) for
    # convenience flat view (model-tagged); routing_stats._load_e2_baseline
    # reads results[model][cond]["baseline"]["rows"] directly and does not
    # consume this.
    flat_rows = []
    for mk in merged["meta"]["models"]:
        for cond, cell in merged["results"][mk].items():
            for r in cell["baseline"]["rows"]:
                flat_rows.append({"model": mk, "condition": cond,
                                  "pair_id": r["pair_id"], "gold_seg": r["gold_seg"],
                                  "correct": r["correct"]})
    merged["rows"] = flat_rows

    for p in (out_repo, out_mcout):
        json.dump(merged, open(p, "w"), indent=2)
        print(f"[e2] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
