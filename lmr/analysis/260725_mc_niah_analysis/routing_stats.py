"""E1: answer position에서 layer별 gold-chunk routing 정확도.

생성 불필요 — 프롬프트 1-pass. 점수는 SSC와 동일 수식으로 재계산:
u = ssc.connector(h); summaries = segment_key_sums(normalize(k)); score = <u, c_i> (head 합).

드라이버는 `--model {mc-5B,mc-30B}` 하나씩 실행되며(sbatch에서 && 로 순차 실행),
매 실행마다 기존 results/e1_routing.json(+per_sample)을 읽어 현재 모델 결과만
갱신·병합한다. 두 번 실행이 끝나면 두 모델이 모두 담긴 최종 JSON/그림이 남는다.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, data as mcdata

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CHUNK, TOPK = load_mc.CHUNK, load_mc.TOPK

DATASETS = {
    "niah_single_1": {"kind": "a", "path": os.path.join(MC_OUT, "data", "2048", "niah_single_1", "validation.jsonl")},
    "niah_multikey_1": {"kind": "a", "path": os.path.join(MC_OUT, "data", "2048", "niah_multikey_1", "validation.jsonl")},
    "paired_S_multi": {"kind": "b", "path": os.path.join(MC_OUT, "data", "paired", "S.jsonl"), "condition": "S"},
    "paired_D_multi": {"kind": "b", "path": os.path.join(MC_OUT, "data", "paired", "D.jsonl"), "condition": "D"},
}


def capture_hidden(model, ids):
    """각 layer의 attn 입력(norm_1 이후) [T,D]를 hook으로 수집."""
    store = {}
    hooks = []
    for i, blk in enumerate(model.transformer.h):
        def mk(i):
            def pre(mod, args, kwargs):
                store[i] = args[0].detach()
            return pre
        hooks.append(blk.attn.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(ids)
    for h in hooks:
        h.remove()
    return [store[i][0] for i in range(len(model.transformer.h))]


@torch.no_grad()
def routing_scores_at(attn, h, t):
    """h [T,D] (bf16 cuda). t 위치의 과거 segment별 routing score [n_seg] (미래/현재는 -inf)."""
    load_mc.bootstrap()
    from dsc.mc_baseline.mc_ssc import segment_key_sums
    hb = h.unsqueeze(0)
    q, k, v, g, b, w = attn._project(hb)
    rk = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    summaries = segment_key_sums(rk, attn.ssc.chunk_size)          # [1,N,H,K]
    u = attn.ssc.connector(hb[:, t:t + 1]).view(1, 1, attn.ssc.num_heads,
                                                attn.ssc.head_qk_dim)
    scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())[0, 0]  # [N]
    cur_seg = t // attn.ssc.chunk_size
    scores[cur_seg:] = float("-inf")
    return scores


def analyze_sample(model, tok, input_text, topk=TOPK):
    """gold-chunk routing 진단. 반환 per_layer 항목의 hit/gold_rank/amongkeys는
    gold_seg가 ELIGIBLE(과거 segment)일 때만 값을 채우고, 아니면 None
    (t==T-1 시점에 gold가 현재 segment에 있어 구조적으로 routing 불가능한 경우)."""
    ann = mcdata.annotate(input_text, tok)
    ids = torch.tensor([tok(input_text, add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    t = T - 1
    cur_seg = t // CHUNK
    eligible = ann["gold_seg"] < cur_seg
    key_segs = sorted({n["seg"] for n in ann["needles"]})
    eligible_key_segs = [s for s in key_segs if s < cur_seg]
    hiddens = capture_hidden(model, ids)
    per_layer = []
    for i, attn in load_mc.mc_layers(model):
        s = routing_scores_at(attn, hiddens[i], t)
        order = torch.argsort(s, descending=True).tolist()
        if eligible:
            gold_rank = order.index(ann["gold_seg"])
            hit = ann["gold_seg"] in order[:topk]
        else:
            gold_rank, hit = None, None
        amongkeys = None
        if len(eligible_key_segs) >= 2:
            best_key_seg = max(eligible_key_segs, key=lambda ks: float(s[ks]))
            amongkeys = (best_key_seg == ann["gold_seg"])
        per_layer.append({"layer": i, "gold_rank": gold_rank, "hit": hit,
                          "amongkeys": amongkeys})
    return {"gold_seg": ann["gold_seg"], "n_seg": ann["n_seg"], "cur_seg": cur_seg,
            "eligible": eligible, "n_eligible_key_segs": len(eligible_key_segs),
            "per_layer": per_layer}


def _rows_for(dsname):
    spec = DATASETS[dsname]
    rows = [json.loads(l) for l in open(spec["path"])]
    if spec["kind"] == "b":
        rows = [r for r in rows if r.get("variant") == "multi"]
    out = []
    for r in rows:
        sid = r.get("index", r.get("pair_id"))
        rec = {"sample_id": sid, "input": r["input"]}
        if spec["kind"] == "b":
            rec["pair_id"] = r["pair_id"]
            rec["condition"] = r.get("condition", spec.get("condition"))
        out.append(rec)
    return out


def run_model(kind):
    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(kind)
    n_layers = len(load_mc.mc_layers(model))
    print(f"[e1] {kind}: n_layers={n_layers}", flush=True)

    per_sample = {}  # dataset -> [ {sample_id, ..., per_layer:[...]} ]
    layer_agg = {}   # dataset -> layer -> accumulators

    for dsname in DATASETS:
        rows = _rows_for(dsname)
        per_sample[dsname] = []
        layer_agg[dsname] = {i: {"hit_sum": 0, "hit_n": 0, "rank_sum": 0.0, "rank_n": 0,
                                  "ak_sum": 0, "ak_n": 0} for i in range(n_layers)}
        n_eligible = 0
        for ri, r in enumerate(rows):
            try:
                res = analyze_sample(model, tok, r["input"], topk=TOPK)
            except Exception as e:
                print(f"[e1][warn] {dsname} sample {r['sample_id']} failed: {e}", flush=True)
                continue
            if res["eligible"]:
                n_eligible += 1
            rec = {"sample_id": r["sample_id"], "gold_seg": res["gold_seg"],
                   "n_seg": res["n_seg"], "cur_seg": res["cur_seg"],
                   "eligible": res["eligible"], "per_layer": res["per_layer"]}
            if "pair_id" in r:
                rec["pair_id"] = r["pair_id"]
                rec["condition"] = r["condition"]
            per_sample[dsname].append(rec)
            for pl in res["per_layer"]:
                acc = layer_agg[dsname][pl["layer"]]
                if pl["hit"] is not None:
                    acc["hit_sum"] += int(pl["hit"]); acc["hit_n"] += 1
                    acc["rank_sum"] += pl["gold_rank"]; acc["rank_n"] += 1
                if pl["amongkeys"] is not None:
                    acc["ak_sum"] += int(pl["amongkeys"]); acc["ak_n"] += 1
            print(f"[e1] {kind}/{dsname} {ri+1}/{len(rows)} eligible={res['eligible']}", flush=True)

        n_total = len(per_sample[dsname])
        per_layer_out = []
        best_layer, best_hit = None, -1.0
        for i in range(n_layers):
            acc = layer_agg[dsname][i]
            hit_rate = acc["hit_sum"] / acc["hit_n"] if acc["hit_n"] else None
            rank_mean = acc["rank_sum"] / acc["rank_n"] if acc["rank_n"] else None
            ak_acc = acc["ak_sum"] / acc["ak_n"] if acc["ak_n"] else None
            per_layer_out.append({"layer": i, "hit_at_2": hit_rate, "gold_rank_mean": rank_mean,
                                  "amongkeys_acc": ak_acc, "amongkeys_n": acc["ak_n"],
                                  "n_eligible": acc["hit_n"], "n_total": n_total})
            if hit_rate is not None and hit_rate > best_hit:
                best_hit, best_layer = hit_rate, i
        layer_agg[dsname] = {"per_layer": per_layer_out, "n_total": n_total,
                             "n_eligible": n_eligible, "n_ineligible": n_total - n_eligible,
                             "best_layer": best_layer, "best_layer_hit_at_2": best_hit if best_layer is not None else None}

        # annotate per-sample records with the dataset-level best-layer hit
        for rec in per_sample[dsname]:
            rec["best_layer"] = best_layer
            rec["best_layer_hit"] = (rec["per_layer"][best_layer]["hit"]
                                     if best_layer is not None else None)

    del model
    torch.cuda.empty_cache()
    return layer_agg, per_sample, n_layers


def _load_e2_baseline():
    """Task 7 산출물 results/e2_oracle.json (baseline rows, gen_eval.run_file 형식:
    {"rows": [{"index"/pair_id, "correct", "condition", ...}]} 류)을 관대하게 파싱해
    (condition, pair_id) -> correct 매핑을 만든다. 파일이 없거나 스키마를 못 알아보면
    None을 반환 — 호출부는 이를 e2_join: null 로 기록한다."""
    path = os.path.join(RES, "e2_oracle.json")
    if not os.path.exists(path):
        path = os.path.join(MC_OUT, "results", "e2_oracle.json")
    if not os.path.exists(path):
        return None
    try:
        obj = json.load(open(path))
        mapping = {}

        def ingest(rows, cond=None):
            for r in rows:
                c = r.get("condition", cond)
                pid = r.get("pair_id", r.get("index"))
                if c is not None and pid is not None and "correct" in r:
                    mapping[(c, pid)] = bool(r["correct"])

        if isinstance(obj, dict) and "rows" in obj:
            ingest(obj["rows"])
        elif isinstance(obj, dict):
            for cond, sub in obj.items():
                if isinstance(sub, dict) and "rows" in sub:
                    ingest(sub["rows"], cond=cond)
                elif isinstance(sub, list):
                    ingest(sub, cond=cond)
        elif isinstance(obj, list):
            ingest(obj)
        return mapping or None
    except Exception as e:
        print(f"[e1][warn] e2_oracle.json found but unparseable: {e}", flush=True)
        return None


def _compute_e2_join(per_sample_all):
    """model -> dataset(paired_*_multi) -> {"hit":{"correct":n,"n":n}, "miss":{...}}"""
    baseline = _load_e2_baseline()
    if baseline is None:
        return None
    out = {}
    for model, by_ds in per_sample_all.items():
        for dsname, recs in by_ds.items():
            if not dsname.startswith("paired_"):
                continue
            for rec in recs:
                key = (rec.get("condition"), rec.get("pair_id"))
                if key not in baseline or rec.get("best_layer_hit") is None:
                    continue
                bucket = "hit" if rec["best_layer_hit"] else "miss"
                out.setdefault(model, {}).setdefault(dsname, {}).setdefault(
                    bucket, {"correct": 0, "n": 0})
                cell = out[model][dsname][bucket]
                cell["n"] += 1
                cell["correct"] += int(baseline[key])
    if not out:
        return None
    # add rates
    for model, by_ds in out.items():
        for dsname, buckets in by_ds.items():
            for b, cell in buckets.items():
                cell["acc"] = cell["correct"] / cell["n"] if cell["n"] else None
    return out


def make_figure(results, out_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(results.keys())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        for dsname, agg in results[model].items():
            layers = [pl["layer"] for pl in agg["per_layer"]]
            hits = [pl["hit_at_2"] if pl["hit_at_2"] is not None else float("nan")
                    for pl in agg["per_layer"]]
            ax.plot(layers, hits, marker="o", label=dsname)
        ax.set_title(model)
        ax.set_xlabel("layer")
        ax.set_ylabel("hit@2")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)

    agg_path_mcout = os.path.join(MC_OUT, "results", "e1_routing.json")
    agg_path_repo = os.path.join(RES, "e1_routing.json")
    per_sample_path = os.path.join(MC_OUT, "results", "e1_per_sample.json")

    results = {}
    per_sample_all = {}
    for p in (agg_path_mcout, agg_path_repo):
        if os.path.exists(p):
            try:
                prev = json.load(open(p))
                results.update(prev.get("results", {}))
            except Exception:
                pass
    if os.path.exists(per_sample_path):
        try:
            per_sample_all = json.load(open(per_sample_path))
        except Exception:
            per_sample_all = {}

    layer_agg, per_sample, n_layers = run_model(a.model)
    results[a.model] = layer_agg
    per_sample_all[a.model] = per_sample

    e2_join = _compute_e2_join(per_sample_all)

    out = {"meta": {"topk": TOPK, "chunk": CHUNK,
                    "models": sorted(results.keys()),
                    "datasets": list(DATASETS.keys())},
           "results": results,
           "e2_join": e2_join}

    for p in (agg_path_mcout, agg_path_repo):
        json.dump(out, open(p, "w"), indent=2)
        print(f"[e1] wrote {p}", flush=True)

    json.dump(per_sample_all, open(per_sample_path, "w"))
    print(f"[e1] wrote {per_sample_path}", flush=True)

    fig_mcout = os.path.join(MC_OUT, "results", "e1_routing.png")
    fig_repo = os.path.join(RES, "e1_routing.png")
    make_figure(results, [fig_mcout, fig_repo])
    print(f"[e1] wrote {fig_mcout} and {fig_repo}", flush=True)

    # sanity print
    for dsname, agg in results.get(a.model, {}).items():
        print(f"[e1][sanity] {a.model}/{dsname}: best_layer={agg['best_layer']} "
              f"hit@2={agg['best_layer_hit_at_2']} n_eligible={agg['n_eligible']}/{agg['n_total']}",
              flush=True)


if __name__ == "__main__":
    main()
