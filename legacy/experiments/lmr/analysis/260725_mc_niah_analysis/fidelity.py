"""E3: gold segment state의 write/read fidelity, layer별·조건별.

r = q·M ~= (q·k*)v* + interference 분해의 각 인자를 직접 측정.

State layout sanity (task-8 brief 요구사항 #3): chunk_gdn2(...,
transpose_state_layout=False[기본값])이 반환하는 final_state의 shape는 학습
코드(dsc/mc_gdn2/ssc.py의 _segment_gdn2_batched 주석 "state_bn: [B*N,H,K,V]")
및 커널 소스(dsc/lit_gpt/gdn2_ops/chunk_kda.py:1637-1638,
`final_state = k.new_zeros(N, H, K, V, ...)`)에서 명시적으로 [B,H,K,V] 순서로
구성됨을 확인했다 (K축이 먼저, V축이 나중 — 변수 이름으로 결정되는 것이라
mc_370M에서 K==V==128이라도 순서는 모호하지 않다). 따라서
`torch.einsum("hk,hkv->hv", k_normed, state)`가 맞는 축 순서다. 아래
`_axis_order_sanity` 는 이를 첫 (모델, layer, 샘플)에서 실측으로도 재확인해
로그에 남긴다 (두 축 순서 중 실제로 v를 더 잘 재구성하는 쪽을 출력).
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, data as mcdata
from routing_stats import capture_hidden

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CHUNK = load_mc.CHUNK

_chunk_gdn2 = None


def _get_chunk_gdn2():
    global _chunk_gdn2
    if _chunk_gdn2 is None:
        load_mc.bootstrap()
        from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
        _chunk_gdn2 = chunk_gdn2
    return _chunk_gdn2


def _scan(q, k, v, g, b, w, sl):
    """학습과 동일한 chunk_gdn2 경로로 [sl] 구간을 zero-state 스캔 -> 최종 state [H,K,V].

    training(_segment_gdn2_batched, dsc/mc_gdn2/ssc.py)과 동일한 인자로 호출:
    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=False, cu_seqlens=None. 학습은 bf16 autocast 아래에서
    _project + 이 스캔을 실행하므로 호출부에서도 동일하게 autocast(bf16) 안에서
    호출한다(호출부 참고: sample_metrics).
    """
    chunk_gdn2 = _get_chunk_gdn2()
    _, state = chunk_gdn2(q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl],
                          b=b[:, sl], w=w[:, sl], initial_state=None,
                          output_final_state=True, use_qk_l2norm_in_kernel=True,
                          use_gate_in_kernel=False, cu_seqlens=None)
    return state[0].float()                                    # [H,K,V]


def _cos(a, b, dim=-1):
    return float(F.cosine_similarity(a.flatten(0, -2), b.flatten(0, -2), dim=dim).mean())


_AXIS_SANITY_DONE = {"done": False}


def _axis_order_sanity(kn_t, m, v_t, tag):
    """(1회) 두 가지 축 해석 중 실제로 v를 더 잘 재구성하는 쪽을 실측·로그.

    해석 A (brief/코드가 쓰는 것): state=[H,K,V], v_hat = einsum('hk,hkv->hv', k, M).
    해석 B (뒤바뀐 경우): state를 [H,V,K]로 읽었다고 가정하고 v_hat = einsum('hk,hvk->hv', k, M).
    코드 레벨에서 이미 [B,H,K,V]임을 확인했지만(주석 참고), 실측으로도 재확인한다.
    """
    if _AXIS_SANITY_DONE["done"]:
        return
    _AXIS_SANITY_DONE["done"] = True
    v_hat_a = torch.einsum("hk,hkv->hv", kn_t, m)
    v_hat_b = torch.einsum("hk,hvk->hv", kn_t, m)
    cos_a = _cos(v_hat_a, v_t)
    cos_b = _cos(v_hat_b, v_t)
    print(f"[e3][axis-sanity]{tag} state.shape={tuple(m.shape)} "
          f"cos(A: hk,hkv->hv)={cos_a:.4f} cos(B: hk,hvk->hv)={cos_b:.4f} "
          f"-> using {'A' if cos_a >= cos_b else 'B (UNEXPECTED, investigate!)'}",
          flush=True)
    if cos_b > cos_a:
        print("[e3][axis-sanity][WARN] interpretation B beats A on this sample — "
              "state axis order may not be [H,K,V] as assumed! Investigate before trusting results.",
              flush=True)


@torch.no_grad()
def sample_metrics(model, tok, row, axis_tag=""):
    ann = mcdata.annotate(row["input"], tok)
    ids = torch.tensor([tok(row["input"], add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    hiddens = capture_hidden(model, ids)
    gold = ann["gold_seg"]
    tgt = next(n for n in ann["needles"] if n["key"] == ann["query_key"])
    v_pos = list(range(tgt["tok_start"], tgt["tok_end"] + 1))   # value 토큰들
    seg_sl = slice(gold * CHUNK, min((gold + 1) * CHUNK, T))
    after_sl = slice(gold * CHUNK, tgt["tok_end"] + 1)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, attn in load_mc.mc_layers(model):
            h = hiddens[i].unsqueeze(0)
            q, k, v, g, b, w = attn._project(h)
            kn = F.normalize(k.float(), p=2, dim=-1)
            m_after = _scan(q, k, v, g, b, w, after_sl)         # [H,K,V]
            m_final = _scan(q, k, v, g, b, w, seg_sl)
            if axis_tag:
                _axis_order_sanity(kn[0, v_pos[-1]], m_final, v[0, v_pos[-1]].float(),
                                   f" layer={i} {axis_tag}")
            # b-1: value 토큰별 k로 재독출 -> 실제 v와 cosine (토큰 평균)
            b1a, b1f = [], []
            for t in v_pos:
                kt, vt = kn[0, t], v[0, t].float()              # [H,K],[H,V]
                b1a.append(_cos(torch.einsum("hk,hkv->hv", kt, m_after), vt))
                b1f.append(_cos(torch.einsum("hk,hkv->hv", kt, m_final), vt))
            # b-2: answer position query
            qn = F.normalize(q[0, T - 1].float(), p=2, dim=-1)   # [H,K]
            t_last = v_pos[-1]
            qk = float((qn * kn[0, t_last]).sum(-1).mean())      # head 평균 정렬도
            r = torch.einsum("hk,hkv->hv", qn, m_final)          # [H,V]
            sig = (qn * kn[0, t_last]).sum(-1, keepdim=True) * v[0, t_last].float()
            interf = float((r - sig).norm() / (sig.norm() + 1e-8))
            out.append({"layer": i, "b1_after": sum(b1a) / len(b1a),
                        "b1_final": sum(b1f) / len(b1f), "b2_qk_align": qk,
                        "b2_interf_ratio": interf,
                        "_r": r.cpu()})                          # single-vs-multi 비교용
    return out


def _rows_for(cond):
    path = os.path.join(MC_OUT, "data", "paired", f"{cond}.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def run_model(kind):
    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(kind)
    n_layers = len(load_mc.mc_layers(model))
    print(f"[e3] {kind}: n_layers={n_layers}", flush=True)

    # cond -> pair_id -> variant -> per_layer (list of dicts incl. "_r")
    raw = {}
    for cond in ("S", "D"):
        rows = _rows_for(cond)
        raw[cond] = {}
        for ri, r in enumerate(rows):
            pid, variant = r["pair_id"], r["variant"]
            tag = f"{kind}/{cond}" if ri == 0 else ""
            try:
                per_layer = sample_metrics(model, tok, r, axis_tag=tag)
            except Exception as e:
                print(f"[e3][warn] {kind}/{cond} pair={pid} variant={variant} failed: {e}",
                      flush=True)
                continue
            raw[cond].setdefault(pid, {})[variant] = per_layer
            print(f"[e3] {kind}/{cond} {ri+1}/{len(rows)} pair={pid} variant={variant}",
                  flush=True)

    # pairing pass: attach b2_read_cos_vs_single to multi records, strip "_r"
    per_sample = {cond: [] for cond in ("S", "D")}
    layer_agg = {cond: {"single": {i: {} for i in range(n_layers)},
                        "multi": {i: {} for i in range(n_layers)}}
                 for cond in ("S", "D")}
    metric_keys = ("b1_after", "b1_final", "b2_qk_align", "b2_interf_ratio")

    for cond in ("S", "D"):
        for pid, variants in raw[cond].items():
            single_pl = variants.get("single")
            for variant, pl in variants.items():
                clean_layers = []
                for i, entry in enumerate(pl):
                    rec = {k: v for k, v in entry.items() if k != "_r"}
                    if variant == "multi" and single_pl is not None:
                        rec["b2_read_cos_vs_single"] = _cos(entry["_r"], single_pl[i]["_r"])
                    clean_layers.append(rec)
                    acc = layer_agg[cond][variant].setdefault(i, {})
                    for mk in metric_keys:
                        acc.setdefault(mk, []).append(rec[mk])
                    if "b2_read_cos_vs_single" in rec:
                        acc.setdefault("b2_read_cos_vs_single", []).append(
                            rec["b2_read_cos_vs_single"])
                per_sample[cond].append({"pair_id": pid, "variant": variant,
                                         "per_layer": clean_layers})

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    agg_out = {cond: {"single": [], "multi": []} for cond in ("S", "D")}
    for cond in ("S", "D"):
        for variant in ("single", "multi"):
            for i in range(n_layers):
                acc = layer_agg[cond][variant].get(i, {})
                entry = {"layer": i, "n": len(acc.get("b1_after", []))}
                for mk in metric_keys:
                    entry[mk] = _mean(acc.get(mk, []))
                if variant == "multi":
                    entry["b2_read_cos_vs_single"] = _mean(acc.get("b2_read_cos_vs_single", []))
                agg_out[cond][variant].append(entry)

    del model
    torch.cuda.empty_cache()
    return agg_out, per_sample, n_layers


def make_figure(results, out_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(results.keys())
    if not models:
        return
    metrics = ["b1_final", "b1_after", "b2_qk_align", "b2_read_cos_vs_single"]
    titles = {"b1_final": "b1_final (write fidelity, full segment)",
              "b1_after": "b1_after (write fidelity, up-to-value)",
              "b2_qk_align": "b2_qk_align (answer-query x value-key alignment)",
              "b2_read_cos_vs_single": "b2_read_cos_vs_single (multi read vs single, same pair)"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.flatten()
    colors = {"S": "tab:blue", "D": "tab:orange"}
    linestyles = {"mc-5B": "--", "mc-30B": "-"}
    for ax, metric in zip(axes, metrics):
        for model in models:
            res = results[model]
            for cond in ("S", "D"):
                if cond not in res:
                    continue
                variants = ("multi",) if metric == "b2_read_cos_vs_single" else ("single", "multi")
                for variant in variants:
                    per_layer = res[cond].get(variant)
                    if not per_layer:
                        continue
                    layers = [pl["layer"] for pl in per_layer]
                    ys = [pl.get(metric) if pl.get(metric) is not None else float("nan")
                          for pl in per_layer]
                    marker = "o" if variant == "multi" else "x"
                    alpha = 1.0 if variant == "multi" else 0.6
                    ax.plot(layers, ys, marker=marker, alpha=alpha,
                           color=colors.get(cond, None),
                           linestyle=linestyles.get(model, "-"),
                           label=f"{model}-{cond}-{variant}")
        ax.set_title(titles[metric], fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
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

    agg_path_mcout = os.path.join(MC_OUT, "results", "e3_fidelity.json")
    agg_path_repo = os.path.join(RES, "e3_fidelity.json")
    per_sample_path = os.path.join(MC_OUT, "results", "e3_per_sample.json")

    results = {}
    for p in (agg_path_mcout, agg_path_repo):
        if os.path.exists(p):
            try:
                prev = json.load(open(p))
                results.update(prev.get("results", {}))
            except Exception:
                pass
    per_sample_all = {}
    if os.path.exists(per_sample_path):
        try:
            per_sample_all = json.load(open(per_sample_path))
        except Exception:
            per_sample_all = {}

    agg_out, per_sample, n_layers = run_model(a.model)
    results[a.model] = agg_out
    per_sample_all[a.model] = per_sample

    out = {"meta": {"chunk": CHUNK, "models": sorted(results.keys()),
                    "conditions": ["S", "D"], "variants": ["single", "multi"],
                    "metrics": ["b1_after", "b1_final", "b2_qk_align",
                               "b2_interf_ratio", "b2_read_cos_vs_single"]},
           "results": results}

    for p in (agg_path_mcout, agg_path_repo):
        json.dump(out, open(p, "w"), indent=2)
        print(f"[e3] wrote {p}", flush=True)

    json.dump(per_sample_all, open(per_sample_path, "w"))
    print(f"[e3] wrote {per_sample_path}", flush=True)

    fig_mcout = os.path.join(MC_OUT, "results", "e3_fidelity.png")
    fig_repo = os.path.join(RES, "e3_fidelity.png")
    make_figure(results, [fig_mcout, fig_repo])
    print(f"[e3] wrote {fig_mcout} and {fig_repo}", flush=True)

    # sanity prints (task-8 brief gates)
    for cond in ("S", "D"):
        single_layers = agg_out[cond]["single"]
        b1a_vals = [pl["b1_after"] for pl in single_layers if pl["b1_after"] is not None]
        mean_b1a = sum(b1a_vals) / len(b1a_vals) if b1a_vals else float("nan")
        n_high = sum(1 for v in b1a_vals if v >= 0.5)
        print(f"[e3][sanity] {a.model}/{cond}/single: mean b1_after={mean_b1a:.4f} "
              f"({n_high}/{len(b1a_vals)} layers >= 0.5)", flush=True)
        if mean_b1a < 0.3:
            print(f"[e3][WARN] {a.model}/{cond}/single b1_after looks collapsed "
                  "(<0.3 mean) -- investigate before trusting results!", flush=True)
        multi_layers = agg_out[cond]["multi"]
        rc_vals = [pl["b2_read_cos_vs_single"] for pl in multi_layers
                  if pl.get("b2_read_cos_vs_single") is not None]
        mean_rc = sum(rc_vals) / len(rc_vals) if rc_vals else float("nan")
        print(f"[e3][sanity] {a.model}/{cond}/multi: mean b2_read_cos_vs_single={mean_rc:.4f}",
              flush=True)
        if cond == "D" and mean_rc < 0.5:
            print(f"[e3][WARN] {a.model}/D b2_read_cos_vs_single looks low (<0.5 mean) -- "
                  "check pair_id join logic!", flush=True)


if __name__ == "__main__":
    main()
