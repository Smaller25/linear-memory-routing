"""E1 / ssketch: answer position에서 layer별 gold-chunk routing 정확도.

생성 불필요 — 프롬프트 1-pass. descriptor를 두 갈래로 두고 나란히 잰다.

  --descriptor meank   (기준선, 논문 Eq.16)
      u = ssc.connector(h); summaries = segment_key_sums(normalize(k))
      score = <u, c_i> (head 합)
  --descriptor ssketch (주 arm, 학습 파라미터 0)
      D_m = S_m @ P  (S_m = 청크 최종 상태, P = 시드 고정 random orthonormal [d_v, r])
      score(t,m) = || concat_h( D_{m,h}^T q_{t,h} ) ||_2      (fp32)
      q_t 는 커널이 실제로 읽을 때 쓰는 그 벡터(L2 정규화 + read_scale).

점수 수식은 `dsc/mc_sketch/scoring.py` 한 곳에만 있고 여기서는
`sketch_bridge`가 그 파일을 그대로 불러 쓴다(모델·커널은 pinned worktree 유지).

결과 파일 배치 (중요 — 이전 버전에서 바뀐 부분)
------------------------------------------------
예전에는 `results[model] = layer_agg` 라 arm을 여러 개 돌리면 조용히
덮어썼다. arm 조합을 키로 바꿔 그건 막았지만, 그 다음 버전은 **한 개의
`{tag}.json` 을 잡 시작 때 읽고 잡 끝에 통째로 다시 썼다**. 잡 10개를 GPU
2장에 흘리면 나중에 끝난 잡이 먼저 끝난 잡의 arm을 통째로 덮어써서 3시간짜리
결과가 경고 한 줄 없이 사라진다(파일 단위 read-modify-write 경쟁).

지금은 **잡마다 자기 파일에만 쓴다. 기존 파일을 읽지 않는다.**

    results/{tag}/{arm_key}.json                      (repo, 집계값)
    $MC_OUT/results/{tag}/{arm_key}.json              (동일 사본)
    $MC_OUT/results/{tag}/{arm_key}_per_sample.json   (per-sample 원자료)

파일명의 `{arm_key}` 는 arm 키의 `|` 를 `__` 로 바꾼 것이다:

    mc-5B|meank                      -> mc-5B__meank.json
    mc-5B|ssketch|r8|s0              -> mc-5B__ssketch__r8__s0.json
    mc-5B|ssketch|r8|s0|wu|raw_norm  -> ... (기본값이 아닌 축만 덧붙음)

각 shard 안의 `arm` 에 (model, descriptor, rank, seed, ...) 원본 값이 들어
있으므로 오프라인 집계는 문자열 파싱 없이 그걸 읽으면 된다. 쓰기는 tmp 파일 +
`os.replace` 원자 교체라 잡이 중간에 죽어도 잘린 JSON이 남지 않는다.

병합(`{tag}.json`)과 그림 생성은 GPU가 필요 없는 오프라인 단계
`ssketch_table.py` 가 shard 디렉터리를 glob 해서 한다.

`--out-tag e1_routing` 으로 이름을 바꿀 수 있다(0024 E1의 옛 단일 파일은
arm 정보가 없는 옛 스키마라 ssketch_table.py 가 legacy 경로로 읽는다).
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, data as mcdata
import sketch_bridge

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CHUNK, TOPK = load_mc.CHUNK, load_mc.TOPK

DATASETS = {
    "niah_single_1": {"kind": "a", "path": os.path.join(MC_OUT, "data", "2048", "niah_single_1", "validation.jsonl")},
    "niah_multikey_1": {"kind": "a", "path": os.path.join(MC_OUT, "data", "2048", "niah_multikey_1", "validation.jsonl")},
    # 0025 X2의 헤드라인 셀. gold needle이 여러 개라 macro_hit2(아래)로 읽는다.
    "niah_multivalue": {"kind": "a", "path": os.path.join(MC_OUT, "data", "2048", "niah_multivalue", "validation.jsonl")},
    "paired_S_multi": {"kind": "b", "path": os.path.join(MC_OUT, "data", "paired", "S.jsonl"), "condition": "S"},
    "paired_D_multi": {"kind": "b", "path": os.path.join(MC_OUT, "data", "paired", "D.jsonl"), "condition": "D"},
}


# ----------------------------------------------------------------------
# arm identity
# ----------------------------------------------------------------------
def arm_key(model, descriptor, rank, seed, router_query, online_score):
    if descriptor == "meank":
        return f"{model}|meank"
    key = f"{model}|ssketch|r{rank}|s{seed}"
    if router_query != "q" or online_score != "read_norm":
        key += f"|{router_query}|{online_score}"
    return key


def arm_slug(key):
    """arm 키를 파일 이름으로. `|` 는 경로에 쓰기 껄끄러워 `__` 로 바꾼다.
    ssketch_table.py 의 동명 함수와 반드시 같은 규칙이어야 한다."""
    return key.replace("|", "__")


def atomic_json_dump(obj, path, **kw):
    """tmp 파일에 쓰고 os.replace 로 원자 교체. 잡이 중간에 죽어도 잘린 JSON이
    남지 않는다(다른 잡이 그걸 읽고 조용히 빈 dict로 시작하는 사고 방지)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, **kw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def arm_meta(model, descriptor, rank, seed, router_query, online_score):
    meta = {"model": model, "descriptor": descriptor}
    if descriptor == "ssketch":
        meta.update(rank=rank, seed=seed, router_query=router_query,
                    online_score=online_score,
                    sketch_root=sketch_bridge.sketch_source_root())
    return meta


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
def routing_scores_at(attn, h, t, arm, P=None):
    """h [T,D] (bf16 cuda). t 위치의 과거 segment별 routing score [n_seg]
    (미래/현재는 -inf) 와 online 점수.

    두 arm 모두 점수는 fp32로 계산한다(프로토콜 CONSTANT 5)."""
    load_mc.bootstrap()
    if arm["descriptor"] == "ssketch":
        return sketch_bridge.routing_scores_ssketch(
            attn, h, t, P=P, router_query=arm["router_query"],
            online_score=arm["online_score"], chunk_size=attn.ssc.chunk_size)

    from dsc.mc_baseline.mc_ssc import segment_key_sums, causal_online_key_sums
    hb = h.unsqueeze(0)
    q, k, v, g, b, w = attn._project(hb)
    rk = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    summaries = segment_key_sums(rk, attn.ssc.chunk_size)          # [1,N,H,K]
    u = attn.ssc.connector(hb[:, t:t + 1]).view(1, 1, attn.ssc.num_heads,
                                                attn.ssc.head_qk_dim)
    scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())[0, 0]  # [N]
    osum = causal_online_key_sums(rk, attn.ssc.chunk_size)[:, t:t + 1]
    online = torch.einsum("bthk,bthk->bt", u.float(), osum.float())[0, 0]
    cur_seg = t // attn.ssc.chunk_size
    scores = scores.clone()
    scores[cur_seg:] = float("-inf")
    return scores, float(online)


def rank_order(scores):
    """PINNED 8: `-scores`에 **stable** argsort → 동점이면 낮은 segment 번호 우선.
    (probe_bilinear_router.hit_at_k 의 numpy `np.argsort(-s, kind="stable")` 와 동일)"""
    return torch.argsort(-scores.float(), stable=True).tolist()


def analyze_sample(model, tok, input_text, arm, P_by_layer=None, topk=TOPK):
    """gold-chunk routing 진단. 반환 per_layer 항목의 hit/gold_rank/amongkeys는
    gold_seg가 ELIGIBLE(과거 segment)일 때만 값을 채우고, 아니면 None
    (t==T-1 시점에 gold가 현재 segment에 있어 구조적으로 routing 불가능한 경우).

    macro_hit2는 multivalue용 보조 DV: 질의된 needle 중 ELIGIBLE인 것들에 대해
    "그 needle의 segment가 top-2 안에 있는가"의 평균 (0025 X2와 동일 정의).
    single/multikey에서는 gold needle이 1개라 hit과 같은 값이 된다."""
    ann = mcdata.annotate(input_text, tok)
    ids = torch.tensor([tok(input_text, add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    t = T - 1
    cur_seg = t // CHUNK
    eligible = ann["gold_seg"] < cur_seg
    key_segs = sorted({n["seg"] for n in ann["needles"]})
    eligible_key_segs = [s for s in key_segs if s < cur_seg]
    eligible_gold_needles = [n for n in ann["gold_needles"] if n["seg"] < cur_seg]
    hiddens = capture_hidden(model, ids)
    per_layer = []
    for i, attn in load_mc.mc_layers(model):
        P = P_by_layer.get(i) if P_by_layer else None
        s, online = routing_scores_at(attn, hiddens[i], t, arm, P=P)
        order = rank_order(s)
        top = set(order[:topk])
        if eligible:
            gold_rank = order.index(ann["gold_seg"])
            hit = ann["gold_seg"] in top
        else:
            gold_rank, hit = None, None
        macro_hit2 = None
        if eligible_gold_needles:
            macro_hit2 = sum(n["seg"] in top for n in eligible_gold_needles) / len(eligible_gold_needles)
        amongkeys = None
        if eligible and len(eligible_key_segs) >= 2:
            # ineligible samples (gold_seg == cur_seg) can never have gold in
            # eligible_key_segs (gold's own segment isn't eligible), so
            # amongkeys would be deterministically False there — gate it out.
            best_key_seg = max(eligible_key_segs, key=lambda ks: float(s[ks]))
            amongkeys = (best_key_seg == ann["gold_seg"])
        # top-1 과거 점수 대비 online 점수 — "online이 게이트를 먹는가" 진단용.
        best_past = float(s[order[0]]) if order and torch.isfinite(s[order[0]]) else None
        per_layer.append({"layer": i, "gold_rank": gold_rank, "hit": hit,
                          "macro_hit2": macro_hit2, "amongkeys": amongkeys,
                          "online_score": online, "best_past_score": best_past})
    return {"gold_seg": ann["gold_seg"], "gold_segs": ann["gold_segs"],
            "n_seg": ann["n_seg"], "cur_seg": cur_seg,
            "eligible": eligible, "n_eligible_key_segs": len(eligible_key_segs),
            "n_eligible_gold_needles": len(eligible_gold_needles),
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


def run_model(kind, arm):
    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(kind)
    layers = load_mc.mc_layers(model)
    n_layers = len(layers)
    print(f"[e1] {kind}: n_layers={n_layers} arm={arm}", flush=True)

    P_by_layer = None
    if arm["descriptor"] == "ssketch":
        # 모든 layer가 같은 P를 쓴다 (layer별로 다른 P를 쓰면 layer 프로파일
        # 비교가 P 노이즈와 뒤섞인다). d_v는 layer마다 같으므로 한 번만 만든다.
        d_v = layers[0][1].base.head_v_dim
        rank = arm["rank"] if arm["rank"] > 0 else d_v
        if rank > d_v:
            raise ValueError(f"rank {rank} > head_v_dim {d_v}")
        P = sketch_bridge.make_P(d_v, rank, arm["seed"], device="cuda")
        P_by_layer = {i: P for i, _ in layers}
        arm = dict(arm, rank=rank, head_v_dim=d_v, full_state=(rank == d_v))
        print(f"[e1] sketch P: d_v={d_v} r={rank} seed={arm['seed']} "
              f"full_state={arm['full_state']} src={sketch_bridge.sketch_source_root()}",
              flush=True)

    per_sample = {}  # dataset -> [ {sample_id, ..., per_layer:[...]} ]
    layer_agg = {}   # dataset -> layer -> accumulators

    for dsname in DATASETS:
        rows = _rows_for(dsname)
        per_sample[dsname] = []
        layer_agg[dsname] = {i: {"hit_sum": 0, "hit_n": 0, "rank_sum": 0.0, "rank_n": 0,
                                  "ak_sum": 0, "ak_n": 0, "mh_sum": 0.0, "mh_n": 0}
                             for i in range(n_layers)}
        n_eligible = 0
        cur_seg_sum = 0.0
        chance_sum = 0.0
        for ri, r in enumerate(rows):
            try:
                res = analyze_sample(model, tok, r["input"], arm,
                                     P_by_layer=P_by_layer, topk=TOPK)
            except Exception as e:
                print(f"[e1][warn] {dsname} sample {r['sample_id']} failed: {e}", flush=True)
                continue
            if res["eligible"]:
                n_eligible += 1
                cur_seg_sum += res["cur_seg"]
                chance_sum += min(1.0, 2.0 / res["cur_seg"])
            rec = {"sample_id": r["sample_id"], "gold_seg": res["gold_seg"],
                   "gold_segs": res["gold_segs"],
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
                if pl["macro_hit2"] is not None:
                    acc["mh_sum"] += pl["macro_hit2"]; acc["mh_n"] += 1
            print(f"[e1] {kind}/{dsname} {ri+1}/{len(rows)} eligible={res['eligible']}", flush=True)

        n_total = len(per_sample[dsname])
        per_layer_out = []
        best_layer, best_hit = None, -1.0
        for i in range(n_layers):
            acc = layer_agg[dsname][i]
            hit_rate = acc["hit_sum"] / acc["hit_n"] if acc["hit_n"] else None
            rank_mean = acc["rank_sum"] / acc["rank_n"] if acc["rank_n"] else None
            ak_acc = acc["ak_sum"] / acc["ak_n"] if acc["ak_n"] else None
            mh = acc["mh_sum"] / acc["mh_n"] if acc["mh_n"] else None
            per_layer_out.append({"layer": i, "hit_at_2": hit_rate, "gold_rank_mean": rank_mean,
                                  "macro_hit2": mh, "macro_hit2_n": acc["mh_n"],
                                  "amongkeys_acc": ak_acc, "amongkeys_n": acc["ak_n"],
                                  "n_eligible": acc["hit_n"], "n_total": n_total})
            if hit_rate is not None and hit_rate > best_hit:
                best_hit, best_layer = hit_rate, i
        mean_cur_seg = cur_seg_sum / n_eligible if n_eligible else None
        chance_hit2 = chance_sum / n_eligible if n_eligible else None
        layer_agg[dsname] = {"per_layer": per_layer_out, "n_total": n_total,
                             "n_eligible": n_eligible, "n_ineligible": n_total - n_eligible,
                             "mean_cur_seg": mean_cur_seg, "chance_hit2": chance_hit2,
                             "best_layer": best_layer,
                             "best_layer_hit_at_2": best_hit if best_layer is not None else None,
                             "best_layer_macro_hit2": (per_layer_out[best_layer]["macro_hit2"]
                                                       if best_layer is not None else None)}

        # annotate per-sample records with the dataset-level best-layer hit
        for rec in per_sample[dsname]:
            rec["best_layer"] = best_layer
            rec["best_layer_hit"] = (rec["per_layer"][best_layer]["hit"]
                                     if best_layer is not None else None)

    del model
    torch.cuda.empty_cache()
    return layer_agg, per_sample, n_layers, arm


def _load_e2_baseline(model):
    """Task 7 산출물 results/e2_oracle.json에서 해당 `model`의 baseline rows만
    읽어 (condition, pair_id) -> correct 매핑을 만든다. 실제 스키마:
    {"results": {model: {condition: {"baseline": {"rows": [...]}, "oracle": {...}}}},
     "rows": [...]}  — top-level "rows"는 전 모델을 model 구분 없이 평평하게
    이어붙인 것이라, (condition, pair_id)만으로 매핑하면 나중에 나오는 모델이
    앞 모델을 덮어쓴다(실측: mc-30B가 먼저, mc-5B가 나중에 나와 mc-5B가 항상
    이김 — Task 7 리뷰에서 발견된 버그). 반드시 results[model][condition]
    ["baseline"]["rows"]로 모델별로 읽는다. 파일/모델 키가 없으면 None
    (호출부는 이를 e2_join: null 로 기록)."""
    path = os.path.join(RES, "e2_oracle.json")
    if not os.path.exists(path):
        path = os.path.join(MC_OUT, "results", "e2_oracle.json")
    if not os.path.exists(path):
        return None
    try:
        obj = json.load(open(path))
        model_results = obj.get("results", {}).get(model)
        if not model_results:
            return None
        mapping = {}
        for cond, sub in model_results.items():
            baseline = sub.get("baseline") if isinstance(sub, dict) else None
            rows = baseline.get("rows", []) if isinstance(baseline, dict) else []
            for r in rows:
                pid = r.get("pair_id", r.get("index"))
                if pid is not None and "correct" in r:
                    mapping[(cond, pid)] = bool(r["correct"])
        return mapping or None
    except Exception as e:
        print(f"[e1][warn] e2_oracle.json found but unparseable for model={model}: {e}",
              flush=True)
        return None


def _compute_e2_join(per_sample_all, arms):
    """arm_key -> dataset(paired_*_multi) -> {"hit":{"correct":n,"n":n}, "miss":{...}}.

    baseline은 arm이 아니라 **모델**에 붙는 값이므로 arms[arm_key]["model"]로
    모델을 되찾아 per-model로 읽는다 (see _load_e2_baseline). 옛 JSON에서 온
    arm 정보 없는 키는 키 자체를 모델 이름으로 간주한다(하위 호환)."""
    out = {}
    cache = {}
    for key, by_ds in per_sample_all.items():
        model = arms.get(key, {}).get("model", key.split("|")[0])
        if model not in cache:
            cache[model] = _load_e2_baseline(model)
        baseline = cache[model]
        if baseline is None:
            continue
        for dsname, recs in by_ds.items():
            if not dsname.startswith("paired_"):
                continue
            for rec in recs:
                k = (rec.get("condition"), rec.get("pair_id"))
                if k not in baseline or rec.get("best_layer_hit") is None:
                    continue
                bucket = "hit" if rec["best_layer_hit"] else "miss"
                out.setdefault(key, {}).setdefault(dsname, {}).setdefault(
                    bucket, {"correct": 0, "n": 0})
                cell = out[key][dsname][bucket]
                cell["n"] += 1
                cell["correct"] += int(baseline[k])
    if not out:
        return None
    for key, by_ds in out.items():
        for dsname, buckets in by_ds.items():
            for b, cell in buckets.items():
                cell["acc"] = cell["correct"] / cell["n"] if cell["n"] else None
    return out


# 그림(layer × hit@2 프로파일)은 여러 arm을 한 장에 겹쳐 그려야 해서 잡 안에서
# 그리면 같은 PNG를 여러 잡이 동시에 덮어쓴다. `ssketch_table.py` 의
# make_figure 로 옮겼다(오프라인, GPU 불필요).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    ap.add_argument("--descriptor", default="meank", choices=["meank", "ssketch"])
    ap.add_argument("--rank", type=int, default=8,
                    help="ssketch 전용. head_v_dim(=128)과 같으면 full-state(P=I). "
                         "0이면 head_v_dim으로 자동 설정.")
    ap.add_argument("--seed", type=int, default=0,
                    help="ssketch 전용 P 시드. random P는 시드 민감 — 표에는 시드 3개 이상 mean±sd로.")
    ap.add_argument("--router-query", default="q", choices=["q", "wu"])
    ap.add_argument("--online-score", default="read_norm",
                    choices=["read_norm", "raw_norm", "legacy"])
    ap.add_argument("--out-tag", default="ssketch_routing",
                    help="shard 디렉터리 이름. results/<tag>/<arm_key>.json 에 이 잡의 "
                         "arm만 쓴다(합본·그림은 ssketch_table.py --tag <tag>).")
    a = ap.parse_args()

    key = arm_key(a.model, a.descriptor, a.rank, a.seed, a.router_query, a.online_score)
    meta = arm_meta(a.model, a.descriptor, a.rank, a.seed, a.router_query, a.online_score)
    slug = arm_slug(key)

    # 잡마다 자기 shard 에만 쓴다. 다른 잡의 파일은 읽지도 쓰지도 않는다
    # (GPU 2장 동시 실행 시의 read-modify-write 경쟁 제거). 병합·그림은
    # ssketch_table.py 가 오프라인에서 한다.
    shard_dirs = [os.path.join(RES, a.out_tag), os.path.join(MC_OUT, "results", a.out_tag)]
    for d in shard_dirs:
        os.makedirs(d, exist_ok=True)
    per_sample_path = os.path.join(MC_OUT, "results", a.out_tag, f"{slug}_per_sample.json")

    layer_agg, per_sample, n_layers, resolved_arm = run_model(a.model, meta)

    e2_join = _compute_e2_join({key: per_sample}, {key: resolved_arm})
    e2_join_arm = e2_join.get(key) if e2_join else None

    out = {"meta": {"topk": TOPK, "chunk": CHUNK, "length": 2048,
                    "arm_key": key,
                    "model": a.model,
                    "n_layers": n_layers,
                    "datasets": list(DATASETS.keys()),
                    "tie_break": "stable argsort on -scores (lower segment wins)",
                    "score_dtype": "fp32",
                    "eval_position": "t = T-1",
                    "worktree": load_mc.WORKTREE,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
           "arm_key": key,
           "arm": resolved_arm,
           "results": layer_agg,
           "e2_join": e2_join_arm}

    # per-sample 원자료 먼저(집계규칙 10) — 집계 shard가 있으면 원자료도 있다는
    # 순서를 보장한다.
    atomic_json_dump({"arm_key": key, "arm": resolved_arm, "per_sample": per_sample},
                     per_sample_path)
    print(f"[e1] wrote {per_sample_path}", flush=True)
    for d in shard_dirs:
        p = os.path.join(d, f"{slug}.json")
        atomic_json_dump(out, p, indent=2)
        print(f"[e1] wrote {p}", flush=True)
    print(f"[e1] merge/figure: python ssketch_table.py --tag {a.out_tag}", flush=True)

    # sanity print
    for dsname, agg in layer_agg.items():
        print(f"[e1][sanity] {key}/{dsname}: best_layer={agg['best_layer']} "
              f"hit@2={agg['best_layer_hit_at_2']} "
              f"macro_hit2={agg['best_layer_macro_hit2']} "
              f"n_eligible={agg['n_eligible']}/{agg['n_total']}",
              flush=True)


if __name__ == "__main__":
    main()
