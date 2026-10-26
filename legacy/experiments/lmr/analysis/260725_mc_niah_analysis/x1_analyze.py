"""X1 CPU analysis (rev2, spec §2 / plan Task 2): H_blind descriptor-only test
(2a) + u_t:=q_t rescoring (2b) + conditional X1c centering + X1c top-1 PC removal.

--- Post-review update (review findings 1-2) ---
Two changes from the first-pass implementation, per code review:

1. **`stock` and the R2 regression target are now the dump's own canonical
   `stock_scores`, not a recompute.** The original implementation derived
   both from `score_from(u, c_full)` on fp16-cast `u`/`c_full` -- a *second*
   fp16 quantization on top of whatever quantization already happened when
   the dump itself was written, since the dump's `stock_scores` were computed
   once (in fp32/bf16, at the model's native forward-pass precision) and then
   stored as f32, while `u`/`c_full` are stored as f16. Re-deriving through
   the f16-cast copies is a "double quantization" that can flip which segment
   wins a near-tied top-2 slot -- observed empirically (e.g. mc-5B/
   niah_single_1 layer 14: 0.911 via recompute vs 0.933 canonical; 10 cells
   affected, up to 6.25pp at n=16). `score_from(u, c_full)` is now used ONLY
   as a sanity-check comparison against the canonical dumped scores (see
   `meta.sanity`), never as a reported condition or regression target.
   `u_eq_q` necessarily stays a recompute (no dumped q-score exists in the
   npz schema) -- `meta.score_provenance` documents which conditions are
   canonical-dumped vs recomputed, and the ≤1-flip-at-n=16 noise convention
   for comparing across that boundary.

2. **Added top-1 PC removal as the informative X1c variant** (spec §2a's
   actual alternative to plain centering, not substituted for it here --
   centering is kept, its rank-invariance note below still holds and is
   still correct algebra). Two new hit2 conditions, `u_recomp_pc1` and
   `u_eq_q_pc1`: per (sample,layer), take the top-1 right singular vector of
   the eligible normalized descriptors (`c_hat`), project it out of both the
   query and every `c_hat_i`, and rescore in that projected space. Unlike
   centering (an additive-constant shift -> provably rank-invariant), this
   is a genuine change of basis and can move rankings. See `x1c.pc1_note`.
--------------------------------------------------------

전부 CPU numpy-only (GPU 불필요). Task 1 덤프($MC_OUT/x1_dump/{model}/{dataset}/
{ri}.npz + {ri}.meta.json)만 읽는다. 추가 forward pass 없음.

설계 결정 (구현 시 확정, 스펙 텍스트가 모호했던 지점):
  - `c_full`(덤프에 이미 저장된 seg.mean(0) 정확값)을 그대로 쓴다.
    `csub_raw`에서 재구성하지 않음 — Task 1이 이미 재구성 오차 <2e-3을 검증했고
    csub_raw는 32배 크기라 불필요한 IO/메모리.
  - 2a 통계(R², pred_jaccard, gram_offdiag)는 `cur_seg >= 2`인 eligible 샘플만
    사용한다 (top-2 집합, 샘플 내 z-score 모두 최소 2개 후보가 필요). hit@2는
    e1과 동일 규약으로 cur_seg>=1인 모든 eligible 샘플에 적용(제약 없음).
  - centering = c_i - mean_j(c_j) (원 c, ĉ 아님) — Task 2 브리프 문구 그대로.
    stock_centered/u_eq_q_centered 점수는 이 centered c로 재채점.
    **중요한 발견 (버그 아님)**: score_i = <query, c_i>가 선형이므로
    <query, c_i - mean> = <query, c_i> - <query, mean>이고 mean은 i에 대해
    상수다. 즉 centering은 그 샘플의 모든 segment 점수에 "동일한" 상수를
    더하는 것과 수학적으로 동치라 **순위(top-2, hit@2)를 절대 바꿀 수
    없다** — renormalize 없이는. 그래서 stock_centered/u_eq_q_centered의
    hit@2는 stock/u_eq_q와 부동소수점 노이즈 이하로 항상 동일하다(실측:
    로컬 백업 덤프 전 layer·전 dataset에서 정확히 일치). rev1 스펙의 원래
    문구는 ĉ(정규화)를 centering한 뒤 재정규화 또는 top-1 PC 제거를
    함의했을 가능성이 있으나(§0.2 "1/(1-⟨cos⟩) 배 개선" 예측이 그 근거),
    이번 브리프는 raw c 위에서의 순수 뺄셈만 명시했으므로 그대로 구현하고
    이 no-op 사실을 결과에 명시한다(x1c.centering_ranking_invariant_note).
    지표를 바꾸지 않는다는 원칙(스펙 §7)을 지키기 위해 임의로 재정규화를
    추가하지 않았음 — 필요하면 후속 지시로 별도 조건을 추가할 것.
  - centered 조건은 항상 계산해두고(싸다), 전역 gram_offdiag 최댓값이
    GRAM_TRIGGER(0.8) 이상인 layer가 하나라도 있으면 최종 JSON에 전부 포함,
    아니면 null + 이유 기록 (X1c 트리거 규약).
  - pos/norm/outlier 각각 argmax-2 방향(high/low)을 모두 시도해 layer 단위로
    더 나은 쪽을 채택하고 그 방향을 기록 (pos뿐 아니라 세 특징 모두 일반화).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
DUMP_ROOT = os.path.join(MC_OUT, "x1_dump")
RES = os.path.join(HERE, "results")

MODELS = ["mc-5B", "mc-30B"]
DATASETS = ["niah_single_1", "niah_multikey_1", "paired_S_multi", "paired_D_multi"]
TOPK = 2
GRAM_TRIGGER = 0.8
MIN_SEGS_2A = 2          # cur_seg >= 2 required for r2/jaccard/gram (need >=2 candidates)
STOCK_SANITY_TOL = 0.5   # abs score-unit gate; empirically observed max diff ~0.05


# ------------------------------------------------------------------
# pure numpy building blocks (unit-tested in tests/lmr/test_mc_niah_x1.py)
# ------------------------------------------------------------------

def flatten_hk(x):
    """[...,H,K] -> [...,H*K]."""
    return x.reshape(*x.shape[:-2], -1)


def score_from(query_flat, c_flat):
    """query_flat: [D] (flattened [H,K]); c_flat: [n,D] (flattened [H,K] per
    segment) -> [n]. score_i = sum_d(query_d * c_i_d).

    This is byte-for-byte the same contraction as routing_stats.routing_scores_at's
    `torch.einsum("bthk,bnhk->btn", u, summaries)`: that einsum has no free h or
    k index on the output, so it sums over BOTH heads and the head_qk_dim
    jointly — there is no separate per-head score followed by a head-reduction
    (max/mean/etc). Flattening [H,K] -> [H*K] and taking a plain dot product
    reproduces exactly that joint sum; no head-reduction *choice* was made or
    needed (coordinator review point 3 / spec §2b ask). u and q are both
    stored [L,H,K] head-split for this reason — flatten_hk() undoes the split
    right before this call, for both conditions identically."""
    return c_flat @ query_flat


def normalize_rows(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def top1_right_singular_vector(x):
    """x: [n,D] (n>=2) -> unit-norm top-1 right singular vector [D], via
    np.linalg.svd on x directly (uncentered -- see module docstring for why
    uncentered-on-c_hat was chosen over centered-on-c_hat: centering c_hat
    first would remove the very shared direction we want SVD to *find and
    report as v1* before we explicitly project it out of query+descriptors
    below; doing it uncentered on already-unit-norm rows makes v1 the
    dominant common direction directly). Sign of the returned vector is
    whatever SVD picks (arbitrary) -- projection removal `x - (x.v)v` is
    invariant to the sign of v, so this is safe."""
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[0]


def project_out(v, x):
    """Remove the component along unit vector v from x. x may be [D] (a
    single flat query) or [n,D] (a stack of descriptor rows) -- handles
    both since `x @ v` broadcasts correctly in either case."""
    coef = x @ v
    if x.ndim == 1:
        return x - coef * v
    return x - np.outer(coef, v)


def top2_set(scores, k=TOPK):
    """Indices of the top-k scores (descending; ties broken by original index
    via a stable sort on -scores)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    return set(order[:k].tolist())


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def pos_feature(n):
    if n <= 1:
        return np.zeros(n)
    return np.arange(n) / (n - 1)


def descriptor_features(c_flat):
    """c_flat: [n,D] raw (un-normalized) descriptors for one sample's eligible
    segments (n>=1). Returns (pos[n], norm[n], outlier[n], c_hat[n,D],
    gram_offdiag: float or nan if n<2)."""
    n = c_flat.shape[0]
    pos = pos_feature(n)
    norm = np.linalg.norm(c_flat, axis=1)
    c_hat = normalize_rows(c_flat)
    mean_hat = c_hat.mean(axis=0)
    denom = np.linalg.norm(mean_hat) + 1e-8
    # Deliberate deviation from spec §2a's literal `1 - <c_hat_i, mean_j(c_hat_j)>`:
    # dividing by `denom` here makes this `1 - cos(c_hat_i, mean_direction)` (cosine
    # to the *normalized* mean direction), not a dot product with the raw
    # (sub-unit-norm) mean vector. The two differ by the scalar factor `denom`
    # (<=1), which only rescales `outlier` uniformly per sample -- it does not
    # change any ranking/argmax-2 use of this feature. Numerically negligible
    # here regardless: observed gram_offdiag is >=0.96 everywhere in this dump,
    # so `denom` = ||mean_hat|| is already close to 1 (mean of near-collinear
    # unit vectors has norm close to 1), making the deviation immaterial.
    outlier = 1.0 - (c_hat @ mean_hat) / denom
    if n > 1:
        gram = c_hat @ c_hat.T
        off = gram[~np.eye(n, dtype=bool)]
        gram_offdiag = float(off.mean())
    else:
        gram_offdiag = float("nan")
    return pos, norm, outlier, c_hat, gram_offdiag


def zscore(x):
    mu, sd = x.mean(), x.std()
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def lstsq_r2(X, y):
    """X: [M,F] (no intercept col), y: [M] -> (coef incl. intercept [F+1], r2, pred[M]).
    r2 = nan if y has zero variance (ss_tot==0)."""
    Xi = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    pred = Xi @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return coef, r2, pred


def feature_top2_jaccard(feat, actual_top2, direction):
    """feat: [n] single-feature values for one sample.
    direction: "high" (argmax-2) or "low" (argmin-2)."""
    if direction == "high":
        order = np.argsort(-np.asarray(feat), kind="stable")
    else:
        order = np.argsort(np.asarray(feat), kind="stable")
    return jaccard(set(order[:TOPK].tolist()), actual_top2)


# ------------------------------------------------------------------
# dump loading
# ------------------------------------------------------------------

def load_dataset_samples(model, dataset):
    """Load all *eligible* samples for (model,dataset). Returns list of dicts:
    {meta, u:[L,H,K] f16, q:[L,H,K] f16, c_full:[L,N,H,K] f16, stock_scores:[L,N] f32}.
    Ineligible samples (meta['eligible'] is False) are skipped entirely, per spec."""
    d = os.path.join(DUMP_ROOT, model, dataset)
    files = sorted(glob.glob(os.path.join(d, "*.npz")),
                    key=lambda p: int(os.path.basename(p)[:-4]))
    out = []
    n_total = 0
    for f in files:
        n_total += 1
        meta = json.load(open(f.replace(".npz", ".meta.json")))
        if not meta.get("eligible", False):
            continue
        z = np.load(f)
        out.append({
            "meta": meta,
            "u": z["u"],
            "q": z["q"],
            "c_full": z["c_full"],
            "stock_scores": z["stock_scores"],
        })
    return out, n_total


# ------------------------------------------------------------------
# per (model,dataset,layer) analysis
# ------------------------------------------------------------------

def analyze_layer(samples, layer, sanity_tracker):
    """samples: list from load_dataset_samples (eligible only).
    Returns a dict with r2/feat_r2/pred_jaccard/gram_offdiag/hit2
    (hit2 always includes *_centered and *_pc1 — caller decides whether to
    keep them, gated on the same X1c trigger).
    Mutates sanity_tracker (dict with 'max_diff','n','worst',
    'n_top2_mismatch') in place with the stock-recompute-vs-dumped
    comparison for this layer (sanity-check role ONLY — see review finding 1:
    the canonical `stock` condition and the R2 regression target both use the
    dump's own `stock_scores`, not this recompute)."""
    pooled_X, pooled_y = [], []          # for combined + per-feature R^2 (cur_seg>=2 only)
    per_sample_feats = []                # for jaccard direction search (cur_seg>=2 only)
    gram_vals = []
    hit = {"stock": [], "u_eq_q": [], "stock_centered": [], "u_eq_q_centered": [],
           "u_recomp_pc1": [], "u_eq_q_pc1": []}

    for s in samples:
        meta = s["meta"]
        cur_seg = meta["cur_seg"]
        gold_seg = meta["gold_seg"]
        if cur_seg < 1:
            continue  # structurally impossible (eligible implies gold_seg<cur_seg<=n_seg-1)

        c_full_layer = flatten_hk(s["c_full"][layer].astype(np.float32))   # [N,D]
        u_flat = s["u"][layer].astype(np.float32).reshape(-1)              # [D]
        q_flat = s["q"][layer].astype(np.float32).reshape(-1)
        dumped_scores = s["stock_scores"][layer]                           # [N] f32, -inf at cur_seg+

        elig_c = c_full_layer[:cur_seg]                                    # [cur_seg,D]

        # CANONICAL stock condition + R2 target: the dump's own stock_scores,
        # exactly as served (fp32 already at dump time -- no fp16 u/c_full
        # re-derivation involved). This is what "stock" and the regression
        # target below use.
        score_stock_canonical = dumped_scores[:cur_seg].astype(np.float32)
        actual_top2 = top2_set(score_stock_canonical)
        hit["stock"].append(gold_seg in actual_top2)

        # sanity gate ONLY: recompute stock from fp16 u/c_full and compare to
        # the canonical dumped score above. Not used for any condition or the
        # R2 target (review finding 1: doing so double-quantizes through
        # fp16 and can flip single samples at n=16, up to 6.25pp).
        score_stock_recomp = score_from(u_flat, elig_c)                    # [cur_seg]
        diff = float(np.abs(score_stock_recomp - score_stock_canonical).max())
        sanity_tracker["n"] += 1
        sanity_tracker.setdefault("n_top2_mismatch", 0)
        if top2_set(score_stock_recomp) != actual_top2:
            sanity_tracker["n_top2_mismatch"] += 1
        if diff > sanity_tracker["max_diff"]:
            sanity_tracker["max_diff"] = diff
            sanity_tracker["worst"] = {"layer": layer, "sample_id": meta.get("sample_id"),
                                        "diff": diff}

        # u_eq_q necessarily stays recomputed -- no dumped q-score exists in
        # the npz schema (only stock_scores, which is u-based).
        score_uq = score_from(q_flat, elig_c)
        hit["u_eq_q"].append(gold_seg in top2_set(score_uq))

        mean_c = elig_c.mean(axis=0)
        c_centered = elig_c - mean_c
        score_stock_c = score_from(u_flat, c_centered)
        score_uq_c = score_from(q_flat, c_centered)
        hit["stock_centered"].append(gold_seg in top2_set(score_stock_c))
        hit["u_eq_q_centered"].append(gold_seg in top2_set(score_uq_c))

        # X1c top-1 PC removal (informative variant, spec §2a "Gram
        # off-diagonal >= 0.8" branch). Lives entirely in recomputed space:
        # v1 is the top-1 right singular vector of the eligible normalized
        # descriptors c_hat (uncentered SVD on c_hat, per module docstring),
        # projected out of BOTH the query (u or q) and every c_hat_i, then
        # rescored with a plain dot product in that projected space. No
        # dumped-score counterpart exists for this (the dump has no
        # PC1-removed stock score), so there is no "stock_pc1" condition --
        # only u_recomp_pc1 (recomputed-u path) and u_eq_q_pc1
        # (recomputed-q path).
        c_hat = normalize_rows(elig_c)
        if cur_seg >= 2:
            v1 = top1_right_singular_vector(c_hat)
            c_hat_pc1 = project_out(v1, c_hat)
            u_pc1 = project_out(v1, u_flat)
            q_pc1 = project_out(v1, q_flat)
        else:
            # PC1 undefined for a single candidate; hit@2 is trivially
            # satisfied when cur_seg==1 (only one eligible candidate, gold
            # must be it) regardless of the score, so this fallback is a
            # no-op for the metric either way.
            c_hat_pc1, u_pc1, q_pc1 = c_hat, u_flat, q_flat
        score_u_recomp_pc1 = score_from(u_pc1, c_hat_pc1)
        score_uq_pc1 = score_from(q_pc1, c_hat_pc1)
        hit["u_recomp_pc1"].append(gold_seg in top2_set(score_u_recomp_pc1))
        hit["u_eq_q_pc1"].append(gold_seg in top2_set(score_uq_pc1))

        if cur_seg < MIN_SEGS_2A:
            continue

        pos, norm, outlier, _, gram_offdiag = descriptor_features(elig_c)
        if not np.isnan(gram_offdiag):
            gram_vals.append(gram_offdiag)

        y = zscore(score_stock_canonical)
        X = np.stack([pos, norm, outlier], axis=1)
        pooled_X.append(X)
        pooled_y.append(y)
        per_sample_feats.append({"pos": pos, "norm": norm, "outlier": outlier,
                                  "top2": actual_top2})

    n_hit = {k: (sum(v) / len(v) if v else None) for k, v in hit.items()}

    if not pooled_X:
        return {
            "r2": None, "feat_r2": {"pos": None, "norm": None, "outlier": None},
            "pred_jaccard": {"pos": None, "norm": None, "outlier": None, "combined": None},
            "gram_offdiag": None, "hit2": n_hit, "n_used_2a": 0,
        }

    Xall = np.concatenate(pooled_X, axis=0)
    yall = np.concatenate(pooled_y, axis=0)

    _, r2_combined, _ = lstsq_r2(Xall, yall)
    feat_r2 = {}
    for j, name in enumerate(("pos", "norm", "outlier")):
        _, r2j, _ = lstsq_r2(Xall[:, j:j + 1], yall)
        feat_r2[name] = r2j

    coef_full, _, _ = lstsq_r2(Xall, yall)

    pred_jaccard = {}
    for name in ("pos", "norm", "outlier"):
        j_hi = np.mean([feature_top2_jaccard(sf[name], sf["top2"], "high")
                        for sf in per_sample_feats])
        j_lo = np.mean([feature_top2_jaccard(sf[name], sf["top2"], "low")
                        for sf in per_sample_feats])
        if j_hi >= j_lo:
            pred_jaccard[name] = {"jaccard": float(j_hi), "direction": "high"}
        else:
            pred_jaccard[name] = {"jaccard": float(j_lo), "direction": "low"}

    combined_js = []
    for sf in per_sample_feats:
        fitted = (coef_full[0] + coef_full[1] * sf["pos"] + coef_full[2] * sf["norm"]
                  + coef_full[3] * sf["outlier"])
        combined_js.append(feature_top2_jaccard(fitted, sf["top2"], "high"))
    pred_jaccard["combined"] = float(np.mean(combined_js))

    gram_offdiag = float(np.mean(gram_vals)) if gram_vals else None

    return {
        "r2": r2_combined, "feat_r2": feat_r2,
        "pred_jaccard": pred_jaccard, "gram_offdiag": gram_offdiag,
        "hit2": n_hit, "n_used_2a": len(pooled_X),
    }


def chance_hit2_of(samples):
    vals = [min(1.0, 2.0 / s["meta"]["cur_seg"]) for s in samples if s["meta"]["cur_seg"] > 0]
    return float(np.mean(vals)) if vals else None


# ------------------------------------------------------------------
# driver
# ------------------------------------------------------------------

def analyze_all():
    sanity_tracker = {"max_diff": 0.0, "n": 0, "worst": None, "n_top2_mismatch": 0}
    results = {}
    max_gram_overall = float("-inf")
    max_gram_loc = None

    for model in MODELS:
        results[model] = {}
        for dataset in DATASETS:
            samples, n_total = load_dataset_samples(model, dataset)
            if not samples:
                print(f"[x1_analyze][warn] no eligible samples for {model}/{dataset}",
                      flush=True)
                continue
            n_layers = samples[0]["u"].shape[0]
            per_layer = []
            for layer in range(n_layers):
                stats = analyze_layer(samples, layer, sanity_tracker)
                stats["layer"] = layer
                per_layer.append(stats)
                if stats["gram_offdiag"] is not None and stats["gram_offdiag"] > max_gram_overall:
                    max_gram_overall = stats["gram_offdiag"]
                    max_gram_loc = f"{model}/{dataset}/layer{layer}"
            results[model][dataset] = {
                "n_total": n_total, "n_eligible": len(samples),
                "mean_cur_seg": float(np.mean([s["meta"]["cur_seg"] for s in samples])),
                "chance_hit2": chance_hit2_of(samples),
                "per_layer": per_layer,
            }
            print(f"[x1_analyze] {model}/{dataset}: n_eligible={len(samples)}/{n_total} "
                  f"n_layers={n_layers}", flush=True)

    triggered = max_gram_overall >= GRAM_TRIGGER
    ALL_COND = ["stock", "u_eq_q", "stock_centered", "u_eq_q_centered",
                "u_recomp_pc1", "u_eq_q_pc1"]
    x1c = {"triggered": triggered, "max_gram_offdiag": (None if max_gram_overall == float("-inf")
                                                          else max_gram_overall),
           "trigger_loc": max_gram_loc if triggered else None,
           "centering_ranking_invariant_note": (
               "centering=c_i-mean_j(c_j) with a plain linear dot-product score "
               "shifts every eligible segment's score in a sample by the SAME "
               "additive constant (<query,mean> is i-independent), so top-2/hit@2 "
               "CANNOT differ from the uncentered condition -- this is an algebraic "
               "identity, not an empirical result. stock_centered/u_eq_q_centered "
               "hit2 values below are expected to equal stock/u_eq_q exactly "
               "(mod float noise) for this reason. This is why the figure omits "
               "stock_centered/u_eq_q_centered (visually-overlapping duplicates) and "
               "shows the informative top-1-PC-removal variants instead -- see "
               "pc1_note below."),
           "pc1_note": (
               "top-1 PC removal is the informative X1c variant (spec §2a: 'top-1 PC "
               "제거' as an alternative to plain centering). Per (sample,layer): v1 = "
               "top-1 right singular vector of the eligible normalized descriptors "
               "c_hat (np.linalg.svd, uncentered on c_hat -- see module docstring), "
               "then v1 is projected out of BOTH the query and every c_hat_i, and the "
               "rescored dot product is taken in that projected space. This is NOT "
               "rank-invariant like plain centering: it is a genuine change of basis, "
               "not an additive-constant shift. Two conditions exist: u_recomp_pc1 "
               "(query=u, recomputed-space path) and u_eq_q_pc1 (query=q, "
               "recomputed-space path). There is no 'stock_pc1' condition -- the "
               "canonical dumped stock_scores has no accessible descriptor-space to "
               "intervene on (only score_from(u, c_full) does, which is the same "
               "recomputed path u_recomp_pc1 already uses), so naming it 'stock_pc1' "
               "would misleadingly imply it shares stock's canonical-dumped "
               "provenance when it does not.")}
    if not triggered:
        x1c["reason"] = (f"max gram_offdiag observed across all model/dataset/layer = "
                          f"{max_gram_overall:.4f} (at {max_gram_loc}) < trigger threshold "
                          f"{GRAM_TRIGGER} — centered/pc1 conditions computed but not reported")
    else:
        x1c["reason"] = (f"gram_offdiag >= {GRAM_TRIGGER} at {max_gram_loc} "
                          f"(max={max_gram_overall:.4f}) — centered/pc1 conditions reported for ALL layers")

    # finalize per-layer output: strip/keep centered+pc1 hit2 depending on
    # trigger, add best-layer summaries per condition
    for model in results:
        for dataset in results[model]:
            entry = results[model][dataset]
            conditions = ["stock", "u_eq_q"] + (
                ["stock_centered", "u_eq_q_centered", "u_recomp_pc1", "u_eq_q_pc1"]
                if triggered else [])
            best_layer = {}
            for cond in ALL_COND:
                if cond not in conditions:
                    best_layer[cond] = None
                    for pl in entry["per_layer"]:
                        pl["hit2"][cond] = None
                    continue
                best_l, best_v = None, -1.0
                for pl in entry["per_layer"]:
                    v = pl["hit2"].get(cond)
                    if v is not None and v > best_v:
                        best_v, best_l = v, pl["layer"]
                best_layer[cond] = {"layer": best_l, "hit2": best_v if best_l is not None else None}
            entry["best_layer"] = best_layer

    sanity = {
        "stock_recompute_max_abs_diff": sanity_tracker["max_diff"],
        "n_compared": sanity_tracker["n"],
        "n_top2_mismatch": sanity_tracker["n_top2_mismatch"],
        "tol": STOCK_SANITY_TOL,
        "pass": sanity_tracker["max_diff"] < STOCK_SANITY_TOL,
        "worst": sanity_tracker["worst"],
        "role": (
            "SANITY CHECK ONLY: compares score_from(u, dumped c_full) (fp16-derived "
            "recompute) against the dump's own canonical stock_scores (fp32 at dump "
            "time) on every eligible segment. NOT used as the stock condition or the "
            "R2 regression target -- both of those use stock_scores directly (see "
            "meta.score_provenance). n_top2_mismatch counts (model,dataset,layer,"
            "sample) tuples where the recomputed top-2 set differs from the "
            "canonical dumped top-2 set (the double-quantization flips review "
            "finding 1 flagged)."),
    }

    out = {
        "meta": {"topk": TOPK, "models": MODELS, "datasets": DATASETS,
                 "gram_trigger_threshold": GRAM_TRIGGER,
                 "min_eligible_segs_for_2a": MIN_SEGS_2A,
                 "head_reduction_note": (
                     "u/q stored [L,H,K] head-split; scoring flattens H,K -> H*K "
                     "and takes a plain dot product with c_full (also flattened), "
                     "which is byte-identical to routing_scores_at's "
                     "einsum('bthk,bnhk->btn', ...) joint sum over heads+K. No "
                     "separate per-head reduction (max/mean) was needed or applied "
                     "for either u or u_eq_q scoring — both go through score_from() "
                     "identically."),
                 "score_provenance": {
                     "stock": (
                         "CANONICAL: the dump's own stock_scores[layer][:cur_seg], "
                         "used as-is (no re-derivation from fp16 u/c_full). This is "
                         "exactly what the serving path computed."),
                     "u_eq_q": (
                         "RECOMPUTED: score_from(q, c_full) -- no dumped q-score "
                         "exists in the npz schema (only u-based stock_scores is "
                         "dumped), so this condition is necessarily recomputed from "
                         "fp16 q/c_full."),
                     "stock_centered": "RECOMPUTED (see x1c.centering_ranking_invariant_note).",
                     "u_eq_q_centered": "RECOMPUTED (see x1c.centering_ranking_invariant_note).",
                     "u_recomp_pc1": "RECOMPUTED (see x1c.pc1_note); query=u.",
                     "u_eq_q_pc1": "RECOMPUTED (see x1c.pc1_note); query=q.",
                     "cross_condition_caveat": (
                         "stock is canonical-dumped while every other condition is "
                         "recomputed from fp16 u/q/c_full. At n=16 eligible samples "
                         "(the typical per-cell count in this dump), a single sample "
                         "flip moves hit@2 by 1/16=6.25pp -- treat cross-condition "
                         "differences of <=1 flip (<=6.25pp at n=16, scale "
                         "proportionally at other n) as noise, not signal. See "
                         "meta.sanity for the measured max recompute-vs-canonical "
                         "discrepancy and top-2 mismatch count."),
                 },
                 "sanity": sanity},
        "x1c": x1c,
        "results": results,
    }
    return out


def _load_e1(model, dataset):
    path = os.path.join(RES, "e1_routing.json")
    if not os.path.exists(path):
        return None
    try:
        e1 = json.load(open(path))
        return e1["results"][model][dataset]
    except Exception:
        return None


def make_figure(out, out_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(m, d) for m in MODELS for d in DATASETS if d in out["results"].get(m, {})]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.2 * len(rows)), squeeze=False)

    for row, (model, dataset) in enumerate(rows):
        entry = out["results"][model][dataset]
        layers = [pl["layer"] for pl in entry["per_layer"]]
        r2 = [pl["r2"] if pl["r2"] is not None else float("nan") for pl in entry["per_layer"]]
        gram = [pl["gram_offdiag"] if pl["gram_offdiag"] is not None else float("nan")
                for pl in entry["per_layer"]]

        axL = axes[row][0]
        axL.plot(layers, r2, marker="o", color="tab:blue", label="R2 (combined)")
        axL.set_ylabel("R2", color="tab:blue")
        axL.set_ylim(-0.1, 1.05)
        axL.set_title(f"{model}/{dataset}: R2 & gram_offdiag")
        axL.grid(alpha=0.3)
        axR = axL.twinx()
        axR.plot(layers, gram, marker="s", color="tab:red", linestyle="--",
                 label="gram_offdiag")
        axR.axhline(GRAM_TRIGGER, color="tab:red", alpha=0.3, linestyle=":")
        axR.set_ylabel("gram_offdiag", color="tab:red")
        axL.set_xlabel("layer")

        axH = axes[row][1]
        # NOTE: stock_centered/u_eq_q_centered are intentionally omitted here
        # -- they are algebraically forced to equal stock/u_eq_q (see
        # x1c.centering_ranking_invariant_note), so plotting them would just
        # be visually-overlapping duplicate lines. u_recomp_pc1/u_eq_q_pc1
        # are the informative X1c variants (genuine change of basis, not an
        # additive-constant shift) and are shown instead.
        for cond, color in [("stock", "tab:blue"), ("u_eq_q", "tab:orange"),
                             ("u_recomp_pc1", "tab:green"), ("u_eq_q_pc1", "tab:purple")]:
            vals = [pl["hit2"].get(cond) for pl in entry["per_layer"]]
            if all(v is None for v in vals):
                continue
            vals = [v if v is not None else float("nan") for v in vals]
            axH.plot(layers, vals, marker="o", color=color, label=cond)
        e1_entry = _load_e1(model, dataset)
        if e1_entry is not None:
            e1_hits = [pl["hit_at_2"] if pl["hit_at_2"] is not None else float("nan")
                       for pl in e1_entry["per_layer"]]
            axH.plot([pl["layer"] for pl in e1_entry["per_layer"]], e1_hits,
                     color="gray", linestyle=":", label="e1 stock (ref)")
        chance = entry.get("chance_hit2")
        if chance is not None:
            axH.axhline(chance, color="black", alpha=0.3, linestyle="--", label="chance")
        axH.set_ylim(0, 1.05)
        axH.set_xlabel("layer")
        axH.set_ylabel("hit@2")
        axH.set_title(f"{model}/{dataset}: hit@2 (stock vs recomputed)")
        axH.legend(fontsize=6, loc="lower right")
        axH.grid(alpha=0.3)

    fig.text(0.5, 0.002,
              "note: stock is the canonical dumped score; u_eq_q/u_recomp_pc1/u_eq_q_pc1 are "
              "recomputed (fp16 u/q/c_full). descriptor centering (c_i-mean_j c_j) is omitted "
              "-- it is algebraically rank-invariant for a linear dot-product score (see JSON "
              "x1c.centering_ranking_invariant_note); pc1 removal (shown) is the informative "
              "variant since it is a genuine change of basis, not an additive shift.",
              ha="center", va="bottom", fontsize=6, wrap=True)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)

    out = analyze_all()

    json_repo = os.path.join(RES, "x1_router_probe.json")
    json_mcout = os.path.join(MC_OUT, "results", "x1_router_probe.json")
    for p in (json_repo, json_mcout):
        json.dump(out, open(p, "w"), indent=2)
        print(f"[x1_analyze] wrote {p}", flush=True)

    png_repo = os.path.join(RES, "x1_router_probe.png")
    png_mcout = os.path.join(MC_OUT, "results", "x1_router_probe.png")
    make_figure(out, [png_repo, png_mcout])
    print(f"[x1_analyze] wrote {png_repo} and {png_mcout}", flush=True)

    print(f"[x1_analyze][sanity] max_abs_diff={out['meta']['sanity']['stock_recompute_max_abs_diff']:.4f} "
          f"(tol={STOCK_SANITY_TOL}) pass={out['meta']['sanity']['pass']}", flush=True)
    print(f"[x1_analyze][x1c] triggered={out['x1c']['triggered']} "
          f"max_gram_offdiag={out['x1c']['max_gram_offdiag']}", flush=True)


if __name__ == "__main__":
    main()
