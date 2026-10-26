"""probe_bilinear_router.py -- CPU-only decisive pre-check for exp 0026 (SRLA).

QUESTION
--------
0025 showed the MC-SSC router's scoring is essentially query-independent, and
that cheap geometric fixes (descriptor centering, top-1 PC removal, u:=q) do
not repair it. 0026/SRLA's central bet is that a *learned* bilinear form does:
SRLA scores `Sim(q_t, f_m) = cos(W_q q_t, W_desc f_m)`, which for ranking
purposes is the bilinear form `q_t^T M f_m` with `M = W_q^T W_desc`, rank <= R.
0025 tested exactly ONE point of that family: `M = I` (the `u:=q` condition,
which failed). Nobody tested whether ANY `M` in the family works.

So: **is gold-chunk identity linearly decodable from (q_t, f_m) by a
rank-<=R bilinear form fitted with gold-chunk supervision, measured on a
HELD-OUT split?**

WHAT IS FITTED
--------------
Exactly SRLA's scoring function, nothing more:

    logits_m = cos(A_q q_t, A_f f_m) / tau,   A_q, A_f in R^{R x HK}

trained by cross-entropy against the gold segment index over the *eligible*
descriptor bank of each sample. tau is FIXED at 0.07 (not learned): for a
fixed (A_q, A_f) the temperature cannot change the ranking, hence cannot
change hit@k -- it only rescales the loss landscape. Fixing it removes one
nuisance degree of freedom from an already tiny-n fit.

Note a consequence of SRLA's cosine that is worth stating explicitly: because
`cos(A_q q, A_f f)` is invariant to rescaling `f`, this family *cannot* use
descriptor norm as a routing feature. That is faithful to SRLA, not a
limitation of the probe.

THREE DESIGN DECISIONS THAT MATTER (read before trusting any number)
-------------------------------------------------------------------
1. **Exact train-span dimension reduction.** HK = H*K = 16*128 = 2048, while
   n_train is 12-35 samples. An unconstrained `M` is 2048^2 = 4.2M parameters
   against ~12 labelled examples. Fitting in raw HK is not just slow, it is
   *unidentifiable*: gradients of A_q live in the row space of the training
   queries (dim <= n_train) and gradients of A_f in the row space of the
   training descriptors, so from small init GD can never place mass outside
   those spans, and any mass there is unconstrained by the data anyway. We
   therefore project q and f onto orthonormal bases of the TRAIN-ONLY spans
   (dq <= n_train, df <= n_train * S_max) and fit there. This is an exact
   reparameterisation of what GD-from-small-init can learn, and it is
   train-only so there is no test leakage.

   A direct corollary, and itself a finding: for these n, the *effective*
   capacity of the whole bilinear family saturates at R = dq <= n_train.
   For paired_D/paired_S (n=16, n_train=12) "R = 64" and "R = full" are
   literally the same model as R = 12. We report the effective rank actually
   achievable per cell and de-duplicate the requested rank grid accordingly,
   rather than printing five columns that are secretly three.

2. **Held-out = repeated grouped K-fold out-of-fold (OOF), not one split.**
   A single 50/50 split of paired_D leaves 8 test samples, i.e. a hit@2
   resolution of 12.5pp -- the brief's own "mark it inconclusive" threshold.
   Instead we use K=4 grouped folds x N_REPEATS reshuffles: every sample gets
   a prediction from a model that never saw it (train 12, test 4 per fold),
   and the pooled OOF evaluation therefore has n_test = n_eligible (16/45/47),
   not n/2. Grouping is by `pair_id` for the paired datasets so a pair is
   never split across train/test (in this dump each pair_id in fact appears
   once *within* a dataset -- the S/D partner lives in the other dataset --
   so group-disjoint reduces to sample-disjoint here, but the grouping is
   implemented rather than assumed). For unpaired datasets the group key is
   the enumerate-position filename, NOT `meta.sample_id` (not unique).

3. **Two controls without which a "win" would be meaningless.** The gold
   position is badly non-uniform in this dump (niah_multikey_1: predicting
   the two most frequent gold positions and ignoring the input entirely
   scores 0.638 in-sample vs 0.486 chance; paired_D: 0.562, which is
   coincidentally almost exactly the stock router's measured 0.563). A fitted
   `M` can absorb that prior through whatever positional drift the segment
   descriptors carry, and it *will* generalise to held-out data, while being
   worth nothing for routing. So we report alongside every fit:
     - `posprior`: train-frequency position prior, no features at all;
     - `shuffled_q`: the identical fit with queries permuted within train and
       within test (destroys the q<->bank correspondence, preserves every
       marginal and any position prior).
   The decisive quantity is real-minus-shuffled_q, paired-bootstrapped, not
   real-minus-chance.

REFERENCES REPORTED (all on the same samples as the OOF evaluation)
------------------------------------------------------------------
  stock  : the dump's canonical `stock_scores` (= 0025's measured number).
  M_eye  : the refuted `u:=q` point, cos-free dot product q . f_m.
  chance : mean over samples of min(1, k/n_eligible).
  posprior / shuffled_q : the two controls above.
Plus fit-on-train hit@k, so overfitting is visible.

HYPERPARAMETERS AND THE "ORACLE" CAVEAT
---------------------------------------
weight decay is swept over {0, 1e-2, 1e-1} and the reported headline picks the
wd with the best *test* hit@2. That is selection on the test metric and is
therefore optimistically biased -- deliberately so. This is a go/no-go gate on
whether the family has ANY held-out signal, so we want its upper bound: a
negative result under oracle wd selection is a strong negative, whereas a
positive result is only a lead and must additionally clear `shuffled_q`.
All wd values are kept in the JSON.

OUTPUTS
-------
  results/probe_bilinear_router.json  (+ mirrored to $MC_OUT/results/)
  results/probe_bilinear_router.png
Written by main(); nothing in 0025's X-series is read except x1_dump/ and
nothing of it is modified.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
DUMP_ROOT = os.path.join(MC_OUT, "x1_dump")
RES = os.path.join(HERE, "results")

MODELS = ["mc-5B", "mc-30B"]
# decisive cases first (the 0025 failures), then the two sanity controls
DATASETS = ["paired_D_multi", "niah_multikey_1", "niah_single_1", "paired_S_multi"]

RANKS = [1, 4, 16, 64, "full"]
WDS = [0.0, 1e-2, 1e-1]
TAU = 0.07
N_FOLDS = 4
N_REPEATS = 5
STEPS = 250
LR = 0.05
SVD_TOL = 1e-6
N_BOOT = 2000
MIN_TEST_N = 8          # brief's inconclusive threshold on the OOF sample count
SMALL_N_WARN = 20       # cells below this get an explicit "small n" flag


# ==================================================================
# loading (conventions copied from x1_analyze.load_dataset_samples)
# ==================================================================

def load_dataset_samples(model, dataset):
    """Eligible samples only, exactly as x1_analyze does it: files sorted by the
    integer enumerate-position filename, `meta['eligible'] is False` skipped
    outright. The group key is that filename index (`ri`), because
    `meta.sample_id` is NOT unique in this dump and must never be used as a key."""
    d = os.path.join(DUMP_ROOT, model, dataset)
    files = sorted(glob.glob(os.path.join(d, "*.npz")),
                   key=lambda p: int(os.path.basename(p)[:-4]))
    out, n_total = [], 0
    for f in files:
        n_total += 1
        meta = json.load(open(f.replace(".npz", ".meta.json")))
        if not meta.get("eligible", False):
            continue
        ri = int(os.path.basename(f)[:-4])
        z = np.load(f)
        out.append({
            "ri": ri,
            "meta": meta,
            "q": z["q"],                      # [L,H,K] f16, L2-normalised retrieval query
            "c_full": z["c_full"],            # [L,N,H,K] f16, MC-SSC descriptor
            "stock_scores": z["stock_scores"],  # [L,N] f32, -inf at >= cur_seg
        })
    return out, n_total


def group_key(s):
    """pair_id when the dataset is paired, else the enumerate-position filename."""
    pid = s["meta"].get("pair_id")
    return ("pair", pid) if pid is not None else ("ri", s["ri"])


# ==================================================================
# pure-numpy helpers
# ==================================================================

def hit_at_k(scores, gold, k):
    """scores: [n_valid] (higher = better). Stable argsort on -scores, matching
    x1_analyze.top2_set's tie convention (ties -> lower segment index first)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return float(gold in set(order[:k].tolist()))


def span_basis(X, tol=SVD_TOL):
    """Orthonormal basis [d, D] of the row space of X [m, D], keeping singular
    directions with s > tol * s_max. Returns (basis, d)."""
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((0, X.shape[1]), dtype=X.dtype), 0
    d = int(np.sum(s > tol * s[0]))
    d = max(d, 1)
    return vt[:d], d


def make_boot_index(groups, n_boot=N_BOOT, seed=0):
    """Precompute a [n_boot, n] resample index matrix for a GROUPED bootstrap.

    Every bootstrap in a cell resamples the same groups, so the index matrix is
    built once and reused by every CI/delta call -- turning thousands of
    python-level resample loops into a couple of vectorised fancy-index means.
    Groups are tuples, hence keyed by str().  Group sizes are equal (1 here) in
    this dump; if they were not, the resampled row count could vary per draw,
    so we resample groups and then truncate/pad to n rows only when sizes are
    uniform, asserting that condition.
    """
    gs = np.array([str(g) for g in groups])
    uniq = sorted(set(gs.tolist()))
    idxs = [np.where(gs == g)[0] for g in uniq]
    sizes = {len(i) for i in idxs}
    assert len(sizes) == 1, f"non-uniform group sizes {sizes}; grouped bootstrap needs a rewrite"
    mat = np.stack(idxs)                      # [G, gsize]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(idxs), size=(n_boot, len(idxs)))
    return mat[pick].reshape(n_boot, -1)      # [n_boot, n]


def bootstrap_ci(per_sample, boot):
    """(mean, se, lo, hi) of a per-sample 0/1-ish vector under `boot`."""
    per_sample = np.asarray(per_sample, dtype=np.float64)
    if per_sample.size == 0:
        return None, None, None, None
    draws = per_sample[boot].mean(axis=1)
    return (float(per_sample.mean()), float(draws.std(ddof=1)),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def paired_delta_ci(a, b, boot):
    """Grouped paired bootstrap of mean(a) - mean(b) (same samples, same order).
    `p_ge_0` is the bootstrap fraction of draws <= 0, i.e. a one-sided
    'no better than the reference' tail -- descriptive, not a calibrated test
    at these n."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    draws = a[boot].mean(axis=1) - b[boot].mean(axis=1)
    return {"delta": float(a.mean() - b.mean()), "se": float(draws.std(ddof=1)),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "p_le_0": float(np.mean(draws <= 0.0))}


# ==================================================================
# the batched bilinear fit
# ==================================================================

def fit_bilinear_batch(qz_tr, fz_tr, mask_tr, gold_tr, wds, rank, steps=STEPS, seed=0):
    """Fit `len(wds)` independent (A_q, A_f) pairs in one batched Adam run.

    Batching is free here (measured: B=6 costs the same wall time as B=1 on
    CPU at these sizes) and keeps every wd on an identical optimisation
    trajectory modulo its own penalty. The loss is the SUM over batch of each
    member's mean CE, so each member sees exactly the gradient magnitude it
    would see if fitted alone -- i.e. LR semantics are unchanged by batching.

    qz_tr [n,dq], fz_tr [n,S,df], mask_tr [n,S] bool, gold_tr [n].
    Returns (Aq [B,R,dq], Af [B,R,df]) as numpy.
    """
    import torch
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    B = len(wds)
    n, dq = qz_tr.shape
    df = fz_tr.shape[2]
    R = int(rank)

    q = torch.from_numpy(qz_tr.astype(np.float32))
    f = torch.from_numpy(fz_tr.astype(np.float32))
    m = torch.from_numpy(mask_tr)
    g = torch.from_numpy(gold_tr.astype(np.int64))
    wd = torch.tensor(wds, dtype=torch.float32).view(B, 1, 1)

    Aq = (torch.randn(B, R, dq) / np.sqrt(dq)).requires_grad_()
    Af = (torch.randn(B, R, df) / np.sqrt(df)).requires_grad_()
    opt = torch.optim.Adam([Aq, Af], lr=LR)
    neg = torch.finfo(torch.float32).min / 4

    for _ in range(steps):
        zq = Fn.normalize(torch.einsum("nd,brd->bnr", q, Aq), dim=-1)
        zf = Fn.normalize(torch.einsum("nsd,brd->bnsr", f, Af), dim=-1)
        lg = (zq[:, :, None, :] * zf).sum(-1) / TAU
        lg = torch.where(m[None], lg, torch.full_like(lg, neg))
        ce = Fn.cross_entropy(lg.reshape(B * n, -1), g.repeat(B), reduction="none")
        ce = ce.view(B, n).mean(dim=1)
        pen = (wd.squeeze(-1).squeeze(-1)
               * ((Aq ** 2).sum(dim=(1, 2)) + (Af ** 2).sum(dim=(1, 2))))
        (ce + pen).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    return Aq.detach().numpy(), Af.detach().numpy()


def score_bilinear(Aq, Af, qz, fz):
    """Aq [B,R,dq], Af [B,R,df], qz [n,dq], fz [n,S,df] -> logits [B,n,S]
    (cosine, tau omitted -- irrelevant to ranking)."""
    zq = np.einsum("nd,brd->bnr", qz, Aq)
    zf = np.einsum("nsd,brd->bnsr", fz, Af)
    zq /= (np.linalg.norm(zq, axis=-1, keepdims=True) + 1e-8)
    zf /= (np.linalg.norm(zf, axis=-1, keepdims=True) + 1e-8)
    return np.einsum("bnr,bnsr->bns", zq, zf)


# ==================================================================
# per (model, dataset, layer)
# ==================================================================

def build_layer_arrays(samples, layer):
    """-> Q [n,HK] f32, Fpad [n,S_max,HK] f32, mask [n,S_max], gold [n], cur [n],
    stock [n,S_max] (-inf padded)."""
    n = len(samples)
    cur = np.array([s["meta"]["cur_seg"] for s in samples], dtype=int)
    gold = np.array([s["meta"]["gold_seg"] for s in samples], dtype=int)
    S = int(cur.max())
    HK = samples[0]["q"].shape[1] * samples[0]["q"].shape[2]
    Q = np.empty((n, HK), dtype=np.float32)
    Fp = np.zeros((n, S, HK), dtype=np.float32)
    mask = np.zeros((n, S), dtype=bool)
    stock = np.full((n, S), -np.inf, dtype=np.float64)
    for i, s in enumerate(samples):
        Q[i] = s["q"][layer].astype(np.float32).reshape(-1)
        c = s["c_full"][layer].astype(np.float32).reshape(s["c_full"].shape[1], -1)
        k = cur[i]
        Fp[i, :k] = c[:k]
        mask[i, :k] = True
        stock[i, :k] = s["stock_scores"][layer][:k].astype(np.float64)
    return Q, Fp, mask, gold, cur, stock


def posprior_predict(gold_tr, cur_te, S):
    """Train-frequency position prior; returns a per-test-sample score vector
    [n_te,S] (count of that position in train, -inf where invalid) so it can go
    through the same hit_at_k path as everything else."""
    counts = np.zeros(S, dtype=np.float64)
    for g in gold_tr:
        counts[g] += 1
    out = np.tile(counts, (len(cur_te), 1))
    for i, k in enumerate(cur_te):
        out[i, k:] = -np.inf
    return out


def analyze_layer(samples, layer, groups, folds_per_repeat, boot):
    Q, Fp, mask, gold, cur, stock = build_layer_arrays(samples, layer)
    n, S = mask.shape
    HK = Q.shape[1]

    # ---- static references on all samples (the OOF eval set is all samples) --
    # `M_eye` is the refuted `u:=q` point of 0025: plain dot product q . f_m,
    # i.e. the bilinear form with M = I. Same contraction as x1_analyze's
    # score_from (joint sum over H and K after flattening).
    #
    # `M_eye_neg` / `stock_neg` are the SIGN-FLIPPED versions. They look like a
    # curiosity and are not: M = -I is a zero-parameter member of the very
    # rank-<=R bilinear family 0026 wants to learn, and 0025 tested M = +I
    # without testing it. Where the sign-flip scores far above chance, the
    # descriptor space demonstrably carries gold-chunk information that the
    # stock scorer is reading with the wrong sign -- which is a much cheaper
    # story than a learned module, and it is also the mechanism behind the
    # probe's near-perfect single-needle numbers (see the .md).
    ref = {}
    eye_scores = np.einsum("nsd,nd->ns", Fp, Q)
    eye_scores = np.where(mask, eye_scores, -np.inf)
    stock_neg = np.where(mask, -np.where(np.isfinite(stock), stock, 0.0), -np.inf)
    eye_neg = np.where(mask, -eye_scores, -np.inf)
    for name, sc in (("stock", stock), ("M_eye", eye_scores),
                     ("stock_neg", stock_neg), ("M_eye_neg", eye_neg)):
        ref[name] = {
            "hit1": np.array([hit_at_k(sc[i][:cur[i]], gold[i], 1) for i in range(n)]),
            "hit2": np.array([hit_at_k(sc[i][:cur[i]], gold[i], 2) for i in range(n)]),
        }
    chance = {"hit1": np.array([min(1.0, 1.0 / c) for c in cur]),
              "hit2": np.array([min(1.0, 2.0 / c) for c in cur])}

    # ---- accumulators, keyed by the REQUESTED rank LABEL ---------------------
    # Keying by label (not by the effective rank actually fitted) matters: dq
    # varies by a couple of dimensions across folds, so keying by effective rank
    # would scatter one label's folds across several buckets, each with partial
    # fold coverage. Instead every label accumulates over all folds, with the
    # per-fold effective rank = min(label, rmax); labels that collapse to the
    # same effective rank in a given fold SHARE one fit (the `cache` below), so
    # the collapse costs nothing and is recorded in `rank_effective_per_label`.
    variants = ("real", "shuffled_q")
    labels = [str(r) for r in RANKS]

    def new_acc():
        return {lb: {w: {"hit1": np.zeros(n), "hit2": np.zeros(n), "cnt": np.zeros(n)}
                     for w in WDS} for lb in labels}

    acc_te = {v: new_acc() for v in variants}
    acc_tr = {v: new_acc() for v in variants}
    acc_pp = {"hit1": np.zeros(n), "hit2": np.zeros(n)}
    seen = np.zeros(n)
    dq_seen, df_seen = [], []
    eff_per_label = {lb: set() for lb in labels}
    fold_test_sizes = []

    for rep, folds in enumerate(folds_per_repeat):
        for fi, (te_idx, tr_idx) in enumerate(folds):
            fold_test_sizes.append(len(te_idx))
            seen[te_idx] += 1

            # position prior (no features at all)
            pp = posprior_predict(gold[tr_idx], cur[te_idx], S)
            for j, i in enumerate(te_idx):
                acc_pp["hit1"][i] += hit_at_k(pp[j][:cur[i]], gold[i], 1)
                acc_pp["hit2"][i] += hit_at_k(pp[j][:cur[i]], gold[i], 2)

            # train-only span bases (see module docstring, decision 1)
            Bq, dq = span_basis(Q[tr_idx])
            Bf, df = span_basis(Fp[tr_idx][mask[tr_idx]])
            dq_seen.append(dq)
            df_seen.append(df)
            qz = Q @ Bq.T                          # [n,dq]
            fz = np.einsum("nsd,ed->nse", Fp, Bf)  # [n,S,df]
            rmax = int(min(dq, df))

            # shuffled_q permutes queries *within* train and *within* test, so
            # the train/test q row-sets (hence the span bases above) are
            # unchanged -- that is why one basis serves both variants.
            rng = np.random.default_rng(1_000_003 * rep + 101 * fi + layer)
            qz_v = {"real": qz, "shuffled_q": qz.copy()}
            qz_v["shuffled_q"][tr_idx] = qz[tr_idx][rng.permutation(len(tr_idx))]
            qz_v["shuffled_q"][te_idx] = qz[te_idx][rng.permutation(len(te_idx))]

            cache = {}
            for lb in labels:
                rk = rmax if lb == "full" else min(int(lb), rmax)
                eff_per_label[lb].add(rk)
                if rk not in cache:
                    cache[rk] = {}
                    for v in variants:
                        Aq, Af = fit_bilinear_batch(
                            qz_v[v][tr_idx], fz[tr_idx], mask[tr_idx], gold[tr_idx], WDS, rk,
                            seed=7919 * rep + 613 * fi + 31 * rk + (0 if v == "real" else 1))
                        cache[rk][v] = (
                            score_bilinear(Aq, Af, qz_v[v][te_idx], fz[te_idx]),
                            score_bilinear(Aq, Af, qz_v[v][tr_idx], fz[tr_idx]))
                for v in variants:
                    lg_te, lg_tr = cache[rk][v]
                    for b, w in enumerate(WDS):
                        for j, i in enumerate(te_idx):
                            s_ = lg_te[b, j][:cur[i]]
                            acc_te[v][lb][w]["hit1"][i] += hit_at_k(s_, gold[i], 1)
                            acc_te[v][lb][w]["hit2"][i] += hit_at_k(s_, gold[i], 2)
                            acc_te[v][lb][w]["cnt"][i] += 1
                        for j, i in enumerate(tr_idx):
                            s_ = lg_tr[b, j][:cur[i]]
                            acc_tr[v][lb][w]["hit1"][i] += hit_at_k(s_, gold[i], 1)
                            acc_tr[v][lb][w]["hit2"][i] += hit_at_k(s_, gold[i], 2)
                            acc_tr[v][lb][w]["cnt"][i] += 1

    # ---- aggregate ------------------------------------------------------
    def per_sample(d):
        c = np.maximum(d["cnt"], 1e-9)
        return d["hit1"] / c, d["hit2"] / c

    pp1 = acc_pp["hit1"] / np.maximum(seen, 1e-9)
    pp2 = acc_pp["hit2"] / np.maximum(seen, 1e-9)
    m, se, lo, hi = bootstrap_ci(pp2, boot)
    out = {
        "layer": layer,
        "n": n, "HK": HK, "S_max": S,
        "dq_mean": float(np.mean(dq_seen)), "df_mean": float(np.mean(df_seen)),
        "rank_cap": int(min(np.min(dq_seen), np.min(df_seen))),
        "rank_effective_per_label": {lb: sorted(eff_per_label[lb]) for lb in labels},
        "ref": {},
        "posprior": {"test_hit1": float(pp1.mean()), "test_hit2": m,
                     "test_hit2_se": se, "test_hit2_ci95": [lo, hi]},
        "ranks": {},
    }
    for name in ("stock", "M_eye", "stock_neg", "M_eye_neg"):
        m_, se_, lo_, hi_ = bootstrap_ci(ref[name]["hit2"], boot)
        out["ref"][name] = {"hit1": float(ref[name]["hit1"].mean()), "hit2": m_,
                            "hit2_se": se_, "hit2_ci95": [lo_, hi_]}
    out["ref"]["chance"] = {"hit1": float(chance["hit1"].mean()),
                            "hit2": float(chance["hit2"].mean())}

    for lb in labels:
        entry = {"by_wd": {}}
        cand = {}
        for w in WDS:
            r1, r2 = per_sample(acc_te["real"][lb][w])
            t1, t2 = per_sample(acc_tr["real"][lb][w])
            s1, s2 = per_sample(acc_te["shuffled_q"][lb][w])
            m_, se_, lo_, hi_ = bootstrap_ci(r2, boot)
            entry["by_wd"][str(w)] = {
                "train_hit1": float(t1.mean()), "train_hit2": float(t2.mean()),
                "test_hit1": float(r1.mean()), "test_hit2": m_,
                "test_hit2_se": se_, "test_hit2_ci95": [lo_, hi_],
                "shuffled_q_test_hit1": float(s1.mean()),
                "shuffled_q_test_hit2": float(s2.mean()),
            }
            cand[str(w)] = (r2, s2)
        best_w = max(WDS, key=lambda w: entry["by_wd"][str(w)]["test_hit2"])
        r2, s2 = cand[str(best_w)]
        # the four paired-delta bootstraps are computed only at the selected wd
        # (they are the expensive part and only the selected wd is reported)
        best = dict(entry["by_wd"][str(best_w)])
        best.update({
            "delta_vs_shuffled_q": paired_delta_ci(r2, s2, boot),
            "delta_vs_M_eye": paired_delta_ci(r2, ref["M_eye"]["hit2"], boot),
            "delta_vs_chance": paired_delta_ci(r2, chance["hit2"], boot),
            "delta_vs_posprior": paired_delta_ci(r2, pp2, boot),
        })
        entry["best_wd"] = str(best_w)
        entry["best"] = best
        entry["rank_effective"] = sorted(eff_per_label[lb])
        out["ranks"][lb] = entry

    out["fold_test_size_min"] = int(min(fold_test_sizes))
    out["fold_test_size_mean"] = float(np.mean(fold_test_sizes))
    return out


TRANSFER_RANK = 4


def transfer_analysis(per_ds, n_layers, rank=TRANSFER_RANK):
    """Cross-DATASET transfer of the fitted scorer, for one model.

    Why this exists: the within-dataset OOF numbers can be large and still be
    worth nothing to 0026. A scorer fitted on one NIAH variant is fitted on data
    where an *artificially inserted* needle sentence makes the gold chunk
    linearly marked in descriptor space; if the fitted `M` is really just that
    variant's needle signature, it will not survive contact with real training
    data. So: fit on ALL samples of dataset `src`, evaluate on ALL samples of
    dataset `dst` (src != dst is a genuine transfer test, src == dst is the
    fit-on-everything diagonal and is an OVERFIT upper bound, not a held-out
    number -- it is here only as the scale reference for the off-diagonals).

    Both `real` and `shuffled_q` are fitted, so the off-diagonal cells also say
    whether whatever transfers is query-dependent. wd is swept and the max over
    wd is reported (same optimistic convention as the main sweep).
    """
    out = {}
    for L in range(n_layers):
        arrs = {ds: build_layer_arrays(s, L) for ds, s in per_ds.items()}
        for src, ssrc in per_ds.items():
            Qs, Fs, ms, gs, cs, _ = arrs[src]
            Bq, dq = span_basis(Qs)
            Bf, df = span_basis(Fs[ms])
            rk = int(min(rank, dq, df))
            qz_s = {"real": Qs @ Bq.T}
            rng = np.random.default_rng(20260726 + L)
            qz_s["shuffled_q"] = qz_s["real"][rng.permutation(len(ssrc))]
            fz_s = np.einsum("nsd,ed->nse", Fs, Bf)
            fitted = {}
            for v in ("real", "shuffled_q"):
                fitted[v] = fit_bilinear_batch(qz_s[v], fz_s, ms, gs, WDS, rk,
                                               seed=104729 + L)
            for dst in per_ds:
                Qd, Fd, md, gd, cd, _ = arrs[dst]
                qz_d = Qd @ Bq.T
                fz_d = np.einsum("nsd,ed->nse", Fd, Bf)
                cell = {"rank_effective": rk, "n_src": len(ssrc), "n_dst": len(gd)}
                for v in ("real", "shuffled_q"):
                    Aq, Af = fitted[v]
                    lg = score_bilinear(Aq, Af, qz_d, fz_d)
                    h2 = [float(np.mean([hit_at_k(lg[b, i][:cd[i]], gd[i], 2)
                                         for i in range(len(gd))])) for b in range(len(WDS))]
                    h1 = [float(np.mean([hit_at_k(lg[b, i][:cd[i]], gd[i], 1)
                                         for i in range(len(gd))])) for b in range(len(WDS))]
                    bi = int(np.argmax(h2))
                    cell[v] = {"hit2": h2[bi], "hit1": h1[bi], "wd": str(WDS[bi]),
                               "hit2_by_wd": dict(zip(map(str, WDS), h2))}
                cell["chance_hit2"] = float(np.mean([min(1.0, 2.0 / c) for c in cd]))
                out.setdefault(str(L), {}).setdefault(src, {})[dst] = cell
    return out


def make_folds(groups, n_folds, n_repeats, seed=0):
    """Grouped K-fold, reshuffled per repeat. Returns list (len n_repeats) of
    lists of (test_idx, train_idx)."""
    keys = sorted(set(str(g) for g in groups))
    gs = np.array([str(g) for g in groups])
    reps = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + 977 * r)
        perm = rng.permutation(len(keys))
        assign = {keys[perm[i]]: i % n_folds for i in range(len(keys))}
        fold_of = np.array([assign[g] for g in gs])
        folds = []
        for f in range(n_folds):
            te = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            if len(te) == 0 or len(tr) < 2:
                continue
            folds.append((te, tr))
        reps.append(folds)
    return reps


def worker(args):
    kind, model, dataset, n_threads = args
    import torch
    torch.set_num_threads(n_threads)

    if kind == "transfer":
        per_ds = {}
        for ds in DATASETS:
            s, _ = load_dataset_samples(model, ds)
            if s:
                per_ds[ds] = s
        n_layers = next(iter(per_ds.values()))[0]["q"].shape[0]
        tr = transfer_analysis(per_ds, n_layers)
        print(f"[probe] {model}/TRANSFER done ({len(per_ds)} datasets x {n_layers} layers)",
              flush=True)
        return kind, model, None, tr

    samples, n_total = load_dataset_samples(model, dataset)
    if not samples:
        return kind, model, dataset, None
    groups = [group_key(s) for s in samples]
    folds = make_folds(groups, N_FOLDS, N_REPEATS)
    boot = make_boot_index(groups, N_BOOT)
    n_layers = samples[0]["q"].shape[0]
    per_layer = [analyze_layer(samples, L, groups, folds, boot) for L in range(n_layers)]
    entry = {
        "n_total": n_total, "n_eligible": len(samples),
        "n_groups": len(set(str(g) for g in groups)),
        "group_key": "pair_id" if samples[0]["meta"].get("pair_id") is not None else "file_index",
        "mean_cur_seg": float(np.mean([s["meta"]["cur_seg"] for s in samples])),
        "rank_cap_min_over_layers": int(min(pl["rank_cap"] for pl in per_layer)),
        "rank_cap_max_over_layers": int(max(pl["rank_cap"] for pl in per_layer)),
        "per_layer": per_layer,
    }
    print(f"[probe] {model}/{dataset}: n_elig={len(samples)}/{n_total} layers={n_layers} "
          f"rank_cap={entry['rank_cap_min_over_layers']}..{entry['rank_cap_max_over_layers']}",
          flush=True)
    return kind, model, dataset, entry


# ==================================================================
# summary / decision
# ==================================================================

def summarize(results):
    summ = {}
    for model, dsets in results.items():
        summ[model] = {}
        for dataset, entry in dsets.items():
            best = None
            for pl in entry["per_layer"]:
                for rk, rke in pl["ranks"].items():
                    v = rke["best"]["test_hit2"]
                    if best is None or v > best["test_hit2"]:
                        best = {
                            "layer": pl["layer"], "rank": rk,
                            "rank_effective": rke["rank_effective"], "wd": rke["best_wd"],
                            "test_hit2": v, "test_hit2_se": rke["best"]["test_hit2_se"],
                            "test_hit2_ci95": rke["best"]["test_hit2_ci95"],
                            "test_hit1": rke["best"]["test_hit1"],
                            "train_hit2": rke["best"]["train_hit2"],
                            "shuffled_q_test_hit2": rke["best"]["shuffled_q_test_hit2"],
                            "delta_vs_shuffled_q": rke["best"]["delta_vs_shuffled_q"],
                            "delta_vs_M_eye": rke["best"]["delta_vs_M_eye"],
                            "delta_vs_chance": rke["best"]["delta_vs_chance"],
                            "delta_vs_posprior": rke["best"]["delta_vs_posprior"],
                            "ref_stock_hit2": pl["ref"]["stock"]["hit2"],
                            "ref_M_eye_hit2": pl["ref"]["M_eye"]["hit2"],
                            "ref_M_eye_neg_hit2": pl["ref"]["M_eye_neg"]["hit2"],
                            "ref_chance_hit2": pl["ref"]["chance"]["hit2"],
                            "ref_posprior_hit2": pl["posprior"]["test_hit2"],
                        }
            # best-layer-per-rank curve for the figure
            curve = {}
            for pl in entry["per_layer"]:
                for rk, rke in pl["ranks"].items():
                    v = rke["best"]["test_hit2"]
                    if rk not in curve or v > curve[rk]["test_hit2"]:
                        curve[rk] = {"layer": pl["layer"], "test_hit2": v,
                                     "se": rke["best"]["test_hit2_se"],
                                     "shuffled_q": rke["best"]["shuffled_q_test_hit2"],
                                     "rank_effective": rke["rank_effective"]}
            # ---- selection-inflation control -------------------------------
            # `best` is a maximum over layers x rank-labels x wds of a metric
            # measured on n=16..47 samples. At n=16 (SE ~0.11) the max over a
            # few hundred configurations is inflated by a lot, so the honest
            # comparison is max-over-the-SAME-grid for the shuffled_q control
            # rather than the fitted value against a single fixed reference.
            n_cfg = 0
            sh_max, sh_at = -1.0, None
            for pl in entry["per_layer"]:
                for rk, rke in pl["ranks"].items():
                    for w, e in rke["by_wd"].items():
                        n_cfg += 1
                        if e["shuffled_q_test_hit2"] > sh_max:
                            sh_max = e["shuffled_q_test_hit2"]
                            sh_at = {"layer": pl["layer"], "rank": rk, "wd": w}
            # less selection-sensitive view: average over all 16 layers at the
            # rank label that won, for real and for shuffled_q
            brk = best["rank"]
            lay_real = [pl["ranks"][brk]["best"]["test_hit2"] for pl in entry["per_layer"]]
            lay_shuf = [pl["ranks"][brk]["best"]["shuffled_q_test_hit2"]
                        for pl in entry["per_layer"]]
            top3 = sorted(((pl["ranks"][brk]["best"]["test_hit2"], pl["layer"])
                           for pl in entry["per_layer"]), reverse=True)[:3]

            # stock's own best layer, for the "does M find signal where stock has
            # none" question
            st = max(entry["per_layer"], key=lambda pl: pl["ref"]["stock"]["hit2"])
            n = entry["n_eligible"]
            summ[model][dataset] = {
                "n_oof_test": n,
                "small_n_flag": n < SMALL_N_WARN,
                "inconclusive": n < MIN_TEST_N,
                "best": best,
                "selection": {
                    "n_configs_maximised_over": n_cfg,
                    "shuffled_q_best_over_same_grid": sh_max,
                    "shuffled_q_best_at": sh_at,
                    "real_minus_shuffled_at_respective_grid_maxima": best["test_hit2"] - sh_max,
                    "note": ("both numbers are grid maxima on the same n samples, so their "
                             "difference is the selection-inflation-matched estimate of what "
                             "query information adds. `best.delta_vs_shuffled_q` is the "
                             "sample-paired version at the real fit's own winning config, "
                             "which is the tighter test but is selected on the real arm."),
                },
                "layerwise_at_best_rank": {
                    "rank": brk,
                    "mean_over_layers_real": float(np.mean(lay_real)),
                    "mean_over_layers_shuffled_q": float(np.mean(lay_shuf)),
                    "top3_layers_real": [{"layer": l, "test_hit2": v} for v, l in top3],
                },
                "per_rank_best_layer": curve,
                "stock_best_layer": {"layer": st["layer"], "hit2": st["ref"]["stock"]["hit2"]},
                "at_best_layer": {
                    "layer": best["layer"],
                    "stock_hit2": best["ref_stock_hit2"],
                    "M_eye_hit2": best["ref_M_eye_hit2"],
                    "M_eye_neg_hit2": best["ref_M_eye_neg_hit2"],
                    "chance_hit2": best["ref_chance_hit2"],
                    "posprior_hit2": best["ref_posprior_hit2"],
                },
                "M_eye_neg_best_layer": max(
                    ({"layer": pl["layer"], "hit2": pl["ref"]["M_eye_neg"]["hit2"]}
                     for pl in entry["per_layer"]), key=lambda x: x["hit2"]),
            }
    return summ


def summarize_transfer(transfer):
    """Per model: the layer whose diagonal (src==dst) mean is highest, the full
    4x4 matrix at that layer, and the headline off-diagonal generalisation gap.

    The number that matters for 0026 is `offdiag_mean_minus_chance`: if fitting
    on one NIAH variant buys nothing on the others, the within-dataset OOF wins
    are a per-variant needle signature, not a routing prior a real training run
    could reuse."""
    out = {}
    for model, per_layer in transfer.items():
        if not per_layer:
            continue
        best_L, best_diag = None, -1.0
        for L, mat in per_layer.items():
            diag = [mat[d][d]["real"]["hit2"] for d in mat if d in mat[d]]
            if diag and float(np.mean(diag)) > best_diag:
                best_diag, best_L = float(np.mean(diag)), L
        mat = per_layer[best_L]
        offd, offd_ch, offd_sh = [], [], []
        for src in mat:
            for dst in mat[src]:
                if src == dst:
                    continue
                offd.append(mat[src][dst]["real"]["hit2"])
                offd_sh.append(mat[src][dst]["shuffled_q"]["hit2"])
                offd_ch.append(mat[src][dst]["chance_hit2"])
        out[model] = {
            "layer_with_best_diagonal": best_L,
            "diagonal_mean_overfit_upper_bound": best_diag,
            "matrix_real_hit2": {s: {d: mat[s][d]["real"]["hit2"] for d in mat[s]} for s in mat},
            "matrix_shuffled_q_hit2": {s: {d: mat[s][d]["shuffled_q"]["hit2"] for d in mat[s]}
                                       for s in mat},
            "matrix_chance_hit2": {s: {d: mat[s][d]["chance_hit2"] for d in mat[s]} for s in mat},
            "offdiag_mean_real": float(np.mean(offd)),
            "offdiag_mean_shuffled_q": float(np.mean(offd_sh)),
            "offdiag_mean_chance": float(np.mean(offd_ch)),
            "offdiag_mean_minus_chance": float(np.mean(offd) - np.mean(offd_ch)),
            "n_offdiag_cells": len(offd),
        }
    return out


def make_figure(out, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = out["summary"]
    rows = [(m, d) for m in MODELS for d in DATASETS if d in summ.get(m, {})]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), squeeze=False)
    for ax_i, (model, dataset) in enumerate(rows):
        ax = axes[ax_i // 4][ax_i % 4]
        s = summ[model][dataset]
        curve = s["per_rank_best_layer"]
        ranks = [str(r) for r in RANKS if str(r) in curve]
        y = [curve[r]["test_hit2"] for r in ranks]
        e = [curve[r]["se"] or 0.0 for r in ranks]
        sq = [curve[r]["shuffled_q"] for r in ranks]
        x = np.arange(len(ranks))
        ax.errorbar(x, y, yerr=e, marker="o", color="tab:blue", capsize=3, zorder=5,
                    label="fitted bilinear (test hit@2, best layer/wd)")
        ax.plot(x, sq, marker="x", color="tab:purple", ls="--", alpha=.85, zorder=4,
                label="same fit, SHUFFLED-q control")
        ab = s["at_best_layer"]
        ax.axhline(ab["stock_hit2"], color="tab:red", ls="-", alpha=.75,
                   label=f"stock ({ab['stock_hit2']:.3f})")
        ax.axhline(ab["M_eye_hit2"], color="tab:orange", ls="--", alpha=.85,
                   label=f"M=I / u:=q ({ab['M_eye_hit2']:.3f})")
        mn = s["M_eye_neg_best_layer"]
        ax.axhline(mn["hit2"], color="tab:brown", ls=(0, (3, 1, 1, 1)), alpha=.85,
                   label=f"M=-I, best layer {mn['layer']} ({mn['hit2']:.3f})")
        ax.axhline(ab["posprior_hit2"], color="tab:green", ls="-.", alpha=.85,
                   label=f"position prior ({ab['posprior_hit2']:.3f})")
        ax.axhline(ab["chance_hit2"], color="black", ls=":", alpha=.7,
                   label=f"chance ({ab['chance_hit2']:.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(={curve[r]['rank_effective'][0]})"
                            if curve[r]["rank_effective"][0] != r else r for r in ranks],
                           fontsize=7)
        ax.set_xlabel("requested rank R (effective rank in parens where capped)")
        ax.set_ylabel("test hit@2 (OOF)")
        ax.set_ylim(0, 1.02)
        b = s["best"]
        ax.set_title(f"{model} / {dataset}   n={s['n_oof_test']}\nbest: layer {b['layer']}, "
                     f"R={b['rank']}, fit {b['test_hit2']:.3f} vs shuf-q "
                     f"{b['shuffled_q_test_hit2']:.3f}", fontsize=9)
        ax.grid(alpha=.3)
        ax.legend(fontsize=6, loc="upper left")
    for j in range(len(rows), 8):
        axes[j // 4][j % 4].axis("off")
    fig.suptitle("0026 pre-check: can a LEARNED rank-<=R bilinear form q^T M f route where M=I cannot?\n"
                 "held-out = repeated grouped 4-fold OOF (5 repeats); wd chosen on test (optimistic upper bound)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.text(0.5, 0.005,
             "Error bars = grouped bootstrap SE over samples. Ranks are capped at the train-span "
             "dimension (<= n_train), so R=64/full collapse onto lower ranks at small n. "
             "The decisive comparison is fitted-vs-shuffled-q (purple), not fitted-vs-chance.",
             ha="center", fontsize=7)
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)

    tasks = ([("cell", m, d, a.threads) for m in MODELS for d in DATASETS]
             + [("transfer", m, None, a.threads) for m in MODELS])
    results = {m: {} for m in MODELS}
    transfer = {m: {} for m in MODELS}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for kind, model, dataset, entry in ex.map(worker, tasks):
            if entry is None:
                continue
            if kind == "transfer":
                transfer[model] = entry
            else:
                results[model][dataset] = entry

    out = {
        "meta": {
            "question": ("is gold-chunk identity linearly decodable from (q_t, f_m) by a "
                         "rank-<=R bilinear form q^T M f (M = A_q^T A_f, SRLA's scoring "
                         "cos(A_q q, A_f f)), trained with gold supervision, on HELD-OUT data?"),
            "scoring": "logits_m = cos(A_q q_t, A_f f_m) / tau",
            "tau": TAU, "tau_note": ("FIXED, not learned: for fixed (A_q,A_f) temperature "
                                     "cannot change ranking hence cannot change hit@k."),
            "ranks_requested": [str(r) for r in RANKS],
            "rank_cap_note": ("ranks are capped at min(dq,df) where dq/df are the TRAIN-span "
                              "dimensions (dq <= n_train). At n_train=12 (paired sets) R=16/64/full "
                              "are all literally the same model as R=12; the grid is de-duplicated "
                              "and `ranks_effective` records what was actually fitted. Full-rank "
                              "unconstrained M in raw HK=2048 (4.2M params vs ~12 labels) is not "
                              "merely infeasible, it is unidentifiable -- see module docstring."),
            "wds": WDS,
            "wd_selection_note": ("headline picks the wd with the best TEST hit@2 -> optimistically "
                                  "biased on purpose. A negative under oracle wd selection is a "
                                  "strong negative; a positive is only a lead and must also beat "
                                  "the shuffled_q control."),
            "cv": {"n_folds": N_FOLDS, "n_repeats": N_REPEATS,
                   "grouping": "pair_id for paired_* datasets, else enumerate-position filename",
                   "note": ("held-out is repeated grouped K-fold OOF: every sample is scored by a "
                            "model that never saw it, so n_test = n_eligible rather than n/2. "
                            "meta.sample_id is never used as a key (not unique in this dump).")},
            "optimizer": {"adam_lr": LR, "steps": STEPS, "init": "N(0, 1/sqrt(d))"},
            "eligibility": ("copied from x1_analyze: meta['eligible'] False -> skipped; candidate "
                            "bank is segments i < cur_seg; label = gold_seg (< cur_seg by "
                            "eligibility)."),
            "descriptor": ("f_m = c_full[layer][m] (per-segment mean of normalised keys, the MC-SSC "
                           "descriptor). NOTE cos(A_q q, A_f f) is invariant to ||f||, so this "
                           "family structurally cannot use descriptor norm -- faithful to SRLA."),
            "controls": {
                "posprior": ("train gold-position frequency prior, no features. The gold position is "
                             "strongly non-uniform in this dump, so this is a real confound a fitted "
                             "M can absorb and still generalise."),
                "shuffled_q": ("identical fit with queries permuted within train and within test: "
                               "destroys q<->bank correspondence, preserves marginals and any "
                               "position prior. real-minus-shuffled_q is the decisive statistic."),
            },
            "bootstrap": {"n_boot": N_BOOT, "resampled_over": "groups"},
            "min_test_n_for_conclusion": MIN_TEST_N,
            "small_n_warn": SMALL_N_WARN,
            "models": MODELS, "datasets": DATASETS,
            "transfer": {
                "rank": TRANSFER_RANK,
                "note": ("fit on ALL samples of `src`, evaluate on ALL samples of `dst`. The "
                         "src==dst diagonal is fit-on-everything and is therefore an OVERFIT "
                         "upper bound, NOT a held-out number -- use the main sweep's OOF for "
                         "that; the diagonal is here only as the scale against which the "
                         "off-diagonal transfer cells are read."),
            },
            "sign_flip_note": (
                "`M_eye_neg` (M = -I) is a zero-parameter member of the fitted family that 0025 "
                "never tested. Where it scores far above chance the descriptor space carries "
                "gold-chunk information the stock scorer reads with the WRONG SIGN, and no "
                "learned module is needed to collect it."),
        },
        "results": results,
        "transfer": transfer,
    }
    out["summary"] = summarize(results)
    out["transfer_summary"] = summarize_transfer(transfer)

    for p in (os.path.join(RES, "probe_bilinear_router.json"),
              os.path.join(MC_OUT, "results", "probe_bilinear_router.json")):
        json.dump(out, open(p, "w"), indent=2)
        print(f"[probe] wrote {p}", flush=True)
    make_figure(out, [os.path.join(RES, "probe_bilinear_router.png"),
                      os.path.join(MC_OUT, "results", "probe_bilinear_router.png")])
    print("[probe] wrote figures", flush=True)

    print("\n=== headline: test hit@2 at best (layer, rank, wd) ===", flush=True)
    for m in MODELS:
        for d in DATASETS:
            s = out["summary"].get(m, {}).get(d)
            if not s:
                continue
            b = s["best"]
            print(f"{m:8s} {d:17s} n={s['n_oof_test']:3d} L{b['layer']:2d} R{b['rank']:>4s} "
                  f"fit={b['test_hit2']:.3f}+-{b['test_hit2_se']:.3f} "
                  f"(train {b['train_hit2']:.3f}) | shufq={b['shuffled_q_test_hit2']:.3f} "
                  f"(gridmax {s['selection']['shuffled_q_best_over_same_grid']:.3f}) "
                  f"stock={b['ref_stock_hit2']:.3f} M=I={b['ref_M_eye_hit2']:.3f} "
                  f"pos={b['ref_posprior_hit2']:.3f} chance={b['ref_chance_hit2']:.3f} | "
                  f"d_shufq={b['delta_vs_shuffled_q']['delta']:+.3f} "
                  f"CI[{b['delta_vs_shuffled_q']['ci95'][0]:+.3f},"
                  f"{b['delta_vs_shuffled_q']['ci95'][1]:+.3f}] "
                  f"| layermean R={s['layerwise_at_best_rank']['rank']}: "
                  f"{s['layerwise_at_best_rank']['mean_over_layers_real']:.3f} vs shufq "
                  f"{s['layerwise_at_best_rank']['mean_over_layers_shuffled_q']:.3f} "
                  f"| M=-I best L{s['M_eye_neg_best_layer']['layer']}="
                  f"{s['M_eye_neg_best_layer']['hit2']:.3f}", flush=True)

    print("\n=== cross-dataset transfer (fit on src, eval on dst; rank "
          f"{TRANSFER_RANK}) ===", flush=True)
    for m, t in out["transfer_summary"].items():
        print(f"{m}: layer {t['layer_with_best_diagonal']} | diagonal(overfit UB) "
              f"{t['diagonal_mean_overfit_upper_bound']:.3f} | off-diagonal real "
              f"{t['offdiag_mean_real']:.3f} shufq {t['offdiag_mean_shuffled_q']:.3f} "
              f"chance {t['offdiag_mean_chance']:.3f} -> transfer above chance "
              f"{t['offdiag_mean_minus_chance']:+.3f} ({t['n_offdiag_cells']} cells)", flush=True)


if __name__ == "__main__":
    main()
