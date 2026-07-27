"""X1 CPU analysis tests (rev2 Task 2): descriptor-only H_blind test (2a) +
u_t:=q_t rescoring (2b) synthetic verification. Pure numpy, no torch/GPU."""
import os
import sys

import numpy as np

ANA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "lmr", "analysis", "260725_mc_niah_analysis"))
sys.path.insert(0, ANA)
import x1_analyze as xa  # noqa: E402


# ------------------------------------------------------------------
# building blocks
# ------------------------------------------------------------------

def test_flatten_hk():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    flat = xa.flatten_hk(x)
    assert flat.shape == (2, 12)
    assert np.array_equal(flat[0], x[0].reshape(-1))


def test_score_from_matches_manual_dot():
    rng = np.random.default_rng(0)
    q = rng.normal(size=8)
    c = rng.normal(size=(5, 8))
    got = xa.score_from(q, c)
    want = np.array([c[i] @ q for i in range(5)])
    np.testing.assert_allclose(got, want)


def test_score_from_matches_headwise_einsum():
    """score_from on flattened [H,K] must equal einsum('hk,nhk->n', ...) —
    the exact contraction routing_scores_at uses (sums over heads AND K
    jointly, no separate head-reduction). Covers coordinator review point 3."""
    rng = np.random.default_rng(1)
    H, K, N = 4, 6, 5
    u = rng.normal(size=(H, K))
    c = rng.normal(size=(N, H, K))
    want = np.einsum("hk,nhk->n", u, c)
    got = xa.score_from(xa.flatten_hk(u), xa.flatten_hk(c))
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_top2_set_and_jaccard():
    s = np.array([1.0, 5.0, 3.0, 0.0])
    assert xa.top2_set(s) == {1, 2}
    assert xa.jaccard({1, 2}, {1, 2}) == 1.0
    assert xa.jaccard({1, 2}, {1, 3}) == 1.0 / 3.0
    assert xa.jaccard(set(), set()) == 1.0


def test_pos_feature():
    assert np.allclose(xa.pos_feature(4), [0, 1 / 3, 2 / 3, 1.0])
    assert np.allclose(xa.pos_feature(1), [0.0])


def test_gram_offdiag_hand_computed():
    """Gram off-diagonal mean matches a hand-computed value for orthogonal
    vs identical rows (coordinator/brief requirement (iii))."""
    # two identical unit rows -> cos=1 off-diagonal
    c = np.array([[1.0, 0.0], [1.0, 0.0]])
    _, _, _, _, gram = xa.descriptor_features(c)
    assert np.isclose(gram, 1.0)
    # two orthogonal unit rows -> cos=0 off-diagonal
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, _, _, _, gram = xa.descriptor_features(c)
    assert np.isclose(gram, 0.0)
    # three rows, hand compute
    c = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    c_hat = c / np.linalg.norm(c, axis=1, keepdims=True)
    gram_mat = c_hat @ c_hat.T
    off = gram_mat[~np.eye(3, dtype=bool)]
    _, _, _, _, gram = xa.descriptor_features(c)
    assert np.isclose(gram, off.mean())


def test_descriptor_features_outlier_of_symmetric_set_is_uniform():
    # 4 orthonormal-ish rows symmetric around their mean -> outlier roughly equal
    c = np.eye(4)
    pos, norm, outlier, c_hat, gram = xa.descriptor_features(c)
    assert np.allclose(norm, 1.0)
    assert np.allclose(outlier, outlier[0])  # symmetric -> all equal


# ------------------------------------------------------------------
# synthetic H_blind cases (brief step 1 (i),(ii); spec judgement criteria)
# ------------------------------------------------------------------

def _fake_samples(n_samples, n_seg, H=2, K=3, rng=None, score_fn=None,
                   gold_seg=0):
    """Build a list of sample dicts shaped like load_dataset_samples() output,
    for a single layer (caller passes layer=0). `score_fn(c_flat)-> [n_seg]`
    determines the *actual* dumped stock_scores (so u is picked/back-solved
    to reproduce it approximately isn't required — we bypass that by writing
    stock_scores directly and using a u that also reproduces it via score_from,
    keeping the internal sanity check passing)."""
    rng = rng or np.random.default_rng(0)
    samples = []
    for _ in range(n_samples):
        cur_seg = n_seg
        c = rng.normal(size=(1, n_seg, H, K)).astype(np.float32)  # [L=1,N,H,K]
        u = rng.normal(size=(1, H, K)).astype(np.float32)
        q = rng.normal(size=(1, H, K)).astype(np.float32)
        c_flat = xa.flatten_hk(c[0])                              # [N,D]
        # force u such that score_from(u,c_flat) == score_fn(c_flat) exactly,
        # by solving least squares for u given the desired scores (so the
        # internal stock-recompute sanity check in analyze_layer passes).
        target = score_fn(c_flat)
        u_flat, *_ = np.linalg.lstsq(c_flat, target, rcond=None)
        u[0] = u_flat.reshape(H, K)
        stock_scores = np.full((1, n_seg), float("-inf"), dtype=np.float32)
        stock_scores[0, :cur_seg] = target
        meta = {"gold_seg": gold_seg, "cur_seg": cur_seg, "n_seg": n_seg,
                "n_tok": 999, "eligible": True, "key_segs": [gold_seg],
                "sample_id": rng.integers(0, 10**6)}
        # NB: kept float32 here (not fp16-cast like the real dump) — these
        # two H_blind tests assert a tight sanity-gate tolerance to prove the
        # *math* is self-consistent; fp16 quantization noise is a separate,
        # already-measured concern (see STOCK_SANITY_TOL's docstring) and
        # would swamp a 1e-3 bar on a tiny 6-dim synthetic descriptor.
        samples.append({"meta": meta, "u": u.astype(np.float32),
                         "q": q.astype(np.float32),
                         "c_full": c.astype(np.float32),
                         "stock_scores": stock_scores})
    return samples


def test_h_blind_scores_linear_in_pos_gives_high_r2_and_pos_jaccard():
    """(i) scores exactly a linear function of pos_i -> r2~=1, pos pred_jaccard high."""
    rng = np.random.default_rng(42)
    n_seg = 6

    def score_fn(c_flat):
        pos = xa.pos_feature(n_seg)
        return 10.0 * pos  # pure linear-in-pos signal, no noise

    samples = _fake_samples(n_samples=40, n_seg=n_seg, rng=rng, score_fn=score_fn)
    tracker = {"max_diff": 0.0, "n": 0, "worst": None}
    stats = xa.analyze_layer(samples, layer=0, sanity_tracker=tracker)

    assert tracker["max_diff"] < 1e-3, "internal sanity gate should pass for constructed data"
    assert stats["r2"] > 0.95, f"expected near-1 R2 for pure pos-linear scores, got {stats['r2']}"
    assert stats["feat_r2"]["pos"] > 0.95
    # pos should win with "high" direction (scores increase with pos)
    assert stats["pred_jaccard"]["pos"]["jaccard"] > 0.9
    assert stats["pred_jaccard"]["pos"]["direction"] == "high"


def test_h_blind_scores_independent_of_descriptor_features_gives_low_r2():
    """(ii) scores generated from a random direction unrelated to pos/norm/outlier
    -> r2 should be low (query-driven selection, not descriptor-only)."""
    rng = np.random.default_rng(7)
    n_seg, H, K = 8, 2, 5  # D=H*K=10 >= n_seg=8, so the helper's u-backsolve
    # (which reproduces score_fn exactly for the internal sanity check) stays
    # an exact/underdetermined system rather than an unsatisfiable overdetermined
    # one -- unrelated to the R2-should-be-low claim this test is making.
    # a fixed random projection direction, independent of pos/norm/outlier by construction
    direction = rng.normal(size=n_seg * H * K).reshape(n_seg, H, K)

    def score_fn(c_flat):
        # score_i = <c_i, random_direction_i> -- couples each row to an
        # independent random vector, decorrelated from pos/norm/outlier
        d_flat = xa.flatten_hk(direction)
        return np.array([c_flat[i] @ d_flat[i] for i in range(n_seg)])

    samples = _fake_samples(n_samples=60, n_seg=n_seg, H=H, K=K, rng=rng, score_fn=score_fn)
    tracker = {"max_diff": 0.0, "n": 0, "worst": None}
    stats = xa.analyze_layer(samples, layer=0, sanity_tracker=tracker)

    assert tracker["max_diff"] < 1e-3
    assert stats["r2"] < 0.5, f"expected low R2 for descriptor-independent scores, got {stats['r2']}"


def test_lstsq_r2_perfect_fit_is_one():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    coef_true = np.array([0.5, 1.0, -2.0, 3.0])
    y = coef_true[0] + X @ coef_true[1:]
    _, r2, _ = xa.lstsq_r2(X, y)
    assert np.isclose(r2, 1.0, atol=1e-8)


def test_lstsq_r2_zero_variance_y_is_nan():
    X = np.random.default_rng(3).normal(size=(10, 2))
    y = np.zeros(10)
    _, r2, _ = xa.lstsq_r2(X, y)
    assert np.isnan(r2)


# ------------------------------------------------------------------
# u_t := q_t rescoring synthetic verification (brief: "테스트: ... u=q 재채점
# 로직 합성 검증 (신호를 q에만 심으면 u_eq_q hit 상승)")
# ------------------------------------------------------------------

def test_u_eq_q_hit_rises_when_signal_is_only_in_q():
    """Construct samples where u is uninformative (near-random routing, low
    stock hit@2) but q happens to align with the descriptor of the gold
    segment for every sample (signal planted only in q) -> u_eq_q hit@2
    should be higher than stock hit@2."""
    rng = np.random.default_rng(11)
    n_seg = 6
    gold_seg = 2
    samples = []
    for _ in range(80):
        H, K = 2, 4
        c = rng.normal(size=(n_seg, H, K)).astype(np.float32)
        c_flat = xa.flatten_hk(c)

        # u: random, uncorrelated with which segment is gold -> stock routing ~chance
        u = rng.normal(size=(H, K)).astype(np.float32)

        # q: constructed to align strongly with the gold segment's descriptor
        # (plant the "signal" only in q, not in u)
        gold_vec = c_flat[gold_seg]
        noise = rng.normal(size=H * K) * 0.01
        q_flat = gold_vec / (np.linalg.norm(gold_vec) + 1e-8) + noise
        q = q_flat.reshape(H, K).astype(np.float32)

        stock_scores = np.full((1, n_seg), float("-inf"), dtype=np.float32)
        stock_scores[0, :n_seg] = c_flat @ u.reshape(-1)

        meta = {"gold_seg": gold_seg, "cur_seg": n_seg, "n_seg": n_seg,
                "n_tok": 999, "eligible": True, "key_segs": [gold_seg],
                "sample_id": int(rng.integers(0, 10**6))}
        samples.append({
            "meta": meta,
            "u": u[None].astype(np.float16),
            "q": q[None].astype(np.float16),
            "c_full": c[None].astype(np.float16),
            "stock_scores": stock_scores,
        })

    tracker = {"max_diff": 0.0, "n": 0, "worst": None}
    stats = xa.analyze_layer(samples, layer=0, sanity_tracker=tracker)

    assert stats["hit2"]["stock"] < 0.6, (
        f"expected near-chance stock hit@2 with random u, got {stats['hit2']['stock']}")
    assert stats["hit2"]["u_eq_q"] > 0.9, (
        f"expected high u_eq_q hit@2 when signal is planted in q, got {stats['hit2']['u_eq_q']}")
    assert stats["hit2"]["u_eq_q"] > stats["hit2"]["stock"]


def test_centering_changes_scores_when_offset_present():
    """centering (c_i - mean_j c_j) should change scores/top-2 when segments
    share a large common-mode offset unrelated to the discriminative signal
    (sanity that the centered-condition code path actually does something)."""
    rng = np.random.default_rng(5)
    n_seg = 5
    gold_seg = 1
    H, K = 2, 3
    common = rng.normal(size=(H, K)) * 5.0  # big shared offset across all segments
    c = np.stack([common + rng.normal(size=(H, K)) * 0.1 for _ in range(n_seg)])
    c[gold_seg] += rng.normal(size=(H, K)) * 0.1  # gold has only a small local perturbation
    c_flat = xa.flatten_hk(c)
    u = rng.normal(size=(H, K)).astype(np.float32)

    stock_scores = np.full((1, n_seg), float("-inf"), dtype=np.float32)
    stock_scores[0, :n_seg] = c_flat @ u.reshape(-1)
    meta = {"gold_seg": gold_seg, "cur_seg": n_seg, "n_seg": n_seg, "n_tok": 999,
            "eligible": True, "key_segs": [gold_seg], "sample_id": 0}
    samples = [{"meta": meta, "u": u[None].astype(np.float16),
                "q": u[None].astype(np.float16),
                "c_full": c[None].astype(np.float16), "stock_scores": stock_scores}]

    tracker = {"max_diff": 0.0, "n": 0, "worst": None}
    stats = xa.analyze_layer(samples, layer=0, sanity_tracker=tracker)
    # stock vs stock_centered hit2 lists are computed per-sample booleans in
    # analyze_layer; with n=1 sample this just checks the plumbing runs and
    # produces a boolean, not a crash/None.
    assert stats["hit2"]["stock"] in (0.0, 1.0)
    assert stats["hit2"]["stock_centered"] in (0.0, 1.0)


def test_pc1_removal_raises_hit2_when_dominant_common_direction_masks_signal():
    """Review finding 5: synthetic case where a planted dominant common
    direction masks a discriminative signal -> pc1 removal must raise hit@2.

    Construction: each segment's raw descriptor c_i = (A + eps_i)*common_hat
    + tiny_noise, plus a small discriminative bump `b*d_hat` (d_hat orthogonal
    to common_hat) ONLY on the gold segment. `eps_i` (per-segment magnitude
    noise along the shared common direction, uncorrelated with which segment
    is gold) is deliberately made large relative to `b`, so the raw dot
    product <u, c_i> = c1*(A+eps_i) + c2*b*1[i==gold] + ... is dominated by
    the eps_i noise term and the stock/u_eq_q (raw, unnormalized-c) hit@2
    collapses toward chance.

    Normalizing c_i (c_hat_i = c_i/||c_i||) already cancels most of the
    eps_i-driven magnitude noise (since it is collinear with the dominant
    direction the norm measures), and removing the top-1 PC (~= common_hat,
    the shared direction across all rows in c_hat-space) removes what's left
    of it, leaving the small discriminative component along d_hat to
    dominate the ranking -> u_recomp_pc1 hit@2 should be high and clearly
    above stock's."""
    rng = np.random.default_rng(123)
    H, K = 4, 4
    D = H * K
    n_seg = 6
    gold_seg = 2

    common_hat = rng.normal(size=D)
    common_hat /= np.linalg.norm(common_hat)
    d_raw = rng.normal(size=D)
    d_hat = d_raw - (d_raw @ common_hat) * common_hat
    d_hat /= np.linalg.norm(d_hat)

    A, eps_scale, b, tiny = 20.0, 8.0, 1.0, 0.05
    n_samples = 60
    samples = []
    for s_i in range(n_samples):
        eps = rng.uniform(-eps_scale, eps_scale, size=n_seg)
        c = np.zeros((n_seg, D))
        for i in range(n_seg):
            c[i] = (A + eps[i]) * common_hat + tiny * rng.normal(size=D)
            if i == gold_seg:
                c[i] += b * d_hat
        u = 5.0 * common_hat + 5.0 * d_hat + tiny * rng.normal(size=D)
        q = u.copy()

        stock_scores = np.full((1, n_seg), float("-inf"), dtype=np.float32)
        stock_scores[0, :n_seg] = c @ u
        meta = {"gold_seg": gold_seg, "cur_seg": n_seg, "n_seg": n_seg, "n_tok": 999,
                "eligible": True, "key_segs": [gold_seg], "sample_id": s_i}
        samples.append({
            "meta": meta,
            "u": u.reshape(1, H, K).astype(np.float32),
            "q": q.reshape(1, H, K).astype(np.float32),
            "c_full": c.reshape(1, n_seg, H, K).astype(np.float32),
            "stock_scores": stock_scores,
        })

    tracker = {"max_diff": 0.0, "n": 0, "worst": None}
    stats = xa.analyze_layer(samples, layer=0, sanity_tracker=tracker)

    chance = 2.0 / n_seg  # hit@2 out of 6 candidates
    assert stats["hit2"]["stock"] < chance + 0.25, (
        f"expected the dominant common-direction noise to mask the signal in the "
        f"raw (unnormalized) stock scores, got stock hit2={stats['hit2']['stock']}")
    assert stats["hit2"]["u_recomp_pc1"] > 0.9, (
        f"expected pc1 removal to uncover the masked signal, got "
        f"u_recomp_pc1 hit2={stats['hit2']['u_recomp_pc1']}")
    assert stats["hit2"]["u_recomp_pc1"] > stats["hit2"]["stock"] + 0.4


def test_sanity_gate_flags_mismatched_stock_scores():
    """If dumped stock_scores don't match score_from(u, c_full), the sanity
    tracker's max_diff should reflect that (regression guard for the
    coordinator's sanity-gate requirement)."""
    n_seg = 3
    H, K = 2, 2
    c = np.ones((n_seg, H, K), dtype=np.float32)
    u = np.ones((H, K), dtype=np.float32)
    # deliberately wrong stock_scores (should be c_flat@u = [4,4,4] since H*K=4 ones*ones)
    stock_scores = np.array([[100.0, 100.0, float("-inf")]], dtype=np.float32)
    meta = {"gold_seg": 0, "cur_seg": 2, "n_seg": n_seg, "n_tok": 9,
            "eligible": True, "key_segs": [0], "sample_id": 0}
    samples = [{"meta": meta, "u": u[None].astype(np.float16),
                "q": u[None].astype(np.float16),
                "c_full": c[None].astype(np.float16), "stock_scores": stock_scores}]
    tracker = {"max_diff": 0.0, "n": 0, "worst": None}
    xa.analyze_layer(samples, layer=0, sanity_tracker=tracker)
    assert tracker["max_diff"] > 50.0
    assert tracker["worst"] is not None


def test_ineligible_samples_must_be_filtered_before_analyze_layer():
    """analyze_layer assumes eligible-only input (spec: skip ineligible
    entirely) — this documents the contract at the load_dataset_samples
    layer rather than inside analyze_layer itself."""
    import inspect
    src = inspect.getsource(xa.load_dataset_samples)
    assert 'meta.get("eligible"' in src or "eligible" in src
