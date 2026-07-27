import os, random, sys

import pytest
import torch

ANA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "lmr", "analysis", "260725_mc_niah_analysis"))
sys.path.insert(0, ANA)
from x4_gen import select_and_score, chance_upper  # noqa: E402


# ---------------------------------------------------------------------------
# select_and_score — pure selection-override logic
# ---------------------------------------------------------------------------
def test_stock_mode_is_real_topk_unchanged():
    scores = [1.0, 5.0, 3.0, 2.0, 4.0]
    idx, sc = select_and_score("stock", scores, topk=2, online_score=0.0)
    assert idx == [1, 4]          # indices of the two highest scores (5, 4)
    assert sc == [5.0, 4.0]       # real scores, no flooring


def test_recent_mode_picks_highest_indices_ignoring_score():
    scores = [9.0, 1.0, 1.0, 1.0, 0.5]  # segment 0 has the best score but is NOT recent
    idx, sc = select_and_score("recent", scores, topk=2, online_score=0.0)
    assert idx == [4, 3]           # most recent first (n-1, n-2, ...)
    # floor = max(real score of CHOSEN slots (0.5, 1.0) -> 1.0, online=0.0) = 1.0
    assert sc == [1.0, 1.0]


def test_recent_mode_floor_uses_online_when_online_is_higher():
    scores = [1.0, 1.0, 1.0]
    idx, sc = select_and_score("recent", scores, topk=1, online_score=9.0)
    assert idx == [2]
    assert sc == [9.0]


def test_random_mode_requires_rng():
    with pytest.raises(ValueError):
        select_and_score("random", [1.0, 2.0], topk=1, online_score=0.0, rng=None)


def test_random_mode_uses_rng_and_floors_score():
    scores = [10.0, 0.0, 0.0, 0.0]  # segment 0 is the "good" one; random should be able to miss it
    rng = random.Random(0)
    idx, sc = select_and_score("random", scores, topk=1, online_score=1.0, rng=rng)
    assert len(idx) == 1 and 0 <= idx[0] < 4
    # floor = max(real score of the chosen slot, online=1.0)
    expected_floor = max(scores[idx[0]], 1.0)
    assert sc == [expected_floor]


def test_random_mode_is_reproducible_given_same_seed():
    scores = list(range(20))
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    idx1, _ = select_and_score("random", scores, topk=3, online_score=0.0, rng=rng1)
    idx2, _ = select_and_score("random", scores, topk=3, online_score=0.0, rng=rng2)
    assert idx1 == idx2


def test_random_mode_different_seeds_usually_differ():
    scores = list(range(50))
    picks = set()
    for seed in range(5):
        idx, _ = select_and_score("random", scores, topk=2, online_score=0.0,
                                  rng=random.Random(seed))
        picks.add(tuple(sorted(idx)))
    assert len(picks) > 1  # 5 different seeds should not all collapse to one draw


def test_topk_clamped_to_available_segments():
    idx, sc = select_and_score("stock", [1.0, 2.0], topk=5, online_score=0.0)
    assert len(idx) == 2 and len(sc) == 2


def test_zero_frozen_segments_returns_empty():
    for mode in ("stock", "recent"):
        idx, sc = select_and_score(mode, [], topk=2, online_score=0.0)
        assert idx == [] and sc == []
    idx, sc = select_and_score("random", [], topk=2, online_score=0.0, rng=random.Random(0))
    assert idx == [] and sc == []


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        select_and_score("bogus", [1.0], topk=1, online_score=0.0)


# ---------------------------------------------------------------------------
# chance_upper
# ---------------------------------------------------------------------------
def test_chance_upper_formula():
    # N=9 -> 2/(9-1)*0.6 = 0.15
    assert chance_upper(9, conditional_em=0.6) == pytest.approx(0.15)


def test_chance_upper_clamped_to_one():
    # N=2 -> 2/(2-1)=2.0, clamped to 1.0, * 0.6 = 0.6
    assert chance_upper(2, conditional_em=0.6) == pytest.approx(0.6)


def test_chance_upper_undefined_for_n_seg_le_1():
    assert chance_upper(1) is None
    assert chance_upper(0) is None


def test_chance_upper_scales_with_length():
    # spec's stated prediction: chance_upper should DECREASE as n_seg grows
    # (more segments to route among by chance)
    vals = [chance_upper(n) for n in (4, 8, 16, 32, 64, 128)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# Vectorized-vs-reference cross-check: OverrideGDN2SSC's forward vectorizes
# select_and_score's rule across a [B,T] batch of tokens using pure torch ops
# (top-k-of-random-values sampling-without-replacement trick + gather/floor).
# We can't import OverrideGDN2SSC itself here (it needs the pinned long-gdn
# worktree + fla, GPU-only), but we CAN replicate exactly the same torch
# expression it uses and check it agrees with select_and_score row-by-row —
# this is the "summary/score bookkeeping" cross-check for the vectorized
# path (see x4_gen.OverrideGDN2SSC.forward for the real, in-situ version).
# ---------------------------------------------------------------------------
def _vectorized_recent_or_random(mode, all_scores, online_score, topk, seed=None):
    """all_scores: [B,T,N] tensor. Mirrors OverrideGDN2SSC.forward's
    recent/random branch + floor rule exactly."""
    B, T, N = all_scores.shape
    route_count = min(topk, N)
    if mode == "recent":
        idx = torch.arange(N - 1, N - 1 - route_count, -1)
        top_indices = idx.view(1, 1, route_count).expand(B, T, -1)
    else:
        gen = torch.Generator()
        gen.manual_seed(seed)
        rnd = torch.rand(B, T, N, generator=gen)
        top_indices = torch.topk(rnd, k=route_count, dim=-1).indices
    selected_scores = torch.gather(all_scores, -1, top_indices)
    floor = torch.maximum(selected_scores.max(-1).values, online_score)
    top_scores = floor.unsqueeze(-1).expand(-1, -1, route_count)
    return top_indices, top_scores


def test_vectorized_recent_matches_reference_per_row():
    torch.manual_seed(0)
    B, T, N, topk = 2, 5, 7, 2
    all_scores = torch.randn(B, T, N)
    online_score = torch.randn(B, T)
    vec_idx, vec_sc = _vectorized_recent_or_random("recent", all_scores, online_score, topk)
    for b in range(B):
        for t in range(T):
            ref_idx, ref_sc = select_and_score(
                "recent", all_scores[b, t].tolist(), topk, float(online_score[b, t]))
            assert sorted(vec_idx[b, t].tolist()) == sorted(ref_idx)
            assert vec_sc[b, t].tolist() == pytest.approx(ref_sc)


def test_vectorized_random_matches_reference_distribution_and_floor_rule():
    # We can't force identical index DRAWS between python's random.Random
    # and torch.Generator (different PRNGs), so we check the invariants
    # select_and_score guarantees instead of bit-identical indices: valid
    # k-subset, and the floor-score rule holds given whatever indices were
    # drawn.
    torch.manual_seed(0)
    B, T, N, topk = 3, 4, 10, 3
    all_scores = torch.randn(B, T, N)
    online_score = torch.randn(B, T)
    vec_idx, vec_sc = _vectorized_recent_or_random(
        "random", all_scores, online_score, topk, seed=123)
    for b in range(B):
        for t in range(T):
            idx = vec_idx[b, t].tolist()
            assert len(idx) == len(set(idx)) == topk  # no replacement
            assert all(0 <= i < N for i in idx)
            expected_floor = max(max(all_scores[b, t, i].item() for i in idx),
                                 float(online_score[b, t]))
            assert vec_sc[b, t, 0].item() == pytest.approx(expected_floor)


def test_random_sampling_without_replacement_is_uniform_ish():
    # top-k-of-iid-uniform-keys trick: sanity-check no index is starved
    # across many draws (weak uniformity check, not a strict distribution
    # test).
    torch.manual_seed(1)
    N, topk, trials = 5, 2, 4000
    gen = torch.Generator()
    gen.manual_seed(7)
    rnd = torch.rand(trials, 1, N, generator=gen)
    idx = torch.topk(rnd, k=topk, dim=-1).indices.reshape(trials, topk)
    counts = torch.bincount(idx.reshape(-1), minlength=N).float()
    # each of N segments should appear in a comparable fraction of the
    # trials*topk draws (uniform would be exactly trials*topk/N each)
    expected = trials * topk / N
    assert (counts - expected).abs().max() < 0.15 * expected
