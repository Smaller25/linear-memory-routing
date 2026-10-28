"""CPU-only tests for x2_null.py's structural_stats_for_sample: the
reviewer's stronger null (needle_null_hit2 = top-2 uniform over
needle-bearing eligible segments) and the structural top-2 ceiling
(max achievable macro hit@2 given one shared top-2 per sample). No
tokenizer/GPU — needles/gold_needles are hand-built dicts shaped like
data.py::annotate()'s output."""
import os
import sys

ANA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "lmr", "analysis", "260725_mc_niah_analysis"))
sys.path.insert(0, ANA)
import x2_null as xn  # noqa: E402


def _needle(seg, key="k"):
    return {"key": key, "seg": seg}


def test_single_needle_null_and_ceiling_are_both_one():
    """1 needle in context, eligible -> M=1, n_e=1, G=1 -> both null and
    ceiling saturate at 1.0 (matches reviewer's reference: single=1.000)."""
    needles = [_needle(3)]
    gold = needles
    stats = xn.structural_stats_for_sample(cur_seg=7, needles=needles, gold_needles=gold)
    assert stats["n_e"] == 1
    assert stats["M"] == 1
    assert stats["needle_null_hit2"] == 1.0
    assert stats["structural_ceiling"] == 1.0


def test_multikey_ceiling_is_one_null_uses_all_needle_segments():
    """multikey_1: 4 needles present (distinct keys), only 1 queried ->
    n_e=1 (G=1<=2 -> ceiling 1.0), but M counts ALL 4 needle-bearing
    segments (any key), not just the queried one -> null = 2/4 = 0.5,
    strictly BELOW plain chance if cur_seg > 4 (the whole point of the
    stronger null: a perfect needle-detector-but-blind-discriminator
    should do noticeably better than uniform-over-all-past-segments)."""
    needles = [_needle(1, "a"), _needle(3, "b"), _needle(5, "c"), _needle(6, "d")]
    gold = [needles[0]]  # only key "a" queried
    stats = xn.structural_stats_for_sample(cur_seg=8, needles=needles, gold_needles=gold)
    assert stats["n_e"] == 1
    assert stats["M"] == 4
    assert stats["needle_null_hit2"] == 0.5
    assert stats["structural_ceiling"] == 1.0  # G=1 <= 2


def test_multiquery_no_collision_ceiling_is_two_over_n():
    """4 distinct queried needles in 4 distinct segments (typical case,
    no birthday-paradox collision) -> G=4 > 2 -> ceiling = top-2 sum(1+1)/4
    = 0.5 (only 2 of 4 needles can ever share the one top-2 slice)."""
    needles = [_needle(1, "a"), _needle(2, "b"), _needle(3, "c"), _needle(4, "d")]
    gold = needles  # all 4 queried
    stats = xn.structural_stats_for_sample(cur_seg=6, needles=needles, gold_needles=gold)
    assert stats["n_e"] == 4
    assert stats["G"] == 4
    assert stats["structural_ceiling"] == 0.5
    assert stats["needle_null_hit2"] == 0.5  # M=4 too (all needles are gold here)


def test_multiquery_with_segment_collision_raises_ceiling():
    """Two of the 4 queried needles happen to land in the SAME segment
    (plausible with small cur_seg — a mini birthday-paradox regime) ->
    G=3 distinct segments with counts [2,1,1] -> best top-2 selection
    covers the count-2 segment plus one count-1 segment = 3 of 4 needles
    -> ceiling = 3/4 = 0.75, strictly above the no-collision 0.5 case."""
    needles = [_needle(1, "a"), _needle(1, "b"), _needle(3, "c"), _needle(4, "d")]
    gold = needles
    stats = xn.structural_stats_for_sample(cur_seg=6, needles=needles, gold_needles=gold)
    assert stats["n_e"] == 4
    assert stats["G"] == 3
    assert stats["structural_ceiling"] == 0.75


def test_two_queried_needles_ceiling_is_one():
    """G=2 (e.g. the q=2 sweep) -> both fit in top-2 simultaneously ->
    ceiling 1.0 regardless of counts (matches reviewer reference: q2=1.0)."""
    needles = [_needle(1, "a"), _needle(2, "b")]
    gold = needles
    stats = xn.structural_stats_for_sample(cur_seg=5, needles=needles, gold_needles=gold)
    assert stats["structural_ceiling"] == 1.0


def test_sample_with_zero_eligible_gold_needles_returns_none():
    """All queried needles land in/after cur_seg (structurally unroutable,
    matches x2_probe's eligibility exclusion) -> excluded entirely, not
    silently zeroed."""
    needles = [_needle(5, "a")]
    gold = needles
    stats = xn.structural_stats_for_sample(cur_seg=3, needles=needles, gold_needles=gold)
    assert stats is None


def test_aggregate_averages_only_eligible_samples():
    """compute_dataset_structural-style aggregation (exercised directly via
    structural_stats_for_sample over a hand-built sample list) must skip
    None (ineligible) entries rather than average them in as 0."""
    samples = [
        (7, [_needle(1, "a"), _needle(2, "b"), _needle(3, "c"), _needle(4, "d")],
         [_needle(1, "a"), _needle(2, "b"), _needle(3, "c"), _needle(4, "d")]),  # G=4, ceiling .5
        (2, [_needle(5, "a")], [_needle(5, "a")]),  # ineligible (seg 5 >= cur_seg 2)
        (3, [_needle(1, "a")], [_needle(1, "a")]),  # single, eligible, ceiling 1.0
    ]
    stats_list = [xn.structural_stats_for_sample(cs, nd, gd) for cs, nd, gd in samples]
    eligible = [s for s in stats_list if s is not None]
    assert len(eligible) == 2  # the seg=5/cur_seg=2 sample is excluded
    ceilings = [s["structural_ceiling"] for s in eligible]
    assert ceilings == [0.5, 1.0]
    assert sum(ceilings) / len(ceilings) == 0.75
