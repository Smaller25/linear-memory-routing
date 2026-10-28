"""X2 (Task 3) CPU-only tests for the pure aggregation math in x2_probe.py:
two-level macro hit@2/hit@k, ineligible-needle exclusion, chance formulas,
best-layer selection. No torch/GPU — analyze_sample_multi (which does need
CUDA for the forward pass) is exercised separately on VESSL; this file only
covers aggregate_dataset, which consumes already-computed per-sample dicts
shaped exactly like analyze_sample_multi's return value."""
import os
import sys

ANA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "lmr", "analysis", "260725_mc_niah_analysis"))
sys.path.insert(0, ANA)
import x2_probe as xp  # noqa: E402


def _sample(cur_seg, n_seg, hit2_by_layer, hitk_by_layer, n_queried=None, n_ineligible=0):
    """hit2_by_layer / hitk_by_layer: {layer: [bool,...]} — same length per
    layer (= number of ELIGIBLE needles in this sample)."""
    n_layers = len(hit2_by_layer)
    per_layer = [{"layer": i, "hit2": hit2_by_layer[i], "hitk": hitk_by_layer[i]}
                for i in range(n_layers)]
    n_eligible = len(next(iter(hit2_by_layer.values()))) if hit2_by_layer else 0
    if n_queried is None:
        n_queried = n_eligible + n_ineligible
    return {"cur_seg": cur_seg, "n_seg": n_seg, "n_queried": n_queried,
            "n_eligible_needles": n_eligible, "n_ineligible_needles": n_ineligible,
            "k": n_queried, "topk_eff": min(n_queried, cur_seg), "per_layer": per_layer}


def test_macro_is_two_level_not_flattened():
    """Sample A: 2 eligible needles [True, False] -> per-sample macro 0.5.
    Sample B: 1 eligible needle [True] -> per-sample macro 1.0.
    Two-level macro = mean(0.5, 1.0) = 0.75, NOT the flattened
    mean([True,False,True]) = 0.667 — samples must not be weighted by how
    many needles they happen to ask about."""
    samples = [
        _sample(cur_seg=5, n_seg=8, hit2_by_layer={0: [True, False]}, hitk_by_layer={0: [True, False]}),
        _sample(cur_seg=5, n_seg=8, hit2_by_layer={0: [True]}, hitk_by_layer={0: [True]}),
    ]
    out, per_sample = xp.aggregate_dataset("t", 2048, n_rows=2, n_layers=1, sample_results=samples)
    assert out["per_layer"][0]["macro_hit2"] == 0.75
    assert abs(out["per_layer"][0]["macro_hit2"] - (1 + 0 + 1) / 3) > 1e-9  # not the flattened value


def test_ineligible_needles_excluded_from_numerator_and_counted():
    """A sample with 1 ineligible needle (gold_seg >= cur_seg, structurally
    unroutable) must exclude it from hit2/hitk lists (already done upstream
    in analyze_sample_multi) but its count must still surface in the
    per-sample record for auditing."""
    samples = [
        _sample(cur_seg=6, n_seg=10, hit2_by_layer={0: [True, True]},
                hitk_by_layer={0: [True, True]}, n_ineligible=1),
    ]
    out, per_sample = xp.aggregate_dataset("t", 2048, n_rows=1, n_layers=1, sample_results=samples)
    assert out["per_layer"][0]["macro_hit2"] == 1.0
    assert per_sample[0]["n_ineligible_needles"] == 1
    assert per_sample[0]["n_queried"] == 3  # 2 eligible + 1 ineligible


def test_sample_with_all_needles_ineligible_is_fully_excluded():
    """gold_seg == cur_seg for every queried needle (structurally unroutable,
    same as E1's single-gold 'eligible: false' case) -> sample contributes
    to neither macro hit2/hitk nor chance, per_layer is None."""
    samples = [
        _sample(cur_seg=0, n_seg=4, hit2_by_layer={}, hitk_by_layer={}, n_queried=1, n_ineligible=1),
        _sample(cur_seg=5, n_seg=8, hit2_by_layer={0: [True]}, hitk_by_layer={0: [True]}),
    ]
    out, per_sample = xp.aggregate_dataset("t", 2048, n_rows=2, n_layers=1, sample_results=samples)
    assert out["n_eligible_samples"] == 1
    assert out["n_ineligible_samples"] == 1
    assert per_sample[0]["per_layer"] is None
    assert out["per_layer"][0]["macro_hit2"] == 1.0  # only sample 2 counted
    assert out["per_layer"][0]["n_samples"] == 1


def test_chance_formulas_match_spec():
    """chance_hit2 = mean(min(1, 2/cur_seg)); chance_hitk = mean(min(1, k/cur_seg))
    (spec §7: multi-gold chance re-derivation). k here = n_queried (4), and
    cur_seg=4 makes k/cur_seg == 1.0 exactly (the k>=n_past saturation case)."""
    samples = [
        _sample(cur_seg=8, n_seg=10, hit2_by_layer={0: [True, False, True, False]},
                hitk_by_layer={0: [True, True, False, False]}),   # k=4, cur_seg=8 -> chance2=0.25, chancek=0.5
        _sample(cur_seg=4, n_seg=6, hit2_by_layer={0: [False, False, False, False]},
                hitk_by_layer={0: [True, False, True, False]}),   # k=4, cur_seg=4 -> chance2=0.5, chancek=1.0 (capped)
    ]
    out, _ = xp.aggregate_dataset("t", 2048, n_rows=2, n_layers=1, sample_results=samples)
    assert abs(out["chance_hit2"] - (0.25 + 0.5) / 2) < 1e-9
    assert abs(out["chance_hitk"] - (0.5 + 1.0) / 2) < 1e-9


def test_best_layer_selects_highest_macro_hit2():
    samples = [
        _sample(cur_seg=5, n_seg=8,
                hit2_by_layer={0: [True, False], 1: [True, True], 2: [False, False]},
                hitk_by_layer={0: [True, False], 1: [True, True], 2: [False, False]}),
    ]
    out, _ = xp.aggregate_dataset("t", 2048, n_rows=1, n_layers=3, sample_results=samples)
    assert out["best_layer"] == 1
    assert out["best_layer_macro_hit2"] == 1.0
    assert out["best_layer_macro_hitk"] == 1.0


def test_n_total_reflects_all_rows_including_failed_samples():
    """n_rows (the total dataset size) is passed independently of
    len(sample_results) so that GPU-side failures (exceptions during
    analyze_sample_multi, caught in run_dataset) don't silently shrink the
    reported denominator."""
    samples = [_sample(cur_seg=5, n_seg=8, hit2_by_layer={0: [True]}, hitk_by_layer={0: [True]})]
    out, _ = xp.aggregate_dataset("t", 2048, n_rows=5, n_layers=1, sample_results=samples)
    assert out["n_total"] == 5
    assert out["n_eligible_samples"] == 1


def test_all_combos_covers_expected_tasks_and_lengths():
    combos = xp.all_combos()
    assert ("niah_multiquery", 2048) in combos
    assert ("niah_multivalue", 8192) in combos
    assert ("niah_multikey_1", 4096) in combos
    assert ("niah_multiquery_q2", 2048) in combos
    assert ("niah_single_2", 2048) in combos
    # q=2 sweep and the essay-haystack single-needle control are 2048-only
    assert ("niah_multiquery_q2", 4096) not in combos
    assert ("niah_multiquery_q2", 8192) not in combos
    assert ("niah_single_2", 4096) not in combos
    n_extra_2048 = len(xp.EXTRA_TASKS_BY_LENGTH[2048])
    assert len(combos) == 3 * len(xp.CORE_TASKS) + n_extra_2048
