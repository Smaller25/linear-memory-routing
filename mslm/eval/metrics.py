"""Scoring: RULER string match (official) and routing diagnostics."""

from __future__ import annotations

import importlib.util
import math
import re

from mslm.envfile import REPO_ROOT

_spec = importlib.util.spec_from_file_location("ruler_eval_metrics", REPO_ROOT / "third_party/ruler/eval_metrics.py")
ruler_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ruler_metrics)
string_match_all = ruler_metrics.string_match_all     # official NIAH metric (mean over items, x100)


def postprocess_pred(text: str) -> str:
    """RULER scripts/eval/evaluate.py postprocess_pred: drop control chars, strip."""
    text = re.sub(r"[\x00-\x1f]", "\n", text)
    return text.strip()


def item_score(pred: str, answers: list[str]) -> float:
    """Per-item version of string_match_all (0..1): fraction of answers found."""
    p = pred.lower()
    return sum(a.lower() in p for a in answers) / len(answers)


def hypergeom_hit_at_k(n_candidates: int, n_gold: int, k: int) -> float:
    """P(at least one gold among k segments drawn uniformly without replacement).

    Random-routing baseline per item: n_candidates = eligible past segments at the
    query position, n_gold = distinct gold segments among them. 1 - C(N-G,k)/C(N,k).
    """
    k = min(k, n_candidates)
    if n_gold <= 0 or k <= 0:
        return 0.0
    if n_candidates - n_gold < k:
        return 1.0
    return 1.0 - math.comb(n_candidates - n_gold, k) / math.comb(n_candidates, k)


def routing_stats(scores_row, chosen_row, gold: list[int]) -> dict:
    """Routing at one query position of one layer.

    scores_row: [N] router scores (-inf = not eligible); chosen_row: [R] chosen ids (-1 invalid).
    Returns hit (any gold chosen), best gold rank (1 = top) among eligible, random baseline.
    """
    import torch
    eligible = torch.isfinite(scores_row)
    n_cand = int(eligible.sum())
    gold_el = sorted({g for g in gold if 0 <= g < len(scores_row) and bool(eligible[g])})
    chosen = {int(c) for c in chosen_row.tolist() if c >= 0}
    k = len(chosen_row)
    if not gold_el:
        return {"n_candidates": n_cand, "n_gold": 0, "hit": None, "gold_rank": None, "random_hit": None}
    order = torch.argsort(scores_row.masked_fill(~eligible, -float("inf")), descending=True).tolist()
    rank = min(order.index(g) for g in gold_el) + 1
    return {"n_candidates": n_cand, "n_gold": len(gold_el), "hit": float(any(g in chosen for g in gold_el)),
            "gold_rank": rank, "random_hit": hypergeom_hit_at_k(n_cand, len(gold_el), k)}
