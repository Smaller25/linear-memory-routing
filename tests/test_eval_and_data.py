import json
import math

import numpy as np
import pytest

from mslm.data.loader import TokenWindows, sample_ids
from mslm.eval import compare
from mslm.eval.metrics import hypergeom_hit_at_k, item_score, postprocess_pred, string_match_all


def test_hypergeom_baseline():
    # 4 gold among 31 eligible segments, k=2: 1 - C(27,2)/C(31,2)
    assert hypergeom_hit_at_k(31, 4, 2) == pytest.approx(1 - math.comb(27, 2) / math.comb(31, 2))
    assert hypergeom_hit_at_k(31, 4, 2) == pytest.approx(0.2452, abs=1e-4)
    assert hypergeom_hit_at_k(31, 1, 4) == pytest.approx(4 / 31)
    assert hypergeom_hit_at_k(3, 1, 4) == 1.0
    assert hypergeom_hit_at_k(10, 0, 2) == 0.0


def test_item_score_matches_official():
    preds = ["The number is 123 and 456", "nothing", "  7\x01"]
    refs = [["123", "456", "789"], ["1"], ["7"]]
    per = [item_score(postprocess_pred(p), r) for p, r in zip(preds, refs)]
    assert round(100 * sum(per) / len(per), 2) == string_match_all([postprocess_pred(p) for p in preds], refs)


def test_sampler_deterministic_disjoint(tmp_path):
    arr = np.arange(10_000, dtype=np.uint16)
    arr.tofile(tmp_path / "train_000.bin")
    arr[:500].tofile(tmp_path / "val_000.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"train_files": ["train_000.bin"], "val_files": ["val_000.bin"]}))
    d = TokenWindows(str(tmp_path), "train", 99, seed=0)
    ids = [sample_ids(3, m, 2, 2, r, 2) for m in range(2) for r in range(2)]
    flat = [i for x in ids for i in x]
    assert len(set(flat)) == len(flat) == 8
    x1, y1 = d.batch(flat)
    x2, _ = TokenWindows(str(tmp_path), "train", 99, seed=0).batch(flat)
    assert (x1 == x2).all() and (y1[:, :-1] == x1[:, 1:]).all()
    starts = {int(r[0]) for r in x1}
    assert all(s % 100 == 0 for s in starts)              # non-overlapping windows


def test_compare_paired(tmp_path):
    for name, scores in {"a": [0, 0, 1, 1], "b": [1, 1, 1, 0]}.items():
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for i, s in enumerate(scores):
                f.write(json.dumps({"route": "learned", "length": 8192, "task": "t", "index": i, "score": s}) + "\n")
    rows = compare.compare(compare.load([str(tmp_path / "a.jsonl")], None), compare.load([str(tmp_path / "b.jsonl")], None))
    assert rows[0]["b_wins"] == 2 and rows[0]["b_losses"] == 1 and rows[0]["diff"] == 25.0
