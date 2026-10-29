"""Self-supervised routing labels from repeated spans (span-copy).

At input position t the model predicts y_t. If the n tokens ending at t already
appeared ending at an earlier position j, and the token after that earlier
occurrence equals y_t (x_{j+1} == y_t), then copying from the earlier occurrence
predicts the next token. The segment holding x_{j+1} is a gold segment for the
router at t. Only segments at least ``min_gap`` segments before t's own segment
count (min_gap=2 drops the immediately preceding segment, which is too easy).
Up to ``max_gold`` distinct gold segments are kept, most recent first; -1 pads.

    python -m mslm.data.span_copy --stats   # label density on the val split (uses CONFIG)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

_P = np.uint64(0x9E3779B97F4A7C15)   # odd 64-bit multiplier (wrapping arithmetic)


def _keys(w: np.ndarray, n: int) -> np.ndarray:
    """key[p] = hash(w[p-n+1..p], w[p+1]) for p in [n-1, len(w)-2]; positions before n-1 are unused."""
    w = w.astype(np.uint64) + np.uint64(1)
    L = len(w) - 1                                   # number of input positions
    h = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):                           # h[p] = sum_i w[p-i] * P^(i+1), wrapping
            h[n - 1:] = h[n - 1:] * _P + w[n - 1 - i: L - i]
        h = h * _P + w[1:]                           # append the next token
    return h


def copy_gold(x: np.ndarray, y: np.ndarray, chunk: int, n: int, min_gap: int = 2, max_gold: int = 4) -> np.ndarray:
    """Gold segments per input position. x, y: [T] (y = x shifted by one). Returns int64 [T, max_gold]."""
    T = len(x)
    out = np.full((T, max_gold), -1, dtype=np.int64)
    if T <= n:
        return out
    w = np.concatenate([x, y[-1:]])
    key = _keys(w, n)[n - 1:]                        # key of position p = n-1+i
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        pos = order[s:e] + (n - 1)                   # ascending positions sharing the key
        for a in range(1, len(pos)):
            t = pos[a]
            seg_t = t // chunk
            golds = []
            for j in pos[max(0, a - 64):a][::-1]:    # most recent first, at most 64 back
                g = (j + 1) // chunk
                if g <= seg_t - min_gap and g not in golds:
                    golds.append(g)
                    if len(golds) == max_gold:
                        break
            if golds:
                out[t, :len(golds)] = golds
    return out


def batch_copy_gold(x: torch.Tensor, y: torch.Tensor, chunk: int, n: int, min_gap: int, max_gold: int) -> torch.Tensor:
    """[B,T] token tensors -> [B,T,max_gold] int64 gold segments (-1 = none)."""
    xs, ys = x.numpy(), y.numpy()
    return torch.from_numpy(np.stack([copy_gold(a, b, chunk, n, min_gap, max_gold) for a, b in zip(xs, ys)]))


def stats(windows, chunk: int, ngrams, min_gap: int, n_windows: int) -> list[dict]:
    """Label density and segment distance over the first n_windows windows of a TokenWindows split."""
    rows = []
    for n in ngrams:
        per_win, dist = [], []
        for i in range(min(n_windows, windows.n)):
            x, y = windows.batch([i])
            g = copy_gold(x[0].numpy(), y[0].numpy(), chunk, n, min_gap, 1)[:, 0]
            lab = np.flatnonzero(g >= 0)
            per_win.append(len(lab))
            dist.extend((lab // chunk - g[lab]).tolist())
        per_win, dist = np.array(per_win), np.array(dist)
        rows.append({"ngram": n, "windows": len(per_win), "labels_per_window_mean": float(per_win.mean()),
                     "labels_per_window_median": float(np.median(per_win)),
                     "windows_with_lt10_labels": float((per_win < 10).mean()),
                     "label_frac": float(per_win.sum() / (len(per_win) * windows.win)),
                     "seg_distance_quartiles": np.percentile(dist, [25, 50, 75]).tolist() if len(dist) else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--ngrams", default="8,16,32")
    ap.add_argument("--windows", type=int, default=256)
    a = ap.parse_args()
    from mslm.data.loader import TokenWindows
    from mslm.settings import Settings
    s = Settings.from_env()
    val = TokenWindows(s.str("TRAIN_DATA_DIR"), "val", s.int("SEQ_LEN"), 0)
    import json
    for r in stats(val, s.int("MC_CHUNK"), [int(v) for v in a.ngrams.split(",")], s.int("COPY_MIN_GAP"), a.windows):
        print(json.dumps(r))


if __name__ == "__main__":
    main()
