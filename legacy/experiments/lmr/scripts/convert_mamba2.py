# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 0: download an official mamba2 checkpoint, convert to FLA, run the logit-match.

GPU/local + network only.

    python -m lmr.scripts.convert_mamba2 --repo state-spaces/mamba2-1.3b
"""

from __future__ import annotations

import argparse

from lmr.converter import logit_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="state-spaces/mamba2-1.3b")
    # 1e-2: FLA scan vs mamba_ssm CUDA kernel differ at the ~1e-3 level on fp32 logits
    # (scale ~7); next-token argmax agreement is 100%, Pearson r=1.0 — pure kernel numerics.
    ap.add_argument("--atol", type=float, default=1e-2)
    args = ap.parse_args()

    max_diff = logit_match(args.repo, atol=args.atol)
    print(f"logit-match OK: max|Δ| = {max_diff:.3e} (< {args.atol})")


if __name__ == "__main__":
    main()
