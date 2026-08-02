# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 3: real recall (FDA / SWDE / SQuAD / NQ) via the FLA lm-eval harness.

Reuses ``evals/harness.py`` (``@register_model('fla')``). This is a thin convenience wrapper
that documents the task set and forwards to ``lm_eval``; run on GPU/local.

    python -m lmr.scripts.eval.eval_real --model-path <converted-fla-mamba2> \
        --tasks fda swde squad_completion nq_open
"""

from __future__ import annotations

import argparse

DEFAULT_TASKS = ["fda", "swde", "squad_completion", "nq_open"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="path to a converted FLA mamba2 checkpoint")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--batch-size", default="auto")
    args = ap.parse_args()

    import fla  # noqa: F401  registers the 'fla' lm-eval model
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="fla",
        model_args=f"pretrained={args.model_path}",
        tasks=args.tasks,
        batch_size=args.batch_size,
    )
    for task, metrics in results["results"].items():
        print(task, metrics)


if __name__ == "__main__":
    main()
