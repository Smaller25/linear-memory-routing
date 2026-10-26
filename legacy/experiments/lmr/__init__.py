# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Linear Memory Routing -- Memory Caching (MC) on FLA linear-attention models.

Adds memory capacity to fixed-state linear models (Mamba2, ...) by caching frozen recurrent
state checkpoints across segments and combining them at read-out (RM / GRM / SSC), plus a MoM
comparison arm. See ``lmr/`` module docstrings and the project plan for details.
"""

from lmr.readout import (
    GatedResidualMemory,
    ResidualMemory,
    SparseSelectiveCaching,
    build_readout,
)
from lmr.segment_runner import run_mixer_with_cache, run_segmented_lm
from lmr.ssd_scan import naive_ssd_scan

__all__ = [
    "naive_ssd_scan",
    "run_mixer_with_cache",
    "run_segmented_lm",
    "ResidualMemory",
    "GatedResidualMemory",
    "SparseSelectiveCaching",
    "build_readout",
]
