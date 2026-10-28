# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic-MoSC: Dynamic Mixture of Segment-Cache (from-scratch track).

A from-scratch linear-RNN architecture (GDN2 backbone) that fuses two axes:
  - temporal: surprisal-driven dynamic checkpointing of recurrent state (``dynamic_chunk``)
  - spatial : a hard top-k router over parallel segment-cache pools (``router``)

This is the *co-trained* sibling of the frozen ``lmr/`` retrofit track (SSC over a frozen backbone).
The two share tasks (``lmr.tasks``) and the hard-top-k selection idea, but nothing else — see
``lmr/mosc/README.md`` and ``report/README.md``.
"""

from lmr.mosc.backbone import GDN2LM

__all__ = ["GDN2LM"]
