# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU test for the linear-decomposition identity the GDN Memory-Caching read-out relies on.

The gated-delta state is affine in its initial state for fixed inputs:
``h_i = decay_i (I - beta_i k_i k_i^T) h_{i-1} + beta_i k_i (x) v_i``. Zeroing the value ``v`` kills
the write term, so the output becomes purely the cached state's contribution. This mirrors
``test_ssd_scan.py`` for Mamba2 and justifies the GDN adapter's ``memory_only`` scan (which zeros
``v``). Uses the pure-PyTorch reference op so it runs on CPU, independent of the Triton kernel.
"""

import torch

from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from lmr.state_utils import gdn_meanpool_state


def _inputs(b=2, t=12, h=3, k=8, v=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, k, generator=g)
    key = torch.randn(b, t, h, k, generator=g)
    val = torch.randn(b, t, h, v, generator=g)
    beta = torch.rand(b, t, h, generator=g)               # in (0, 1), like sigmoid(b_proj)
    decay = -torch.rand(b, t, h, generator=g) * 0.5 - 0.01  # negative -> exp in (0, 1)
    s0 = torch.randn(b, h, k, v, generator=g)
    return q, key, val, beta, decay, s0


def test_superposition_in_state_and_value():
    """scan(v, init=S0) == scan(v=0, init=S0) + scan(v, init=0)."""
    q, k, v, beta, g, s0 = _inputs()

    y_full, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g, initial_state=s0)
    y_state, _ = naive_recurrent_gated_delta_rule(q, k, torch.zeros_like(v), beta, g, initial_state=s0)
    y_input, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g, initial_state=None)

    assert torch.allclose(y_full, y_state + y_input, atol=1e-4, rtol=1e-3)


def test_gdn_meanpool_state_shape():
    b, hv, k, v = 2, 3, 8, 10
    state = torch.randn(b, hv, k, v)
    d = gdn_meanpool_state(state)
    assert d.shape == (b, hv * k)
    assert torch.allclose(d, state.mean(dim=3).reshape(b, hv * k))
