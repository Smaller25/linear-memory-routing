# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the linear-decomposition identity that RM relies on."""

import torch

from lmr.ssd_scan import naive_ssd_scan


def _inputs(b=2, seqlen=12, h=4, p=8, n=16, chunk=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, seqlen, h, p, generator=g, dtype=torch.float64)
    dt = torch.randn(b, seqlen, h, generator=g, dtype=torch.float64)
    A = -torch.rand(h, generator=g, dtype=torch.float64) - 0.1  # negative
    B = torch.randn(b, seqlen, h, n, generator=g, dtype=torch.float64)
    C = torch.randn(b, seqlen, h, n, generator=g, dtype=torch.float64)
    D = torch.randn(h, p, generator=g, dtype=torch.float64)
    dt_bias = torch.randn(h, generator=g, dtype=torch.float64)
    h0 = torch.randn(b, h, p, n, generator=g, dtype=torch.float64)
    return x, dt, A, B, C, D, dt_bias, h0, chunk


def test_linearity_in_state_and_input():
    """scan(x, init=h0) == scan(x, init=0) + scan(0, init=h0)."""
    x, dt, A, B, C, D, dt_bias, h0, chunk = _inputs()
    kw = dict(D=D, dt_bias=dt_bias, dt_softplus=True)

    y_full, _ = naive_ssd_scan(x, dt, A, B, C, chunk, initial_states=h0, **kw)
    y_input, _ = naive_ssd_scan(x, dt, A, B, C, chunk, initial_states=None, **kw)
    # state-only contribution: zero input so the D skip drops out too.
    y_state, _ = naive_ssd_scan(torch.zeros_like(x), dt, A, B, C, chunk, initial_states=h0,
                                D=None, dt_bias=dt_bias, dt_softplus=True)

    assert torch.allclose(y_full, y_input + y_state, atol=1e-9, rtol=1e-6)


def test_none_equals_zero_state():
    x, dt, A, B, C, D, dt_bias, h0, chunk = _inputs()
    kw = dict(D=D, dt_bias=dt_bias, dt_softplus=True)
    y_none, _ = naive_ssd_scan(x, dt, A, B, C, chunk, initial_states=None, **kw)
    zeros = torch.zeros_like(h0)
    y_zero, _ = naive_ssd_scan(x, dt, A, B, C, chunk, initial_states=zeros, **kw)
    assert torch.allclose(y_none, y_zero, atol=1e-9)


def test_final_state_shape():
    x, dt, A, B, C, D, dt_bias, h0, chunk = _inputs()
    _, final = naive_ssd_scan(x, dt, A, B, C, chunk, D=D, dt_bias=dt_bias, initial_states=h0)
    assert final.shape == h0.shape
