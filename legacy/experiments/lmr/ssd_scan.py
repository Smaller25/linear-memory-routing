# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-torch SSD (Mamba2) scan with explicit ``initial_states`` support.

This mirrors the *prefill* branch of :meth:`fla.layers.mamba2.Mamba2Mixer.torch_forward`
(``fla/layers/mamba2.py:588-669``) but exposes the chunk-level ``previous_states`` seed
(line 635 of that file, hardcoded to zeros there) as an argument. That single change is
what Memory Caching needs: it lets us run the scan on a segment with an *injected* frozen
state checkpoint.

The scan is *jointly linear* in ``(x, initial_states)``::

    y(x, h0) == y(x, 0) + y(0, h0)

which is the identity the Residual-Memory (RM) read-out relies on (see
:mod:`lmr.readout`). ``tests/lmr/test_ssd_scan.py`` asserts it numerically.

Runs on CPU (no Triton/CUDA) so it doubles as the reference implementation and the
CPU-testable substrate for the whole MC mechanism. On CUDA the segment runner can
dispatch to ``mamba_ssm.mamba_chunk_scan_combined`` instead (same math, faster).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Reuse FLA's own chunking helpers for exact numerical parity with torch_forward.
from fla.layers.mamba2 import pad_tensor_by_size, reshape_into_chunks, segment_sum


def naive_ssd_scan(
    hidden_states: torch.Tensor,   # [b, l, h, p]  raw x (NOT discretized)
    dt: torch.Tensor,              # [b, l, h]     raw dt (pre-softplus)
    A: torch.Tensor,               # [h]           raw A = -exp(A_log)
    B: torch.Tensor,               # [b, l, h, n]  already repeated to num_heads
    C: torch.Tensor,               # [b, l, h, n]  already repeated to num_heads
    chunk_size: int,
    D: torch.Tensor | None = None,         # [h, p] or [h]; skip connection
    dt_bias: torch.Tensor | None = None,   # [h]
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: torch.Tensor | None = None,  # [b, h, p, n] frozen seed state
    return_final_state: bool = True,
):
    """Naive chunked SSD scan with an optional injected initial state.

    Returns ``(y, final_state)`` where ``y`` is ``[b, l, h, p]`` (the SSM output *before*
    the output gate / RMSNorm) and ``final_state`` is ``[b, h, p, n]`` (or ``None``).

    All math is done in fp32 for parity with ``torch_forward``.
    """
    b, seqlen, nheads, headdim = hidden_states.shape
    orig_dtype = hidden_states.dtype

    # Compute in fp32 for parity with torch_forward / the mamba_ssm kernel, but honour fp64
    # inputs so the linearity identity can be checked at full precision.
    compute_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
    hidden_states = hidden_states.to(compute_dtype)
    B = B.to(compute_dtype)
    C = C.to(compute_dtype)
    A = A.to(compute_dtype)

    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = nn.functional.softplus(dt)
    dt = torch.clamp(dt, dt_limit[0], dt_limit[1]).to(compute_dtype)

    pad_size = (chunk_size - seqlen % chunk_size) % chunk_size

    D_residual = None
    if D is not None:
        D = D.to(compute_dtype)
        if D.dim() == 1:  # [h] -> [h, p]
            D = D[:, None].expand(nheads, headdim)
        D_residual = D * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A.
    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt  # [b, l, h]

    # Rearrange into chunks.
    hidden_states, A, B, C = (reshape_into_chunks(t, pad_size, chunk_size) for t in (hidden_states, A, B, C))

    A = A.permute(0, 3, 1, 2)  # [b, h, c, chunk]
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Intra-chunk (diagonal) outputs.
    L = torch.exp(segment_sum(A))
    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
    G = G_intermediate.sum(dim=-1)
    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Per-chunk states.
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)  # [b, c, h, p, n]

    # 3. Inter-chunk recurrence -- seed with the injected initial state (the MC hook).
    if initial_states is not None:
        previous_states = initial_states[:, None].to(states.dtype)  # [b, 1, h, p, n]
    else:
        previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. State -> output (off-diagonal) contribution.
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    y = Y_diag + Y_off
    y = y.reshape(b, -1, nheads, headdim)
    if D_residual is not None:
        y = y + D_residual
    if pad_size > 0:
        y = y[:, :seqlen, :, :]

    y = y.to(orig_dtype)
    if not return_final_state:
        final_state = None
    return y, final_state
