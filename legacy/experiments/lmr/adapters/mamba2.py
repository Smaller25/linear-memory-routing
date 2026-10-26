# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Mamba2 adapter for the Memory-Caching segment runner.

The per-segment mixer math reproduces the prefill branch of
:meth:`fla.layers.mamba2.Mamba2Mixer.torch_forward` (projection + causal conv + SSD scan + output
gate) *without editing* ``fla/``; it reuses the mixer's own submodules. The scan dispatches to
:func:`lmr.ssd_scan.naive_ssd_scan` (CPU) or, when available and on CUDA,
``mamba_ssm.mamba_chunk_scan_combined`` (same math, faster).

This module preserves the original ``segment_runner`` math exactly -- the helpers below are moved
verbatim from the pre-refactor ``lmr/segment_runner.py``; the adapter just wires them into the
generic :func:`lmr.adapters.base.run_mixer_with_cache`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lmr.adapters.base import Adapter, run_mixer_with_cache
from lmr.ssd_scan import naive_ssd_scan
from lmr.state_utils import meanpool_state

try:  # optional, CUDA-only fast path
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except Exception:  # pragma: no cover - absent on CPU-only boxes
    mamba_chunk_scan_combined = None


def _project_segment(mixer, x_in: torch.Tensor):
    """Run in_proj + causal conv and split into the SSD inputs.

    Returns ``(x, dt, B_g, C_g, gate)`` with shapes
    ``x:[b,l,h,p]``, ``dt:[b,l,h]``, ``B_g/C_g:[b,l,n_groups,n]``, ``gate:[b,l,intermediate]``.
    Mirrors ``torch_forward`` lines 471-518 (prefill, no cache, no padding mask).
    """
    b, seqlen, _ = x_in.shape
    projected = mixer.in_proj(x_in)
    d_mlp = (projected.shape[-1] - 2 * mixer.intermediate_size
             - 2 * mixer.n_groups * mixer.ssm_state_size - mixer.num_heads) // 2
    _, _, gate, hidden_states_B_C, dt = projected.split(
        [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1,
    )
    hidden_states_B_C = mixer.act(
        mixer.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seqlen].transpose(1, 2),
    )
    hidden_states, B, C = torch.split(
        hidden_states_B_C,
        [mixer.intermediate_size, mixer.n_groups * mixer.ssm_state_size, mixer.n_groups * mixer.ssm_state_size],
        dim=-1,
    )
    x = hidden_states.reshape(b, seqlen, mixer.num_heads, mixer.head_dim)
    B_g = B.reshape(b, seqlen, mixer.n_groups, mixer.ssm_state_size)
    C_g = C.reshape(b, seqlen, mixer.n_groups, mixer.ssm_state_size)
    return x, dt, B_g, C_g, gate


def _run_scan(mixer, x, dt, B_g, C_g, A, D, initial_states, backend):
    """Dispatch one SSD scan. Returns ``(y:[b,l,h,p], final_state:[b,h,p,n])``."""
    if backend == "cuda" and mamba_chunk_scan_combined is not None and x.is_cuda:
        # The mamba_ssm backward kernel asserts D.stride(-1) == 1; _get_D(expand_to_head_dim)
        # returns a non-contiguous expand, so make it contiguous for the training (backward) path.
        if D is not None:
            D = D.contiguous()
        y, final_state = mamba_chunk_scan_combined(
            x, dt, A, B_g, C_g,
            chunk_size=mixer.chunk_size, D=D, z=None, seq_idx=None,
            return_final_states=True, dt_bias=mixer.dt_bias, dt_softplus=True,
            initial_states=initial_states,
            **({} if mixer.dt_limit == (0.0, float("inf")) else {"dt_limit": mixer.dt_limit}),
        )
        return y, final_state
    rep = mixer.num_heads // mixer.n_groups
    B = B_g.repeat(1, 1, rep, 1)
    C = C_g.repeat(1, 1, rep, 1)
    return naive_ssd_scan(
        x, dt, A, B, C, mixer.chunk_size, D=D, dt_bias=mixer.dt_bias,
        dt_softplus=True, dt_limit=mixer.dt_limit, initial_states=initial_states,
    )


def _finalize(mixer, y, gate, dtype):
    """Apply the output gate / RMSNorm and final projection (torch_forward lines 660-668)."""
    b, seqlen = y.shape[0], y.shape[1]
    y = y.reshape(b, seqlen, -1)
    if mixer.rmsnorm:
        scan_output = mixer.norm(y, gate)
    else:
        scan_output = y * mixer.act(gate)
    return mixer.out_proj(scan_output.to(dtype))


@dataclass
class _Mamba2Proj:
    x: torch.Tensor
    dt: torch.Tensor
    B_g: torch.Tensor
    C_g: torch.Tensor
    gate: torch.Tensor
    A: torch.Tensor
    D: torch.Tensor


class Mamba2Adapter(Adapter):
    name = "mamba2"

    def project(self, mixer, x_in):
        x, dt, B_g, C_g, gate = _project_segment(mixer, x_in)
        A = -torch.exp(mixer.A_log.float())
        D = mixer._get_D(expand_to_head_dim=True)
        return _Mamba2Proj(x, dt, B_g, C_g, gate, A, D)

    def scan(self, mixer, proj, initial_state, backend, *, memory_only):
        if memory_only:
            # Zero the value path and drop the D skip so the output is purely the cached state's
            # contribution -- exact by SSD linearity y(x, h0) = y(x, 0) + y(0, h0).
            return _run_scan(mixer, torch.zeros_like(proj.x), proj.dt, proj.B_g, proj.C_g,
                             proj.A, None, initial_state, backend)
        return _run_scan(mixer, proj.x, proj.dt, proj.B_g, proj.C_g, proj.A, proj.D,
                         initial_state, backend)

    def descriptor(self, state):
        return meanpool_state(state)

    def descriptor_dim(self, mixer):
        return mixer.num_heads * mixer.ssm_state_size

    def finalize(self, mixer, proj, y, dtype):
        return _finalize(mixer, y, proj.gate, dtype)

    def blocks(self, model):
        return model.backbone.layers

    def mixer_of(self, block):
        return block.mixer

    def embed(self, model, input_ids):
        return model.backbone.embeddings(input_ids)

    def final_norm(self, model, hidden):
        return model.backbone.norm_f(hidden)

    def vanilla_hidden(self, model, input_ids):
        return model.backbone(input_ids).last_hidden_state

    def lm_head(self, model):
        return model.lm_head

    def run_block(self, block, hidden, cached_states, readout, backend):
        residual = hidden
        normed = block.norm(hidden)
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)
        out, final_state, aux = run_mixer_with_cache(self, self.mixer_of(block), normed, cached_states,
                                                      readout, backend=backend)
        hidden = residual + out
        if block.residual_in_fp32:
            hidden = hidden.to(dtype=block.norm.weight.dtype)
        return hidden, final_state, aux
