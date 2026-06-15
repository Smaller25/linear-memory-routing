# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Segment-by-segment Memory-Caching runner for an FLA ``Mamba2ForCausalLM``.

The model is run in segments of ``chunk_size`` tokens. Within a segment every Mamba2 layer
runs from a *zero* SSM state (the segment is independent); at each segment boundary the layer's
final state is frozen and appended to that layer's checkpoint cache. The read-out
(:mod:`lmr.readout`) then augments each later segment's online output with the contributions of
all earlier frozen checkpoints.

The per-segment mixer math reproduces the prefill branch of
:meth:`fla.layers.mamba2.Mamba2Mixer.torch_forward` (projection + causal conv + SSD scan +
output gate) *without editing* ``fla/``; it reuses the mixer's own submodules. The scan itself
dispatches to :func:`lmr.ssd_scan.naive_ssd_scan` (CPU) or, when available and on CUDA,
``mamba_ssm.mamba_chunk_scan_combined`` (same math, faster).
"""

from __future__ import annotations

import torch

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


def run_mixer_with_cache(mixer, x_in, cached_states, readout, backend="naive"):
    """Run one Mamba2 mixer over a segment, augmenting with cached checkpoints.

    ``cached_states``: list of frozen ``[b, h, p, n]`` checkpoints from earlier segments.
    ``readout``: an :mod:`lmr.readout` head combining online + cached contributions.
    Returns ``(out:[b,l,hidden], final_state:[b,h,p,n], aux)``.
    """
    dtype = x_in.dtype
    x, dt, B_g, C_g, gate = _project_segment(mixer, x_in)
    A = -torch.exp(mixer.A_log.float())
    D = mixer._get_D(expand_to_head_dim=True)

    y_main, final_state = _run_scan(mixer, x, dt, B_g, C_g, A, D, None, backend)

    y_cached, descriptors = [], []
    if cached_states:
        zeros_x = torch.zeros_like(x)
        for h_i in cached_states:
            y_i, _ = _run_scan(mixer, zeros_x, dt, B_g, C_g, A, None, h_i, backend)
            y_cached.append(y_i)
            descriptors.append(meanpool_state(h_i))
    descriptors = torch.stack(descriptors, dim=1) if descriptors else None

    y, aux = readout(y_main, y_cached, x_in, descriptors)
    out = _finalize(mixer, y, gate, dtype)
    return out, final_state, aux


def segment_lengths(total: int, chunk_size: int) -> list[int]:
    n_full, rem = divmod(total, chunk_size)
    lens = [chunk_size] * n_full
    if rem:
        lens.append(rem)
    return lens


def run_segmented_lm(model, input_ids, readouts, chunk_size, backend="naive"):
    """Run an FLA ``Mamba2ForCausalLM`` segment-by-segment with Memory Caching.

    ``readouts``: one read-out head per layer (e.g. a ``ModuleList``); RM heads are shared/
    param-free, GRM/SSC heads carry per-layer params.
    Returns ``(logits:[b, L, vocab], aux_total)``. ``aux_total`` sums any SSC load-balance losses.
    """
    backbone = model.backbone
    layers = backbone.layers
    caches: list[list[torch.Tensor]] = [[] for _ in layers]

    seg_logits = []
    aux_total = input_ids.new_zeros((), dtype=torch.float32)
    offset = 0
    for seg_len in segment_lengths(input_ids.shape[1], chunk_size):
        seg_ids = input_ids[:, offset:offset + seg_len]
        offset += seg_len

        hidden = backbone.embeddings(seg_ids)
        new_finals = []
        for li, block in enumerate(layers):
            residual = hidden
            normed = block.norm(hidden)
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)
            out, final_state, aux = run_mixer_with_cache(
                block.mixer, normed, caches[li], readouts[li], backend=backend,
            )
            hidden = residual + out
            if block.residual_in_fp32:
                hidden = hidden.to(dtype=block.norm.weight.dtype)
            new_finals.append(final_state.detach())
            if aux is not None:
                aux_total = aux_total + aux

        hidden = backbone.norm_f(hidden)
        seg_logits.append(model.lm_head(hidden))

        for li, fs in enumerate(new_finals):
            caches[li].append(fs)

    return torch.cat(seg_logits, dim=1), aux_total
