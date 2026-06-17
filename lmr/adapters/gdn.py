# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gated-DeltaNet adapter for the Memory-Caching segment runner.

The per-segment mixer math mirrors :meth:`fla.layers.gated_deltanet.GatedDeltaNet.forward`
(``gated_deltanet.py:230-312``) without editing ``fla/``: it reuses the mixer's own projections,
short convolutions and the ``chunk_gated_delta_rule`` op (which accepts both ``initial_state`` and
``output_final_state``). Conv state is *not* carried across segments -- segments are independent
except for the recurrent state, which is the quantity Memory-Caching freezes and re-combines (same
approximation as the Mamba2 adapter).

GDN blocks (``GatedDeltaNetBlock``) add an MLP sublayer after the recurrent mixer, so ``run_block``
runs ``attn_norm -> mixer-with-cache -> +res -> mlp_norm -> mlp -> +res``. The MLP is
recurrence-free, so it runs per-segment unchanged. The explicit (non-fused) residual path here is
mathematically identical to the block's fused-norm path.

Memory-only scan: zeroing the value ``v`` makes the delta-rule write term ``beta * v k^T`` vanish,
leaving ``o_t = q_t (M_t...M_1) S0`` -- exactly the cached state's contribution. This holds because
the gated-delta state is affine in its initial state for fixed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from lmr.adapters.base import Adapter, run_mixer_with_cache

# ``fla`` GDN ops are Triton kernels; import them lazily inside the methods so that merely importing
# ``lmr`` (e.g. for the CPU-only read-out tests) does not require Triton.


@dataclass
class _GDNProj:
    q: torch.Tensor          # [b, l, HV, head_k_dim]
    k: torch.Tensor          # [b, l, HV, head_k_dim]
    v: torch.Tensor          # [b, l, HV, head_v_dim]
    beta: torch.Tensor       # [b, l, HV]
    g: torch.Tensor          # [b, l, HV]  (a_proj kernel gate input)
    out_gate: torch.Tensor | None  # [b, l, value_dim] or None


class GDNAdapter(Adapter):
    name = "gdn"

    def project(self, mixer, x_in):
        if mixer.use_short_conv:
            q, _ = mixer.q_conv1d(x=mixer.q_proj(x_in), cache=None, output_final_state=False)
            k, _ = mixer.k_conv1d(x=mixer.k_proj(x_in), cache=None, output_final_state=False)
            v, _ = mixer.v_conv1d(x=mixer.v_proj(x_in), cache=None, output_final_state=False)
        else:
            q = F.silu(mixer.q_proj(x_in))
            k = F.silu(mixer.k_proj(x_in))
            v = F.silu(mixer.v_proj(x_in))
        q = rearrange(q, '... (h d) -> ... h d', d=mixer.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=mixer.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=mixer.head_v_dim)
        beta = mixer.b_proj(x_in).sigmoid()
        if mixer.allow_neg_eigval:
            beta = beta * 2.
        g = mixer.a_proj(x_in)
        out_gate = mixer.g_proj(x_in) if mixer.use_gate else None
        return _GDNProj(q, k, v, beta, g, out_gate)

    def scan(self, mixer, proj, initial_state, backend, *, memory_only):
        # Forward / eval (no-grad) uses the chunk kernel — fast and correct everywhere.
        # NOTE: GDN router TRAINING (backward) is currently blocked across all FLA backends on our
        # hardware (see report/0006-0007):
        #   - chunk backward: OOMs A100 shared memory (head_dim=256 -> 225KB > 167KB); on
        #     Hopper+Triton>=3.4 it is miscomputed (#640) and needs tilelang, which fails to import
        #     on the py3.13 container (tvm-ffi).
        #   - fused_recurrent backward: NotImplementedError in FLA (can't compute dg).
        #   - naive (pure-torch) recurrence: differentiable but a per-token Python loop -> too slow.
        # So GDN is a forward/RM-only arm for now; trained-router GDN awaits a differentiable
        # chunked scan or a fixed tilelang/H100 env.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        v = torch.zeros_like(proj.v) if memory_only else proj.v
        o, final_state = chunk_gated_delta_rule(
            q=proj.q, k=proj.k, v=v, g=proj.g, beta=proj.beta,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            A_log=mixer.A_log, dt_bias=mixer.dt_bias,
        )
        return o, final_state

    def descriptor(self, state):
        # state: [b, HV, K, V] -> pool over V (the non-read-out axis) -> [b, HV*K]
        from lmr.state_utils import gdn_meanpool_state
        return gdn_meanpool_state(state)

    def descriptor_dim(self, mixer):
        return mixer.num_v_heads * mixer.head_k_dim

    def finalize(self, mixer, proj, y, dtype):
        if mixer.use_gate:
            g = rearrange(proj.out_gate, '... (h d) -> ... h d', d=mixer.head_v_dim)
            o = mixer.o_norm(y, g)
        else:
            o = mixer.o_norm(y)
        o = rearrange(o, 'b t h d -> b t (h d)')
        return mixer.o_proj(o).to(dtype)

    def blocks(self, model):
        return model.model.layers

    def mixer_of(self, block):
        return block.attn

    def embed(self, model, input_ids):
        return model.model.embeddings(input_ids)

    def final_norm(self, model, hidden):
        return model.model.norm(hidden)

    def vanilla_hidden(self, model, input_ids):
        return model.model(input_ids).last_hidden_state

    def lm_head(self, model):
        return model.lm_head

    def run_block(self, block, hidden, cached_states, readout, backend):
        from fla.layers.gated_deltanet import GatedDeltaNet
        if not isinstance(block.attn, GatedDeltaNet):
            raise TypeError(
                "GDN segment runner requires every block's mixer to be GatedDeltaNet; "
                f"got {type(block.attn).__name__} (interleaved full attention is not supported)."
            )
        if getattr(block, "use_attnres", False):
            raise NotImplementedError("attnres blocks are not supported by the GDN segment runner.")

        residual = hidden
        normed = block.attn_norm(hidden)
        out, final_state, aux = run_mixer_with_cache(self, self.mixer_of(block), normed, cached_states,
                                                     readout, backend=backend)
        hidden = residual + out

        # MLP sublayer: recurrence-free, runs per-segment unchanged. Equivalent to the block's
        # fused-norm path (mlp_norm folds the post-attn residual add internally).
        residual = hidden
        normed = block.mlp_norm(hidden)
        hidden = residual + block.mlp(normed)
        return hidden, final_state, aux
