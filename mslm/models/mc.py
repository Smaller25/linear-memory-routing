"""Memory Caching with Sparse Selective Caching (SSC) on a GDN-2 mixer.

Follows long-gdn/dsc/mc_gdn2 + mc_baseline/mc_ssc.py (paper arXiv:2602.24281
Eq. 16-17), with the five parts looked up in ``registry``:

    online, M_1..M_N = connect(GDN-2 over CHUNK-token segments)
    c_i, c_t^on      = descriptor(normalised keys)
    r_t^i, r_t^on    = router(x_t, c_i, c_t^on)
    R_t              = TopK_{i < seg(t)} r_t^i            (or oracle segments)
    gamma            = gate(r_t^on, r_t^{R_t})
    y_t              = gamma^on * online_t + sum_{i in R_t} gamma^i M_i(q_t)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mslm.models import registry
from mslm.models.config import MCConfig
from mslm.models.gdn2 import GDN2Mixer, gdn_kernel


@dataclass
class RoutingRecord:
    scores: torch.Tensor        # [B,T,N] router scores, -inf where not eligible (segment >= own)
    indices: torch.Tensor       # [B,T,R] chosen segments, -1 where invalid
    weights: torch.Tensor       # [B,T,1+R] gate weights, column 0 = online memory
    seg_of_token: torch.Tensor  # [T] segment id of each position


def _pad_to_multiple(tensors, chunk):
    T = tensors[0].shape[1]
    pad = (-T) % chunk
    if pad == 0:
        return tensors, T
    # Padding goes at the END: positions < T never read the padded segment
    # (they only read segments strictly before their own), so it is exact.
    return [F.pad(t, (0, 0, 0, 0, 0, pad)) for t in tensors], T


def read_memories(q, memories, indices, weights, *, kernel: str):
    """sum_r weights[...,r] * M_{indices[...,r]}(normalize(q) / sqrt(K))."""
    scale = q.shape[-1] ** -0.5
    use_triton = kernel == "triton" or (kernel == "auto" and q.is_cuda)
    from mslm.kernels import gather_read
    if use_triton:
        return gather_read.ssc_gather_read(q, memories, indices, weights, scale=scale, normalize_queries=True)
    return gather_read._ssc_gather_read_torch(q, memories, indices, weights, scale, True)


class MCMixer(nn.Module):
    def __init__(self, d_model: int, n_head: int, head_dim: int, mc: MCConfig, **gdn_kw):
        super().__init__()
        self.base = GDN2Mixer(d_model, n_head, head_dim, **gdn_kw)
        self.cfg = mc
        self.descriptor = registry.get("descriptor", mc.descriptor)
        self.router = registry.get("router", mc.router)(d_model, n_head, head_dim, mc)
        gate = registry.get("gate", mc.gate)
        self.gate_is_module = isinstance(gate, type)
        self.gate = gate(d_model, n_head, head_dim, mc) if self.gate_is_module else gate
        if self.gate_is_module and mc.aux_loss == "none":
            raise ValueError(f"MC_GATE={mc.gate} decouples the router from the LM loss; "
                             f"set MC_AUX_LOSS so the router is trained at all")
        self.connect = registry.get("connect", mc.connect)
        # Set by LM.forward for one call:
        self.oracle_segments: torch.Tensor | None = None   # [B,G] gold segment ids, -1 padded
        self.capture = False
        self.record: RoutingRecord | None = None

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        C, topk = self.cfg.chunk, self.cfg.topk
        q, k, v, g, b, w = self.base.project(x)
        (qp, kp, vp, gp, bp, wp), T = _pad_to_multiple([q, k, v, g, b, w], C)
        online, mems = self.connect(qp, kp, vp, gp, bp, wp, C, gdn_kernel(self.base.backend(x)))
        kn = F.normalize(kp.float(), p=2, dim=-1)
        seg_desc, online_desc = self.descriptor(kn, C)
        online, online_desc = online[:, :T], online_desc[:, :T]
        N = mems.shape[1]

        scores, online_score = self.router(x, seg_desc, online_desc)      # [B,T,N], [B,T]
        seg_of_token = torch.arange(T, device=x.device) // C
        eligible = torch.arange(N, device=x.device)[None, :] < seg_of_token[:, None]
        scores = scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)

        R = min(topk, N)
        if R > 0:
            select = scores
            if self.oracle_segments is not None:
                select = scores + self._oracle_bias(scores, eligible)
            _, idx = torch.topk(select, k=R, dim=-1)
            sel = scores.gather(-1, idx)
            valid = torch.isfinite(sel)
        else:
            idx = torch.zeros(*scores.shape[:2], 0, dtype=torch.long, device=x.device)
            sel = scores.new_zeros(*scores.shape[:2], 0)
            valid = torch.zeros_like(sel, dtype=torch.bool)
        if self.gate_is_module:
            weights = self.gate(x, seg_desc, online_desc, idx, valid)      # [B,T,1+R]
        else:
            weights = self.gate(online_score, sel, valid)                  # [B,T,1+R]
        out = weights[..., :1].to(online.dtype).unsqueeze(-1) * online
        if R > 0:
            cached = read_memories(q, mems, idx.masked_fill(~valid, 0),
                                   weights[..., 1:].to(online.dtype), kernel=self.cfg.read_kernel)
            out = out + cached.to(online.dtype)
        if self.capture or self.cfg.aux_loss != "none":
            self.record = RoutingRecord(scores, idx.masked_fill(~valid, -1), weights, seg_of_token)
        return self.base.output(out, x)

    def _oracle_bias(self, scores, eligible):
        """+1e4 on gold segments that are eligible, so top-k takes them first;
        remaining slots (if k > #gold) fall back to the learned ranking.
        Gate weights still use the learned scores. Evaluation only."""
        B, T, N = scores.shape
        gold = self.oracle_segments.to(scores.device)
        onehot = torch.zeros(B, N + 1, device=scores.device)
        onehot.scatter_(1, gold.clamp(min=-1) % (N + 1), 1.0)   # -1 -> slot N (dropped)
        bias = onehot[:, :N].unsqueeze(1) * eligible.unsqueeze(0)
        return bias * 1e4
