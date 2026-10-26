"""GDN-2 token mixer.

Same parameters, names and initialisation as ``GatedDeltaNet2`` in
long-gdn/dsc/lit_gpt/gdn2.py (q/k/v proj + short conv, channel-wise decay
f_proj/A_log/dt_bias, erase gate b_proj, write gate w_proj, gated RMS-norm
output). Split into ``project`` / ``recur`` / ``output`` so the MC layer can
reuse the projections and replace only the recurrence/read.

Backends: ``fla`` = vendored Triton kernels + fla fused ops (CUDA);
``torch`` = pure PyTorch reference (CPU, tests).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def resolve_backend(name: str, device: torch.device) -> str:
    if name == "auto":
        return "fla" if device.type == "cuda" else "torch"
    if name not in ("fla", "torch"):
        raise ValueError(f"GDN_BACKEND={name!r}; choose auto | fla | torch")
    return name


def gdn_kernel(backend: str):
    if backend == "fla":
        from mslm.kernels.gdn2_ops.chunk_gdn2 import chunk_gdn2
        return chunk_gdn2
    from mslm.kernels.naive_gdn2 import naive_gdn2
    return naive_gdn2


class ShortConv(nn.Conv1d):
    """Depthwise causal conv + SiLU (weights laid out like fla.ShortConvolution)."""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=False)

    def forward(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        if backend == "fla":
            from fla.modules.conv import causal_conv1d
            y, _ = causal_conv1d(x=x, weight=self.weight.squeeze(1), bias=None,
                                 activation="silu", backend="triton")
            return y
        T = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), self.weight.to(x.dtype), None,
                     padding=self.kernel_size[0] - 1, groups=self.groups)[..., :T]
        return F.silu(y).transpose(1, 2)


class GatedRMSNorm(nn.Module):
    """y = RMSNorm(x) * weight * silu(gate)  (fla FusedRMSNormSwishGate)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate, backend: str):
        if backend == "fla":
            from fla.modules.fused_norm_gate import rms_norm_gated
            return rms_norm_gated(x, gate, self.weight, None, "swish", eps=self.eps)
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * F.silu(gate.float())).to(x.dtype)


class GDN2Mixer(nn.Module):
    def __init__(self, d_model: int, n_head: int, head_dim: int, *, conv_size: int = 4,
                 norm_eps: float = 1e-5, backend: str = "auto"):
        super().__init__()
        self.d_model, self.n_head, self.head_dim = d_model, n_head, head_dim
        self.backend_name = backend
        D = n_head * head_dim
        self.q_proj = nn.Linear(d_model, D, bias=False)
        self.k_proj = nn.Linear(d_model, D, bias=False)
        self.v_proj = nn.Linear(d_model, D, bias=False)
        self.q_conv1d = ShortConv(D, conv_size)
        self.k_conv1d = ShortConv(D, conv_size)
        self.v_conv1d = ShortConv(D, conv_size)
        self.f_proj = nn.Sequential(nn.Linear(d_model, head_dim, bias=False),
                                    nn.Linear(head_dim, D, bias=False))
        self.b_proj = nn.Linear(d_model, D, bias=False)
        self.w_proj = nn.Linear(d_model, D, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(n_head).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(d_model, head_dim, bias=False),
                                    nn.Linear(head_dim, D, bias=True))
        self.o_norm = GatedRMSNorm(head_dim, norm_eps)
        self.o_proj = nn.Linear(D, d_model, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2 ** -2.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                m._mslm_initialized = True

    def backend(self, x: torch.Tensor) -> str:
        return resolve_backend(self.backend_name, x.device)

    def project(self, x: torch.Tensor):
        """Return q,k,g,b [B,T,H,K] and v,w [B,T,H,V] exactly as GDN-2 feeds its kernel."""
        be = self.backend(x)
        B, T, _ = x.shape
        H, K = self.n_head, self.head_dim
        q = self.q_conv1d(self.q_proj(x), be)
        k = self.k_conv1d(self.k_proj(x), be)
        v = self.v_conv1d(self.v_proj(x), be)
        g = -self.A_log.float().exp().repeat_interleave(K) * F.softplus(self.f_proj(x).float() + self.dt_bias)
        b = self.b_proj(x).sigmoid()
        w = self.w_proj(x).sigmoid()
        return tuple(t.reshape(B, T, H, K) for t in (q, k, v, g, b, w))

    def output(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        gate = self.g_proj(x).view(B, T, self.n_head, self.head_dim)
        o = self.o_norm(o, gate, self.backend(x))
        return self.o_proj(o.reshape(B, T, -1).to(x.dtype))

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        q, k, v, g, b, w = self.project(x)
        o, _ = gdn_kernel(self.backend(x))(q=q, k=k, v=v, g=g, b=b, w=w, initial_state=None,
                                           output_final_state=False, use_qk_l2norm_in_kernel=True,
                                           use_gate_in_kernel=False, cu_seqlens=None)
        return self.output(o, x)
