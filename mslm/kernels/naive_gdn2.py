"""Pure-PyTorch GDN-2 recurrence (CPU smoke tests and kernel correctness checks).

Same math as ``fla/ops/gdn2/naive.py`` (MIT, fla-org) and as the vendored
``chunk_gdn2`` kernel called with ``use_qk_l2norm_in_kernel=True``:

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
    o_t = S_t^T (q_t * scale)

Shapes: q,k,g,b [B,T,H,K]; v,w [B,T,H,V]; state [B,H,K,V] (float32).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def naive_gdn2(q, k, v, g, b, w, *, initial_state=None, output_final_state=True,
               use_qk_l2norm_in_kernel=True, scale=None, **_unused):
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), p=2, dim=-1)
        k = F.normalize(k.float(), p=2, dim=-1)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    dtype = v.dtype
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    B, T, H, K = k.shape
    V = v.shape[-1]
    h = (torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
         if initial_state is None else initial_state.float())
    outs = []
    for t in range(T):
        h = h * g[:, t].exp().unsqueeze(-1)
        erase = ((b[:, t] * k[:, t]).unsqueeze(-1) * h).sum(-2)
        v_new = w[:, t] * v[:, t] - erase
        h = h + k[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
        outs.append(((q[:, t] * scale).unsqueeze(-1) * h).sum(-2))
    o = torch.stack(outs, dim=1).to(dtype)
    return o, (h if output_final_state else None)
