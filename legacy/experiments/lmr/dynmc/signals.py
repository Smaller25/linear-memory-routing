# -*- coding: utf-8 -*-
"""Boundary-trigger signals, evaluated at 64-token chunk boundaries (plan §2.4).

Per token t (GDN-1 conventions, per v-head scalars):
    alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))   # decay gate in (0,1)
    beta_t  = sigmoid(b_t) [* 2 if allow_neg_eigval]       # write gate
    k_t     = l2norm(post-conv k), v_t = post-conv v        # as seen by the kernel

Signals accumulated over the chunk ending at boundary c (all per-layer,
per-head, then head-mean by the caller):
    drift      ‖S_c − S_{c−1}‖_F / (‖S_{c−1}‖_F + ε)            (+보조 Σ β_t‖e_t‖)
    erasure    Σ_t [ β_t‖α_t k_tᵀ S_{t−1}‖₂ + (1−α_t)‖S_{t−1}‖_F ]
    saturation sr(S_c)/d_k,  sr = ‖S‖_F² / ‖S‖₂²  (‖S‖₂: power iteration 5회)
    surprise   Σ_t ‖e_t‖₂ / ‖v_t‖₂,  e_t = v_t − α_t k_tᵀ S_{t−1}

Chunk-granularity approximation: within a chunk, S_{t−1} is approximated by the
state at the previous boundary S_{c−1} (the kernel materializes states only
every 64 tokens; plan §1 requires evaluation at those boundaries only). Extra
cost is one k @ S per chunk.

States are `[H, V, K]` (state_v_first=True, GDN-1 fla convention); k_tᵀ S_{t−1}
therefore contracts K: einsum('thk,hvk->thv').
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


@torch.no_grad()
def spectral_norm_power_iter(S: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """S: [H, V, K] -> ‖S_h‖₂ per head, via power iteration (plan: 5회)."""
    H, V, K = S.shape
    v = torch.randn(H, K, 1, device=S.device, dtype=torch.float32)
    v = v / (v.norm(dim=1, keepdim=True) + EPS)
    A = S.float()
    for _ in range(iters):
        u = A @ v                                   # [H, V, 1]
        u = u / (u.norm(dim=1, keepdim=True) + EPS)
        v = A.transpose(1, 2) @ u                   # [H, K, 1]
        v = v / (v.norm(dim=1, keepdim=True) + EPS)
    return (A @ v).norm(dim=(1, 2))                 # [H]


@torch.no_grad()
def chunk_signals(prev_state: torch.Tensor | None,
                  curr_state: torch.Tensor,
                  alpha: torch.Tensor,     # [T_c, H]
                  beta: torch.Tensor,      # [T_c, H]
                  k: torch.Tensor,         # [T_c, H, K] (l2-normalized)
                  v: torch.Tensor,         # [T_c, H, V]
                  ) -> dict[str, torch.Tensor]:
    """Signals for one chunk. Returns per-head tensors [H] (fp32)."""
    H = curr_state.shape[0]
    dev = curr_state.device
    curr = curr_state.float()
    prev = prev_state.float() if prev_state is not None else torch.zeros_like(curr)

    prev_f = prev.norm(dim=(1, 2))                                   # [H]
    # (a) drift — 대표 변형: chunk state 상대 변화량.
    # 분모는 max(prev, curr) — 첫 chunk(prev=0)에서 ε-나눗셈 폭발 방지, 값 ~[0,1]
    curr_f = curr.norm(dim=(1, 2))
    drift = (curr - prev).norm(dim=(1, 2)) / torch.clamp(torch.maximum(prev_f, curr_f), min=EPS)

    # k_tᵀ S_{c−1}: [T,H,K] × [H,V,K] → [T,H,V]
    kS = torch.einsum("thk,hvk->thv", k.float(), prev)
    kS_norm = kS.norm(dim=-1)                                        # [T, H]
    a, b = alpha.float(), beta.float()

    # (b) erasure = overwrite + decay
    erasure = (b * a * kS_norm).sum(0) + ((1.0 - a) * prev_f.unsqueeze(0)).sum(0)

    # (d) surprise + (a-보조) weighted-error drift
    e = v.float() - a.unsqueeze(-1) * kS                             # [T, H, V]
    e_norm = e.norm(dim=-1)                                          # [T, H]
    surprise = (e_norm / (v.float().norm(dim=-1) + EPS)).sum(0)
    drift_err = (b * e_norm).sum(0)

    # (c) saturation — stable rank / d_k
    fro2 = (curr ** 2).sum(dim=(1, 2))
    spec = spectral_norm_power_iter(curr_state)
    d_k = curr_state.shape[2]
    saturation = fro2 / (spec ** 2 + EPS) / d_k

    return {"drift": drift, "drift_err": drift_err, "erasure": erasure,
            "surprise": surprise, "saturation": saturation}


class GDNSignalRecorder:
    """Chunked-prefill driver: run any fla GatedDeltaNet LM 64 tokens at a time,
    threading the cache, and record all four signals at every chunk boundary.

    No kernel or model modification: per-chunk gate/k/v values are recomputed
    from the layer's own projections on the chunk's hidden states (captured
    with forward pre-hooks), and boundary states come from the fla Cache.

    Used for: G0 (signal correlation/Jaccard on a real checkpoint), 실험 0-ii
    (0024 mid-training checkpoints), and inference trigger evaluation.
    """

    def __init__(self, model, chunk: int = 64):
        self.model = model
        self.chunk = chunk
        self.layers = [blk.attn for blk in model.model.layers]
        self._hidden: dict[int, torch.Tensor] = {}
        self._handles = []
        for li, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_pre_hook(
                self._make_hook(li), with_kwargs=True))

    def _make_hook(self, li):
        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            self._hidden[li] = hs.detach()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()

    @torch.inference_mode()
    def _layer_chunk_inputs(self, layer, hs, conv_cache):
        """Recompute (alpha, beta, k, v) for one chunk from layer projections."""
        from fla.modules.l2norm import l2norm
        k_raw = layer.k_proj(hs)
        v_raw = layer.v_proj(hs)
        # short conv (cache threading; hook 시점의 conv cache 사용)
        k_conv, ck = layer.k_conv1d(x=k_raw, cache=conv_cache[0], output_final_state=True)
        v_conv, cv = layer.v_conv1d(x=v_raw, cache=conv_cache[1], output_final_state=True)
        H, Dk = layer.num_v_heads, layer.head_k_dim
        k = l2norm(k_conv.view(k_conv.shape[1], layer.num_heads, Dk))
        if layer.num_v_heads > layer.num_heads:
            k = k.repeat_interleave(layer.num_v_heads // layer.num_heads, dim=-2)
        v = v_conv.view(v_conv.shape[1], H, layer.head_v_dim)
        a_t = layer.a_proj(hs).squeeze(0).float()               # [T, H]
        alpha = torch.exp(-layer.A_log.float().exp() * F.softplus(a_t + layer.dt_bias.float()))
        beta = layer.b_proj(hs).squeeze(0).float().sigmoid()
        if layer.allow_neg_eigval:
            beta = beta * 2.0
        return alpha, beta, k, v, (ck, cv)

    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor):
        """input_ids: [1, T]. Returns signals[name]: [num_chunks, L, H] fp32 (cpu)."""
        from fla.models.utils import Cache
        T = input_ids.shape[1]
        n_chunks = (T + self.chunk - 1) // self.chunk
        L = len(self.layers)
        past = Cache()
        prev_states = [None] * L
        conv_caches = [(None, None)] * L
        out: dict[str, list] = {}
        for c in range(n_chunks):
            ids = input_ids[:, c * self.chunk:(c + 1) * self.chunk]
            self._hidden.clear()
            self.model(ids, past_key_values=past, use_cache=True)
            per_layer = {}
            for li, layer in enumerate(self.layers):
                state = past[li]["recurrent_state"]              # [1, H, V, K]
                curr = state.squeeze(0)
                alpha, beta, k, v, conv_caches[li] = self._layer_chunk_inputs(
                    layer, self._hidden[li], conv_caches[li])
                sig = chunk_signals(prev_states[li], curr, alpha, beta, k, v)
                prev_states[li] = curr.clone()
                for name, val in sig.items():
                    per_layer.setdefault(name, []).append(val)
            for name, vals in per_layer.items():
                out.setdefault(name, []).append(torch.stack(vals))  # [L, H]
        return {name: torch.stack(vals).cpu() for name, vals in out.items()}
