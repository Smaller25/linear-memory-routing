"""ReLU Dynamic Selection for Memory Caching (fluid multi-state routing).

Hard Top-k SSC (``mc_ssc.SparseSelectiveCaching``) reads a FIXED number of
cached segment states per token (R = topk), which caps needle coverage when
the number of distinct keys in context exceeds k.  This module replaces the
``topk + joint softmax`` gate with a ReLU gate over ALL completed segments,
so the number of ACTIVE cached states per token is input-dependent (fluid),
in the spirit of ReLU-routed MoE (ReMoE, arXiv:2412.14711).

Everything upstream of the gate is IDENTICAL to the SSC/GRM score path:
    * per-head connector ``u_t = x_t W_u`` (Eq. 10/16),
    * segment descriptors = mean of L2-normalized segment keys
      (``segment_key_sums``, the meanpool convention shared by all arms),
    * causal online descriptor (``causal_online_key_sums``),
    * strict causal eligibility (only completed past segments),
    * the SSC-v2 fused read kernel via ``dense_cached_memory_read``
      (fixed-R Triton kernel applied in bounded route blocks; zero-weight
      routes contribute exactly zero by linearity — no kernel change).

Gate normalization ("relu_norm", the documented choice for the
``mc_relu_370M`` preset):

    a_on  = softplus(online_score)            > 0, online always active
    a_i   = relu(past_score_i)                >= 0, i over completed segments
    denom = a_on + sum_i a_i + eps
    online_weight = a_on / denom
    route_weight_i = a_i / denom

Properties:
    * Total gate mass stays <= 1 (same scale as the SSC softmax gate), so the
      output magnitude regime matches the Hard Top-k arms.
    * When every cached score is <= 0 the read degenerates EXACTLY to the
      online (single-state, vanilla GDN-2) path.
    * The active set {i : a_i > 0} grows with the number of segments the
      input scores positively — the fluidity that Hard Top-k cannot express.

ReMoE-style L1 sparsity regularization is intentionally NOT applied inside
the layer; the mean L1 gate mass is exposed as ``gate_l1`` so a trainer MAY
add it as an auxiliary loss.  The paper-matched pretraining recipe for this
experiment leaves it off.

Tensor convention (same as mc_ssc/mc_grm):
    hidden_states: [B, T, D]
    queries/keys:  [B, T, H, K]
    online_output: [B, T, H, V]
    memories:      [B, N, H, K, V]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .mc_grm import GatedResidualMemory, dense_cached_memory_read
from .mc_ssc import SSCOutput, causal_online_key_sums, segment_key_sums


@dataclass
class ReLUSSCOutput(SSCOutput):
    """SSCOutput plus fluid-routing diagnostics.

    active_counts: [B, T] int64 — number of cached states with weight > 0.
    gate_l1:       scalar — mean over tokens of sum_i relu(score_i)
                   (pre-normalization mass, the ReMoE L1 handle).
    """

    active_counts: torch.Tensor = None
    gate_l1: torch.Tensor = None


class ReLUSelectiveCaching(GatedResidualMemory):
    """ReLU-gated dense Memory Caching with input-dependent active state count.

    Inherits the GRM constructor (connector ``W_u``, chunk/read constants) and
    overrides only the gate.  The score path is bit-identical to GRM/SSC up to
    the gating nonlinearity, so a comparison against Hard Top-k isolates the
    gate as the only changed factor.
    """

    GATE_EPS = 1e-6

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_qk_dim: int,
        *,
        chunk_size: int = 256,
        normalize_queries: bool = False,
        read_scale: float | None = None,
        route_block_size: int = 16,
        normalize_gate: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            head_qk_dim,
            chunk_size=chunk_size,
            normalize_queries=normalize_queries,
            read_scale=read_scale,
            route_block_size=route_block_size,
        )
        # normalize_gate=True : joint L1 norm over online+cached (sum-to-1).
        # normalize_gate=False: raw ReLU weights (ReMoE original design) —
        #   each cached segment contributes independently, online weight is 1.
        #   Removes the zero-sum dilution that starves long-context reads
        #   (observed: 8K gold-hit DECREASING during normalized-gate FT).
        #   No parameters change either way: checkpoints are interchangeable.
        self.normalize_gate = normalize_gate
        # Evaluation-time diagnostics hook (auxiliary DV of the diverse-key
        # NIAH experiment).  Off by default: training pays nothing.
        self.log_active_states = False
        self.last_active_counts: torch.Tensor | None = None
        self.last_final_route_weights: torch.Tensor | None = None
        self.last_final_online_weight: torch.Tensor | None = None
        self.last_route_weights: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        online_output: torch.Tensor,
        memories: torch.Tensor,
    ) -> ReLUSSCOutput:
        """Apply the GRM score path with a fluid ReLU gate (strictly causal)."""
        if hidden_states.ndim != 3 or queries.ndim != 4:
            raise ValueError("expected hidden [B,T,D] and queries [B,T,H,K]")
        if keys.shape != queries.shape:
            raise ValueError("keys must match queries shape [B,T,H,K]")

        batch, length, hidden_size = hidden_states.shape
        query_batch, query_length, heads, key_dim = queries.shape
        if (query_batch, query_length) != (batch, length):
            raise ValueError("hidden_states and queries must share [B,T]")
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden size {hidden_size} != configured {self.hidden_size}"
            )
        if (heads, key_dim) != (self.num_heads, self.head_qk_dim):
            raise ValueError(
                f"query heads/dim {(heads, key_dim)} != configured "
                f"{(self.num_heads, self.head_qk_dim)}"
            )
        if online_output.shape[:3] != (batch, length, heads):
            raise ValueError("online_output must be [B,T,H,V]")

        expected_segments = (length + self.chunk_size - 1) // self.chunk_size
        if (
            memories.ndim != 5
            or memories.shape[:2] != (batch, expected_segments)
            or memories.shape[2:4] != (heads, key_dim)
            or memories.shape[-1] != online_output.shape[-1]
        ):
            raise ValueError(
                "memories must contain one [H,K,V] state per sequence segment"
            )

        num_segments = memories.shape[1]
        # Identical score path to GRM/SSC (Eq. 16): u_t vs mean-pooled keys.
        u = self.connector(hidden_states).view(batch, length, heads, key_dim)
        summaries = segment_key_sums(keys, self.chunk_size)
        all_scores = torch.einsum(
            "bthk,bnhk->btn", u.float(), summaries.float()
        )

        segment_ids = (
            torch.arange(length, device=queries.device) // self.chunk_size
        )
        memory_ids = torch.arange(num_segments, device=queries.device)
        eligible = memory_ids[None, :] < segment_ids[:, None]
        past_scores = all_scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)

        online_summary = causal_online_key_sums(keys, self.chunk_size)
        online_score = torch.einsum(
            "bthk,bthk->bt", u.float(), online_summary.float()
        )

        # --- the ONLY departure from GRM/SSC: the ReLU gate ---
        # relu(-inf) == 0, so ineligible segments drop out exactly.
        cached_activation = F.relu(past_scores)
        if self.normalize_gate:
            online_activation = F.softplus(online_score)
            # clamp_min (not +eps) keeps the vanilla fallback EXACT: when every
            # cached score is <= 0, online_weight == a_on / a_on == 1. The clamp
            # only guards the pathological case where softplus underflows to 0.
            denom = (
                online_activation + cached_activation.sum(dim=-1)
            ).clamp_min(self.GATE_EPS)
            online_weight = (online_activation / denom).unsqueeze(-1).to(
                online_output.dtype
            )
            route_weights = (cached_activation / denom.unsqueeze(-1)).to(
                online_output.dtype
            )
        else:
            # Raw ReLU (ReMoE): unnormalized, independent contributions.
            # online_weight == 1 keeps the vanilla fallback exact (all cached
            # scores <= 0 -> output == online_output, no division involved).
            online_weight = torch.ones(
                batch, length, 1,
                device=online_output.device, dtype=online_output.dtype,
            )
            route_weights = cached_activation.to(online_output.dtype)
        active_counts = (cached_activation > 0).sum(dim=-1)
        gate_l1 = cached_activation.sum(dim=-1).mean()

        if self.log_active_states:
            self.last_active_counts = active_counts.detach()
            self.last_final_route_weights = route_weights[:, -1].detach().float()
            self.last_final_online_weight = (
                online_weight[:, -1, 0].detach().float()
            )
            # Full [B, T, N] weights so batched (right-padded) eval can read
            # each row at ITS OWN final position; the [:, -1] slices above
            # land on pad positions for all but the longest row.
            self.last_route_weights = route_weights.detach()

        route_indices = memory_ids.view(1, 1, num_segments).expand(
            batch, length, num_segments
        )
        route_valid = eligible.unsqueeze(0).expand(batch, length, num_segments)
        safe_indices = route_indices.masked_fill(~route_valid, 0)

        cached_output = dense_cached_memory_read(
            queries,
            memories,
            safe_indices,
            route_weights,
            scale=self.read_scale,
            normalize_queries=self.normalize_queries,
            route_block_size=self.route_block_size,
        ).to(online_output.dtype)
        output = online_weight.unsqueeze(-1) * online_output + cached_output

        return ReLUSSCOutput(
            output=output,
            online_output=online_output,
            cached_output=cached_output,
            route_indices=route_indices.masked_fill(~route_valid, -1),
            route_weights=route_weights,
            online_weight=online_weight,
            route_scores=past_scores,
            active_counts=active_counts,
            gate_l1=gate_l1,
        )


# Concise public name.
ReLUDynamicSelection = ReLUSelectiveCaching
