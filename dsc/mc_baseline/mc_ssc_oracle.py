"""Oracle-routing SSC — P3 upper-bound arm (EVAL ONLY, never trained).

Answers ONE question: if the gold segment were always selected, does the 8K
READ path deliver the answer? This separates selection failure (a better
gate/calibration can fix it) from read/compression failure (it cannot).

Implementation: an eval-only COPY of ``SparseSelectiveCaching.forward``
(the no-change covenant forbids editing mc_ssc.py; extensions go through
inheritance or copies) with a 6-line insertion: per-row gold segments get
their routing score raised to the row/token-wise max eligible score + 1.0
wherever they are eligible, guaranteeing top-1 selection with an
in-distribution weight scale (softmax margin ~1 logit, not a delta spike).

Deviations from the source copy, both eval-safe:
  * kernel dispatch hardcoded to v2 (every eval in this campaign pins
    MC_KERNEL_VERSION=v2);
  * ``oracle_gold`` attribute ([B] long tensor, -1 = no boost for that row)
    is consumed at each forward; the eval harness sets it per batch.
"""

from __future__ import annotations

import torch

from dsc.mc_baseline.cached_memory_read import ssc_gather_read
from dsc.mc_baseline.mc_ssc import (
    SSCOutput,
    causal_online_key_sums,
    segment_key_sums,
)
from dsc.mc_gdn2.ssc import GDN2SSC


class OracleGDN2SSC(GDN2SSC):
    oracle_gold: torch.Tensor | None = None

    def forward(self, hidden_states, queries, keys, online_output, memories):
        if hidden_states.ndim != 3 or queries.ndim != 4 or keys.shape != queries.shape:
            raise ValueError("expected hidden [B,T,D] and matching query/key [B,T,H,K]")
        batch, length, heads, key_dim = queries.shape
        if (heads, key_dim) != (self.num_heads, self.head_qk_dim):
            raise ValueError("query heads/dim mismatch")
        if online_output.shape[:3] != (batch, length, heads):
            raise ValueError("online_output must be [B,T,H,V]")
        if memories.ndim != 5 or memories.shape[:2] != (
                batch, (length + self.chunk_size - 1) // self.chunk_size):
            raise ValueError("memories must contain one [H,K,V] state per segment")

        num_segments = memories.shape[1]
        u = self.connector(hidden_states).view(batch, length, heads, key_dim)
        summaries = segment_key_sums(keys, self.chunk_size)
        all_scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())

        segment_ids = torch.arange(length, device=queries.device) // self.chunk_size
        eligible = torch.arange(num_segments, device=queries.device)[None, :] < segment_ids[:, None]
        past_scores = all_scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)

        # --- ORACLE INSERTION (the only change vs the source forward) ---
        gold = self.oracle_gold
        if gold is not None:
            row_max = past_scores.masked_fill(
                ~torch.isfinite(past_scores), -1e30).amax(dim=-1)  # [B, T]
            for b in range(batch):
                g = int(gold[b])
                if 0 <= g < num_segments:
                    col = past_scores[b, :, g]
                    past_scores[b, :, g] = torch.where(
                        torch.isfinite(col), row_max[b] + 1.0, col)
        # -----------------------------------------------------------------

        route_count = min(self.topk, num_segments)
        if route_count:
            top_scores, top_indices = torch.topk(past_scores, k=route_count, dim=-1)
            valid = torch.isfinite(top_scores)
            safe_indices = top_indices.masked_fill(~valid, 0)
        else:
            top_scores = all_scores.new_empty(batch, length, 0)
            top_indices = torch.empty(batch, length, 0, device=queries.device, dtype=torch.long)
            valid = torch.empty(batch, length, 0, device=queries.device, dtype=torch.bool)
            safe_indices = top_indices

        online_summary = causal_online_key_sums(keys, self.chunk_size)
        online_score = torch.einsum("bthk,bthk->bt", u.float(), online_summary.float())
        gate_logits = torch.cat([online_score.unsqueeze(-1), top_scores], dim=-1)
        gate_valid = torch.cat(
            [torch.ones(batch, length, 1, device=queries.device, dtype=torch.bool), valid],
            dim=-1,
        )
        gate_logits = gate_logits.masked_fill(~gate_valid, -torch.inf)
        gates = torch.softmax(gate_logits, dim=-1).to(online_output.dtype)

        online_weight = gates[..., :1]
        route_weights = gates[..., 1:]
        if route_count:
            cached_output = ssc_gather_read(
                queries, memories, safe_indices, route_weights,
                scale=self.read_scale, normalize_queries=self.normalize_queries,
            ).to(online_output.dtype)
        else:
            cached_output = torch.zeros_like(online_output)
        output = online_weight.unsqueeze(-1) * online_output + cached_output
        return SSCOutput(
            output=output,
            online_output=online_output,
            cached_output=cached_output,
            route_indices=top_indices.masked_fill(~valid, -1),
            route_weights=route_weights,
            online_weight=online_weight,
            route_scores=top_scores.masked_fill(~valid, -torch.inf),
        )


def enable_oracle_routing(model) -> list:
    """Swap every SSC aggregator instance onto an oracle subclass in place.

    Match by CLASS NAME and derive the subclass dynamically from each
    instance's own class: the repo is importable both as ``mc_gdn2.ssc``
    (PYTHONPATH=dsc) and ``dsc.mc_gdn2.ssc``, which yields two distinct
    class objects for the same source — an identity check silently matched
    nothing (observed: '[hook] ORACLE routing on 0 SSC aggregators' while
    the arm ran as a plain top-2 baseline). Returns the affected
    aggregators (set ``agg.oracle_gold`` per batch; None disables).
    """
    aggs = []
    oracle_cls_cache: dict[type, type] = {}
    for module in model.modules():
        if module.__class__.__name__ == "GDN2SSC":
            base = module.__class__
            if base not in oracle_cls_cache:
                oracle_cls_cache[base] = type(
                    "OracleGDN2SSC_dyn", (base,),
                    {"forward": OracleGDN2SSC.forward, "oracle_gold": None})
            module.__class__ = oracle_cls_cache[base]
            module.oracle_gold = None
            aggs.append(module)
    return aggs
