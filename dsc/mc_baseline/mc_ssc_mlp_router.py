"""MLP-router SSC — a-1 stage 2 arm (EVAL ONLY, router trained offline).

Answers ONE question: the offline-trained MLP router lifted routing gold-AUC
well above the native linear router's chance level — does that ranking
improvement translate into diverse-key SCORE at 8K?

Three modes.

``observe``. The MLP score is computed and logged, but selection and gating
stay on the native path — the forward is the original math plus one stash.
This is what the wiring gate needs: the router was fitted on hidden states
produced by a NATIVELY routed model, so injecting every layer at once shifts
the input of every layer past the first, and a deep layer's in-model AUC then
drifts from its training value even when the wiring is perfect. ``observe``
holds the activations fixed so AUC reproduction isolates the wiring.

``select`` (default). The MLP score decides WHICH segments enter the top-k;
the gate logits are the NATIVE scores gathered at those indices. The read
path therefore sees exactly the score scale it was trained with.

``full``. The MLP score also becomes the gate logit (online branch included,
scored against the same cosine formula).

``full`` is offered for the record, not as the primary arm, because it is
built to fail for a reason unrelated to ranking quality. The trained score is
sum over 16 heads of a cosine, times Kd^-0.5 = 0.088, so it spans about
±1.4, while the native logits are raw inner products against a mean of 256
L2-normalized keys. Substituting the former flattens softmax(gate_logits)
toward uniform, which is precisely the dilution that made the P3 dense arm
score 0 while containing the gold segment 100% of the time. The oracle arm
handled the same hazard by raising the gold score only to row_max + 1.0,
keeping the weight in distribution; ``select`` is that discipline applied
here.

Implementation follows the no-change covenant: this is an eval-only copy of
``SparseSelectiveCaching.forward`` (mc_ssc.py is never edited), with kernel
dispatch pinned to v2 as every eval in this campaign sets
MC_KERNEL_VERSION=v2, exactly as mc_ssc_oracle.py does.
"""

from __future__ import annotations

import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsc.mc_baseline.cached_memory_read import ssc_gather_read
from dsc.mc_baseline.mc_ssc import (
    SSCOutput,
    causal_online_key_sums,
    segment_key_sums,
)


def _router_meta(router_dir: str, blocks: int) -> int:
    """Descriptor granularity travels with the checkpoint.

    A head fitted against 8 sub-block descriptors and injected as blocks=1
    would score a formula it was never trained under and fail silently, so
    the trainer writes meta.json and the loader believes it over its own
    default.
    """
    path = os.path.join(router_dir, "meta.json")
    if not os.path.exists(path):
        return blocks
    with open(path) as fh:
        m = int(json.load(fh).get("blocks", blocks))
    if blocks != 1 and blocks != m:
        raise RuntimeError(
            f"{path} says blocks={m} but blocks={blocks} was requested")
    return m


def block_summaries(keys: torch.Tensor, chunk: int, blocks: int) -> torch.Tensor:
    """[B,T,H,Kd] -> [B,Nseg,blocks,H,Kd]: the mean of each sub-block.

    blocks=1 reproduces segment_key_sums exactly, so this generalizes the
    deployed descriptor rather than replacing it. Scoring a segment by its
    best-matching block undoes some of the 1/chunk dilution a single key
    suffers in the segment mean: held out, hit@2 goes 0.250 -> 0.306 -> 0.316
    -> 0.357 at m = 1 / 2 / 4 / 8, and hit@1 0.128 -> 0.204.

    The memory is close to free. A segment already stores a [H,K,V] state —
    16 x 128 x 128 = 262k values here — against a [H,Kd] descriptor of 2048,
    so eight descriptors add about 6% to what a segment costs. The extra
    compute is eight times a routing einsum that was 0.004% of the forward.
    """
    b, t, h, kd = keys.shape
    nseg = (t + chunk - 1) // chunk
    pad = nseg * chunk - t
    if pad:
        keys = torch.cat([keys, keys.new_zeros(b, pad, h, kd)], dim=1)
    if chunk % blocks:
        raise ValueError(f"chunk {chunk} not divisible by blocks {blocks}")
    per = chunk // blocks
    return keys.view(b, nseg, blocks, per, h, kd).mean(dim=3)


class MLPRouterHead(nn.Module):
    """Router u = L2norm(MLP(h)), architecture-identical to the trained one.

    Module and parameter names must match train_mlp_router.MLPRouter so the
    checkpoint loads with strict=True — a silently partial load would leave a
    randomly initialized router scoring plausible-looking garbage.
    """

    def __init__(self, D: int, H: int, Kd: int, hidden: int | None = None,
                 scorer: str = "cos", blocks: int = 1) -> None:
        super().__init__()
        if scorer not in ("cos", "dot"):
            raise ValueError(f"scorer must be 'cos' or 'dot', got {scorer!r}")
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
        hidden = hidden or D
        self.H, self.Kd, self.scorer, self.blocks = H, Kd, scorer, blocks
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(),
                                 nn.Linear(hidden, H * Kd))
        self.scale = Kd ** -0.5
        if scorer == "dot":
            self.logit_scale = nn.Parameter(torch.zeros(1))

    def u(self, hidden_states: torch.Tensor,
          normalize: bool = True) -> torch.Tensor:
        """[B,T,D] -> L2-normalized [B,T,H,Kd], in the training precision.

        autocast is disabled here on purpose. The eval harness runs the whole
        forward under torch.autocast(bfloat16), which would silently run this
        Linear in bf16 while the router was fitted on fp32 cached tensors.
        The observed cost was ~8e-3 of score noise — small in absolute terms,
        but this is a ranking decision between segments whose scores can sit
        much closer than that, so there is nothing to buy by keeping it.
        """
        batch, length = hidden_states.shape[:2]
        wdtype = next(self.net.parameters()).dtype
        with torch.autocast("cuda", enabled=False):
            u = self.net(hidden_states.to(wdtype)).float().view(
                batch, length, self.H, self.Kd)
        return F.normalize(u, dim=-1) if normalize else u

    def scores(self, hidden_states: torch.Tensor,
               summaries: torch.Tensor) -> torch.Tensor:
        """[B,T,D] x descriptors -> [B,T,N].

        ``summaries`` is [B,N,H,Kd] for blocks=1 and [B,N,m,H,Kd] otherwise;
        with m blocks a segment scores as its best-matching block.
        """
        with torch.autocast("cuda", enabled=False):
            g = summaries.float()
            per_block = g.ndim == 5
            if per_block != (self.blocks > 1):
                raise RuntimeError(
                    f"head has blocks={self.blocks} but got descriptors with "
                    f"{g.ndim} dims — the caller and the head disagree on the "
                    "descriptor layout")
            if self.scorer == "dot":
                u = self.u(hidden_states, normalize=False)
                sc = (torch.einsum("bthk,bnmhk->btnm", u, g)
                      if per_block else
                      torch.einsum("bthk,bnhk->btn", u, g))
                sc = sc * self.logit_scale.exp()
            else:
                u = self.u(hidden_states)
                gn = F.normalize(g, dim=-1)
                sc = (torch.einsum("bthk,bnmhk->btnm", u, gn)
                      if per_block else
                      torch.einsum("bthk,bnhk->btn", u, gn))
                sc = sc * self.scale
            return sc.max(dim=-1).values if per_block else sc


class MLPRoutedGDN2SSC:
    """Namespace holding the replacement forward (mixed into a dyn subclass).

    ``mlp_router`` is deliberately absent here: it is assigned per instance
    and lands in nn.Module._modules, which __getattr__ only reaches after
    normal lookup fails — a class-level default would shadow it permanently.
    """

    mlp_router_mode: str = "select"
    log_mlp_scores: bool = False
    last_mlp_scores: torch.Tensor | None = None

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
        elig_mask = ~eligible.unsqueeze(0)
        past_scores = all_scores.masked_fill(elig_mask, -torch.inf)

        router = getattr(self, "mlp_router", None)
        if router is None:
            raise RuntimeError("mlp_router is None on an MLP-routed aggregator")
        # Scored on the SAME tensors the offline capture recorded: the layer's
        # own hidden_states, and summaries pooled from the already-normalized
        # routing keys this forward receives.
        mlp_scores = router.scores(hidden_states, summaries)
        if self.log_mlp_scores:
            self.last_mlp_scores = mlp_scores.detach().float().cpu()
        rank_scores = mlp_scores.masked_fill(elig_mask, -torch.inf)
        if self.mlp_router_mode == "observe":
            rank_scores = past_scores

        route_count = min(self.topk, num_segments)
        if route_count:
            _, top_indices = torch.topk(rank_scores, k=route_count, dim=-1)
            # Validity and gate logits both come from the NATIVE scores at the
            # selected indices, so the read weights keep their trained scale.
            native_at_top = torch.gather(past_scores, -1, top_indices)
            valid = torch.isfinite(native_at_top)
            if self.mlp_router_mode == "full":
                top_scores = torch.gather(rank_scores, -1, top_indices)
            else:
                top_scores = native_at_top
            safe_indices = top_indices.masked_fill(~valid, 0)
        else:
            top_scores = all_scores.new_empty(batch, length, 0)
            top_indices = torch.empty(batch, length, 0, device=queries.device, dtype=torch.long)
            valid = torch.empty(batch, length, 0, device=queries.device, dtype=torch.bool)
            safe_indices = top_indices

        online_summary = causal_online_key_sums(keys, self.chunk_size)
        if self.mlp_router_mode == "full":
            gam_on = F.normalize(online_summary.float(), dim=-1)
            online_score = torch.einsum(
                "bthk,bthk->bt", router.u(hidden_states), gam_on) * router.scale
        else:
            online_score = torch.einsum("bthk,bthk->bt", u.float(),
                                        online_summary.float())
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


def enable_mlp_router(model, router_dir: str, layers: list[int] | None = None,
                      mode: str = "select", device: str = "cuda",
                      blocks: int = 1) -> list:
    """Attach trained routers to the matching SSC aggregators, in place.

    Layer indexing is the enumeration order of MemoryCachingGDN2Layer in
    ``model.modules()``, and each layer's ``.ssc`` is the aggregator — the same
    order train_mlp_router.py used, so router_L9.pt lands on the tenth MC
    layer and not merely on the tenth GDN2SSC found by a different walk.

    Match by CLASS NAME and derive the subclass from each instance's own
    class: the repo is importable both as ``mc_gdn2.ssc`` and
    ``dsc.mc_gdn2.ssc``, so an identity check silently matches nothing (that
    exact failure once let an oracle arm run as a plain baseline).

    Returns the attached layer indices. Raises if a requested layer has no
    weights, since a missing file would leave that layer on the chance-level
    linear router and quietly turn a clean arm into a partial one.
    """
    if mode not in ("observe", "select", "full"):
        raise ValueError(
            f"mode must be 'observe', 'select' or 'full', got {mode!r}")
    blocks = _router_meta(router_dir, blocks)
    mc_layers = [m for m in model.modules()
                 if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    if not mc_layers:
        raise RuntimeError("found 0 MemoryCachingGDN2Layer — nothing to inject")

    available = sorted(int(m.group(1)) for f in os.listdir(router_dir)
                       if (m := re.fullmatch(r"router_L(\d+)\.pt", f)))
    if not available:
        raise RuntimeError(f"no router_L*.pt in {router_dir}")
    want = sorted(layers) if layers else available
    missing = [L for L in want if L not in available]
    if missing:
        raise RuntimeError(
            f"no weights for layers {missing} in {router_dir} (have "
            f"{available}); those layers would fall back to the linear router")
    if len(want) < len(mc_layers):
        print(f"[warn] injecting {len(want)}/{len(mc_layers)} MC layers — the "
              f"remaining {sorted(set(range(len(mc_layers))) - set(want))} keep "
              "the native linear router, so a flat result is ambiguous",
              flush=True)

    cls_cache: dict[type, type] = {}
    attached = []
    for L in want:
        if L >= len(mc_layers):
            raise RuntimeError(f"layer {L} >= {len(mc_layers)} MC layers")
        agg = mc_layers[L].ssc
        sd = torch.load(os.path.join(router_dir, f"router_L{L}.pt"),
                        map_location="cpu")
        # Infer the scorer from the checkpoint. A dot-scored head loaded as
        # cos would drop its learned temperature and renormalize, i.e. score
        # with a formula it was never fitted under, and strict=True would not
        # catch it if the key were simply ignored.
        head = MLPRouterHead(agg.hidden_size, agg.num_heads, agg.head_qk_dim,
                             scorer="dot" if "logit_scale" in sd else "cos",
                blocks=blocks)
        head.load_state_dict(sd, strict=True)
        head = head.to(device).float().eval()
        for p in head.parameters():
            p.requires_grad_(False)

        base = agg.__class__
        if base not in cls_cache:
            cls_cache[base] = type("MLPRoutedGDN2SSC_dyn", (base,),
                                   {"forward": MLPRoutedGDN2SSC.forward,
                                    "mlp_router_mode": "select",
                                    "log_mlp_scores": False,
                                    "last_mlp_scores": None})
        agg.__class__ = cls_cache[base]
        agg.mlp_router = head
        if getattr(agg, "mlp_router", None) is not head:
            raise RuntimeError(
                f"router for layer {L} did not stick on the aggregator "
                "(shadowed attribute?) — refusing to run a half-injected arm")
        agg.mlp_router_mode = mode
        attached.append(L)
    return attached


class RouteBus:
    """One forward's routing decision, published by the source layer.

    Broadcast means every layer reads the SAME segment indices. Segment i is
    the same token span at every layer (segments are cut by token position),
    so sharing indices is well defined even though each layer's memories are
    its own.

    ``seq`` is a publish counter, not decoration. A consumer that silently
    reused a previous forward's indices would route on stale spans and still
    produce a complete run, which is the failure mode this whole campaign
    keeps hitting. Each consumer runs exactly once per forward and after the
    source, so requiring a strictly newer seq catches a missed publish.
    """

    def __init__(self) -> None:
        self.indices: torch.Tensor | None = None
        self.shape: tuple | None = None
        self.seq = 0

    def publish(self, indices: torch.Tensor) -> None:
        self.indices = indices
        self.shape = tuple(indices.shape)
        self.seq += 1

    def take(self, batch: int, length: int, route_count: int,
             last_seq: int) -> torch.Tensor:
        if self.indices is None:
            raise RuntimeError(
                "broadcast consumer ran before the source published — is the "
                "source layer index greater than a consumer's?")
        if self.seq <= last_seq:
            raise RuntimeError(
                f"broadcast bus is stale (seq={self.seq}, consumer last saw "
                f"{last_seq}) — the source layer did not run this forward")
        if self.shape != (batch, length, route_count):
            raise RuntimeError(
                f"broadcast shape mismatch: bus {self.shape} vs consumer "
                f"{(batch, length, route_count)}")
        return self.indices


class BroadcastGDN2SSC:
    """Namespace holding the broadcast forward (mixed into a dyn subclass).

    ``gate_mode`` decides how much of the read the selected segments get, and
    which of them dominates. It exists because selection stopped being the
    bottleneck: routing hit rose 20.5% -> 32.5% while the score stayed flat,
    and the decomposition put P(correct | gold selected) at 0.295 (N=4) and
    0.095 (N=16) against the oracle's 0.67 and 0.69. Under ``native`` gold
    receives 0.123 of the gate weight while the segment picked beside it gets
    0.242 — gold is the weaker of the two — against the oracle's 0.536.

    native  gate logits are each layer's own scores at the shared indices.
            The read keeps exactly the scale it was trained with, and gold
            is often the quieter of the two.
    order   the same native magnitudes, reassigned so the ROUTER's first pick
            carries the largest of them. No new hyperparameter and no new
            scale: the multiset of logits is untouched, only which selected
            segment gets which. Addresses "selected but read too quietly"
            directly, and is exactly right whenever the router ranks gold
            first.
    top1    the router's first pick is raised to the row's maximum eligible
            score plus ``gate_margin``, which is what the oracle arm does
            with the true gold. Oracle-like weight when the router is right,
            and amplifies a wrong segment when it is not.
    boost   every selected segment's logit gains ``gate_margin``, leaving
            their relative split alone. This moves weight from the online
            branch (0.55-0.65 here, 0.27 under the oracle) into the cache
            without choosing a winner, and is the variant that could make a
            wider k affordable.
    """

    bus: RouteBus | None = None
    is_bcast_source: bool = False
    bcast_source_kind: str = "mlp"
    bcast_last_seq: int = 0
    gate_mode: str = "native"
    gate_margin: float = 1.0
    gate_scope: str = "all"

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
        elig_mask = ~eligible.unsqueeze(0)
        past_scores = all_scores.masked_fill(elig_mask, -torch.inf)

        route_count = min(self.topk, num_segments)
        if route_count:
            if self.is_bcast_source:
                if self.bcast_source_kind == "mlp":
                    router = getattr(self, "mlp_router", None)
                    if router is None:
                        raise RuntimeError("broadcast source has no mlp_router")
                    desc = summaries
                    if router.blocks > 1:
                        # Sub-block descriptors are a routing-side change
                        # only: the cached states, the gate logits and the
                        # read are all untouched, so this cannot move the
                        # score except through which segments get picked.
                        desc = block_summaries(keys, self.chunk_size,
                                               router.blocks)
                    rank = router.scores(hidden_states, desc).masked_fill(
                        elig_mask, -torch.inf)
                else:
                    rank = past_scores
                _, top_indices = torch.topk(rank, k=route_count, dim=-1)
                self.bus.publish(top_indices)
                self.bcast_last_seq = self.bus.seq
            else:
                top_indices = self.bus.take(batch, length, route_count,
                                            self.bcast_last_seq)
                self.bcast_last_seq = self.bus.seq
            # Gate logits start as each layer's OWN native scores at the
            # shared indices, so the read keeps the scale it was trained with.
            top_scores = torch.gather(past_scores, -1, top_indices)
            valid = torch.isfinite(top_scores)
            mode = self.gate_mode
            if mode == "order":
                # Reassign the same magnitudes in the router's order. -inf
                # sorts last and the finite entries occupy the leading slots
                # in router order already (ineligible picks rank -inf), so
                # the valid mask still lines up with top_indices.
                top_scores = torch.sort(top_scores, dim=-1,
                                        descending=True).values
            elif mode == "top1":
                row_max = past_scores.masked_fill(
                    ~torch.isfinite(past_scores), -1e30).amax(dim=-1)
                boosted = row_max + self.gate_margin
                first = top_scores[..., 0]
                top_scores = torch.cat(
                    [torch.where(torch.isfinite(first), boosted,
                                 first).unsqueeze(-1),
                     top_scores[..., 1:]], dim=-1)
            elif mode == "boost":
                m = self.gate_margin
                if self.gate_scope == "last_segment":
                    # Applying the margin at every position rewrites the
                    # whole prefill: 8k positions each read their top-2 with
                    # most of the weight, where the model was trained with
                    # the online branch holding about 0.65. Measured at the
                    # query position the weights looked oracle-like and the
                    # score still collapsed to 0.0 at N=16, which is the
                    # signature of a corrupted residual stream rather than a
                    # bad read. The answer is produced in the final segment,
                    # so boost only there and leave the prefill alone.
                    here = (segment_ids == (num_segments - 1)).view(1, -1, 1)
                    m = torch.where(here, torch.full_like(top_scores, m),
                                    torch.zeros_like(top_scores))
                top_scores = torch.where(valid, top_scores + m, top_scores)
            elif mode != "native":
                raise RuntimeError(f"unknown gate_mode {mode!r}")
            safe_indices = top_indices.masked_fill(~valid, 0)
        else:
            top_scores = all_scores.new_empty(batch, length, 0)
            top_indices = torch.empty(batch, length, 0, device=queries.device, dtype=torch.long)
            valid = torch.empty(batch, length, 0, device=queries.device, dtype=torch.bool)
            safe_indices = top_indices

        online_summary = causal_online_key_sums(keys, self.chunk_size)
        online_score = torch.einsum("bthk,bthk->bt", u.float(),
                                    online_summary.float())
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


GATE_MODES = ("native", "order", "top1", "boost")


def enable_broadcast_routing(model, source_layer: int = 0, source: str = "mlp",
                             router_dir: str | None = None,
                             device: str = "cuda", gate_mode: str = "native",
                             gate_margin: float = 1.0, blocks: int = 1,
                             gate_scope: str = "all") -> list:
    """Make every MC layer route on the source layer's top-k picks.

    Tests the rung missing between the a-1 result (gold reached the top-k of
    about 3 of 16 layers, score 4) and the oracle arm (gold at top-1 of all 16,
    score 82): does making gold available at EVERY layer recover the score,
    at a routing quality a real router can deliver?

    ``source_layer`` may be greater than 0, because the best eligible source
    is not always layer 0: with the harmful needle aux removed, layer 2 ranks
    gold into the top 2 at 0.260 for N=16 against layer 0's 0.160, while layer
    1 leads at N=4. Layers BEFORE the source cannot read the bus, so they run
    their own trained router in select mode. That is a real impurity and it is
    bounded on purpose — a-1 already showed that leaving most of the stack out
    of the shared decision costs everything, so keep source_layer small (2
    means 2 of 16 layers route on their own).

    ``source="native"`` broadcasts the untouched linear router's picks. That is
    the coherence control: a chance-level router shared across layers gives
    the same expected number of layer-hits as 16 independent chance picks, but
    all-or-nothing per sample instead of one layer at a time. It separates
    "layers agreeing" from "the router being better".
    """
    if source not in ("mlp", "native"):
        raise ValueError(f"source must be 'mlp' or 'native', got {source!r}")
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode must be one of {GATE_MODES}, got "
                         f"{gate_mode!r}")
    if gate_scope not in ("all", "last_segment"):
        raise ValueError(f"gate_scope must be 'all' or 'last_segment', got "
                         f"{gate_scope!r}")
    if source_layer < 0:
        raise ValueError(f"source_layer must be >= 0, got {source_layer}")
    if router_dir:
        blocks = _router_meta(router_dir, blocks)
    mc_layers = [m for m in model.modules()
                 if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    if not mc_layers:
        raise RuntimeError("found 0 MemoryCachingGDN2Layer — nothing to inject")
    if source_layer >= len(mc_layers) - 1:
        raise ValueError(
            f"source_layer {source_layer} leaves nothing to broadcast to "
            f"({len(mc_layers)} MC layers)")
    if source_layer > 3:
        raise ValueError(
            f"source_layer {source_layer} would leave {source_layer} of "
            f"{len(mc_layers)} layers outside the shared decision — that is "
            "the configuration a-1 already measured as worthless")

    bus = RouteBus()
    cls_cache: dict[type, type] = {}
    pre_cache: dict[type, type] = {}
    attached = []
    for idx, lyr in enumerate(mc_layers):
        agg = lyr.ssc
        base = agg.__class__
        if idx < source_layer:
            # Cannot read the bus yet. Give it its own trained router in
            # select mode rather than leaving it on the chance-level linear
            # one, and require the weights to exist so the fallback is never
            # silently a different policy than intended.
            if source != "mlp" or not router_dir:
                raise RuntimeError(
                    f"source_layer={source_layer} needs --mlp-router weights "
                    f"for the {source_layer} pre-source layers")
            if base not in pre_cache:
                pre_cache[base] = type(
                    "MLPRoutedGDN2SSC_dyn", (base,),
                    {"forward": MLPRoutedGDN2SSC.forward,
                     "mlp_router_mode": "select",
                     "log_mlp_scores": False,
                     "last_mlp_scores": None})
            sd = torch.load(os.path.join(router_dir, f"router_L{idx}.pt"),
                            map_location="cpu")
            head = MLPRouterHead(
                agg.hidden_size, agg.num_heads, agg.head_qk_dim,
                scorer="dot" if "logit_scale" in sd else "cos",
                blocks=blocks)
            head.load_state_dict(sd, strict=True)
            head = head.to(device).float().eval()
            for prm in head.parameters():
                prm.requires_grad_(False)
            agg.__class__ = pre_cache[base]
            agg.mlp_router = head
            agg.mlp_router_mode = "select"
            if getattr(agg, "mlp_router", None) is not head:
                raise RuntimeError(f"pre-source router L{idx} did not stick")
            continue
        if base not in cls_cache:
            cls_cache[base] = type("BroadcastGDN2SSC_dyn", (base,),
                                   {"forward": BroadcastGDN2SSC.forward,
                                    "bus": None,
                                    "is_bcast_source": False,
                                    "bcast_source_kind": source,
                                    "bcast_last_seq": 0})
        agg.__class__ = cls_cache[base]
        agg.bus = bus
        agg.is_bcast_source = (idx == source_layer)
        if agg.is_bcast_source and idx != 0:
            print(f"[warn] broadcast source is L{idx}: layers "
                  f"{list(range(idx))} route on their own trained router",
                  flush=True)
        agg.bcast_source_kind = source
        agg.bcast_last_seq = 0
        agg.gate_mode = gate_mode
        agg.gate_margin = gate_margin
        agg.gate_scope = gate_scope
        if agg.is_bcast_source and source == "mlp":
            if not router_dir:
                raise RuntimeError("source='mlp' needs --mlp-router DIR")
            sd = torch.load(os.path.join(router_dir,
                                         f"router_L{source_layer}.pt"),
                            map_location="cpu")
            head = MLPRouterHead(
                agg.hidden_size, agg.num_heads, agg.head_qk_dim,
                scorer="dot" if "logit_scale" in sd else "cos",
                blocks=blocks)
            head.load_state_dict(sd, strict=True)
            head = head.to(device).float().eval()
            for p in head.parameters():
                p.requires_grad_(False)
            agg.mlp_router = head
            if getattr(agg, "mlp_router", None) is not head:
                raise RuntimeError("broadcast source router did not stick")
        attached.append(idx)
    return attached
