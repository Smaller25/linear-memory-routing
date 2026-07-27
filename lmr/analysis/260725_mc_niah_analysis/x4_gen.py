"""X4: random-routing MC ablation — incremental generation engine + selection
overrides (stock / recent / random).

Goal (spec §5): separate "MC's long-context gain comes from routing quality"
from "from bounded per-segment write load (256 tokens/segment, regardless of
what's written)". We replace top-k SEGMENT SELECTION only — no weight
retraining, no change to the write path, no change to gating math beyond the
E2-style score-floor rule for non-stock selections (see `select_and_score`).

Why an incremental engine at all (not just re-forward each step like
gen_eval.greedy_generate): MC uses checkpoint_mode="independent" — every
segment's own delta-rule recurrence starts from a ZERO state; segments only
talk to each other through the cached (frozen) memories/summaries selected by
routing. That means, once a segment is complete, its contribution to any
LATER position is fully captured by two small per-layer tensors (final state
[H,K,V], descriptor summary [H,K]) — nothing else about that segment's
tokens matters again. So instead of re-forwarding the WHOLE growing sequence
on every generated token (gen_eval.greedy_generate's approach — correct but
means an O(T) forward at every one of 128 steps, T up to 32768), we can
freeze completed segments once and only replay the CURRENT (still-partial)
segment through the stack at each step. This is mathematically equivalent to
full re-forward for the STOCK selection rule (verified by the equivalence
gate in `equivalence_gate()` below — see report/0025 for the empirical
result) and is what makes the random/recent selection overrides
(orthogonal to this optimization) cheap enough to grid over 6 lengths.

Conv lookback correctness note (read before touching `IncrementalEngine`):
GatedDeltaNet2's q/k/v projections apply a depthwise causal conv (kernel_size
4 by default, `dsc/lit_gpt/gdn2.py`) over the FULL un-segmented hidden-state
sequence in the stock/training path — i.e. the conv genuinely looks across a
segment boundary, even though the delta-rule STATE SCAN resets to zero
there. Feeding only the current segment's own tokens through `_project`
naively would zero-pad that context instead of using the true previous
segment's tail, giving WRONG q/k/v for the first `kernel_size-1` tokens of
every new segment (verified: `fla.modules.conv.short_conv.ShortConvolution`
zero-pads when `cache=None`, which is what `_project` always passes). We fix
this by caching, per layer, the EXACT norm_1 ("hidden_states" argument to
`_project`) tensor for the last `kernel_size-1` positions of the
just-completed segment (`_lookback_n1`), and prepending that stored tensor
(not re-derived from token ids — re-deriving from ids only reproduces the
correct value at layer 0, where hidden_states are pure token embeddings;
starting at layer 1 the residual-stream input depends on the FULL stack's
processing of the historical context, which we must reuse from the exact
values already computed when that segment was itself processed, not
recompute from scratch) before calling `_project`, then slicing the conv
output back down to the current segment's own length before the (zero-init)
segment scan and before combining with frozen memories.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_GEN_DEFAULT = 128
CHUNK_DEFAULT = 256
TOPK_DEFAULT = 2
MODES = ("stock", "recent", "random")
SEEDS_DEFAULT = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Pure logic (no torch/model needed) — selection override + score-floor rule.
# This is the REFERENCE implementation for a single (one-token) routing
# decision; `OverrideGDN2SSC.forward` below vectorizes the identical rule
# across [B,T]. tests/lmr/test_mc_niah_x4.py checks the two agree.
# ---------------------------------------------------------------------------
def select_and_score(mode, scores, topk, online_score, rng=None):
    """scores: list[float], the REAL routing score of each of n_frozen past
    segments for one query token (all unconditionally eligible — every
    entry is, by construction, a strictly-completed segment before the
    current partial one; no segment_ids masking needed, unlike stock
    SparseSelectiveCaching which computes eligibility from an absolute
    intra-call segment index).

    Returns (indices: list[int], route_scores: list[float]), both length
    min(topk, n_frozen).

    mode == "stock": real top-k by score, unchanged (matches training/E1/E2
        exactly — this is the sanity/no-op path used by the equivalence
        gate).
    mode == "recent": always the k MOST RECENT segments (highest index
        first) — a position-only baseline, ignoring score entirely.
    mode == "random": k indices drawn uniformly WITHOUT replacement from all
        n_frozen past segments via `rng` (a `random.Random`, required).

    For recent/random, the selected slots' GATE score is replaced by
    `max(max(real score of the slots actually chosen), online_score)` — the
    same rule 0024's E2 oracle used to force-inject the gold segment
    (`oracle.inject_gold`: `best = max(top_scores.max(-1), online_score)`).
    Rationale: if we left the chosen (non-optimal) slots' own real score in
    place, the softmax gate would systematically down-weight them (since
    they weren't picked for being high-scoring), which would confound "the
    SELECTION was bad" with "the GATE additionally punished it for being
    unscored" — two different failure modes. Flooring isolates the former
    (selection quality), which is what X4 is testing.
    """
    n = len(scores)
    k = min(topk, n)
    if k == 0:
        return [], []
    if mode == "stock":
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        idx = order[:k]
        return idx, [scores[i] for i in idx]
    if mode == "recent":
        idx = list(range(n - 1, n - 1 - k, -1))
    elif mode == "random":
        if rng is None:
            raise ValueError("mode='random' requires an rng (random.Random)")
        idx = rng.sample(range(n), k)
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    floor = max(max(scores[i] for i in idx), online_score)
    return idx, [floor] * len(idx)


def chance_upper(n_seg, conditional_em=0.6):
    """Spec §5/§7: chance-routing upper bound = 2/(N-1) * conditional_EM,
    N = ctx/256 (= n_seg here). conditional_EM approximates 0024's oracle
    result range [0.31, 0.56], upper-bounded to the spec's literal 0.6
    constant. n_seg<=1 -> no past segments exist at all -> undefined (None).
    """
    if n_seg <= 1:
        return None
    return min(1.0, 2.0 / (n_seg - 1)) * conditional_em


# ---------------------------------------------------------------------------
# GPU-side override module (lazy dsc/fla imports — only usable where the
# pinned long-gdn worktree + fla are importable, i.e. greenbeard/VESSL).
# ---------------------------------------------------------------------------
def _make_override_class():
    import load_mc
    load_mc.bootstrap()
    from dsc.mc_gdn2.ssc import GDN2SSC
    from dsc.mc_baseline.mc_ssc import SSCOutput, causal_online_key_sums
    from dsc.mc_baseline.cached_memory_read import ssc_gather_read

    class OverrideGDN2SSC(GDN2SSC):
        """SSC that reads FROZEN past-segment memories/summaries (set
        externally by IncrementalEngine, one instance per MC layer) instead
        of the `memories` arg a normal forward call would compute from
        whatever hidden_states it happens to be fed (which, under
        incremental generation, is only ever the current still-partial
        segment — see IncrementalEngine._run_segment). Every entry of
        frozen_summaries/frozen_memories is unconditionally eligible (all
        strictly-completed past segments), so no eligibility masking is
        needed at all, unlike SparseSelectiveCaching.forward.
        """

        selection_mode = "stock"
        seed = 0
        layer_idx = 0
        frozen_summaries = None  # [n_frozen,H,K] (dtype matches routing keys) or None
        frozen_memories = None   # [n_frozen,H,K,V] (dtype matches chunk_gdn2 state) or None
        _gen = None              # torch.Generator, lazily (re)built per (seed,layer_idx,device)
        _gen_device = None

        def _generator(self, device):
            if self._gen is None or self._gen_device != device:
                self._gen = torch.Generator(device=device)
                self._gen.manual_seed(int(self.seed) * 100003 + int(self.layer_idx) + 1)
                self._gen_device = device
            return self._gen

        def forward(self, hidden_states, queries, keys, online_output, memories):
            batch, length, heads, key_dim = queries.shape
            n_frozen = 0 if self.frozen_summaries is None else self.frozen_summaries.shape[0]
            u = self.connector(hidden_states).view(batch, length, heads, key_dim)
            online_summary = causal_online_key_sums(keys, self.chunk_size)
            online_score = torch.einsum("bthk,bthk->bt", u.float(), online_summary.float())

            route_count = min(self.topk, n_frozen)
            if route_count == 0:
                top_scores = online_score.new_zeros(batch, length, 0)
                top_indices = torch.zeros(batch, length, 0, dtype=torch.long,
                                          device=queries.device)
            else:
                all_scores = torch.einsum("bthk,nhk->btn", u.float(),
                                          self.frozen_summaries.float())  # [B,T,n_frozen]
                if self.selection_mode == "stock":
                    top_scores, top_indices = torch.topk(all_scores, k=route_count, dim=-1)
                else:
                    if self.selection_mode == "recent":
                        idx = torch.arange(n_frozen - 1, n_frozen - 1 - route_count, -1,
                                           device=queries.device)
                        top_indices = idx.view(1, 1, route_count).expand(batch, length, -1)
                    elif self.selection_mode == "random":
                        # top-k of iid uniform random keys == a uniform
                        # sample WITHOUT replacement of `route_count` out of
                        # n_frozen (standard trick) — vectorized over [B,T]
                        # instead of a per-token python loop. Seeded
                        # per-(seed,layer) generator -> reproducible.
                        rnd = torch.rand(batch, length, n_frozen, device=queries.device,
                                         generator=self._generator(queries.device))
                        top_indices = torch.topk(rnd, k=route_count, dim=-1).indices
                    else:
                        raise ValueError(f"unknown selection_mode {self.selection_mode!r}")
                    selected_scores = torch.gather(all_scores, -1, top_indices)
                    floor = torch.maximum(selected_scores.max(-1).values, online_score)
                    top_scores = floor.unsqueeze(-1).expand(-1, -1, route_count)

            gate_logits = torch.cat([online_score.unsqueeze(-1), top_scores], dim=-1)
            gates = torch.softmax(gate_logits, dim=-1).to(online_output.dtype)
            online_weight, route_weights = gates[..., :1], gates[..., 1:]
            if route_count:
                mem = self.frozen_memories.unsqueeze(0).expand(batch, -1, -1, -1, -1)
                cached_output = ssc_gather_read(
                    queries, mem, top_indices, route_weights,
                    scale=self.read_scale, normalize_queries=self.normalize_queries,
                ).to(online_output.dtype)
            else:
                cached_output = torch.zeros_like(online_output)
            output = online_weight.unsqueeze(-1) * online_output + cached_output
            return SSCOutput(output=output, online_output=online_output,
                             cached_output=cached_output, route_indices=top_indices,
                             route_weights=route_weights, online_weight=online_weight,
                             route_scores=top_scores)

    return OverrideGDN2SSC


def patch_override(model, mode="stock", seed=0):
    """Replace every MC layer's `attn.ssc` with a fresh OverrideGDN2SSC
    (weights copied, same pattern as oracle.patch_oracle). Returns the list
    of override instances in layer order (== load_mc.mc_layers order, which
    is ascending block index — all 16 blocks are MC layers for mc_370M, see
    dsc/lit_gpt/model.py Block.__init__ / config gdn2_per_layer=1)."""
    import load_mc
    cls = _make_override_class()
    overrides = []
    for i, attn in load_mc.mc_layers(model):
        old = attn.ssc
        new = cls(old.hidden_size, old.num_heads, old.head_qk_dim,
                  topk=old.topk, chunk_size=old.chunk_size)
        new.load_state_dict(old.state_dict())
        new = new.to(next(old.parameters()).device, next(old.parameters()).dtype)
        new.selection_mode = mode
        new.seed = seed
        new.layer_idx = i
        attn.ssc = new
        overrides.append(new)
    return overrides


def set_mode(overrides, mode, seed=0):
    """Mutate an already-patched set of override instances in place (avoids
    re-patching/reloading weights between grid cells)."""
    for o in overrides:
        o.selection_mode = mode
        o.seed = seed
        o._gen = None


def reset_overrides(overrides):
    for o in overrides:
        o.frozen_summaries = None
        o.frozen_memories = None
        o._gen = None


class IncrementalEngine:
    """Segment-by-segment incremental forward for MC models. See module
    docstring for why this is mathematically equivalent to full re-forward
    under checkpoint_mode="independent", and for the conv-lookback fix."""

    def __init__(self, model, overrides, chunk_size=CHUNK_DEFAULT):
        import load_mc
        load_mc.bootstrap()
        from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
        from dsc.mc_baseline.mc_ssc import segment_key_sums

        self.model = model
        self.overrides = overrides
        self.blocks = list(model.transformer.h)
        self.chunk_size = chunk_size
        self._chunk_gdn2 = chunk_gdn2
        self._segment_key_sums = segment_key_sums
        self._lookback_n1 = [None] * len(self.blocks)
        self._kernel_lb = []
        for blk in self.blocks:
            base = blk.attn.base
            kdim = base.q_conv1d.kernel_size[0] - 1 if base.use_short_conv else 0
            self._kernel_lb.append(kdim)

    def reset(self):
        reset_overrides(self.overrides)
        self._lookback_n1 = [None] * len(self.blocks)

    @torch.no_grad()
    def _run_segment(self, x, commit):
        """x: [1,Lc,D] token embeddings for the CURRENT segment's own
        tokens (segment-start .. now); Lc<=chunk_size. Returns logits_last
        [vocab] at the final fed position. If `commit`, freezes this
        segment's final per-layer state + descriptor summary + conv
        lookback for reuse by the NEXT segment — caller must only pass
        commit=True when Lc == chunk_size (a genuinely complete segment);
        asserted below."""
        if commit:
            assert x.shape[1] == self.chunk_size, (
                f"commit=True requires a full {self.chunk_size}-token segment, "
                f"got Lc={x.shape[1]}")
        for i, blk in enumerate(self.blocks):
            n1 = blk.norm_1(x)
            lb = self._lookback_n1[i]
            combined = n1 if lb is None else torch.cat([lb, n1], dim=1)
            q, k, v, g, b, w = blk.attn._project(combined)
            Lb = 0 if lb is None else lb.shape[1]
            if Lb:
                q, k, v, g, b, w = (t[:, Lb:] for t in (q, k, v, g, b, w))
            online_output, state = self._chunk_gdn2(
                q=q, k=k, v=v, g=g, b=b, w=w, initial_state=None,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False, cu_seqlens=None)
            routing_keys = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
            ssc_out = blk.attn.ssc(n1, q, routing_keys, online_output, None)
            gate = rearrange(blk.attn.base.g_proj(n1), "... (h d) -> ... h d",
                             d=blk.attn.base.head_v_dim)
            gated = blk.attn.base.o_norm(ssc_out.output, gate)
            attn_out = blk.attn.base.o_proj(rearrange(gated, "b t h d -> b t (h d)"))
            x = x + attn_out
            n2 = blk.norm_2(x)
            x = x + blk.mlp(n2)

            if commit:
                kern_lb = self._kernel_lb[i]
                self._lookback_n1[i] = n1[:, -kern_lb:].clone() if kern_lb else None
                summary = self._segment_key_sums(routing_keys, routing_keys.shape[1])[:, 0]  # [1,H,K]
                ov = self.overrides[i]
                ov.frozen_summaries = (summary if ov.frozen_summaries is None
                                       else torch.cat([ov.frozen_summaries, summary], dim=0))
                st = state[0]  # [H,K,V] -> keep exact dtype chunk_gdn2 returned (matches stock,
                               # which never explicitly casts the memories tensor at storage time)
                ov.frozen_memories = (st.unsqueeze(0) if ov.frozen_memories is None
                                      else torch.cat([ov.frozen_memories, st.unsqueeze(0)], dim=0))
        x = self.model.transformer.ln_f(x)
        return self.model.lm_head(x[:, -1])


def _embed(model, ids_1d, device):
    ids = torch.tensor([ids_1d], device=device)
    return model.transformer.wte(ids)


@torch.no_grad()
def incremental_generate(model, engine, prompt_ids, n_gen=N_GEN_DEFAULT, eos_id=2,
                         chunk_size=CHUNK_DEFAULT):
    """prompt_ids: python list[int]. Greedy generation via IncrementalEngine
    (segment-by-segment prefill + per-step current-segment replay). Same
    stopping rule as gen_eval.greedy_generate (stop at eos_id or n_gen
    tokens) so outputs are directly comparable in the equivalence gate."""
    engine.reset()
    device = next(model.parameters()).device
    ids = list(prompt_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pos = 0
        logits = None
        while pos < len(ids):
            seg_end = min(pos + chunk_size, len(ids))
            complete = (seg_end - pos) == chunk_size
            x = _embed(model, ids[pos:seg_end], device)
            logits = engine._run_segment(x, commit=complete)
            if complete:
                pos = seg_end
            else:
                break
        cur_ids = list(ids[pos:])

        out = []
        for _ in range(n_gen):
            if cur_ids:
                x = _embed(model, cur_ids, device)
                logits = engine._run_segment(x, commit=(len(cur_ids) == chunk_size))
                if len(cur_ids) == chunk_size:
                    cur_ids = []
            nxt = int(logits.float().argmax())
            if nxt == eos_id:
                break
            out.append(nxt)
            cur_ids.append(nxt)
    return out


# ---------------------------------------------------------------------------
# Equivalence gate (mandatory before the grid — spec §5 / brief).
#
# IMPORTANT design note (found + fixed during the gate's first run):
# OverrideGDN2SSC is NOT a drop-in replacement for the plain full-sequence
# forward path the way oracle.OracleGDN2SSC is. OracleGDN2SSC still consumes
# the `memories` argument gdn2_ssc_forward computes from whatever it's fed,
# so patching it into the model and calling the model's ordinary forward
# (gen_eval.greedy_generate) works unmodified. OverrideGDN2SSC instead reads
# `self.frozen_summaries`/`self.frozen_memories`, which only IncrementalEngine
# ever populates -- calling the model's plain forward with OverrideGDN2SSC
# patched in leaves those None (route_count=0, online-only), which is NOT a
# comparable computation at all. The gate therefore keeps the ORIGINAL stock
# ssc instances around and swaps them in specifically for the "ground truth"
# full-reforward call, swapping the override instances back in for the
# IncrementalEngine call. This subsumes the brief's separate "patched-stock
# == unpatched" sanity check: for this design that check IS the main
# equivalence check (there is no meaningful third calling convention to
# additionally exercise), so a single swap-and-compare loop covers both.
# ---------------------------------------------------------------------------
def equivalence_gate(model_kind, n_samples=8, n_gen=32, task="niah_single_1", length=2048):
    """Mandatory gate (spec §5 / task brief): incremental(stock), driven by
    IncrementalEngine, must give byte-identical greedy tokens to
    gen_eval.greedy_generate (full re-forward) on the TRUE unpatched stock
    model, for `n_samples` rows at `length`. Returns a dict report; does NOT
    raise -- caller decides whether to proceed/abort.
    """
    import load_mc, gen_eval
    from dsc.mc_gdn2.ssc import GDN2SSC

    MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
    path = os.path.join(MC_OUT, "data", str(length), task, "validation.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()][:n_samples]

    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(model_kind)

    stock_pairs = [(attn, attn.ssc) for _, attn in load_mc.mc_layers(model)]  # true stock, before any patching
    overrides = patch_override(model, mode="stock", seed=0)  # mutates attn.ssc in place; same attn order as stock_pairs
    engine = IncrementalEngine(model, overrides)

    def use_stock():
        for attn, ssc in stock_pairs:
            attn.ssc = ssc

    def use_override():
        for (attn, _), ov in zip(stock_pairs, overrides):
            attn.ssc = ov

    per_sample = []
    all_match = True
    for i, r in enumerate(rows):
        ids_list = tok(r["input"], add_special_tokens=False).input_ids
        use_stock()
        full = gen_eval.greedy_generate(
            model, torch.tensor([ids_list], device="cuda"), n_gen=n_gen)
        use_override()
        set_mode(overrides, "stock", seed=0)
        inc = incremental_generate(model, engine, ids_list, n_gen=n_gen)
        match = full == inc
        all_match = all_match and match
        per_sample.append({"sample": i, "match": match, "n_tok_full": len(full),
                           "n_tok_inc": len(inc),
                           "full_head": full[:10], "inc_head": inc[:10]})
        print(f"[x4][gate] sample {i}: match={match} "
              f"(full={full[:10]} inc={inc[:10]})", flush=True)
        if not match:
            first_div = next((j for j in range(min(len(full), len(inc)))
                              if full[j] != inc[j]), min(len(full), len(inc)))
            per_sample[-1]["first_divergence_index"] = first_div

    use_stock()  # leave the model in a clean (unpatched) state
    del model
    torch.cuda.empty_cache()
    return {"model": model_kind, "task": task, "length": length, "n_gen": n_gen,
           "all_match": all_match, "per_sample": per_sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    ap.add_argument("--gate", action="store_true", help="run the equivalence gate and exit")
    ap.add_argument("--gate-n-samples", type=int, default=8)
    ap.add_argument("--gate-n-gen", type=int, default=32)
    a = ap.parse_args()

    if a.gate:
        report = equivalence_gate(a.model, n_samples=a.gate_n_samples, n_gen=a.gate_n_gen)
        HERE = os.path.dirname(os.path.abspath(__file__))
        RES = os.path.join(HERE, "results")
        os.makedirs(RES, exist_ok=True)
        MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
        os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)
        out_path_repo = os.path.join(RES, f"x4_gate_{a.model}.json")
        out_path_mcout = os.path.join(MC_OUT, "results", f"x4_gate_{a.model}.json")
        for p in (out_path_repo, out_path_mcout):
            json.dump(report, open(p, "w"), indent=2)
        status = "PASS" if report["all_match"] else "FAIL"
        print(f"[x4][gate] {a.model}: {status} (all_match={report['all_match']})", flush=True)
        if status == "FAIL":
            sys.exit(1)
        return
    print("nothing to do (pass --gate, or use x4_run.py for the grid)", flush=True)


if __name__ == "__main__":
    main()
