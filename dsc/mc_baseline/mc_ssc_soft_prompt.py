"""Track 2 of report 0027: learned soft prompts that bias what a segment keeps.

The premise. A linear-recurrent layer accumulates its state left to right, so
whatever sits at the front of a segment is read before the text and shapes the
state the text then writes into. Prepending a few trainable embeddings to every
segment therefore lets an objective say "keep the kind of token a diverse-key
query will ask for" without touching a single backbone weight -- only p*D
numbers are trained.

Three facts about this backbone make the implementation simpler than it looks,
and all three were checked rather than assumed:

* **Positions do not shift.** `Block.forward` passes the RoPE cache only to
  `CausalSelfAttention`; a GDN-2 block is called as `self.attn(n_1,
  attention_mask=None)`. With `gdn2_per_layer=1` every layer is GDN-2, so no
  layer consumes positional encodings and inserting tokens cannot misalign
  them.
* **There is exactly one embedding site.** `GPT.forward` does
  `x = self.transformer.wte(idx)` once, and takes ids rather than embeddings,
  so wrapping `wte` is the whole injection.
* **Segment membership is arithmetic.** SSC segments a sequence by
  `position // chunk_size`, so inserting p slots before every chunk of real
  tokens moves every downstream index by a known amount. `PositionMap` carries
  that remapping instead of leaving callers to recompute it, because the gold
  segment index, the needle span and the answer position all depend on it and
  a silent off-by-one there would look exactly like a routing result.

What this is NOT is a zero-initialisable intervention. A prefix of zero
*vectors* is still a token: it runs through the recurrence and produces its own
gate and key, so `prefix -> 0` does not recover the unmodified model. The only
exact control is `p = 0`. Warm-starting from real vocabulary embeddings is
therefore not a nicety but the actual mitigation for initialisation shock, and
the distribution-shift risk is real: this is the same class of intervention as
the uniform gate margin, which looked oracle-like at the query position and
still destroyed the score by rewriting every prefill position (0027 §5).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PositionMap:
    """How an expanded sequence relates to the original one.

    ``new_of_old[i]`` is where original token ``i`` now lives. ``slot_mask``
    marks the inserted positions. ``chunk_out`` is the segment size the SSC
    layer must be configured with so that one state still covers one group of
    ``chunk_in`` real tokens.
    """

    new_of_old: torch.Tensor      # [T] long
    slot_mask: torch.Tensor       # [T_out] bool
    chunk_in: int
    chunk_out: int
    n_prefix: int
    n_suffix: int

    @property
    def length_out(self) -> int:
        return int(self.slot_mask.numel())

    def segment_of_old(self, pos: int) -> int:
        """Segment index of an original position, after expansion.

        Equal to ``pos // chunk_in`` by construction; computed through the map
        so that a change to the layout cannot silently invalidate callers.
        """
        return int(self.new_of_old[pos].item()) // self.chunk_out


def plan_expansion(length: int, chunk_in: int, n_prefix: int,
                   n_suffix: int = 0, device=None) -> PositionMap:
    """Work out the expanded layout for one sequence length.

    Each group of ``chunk_in`` real tokens becomes
    ``n_prefix + chunk_in + n_suffix`` positions. The final group is expanded
    the same way even when it is short, so the segment grid stays uniform and
    ``position // chunk_out`` keeps agreeing with the original
    ``position // chunk_in`` grouping.
    """
    if n_prefix < 0 or n_suffix < 0:
        raise ValueError("prompt lengths must be non-negative")
    if chunk_in <= 0:
        raise ValueError(f"chunk_in must be positive, got {chunk_in}")
    chunk_out = n_prefix + chunk_in + n_suffix
    nseg = (length + chunk_in - 1) // chunk_in
    last = length - (nseg - 1) * chunk_in if nseg else 0
    # The final segment is NOT padded out to chunk_out. Padding it would feed
    # the backbone up to chunk_in filler tokens that the original sequence
    # never had, and `position // chunk_out` keeps assigning the short tail to
    # the right segment anyway because the tail starts at (nseg-1)*chunk_out
    # and is shorter than chunk_out.
    t_out = (nseg - 1) * chunk_out + n_prefix + last + n_suffix if nseg else 0
    slot = torch.zeros(t_out, dtype=torch.bool, device=device)
    new_of_old = torch.empty(length, dtype=torch.long, device=device)
    for seg in range(nseg):
        base = seg * chunk_out
        lo = seg * chunk_in
        hi = min(lo + chunk_in, length)
        slot[base:base + n_prefix] = True
        body = base + n_prefix
        if hi > lo:
            idx = torch.arange(lo, hi, device=device)
            new_of_old[lo:hi] = body + (idx - lo)
        tail = body + (hi - lo)
        slot[tail:tail + n_suffix] = True
    return PositionMap(new_of_old=new_of_old, slot_mask=slot,
                       chunk_in=chunk_in, chunk_out=chunk_out,
                       n_prefix=n_prefix, n_suffix=n_suffix)


def expand_ids(ids: torch.Tensor, pm: PositionMap,
               fill_id: int = 0) -> torch.Tensor:
    """[B,T] -> [B,T_out] with `fill_id` parked in the inserted slots.

    The filler never reaches the model: ``SoftPromptEmbedding`` overwrites
    every slot position with a trained vector. It exists so the wrapped
    embedding still receives a valid index.
    """
    if ids.ndim != 2:
        raise ValueError(f"expected ids [B,T], got {tuple(ids.shape)}")
    b, t = ids.shape
    if t != pm.new_of_old.numel():
        raise ValueError(f"ids length {t} does not match the plan's "
                         f"{pm.new_of_old.numel()}")
    out = ids.new_full((b, pm.length_out), fill_id)
    out[:, pm.new_of_old] = ids
    if int(pm.slot_mask.sum()) + t != pm.length_out:
        raise RuntimeError(
            f"layout leaves {pm.length_out - t - int(pm.slot_mask.sum())} "
            "positions that are neither a real token nor a prompt slot; the "
            "backbone would read filler there")
    return out


class SoftPromptEmbedding(nn.Module):
    """Wraps `wte`, substituting trained vectors at the planned slots.

    Only `prefix` (and `suffix`) require gradients; the wrapped embedding is
    used under `no_grad` so a frozen backbone stays frozen even if the caller
    forgets to freeze it.
    """

    def __init__(self, wte: nn.Module, n_prefix: int, n_suffix: int = 0,
                 warm_start_ids: list[int] | None = None) -> None:
        super().__init__()
        self.wte = wte
        self.n_prefix, self.n_suffix = n_prefix, n_suffix
        d = wte.weight.shape[1]
        dev, dt = wte.weight.device, wte.weight.dtype

        def _init(n: int, ids: list[int] | None) -> nn.Parameter | None:
            if n == 0:
                return None
            if ids:
                if len(ids) < n:
                    ids = (ids * ((n // len(ids)) + 1))[:n]
                w = wte.weight.detach()[torch.tensor(ids[:n], device=dev)]
            else:
                # Random init is the documented way to lose the first
                # thousand steps, so it is opt-in by passing no ids and
                # deliberately small rather than N(0,1).
                w = torch.randn(n, d, device=dev, dtype=dt) * 0.02
            return nn.Parameter(w.clone().float())

        self.prefix = _init(n_prefix, warm_start_ids)
        self.suffix = _init(n_suffix, warm_start_ids)
        self.slot_mask: torch.Tensor | None = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.wte(idx)
        m = self.slot_mask
        if m is None:
            if self.n_prefix or self.n_suffix:
                raise RuntimeError(
                    "soft prompt is attached but no slot_mask was set; call "
                    "set_plan() before the forward or the trained vectors "
                    "are silently unused")
            return x
        b, t, d = x.shape
        if m.numel() != t:
            raise ValueError(f"slot_mask length {m.numel()} does not match "
                             f"the forward's T={t}")
        per = self.n_prefix + self.n_suffix
        if per == 0:
            return x
        x = x.clone()
        # Slots repeat with the segment period, so a single tiled write covers
        # every segment without a Python loop over them.
        filled = int(m.sum())
        if filled % per:
            raise ValueError(f"{filled} slots is not a multiple of "
                             f"{per} prompt vectors per segment")
        rows = [v for v in (self.prefix, self.suffix) if v is not None]
        tile = torch.cat(rows, dim=0).to(x.dtype)
        reps = filled // per
        x[:, m] = tile.repeat(reps, 1).unsqueeze(0).expand(b, -1, -1)
        return x

    def set_plan(self, pm: PositionMap | None) -> None:
        self.slot_mask = None if pm is None else pm.slot_mask.to(
            self.wte.weight.device)


def attach_soft_prompt(model, n_prefix: int, n_suffix: int = 0,
                       warm_start_ids: list[int] | None = None
                       ) -> SoftPromptEmbedding:
    """Replace `model.transformer.wte` with the wrapper and freeze everything else."""
    wte = model.transformer.wte
    if isinstance(wte, SoftPromptEmbedding):
        raise RuntimeError("a soft prompt is already attached")
    sp = SoftPromptEmbedding(wte, n_prefix, n_suffix, warm_start_ids)
    for p in model.parameters():
        p.requires_grad_(False)
    # NOT sp.parameters(): the wrapped embedding is a submodule of sp, so
    # unfreezing the whole wrapper would hand the backbone's embedding table
    # back to the optimizer -- 33M parameters masquerading as a 8k-parameter
    # intervention.
    for v in (sp.prefix, sp.suffix):
        if v is not None:
            v.requires_grad_(True)
    model.transformer.wte = sp
    return sp


def prompt_state_dict(sp: SoftPromptEmbedding) -> dict:
    """Just the trained vectors, so a checkpoint cannot smuggle the backbone."""
    return {k: v.detach().cpu() for k, v in
            (("prefix", sp.prefix), ("suffix", sp.suffix)) if v is not None}


def trainable_report(sp: SoftPromptEmbedding) -> str:
    n = sum(v.numel() for v in (sp.prefix, sp.suffix)
            if v is not None and v.requires_grad)
    d = sp.wte.weight.shape[1]
    return (f"soft prompt: {sp.n_prefix} prefix + {sp.n_suffix} suffix "
            f"vectors of dim {d} = {n} trainable parameters")
