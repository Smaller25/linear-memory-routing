# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper exposing FLA's Mixture-of-Memories (MoM) as a comparison arm.

MoM (``fla/layers/mom.py``, ``fla/models/mom/``) is the *secondary baseline*: M independent
gated-delta memory states + a top-k router + a shared always-on memory, with a Switch-style
load-balance aux loss. Its router routes *tokens -> memory slots* (contrast with SSC, which
routes *tokens -> cached checkpoints*), so it is a baseline, not a re-implementation of MC.

This module just builds the model and exposes the aux loss uniformly so the eval/training
scripts can treat MoM and the MC variants through one interface.
"""

from __future__ import annotations

from fla.models.mom import MomConfig, MomForCausalLM


def build_mom(
    hidden_size: int = 2048,
    num_hidden_layers: int = 24,
    vocab_size: int = 32000,
    num_memories: int = 4,
    topk: int = 2,
    shared_mem: bool = True,
    aux_loss_scale: float = 0.01,
    **kw,
) -> MomForCausalLM:
    """Build a MoM causal LM with the paper defaults (M=4, top-2, shared memory)."""
    config = MomConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        num_memories=num_memories,
        topk=topk,
        shared_mem=shared_mem,
        aux_loss_scale=aux_loss_scale,
        **kw,
    )
    return MomForCausalLM(config)


def forward_with_aux(model: MomForCausalLM, input_ids, labels=None):
    """Forward pass returning ``(logits, aux_loss)`` for a uniform training interface.

    MoM's ``MomForCausalLM`` computes the load-balance aux loss internally and returns it as
    ``aux_loss`` when ``output_router_logits`` is set.
    """
    out = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    return out.logits, getattr(out, "aux_loss", None)
