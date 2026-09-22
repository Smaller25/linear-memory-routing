# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 64
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    local_window: int = -1
    mlp: bool = True
    gdn2_per_layer: int = -1
    # DSC (Dynamic Sparse Caching) — when True, every GDN-2 block becomes a DSCGatedDeltaNet2.
    dsc: bool = False
    dsc_chunk_size: int = 256
    dsc_topk: int = 2
    # MC (Memory Caching) — every GDN-2 block uses a paper MC wrapper.
    # SSC is sparse top-k; GRM and Memory Soup are dense and exactly equivalent
    # for GDN-2's linear matrix-state read.
    # "relu" = ReLU Dynamic Selection: dense read, fluid active-state count
    # (mc_baseline/mc_relu.py). Gate normalization is "relu_norm":
    # softplus(online) + relu(cached), jointly L1-normalized — see module doc.
    mc: bool = False
    mc_variant: Literal["ssc", "grm", "memory_soup", "relu", "relu_raw"] = "ssc"
    mc_chunk_size: int = 256
    mc_topk: int = 2
    mc_route_block_size: int = 16
    # Section 3.4 has two compressor modes. "independent" resets the
    # recurrence at every segment boundary, which is what every checkpoint
    # here was trained with and stays the default. "chained" carries each
    # segment's final state into the next, which is the control for how much
    # the fragmentation costs.
    mc_checkpoint_mode: str = "independent"
    # GDN-2 head geometry. None keeps the library defaults (num_heads=16,
    # head_dim=128), which is what every existing preset was built and
    # trained with, so leaving these unset changes nothing. They exist for
    # the scale ladder: GatedDeltaNet2 was constructed from hidden_size
    # alone, so shrinking n_embd cut the q/k/v/o projections only linearly
    # and a "small" model stayed large. Scale num_heads and hold head_dim at
    # 128 -- the routing descriptor is [num_heads, head_dim], so a fixed
    # head_dim keeps the descriptor geometry, and every finding about
    # pooling inside a 128-dim head, identical to the 370M anchor.
    gdn_num_heads: Optional[int] = None
    gdn_head_dim: Optional[int] = None
    # Log-Linear GDN-2 — weak/base-2 Fenwick hierarchy over the unchanged
    # GDN-2 transition. This is independent from the Memory Caching wrappers.
    log_linear_gdn2: bool = False
    log_linear_lambda_mode: Literal["positive"] = "positive"
    log_linear_checkpoint_levels: bool = True
    # Per-block activation checkpointing (training only). Required for MC SSC to
    # bound VRAM — without it, 16 segments × ~5 GB/segment per layer = OOM.
    block_activation_checkpoint: bool = False
    nope: bool = False
    mamba_init: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm", "FusedRMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["LLaMAMLP"] = "LLaMAMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm

            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)


configs = [
    # ---------------- GatedDeltaNet2 (gdn2) ----------------
    dict(
        org="NVIDIA",
        name="gdn2_1.3B", # Total parameters 1,302,638,112
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6208,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="swa_gdn2_1.3B", # Total parameters 1,300,314,384
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=2,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6784,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="gdn2_370M", # Total parameters ~370M, matches Mamba-370M recurrent state (H=8 * d_k=128 * d_v=128 = 131,072)
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,  # pure recurrent (no SWA)
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
    ),
    # ---------------- Log-Linear GDN-2 (arXiv:2506.04761 lifted to GDN-2) ----------------
    # Same GDN-2 projection/transition/output path as gdn2_370M. The only
    # learned addition is per-token/per-head/per-level lambda. Per-level
    # checkpointing bounds the activation cost of O(log T) primitive calls.
    dict(
        org="LLM-OS-Models",
        name="log_linear_gdn2_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        log_linear_gdn2=True,
        log_linear_lambda_mode="positive",
        log_linear_checkpoint_levels=True,
        block_activation_checkpoint=True,
    ),
    # ---------------- DSC-GatedDeltaNet2 (dsc) ----------------
    # Same backbone as gdn2_370M, with DSC enabled (chunk-boundary cache + multi-res
    # descriptor + top-k routing). Parameter count grows by ~3M (router + combine_alpha).
    dict(
        org="LLM-OS-Models",
        name="dsc_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        dsc=True,
        dsc_chunk_size=256,
        dsc_topk=2,
    ),
    # ---------------- MC-GDN2 (Memory Caching paper SSC, arXiv:2602.24281) ----------------
    # Same backbone as gdn2_370M, with every GDN-2 layer wrapped in MemoryCachingGDN2Layer.
    # Adds only connector W_u (1 linear) per layer on top of vanilla GDN-2.
    # block_activation_checkpoint=True: MC SSC's gather path retains ~64 GB of intermediate
    # per layer (num_v_heads=16 with K=V=128 → 8*4096*2*16*128*128 float32 = 64 GB) for the
    # full T=4096 case, plus chunk_gdn2 saves ~60 GB of backward tensors. Per-Block
    # checkpoint bounds peak to one layer's worth (~10 GB) during backward recompute.
    # The earlier 27x slowdown came from stacking 3 checkpoint layers; this single layer
    # adds ~1.5-2x overhead, consistent with the paper's "minimal overhead" up to constant.
    dict(
        org="LLM-OS-Models",
        name="mc_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        mc=True,
        mc_variant="ssc",
        mc_chunk_size=256,
        mc_topk=4,
        block_activation_checkpoint=False,
    ),
    # Scale-ladder rung below mc_370M. 52.1M non-embedding, 76.7M total --
    # the vocabulary is held at 32k for comparability, so it alone is 24.6M
    # and "50M total" is not reachable at any usable width. Non-embedding is
    # the axis that scales.
    #
    # Every ratio of the anchor is preserved: intermediate_size = 2*n_embd,
    # gdn_num_heads*head_dim = 2*n_embd, head_dim = 128, n_layer = 16.
    # Depth is held rather than shrunk on purpose -- the finding this ladder
    # exists to test is about a routing decision shared across 16 layers,
    # and a shallower model would not have the structure. head_dim is held
    # for the same reason: the routing descriptor is [num_heads, head_dim],
    # so a fixed 128 keeps every result about pooling inside a head
    # comparable to the 370M.
    dict(
        org="LLM-OS-Models",
        name="mc_50M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=3,
        n_embd=384,
        gdn_num_heads=6,
        gdn_head_dim=128,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=768,
        local_window=2048,
        mamba_init=True,
        mc=True,
        mc_variant="ssc",
        mc_chunk_size=256,
        mc_topk=4,
        block_activation_checkpoint=False,
    ),
    # ---------------- MC-GDN2 GRM / Memory Soup (dense, v2 kernel path) ----------------
    # These two names deliberately instantiate the same linear-memory math.
    # Keeping both names makes experiment metadata match the paper terminology;
    # their state_dict layouts are interchangeable.
    dict(
        org="LLM-OS-Models",
        name="mc_grm_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        mc=True,
        mc_variant="grm",
        mc_chunk_size=256,
        mc_route_block_size=16,
        block_activation_checkpoint=False,
    ),
    dict(
        org="LLM-OS-Models",
        name="mc_memory_soup_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        mc=True,
        mc_variant="memory_soup",
        mc_chunk_size=256,
        mc_route_block_size=16,
        block_activation_checkpoint=False,
    ),
    # ---------------- MC-GDN2 ReLU Dynamic Selection (fluid multi-state) ----------------
    # Same backbone/score path as mc_370M (chunk_size=256, independent
    # compressors, meanpool descriptors, connector W_u). The ONLY change vs the
    # Hard Top-k SSC arms is the gate: instead of topk selection + joint
    # softmax, ALL completed segments are gated with
    #     a_on = softplus(online_score); a_i = relu(score_i);
    #     weight = a / (a_on + sum_i a_i + eps)     ["relu_norm"]
    # so the number of ACTIVE cached states per token is input-dependent
    # (ReMoE-style fluid routing, arXiv:2412.14711). When all cached scores
    # are <= 0 the layer degenerates exactly to vanilla single-state GDN-2.
    # Read path = dense (SSC-v2 kernel in route blocks); no topk parameter.
    dict(
        org="LLM-OS-Models",
        name="mc_relu_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        mc=True,
        mc_variant="relu",
        mc_chunk_size=256,
        mc_route_block_size=16,
        block_activation_checkpoint=False,
    ),
    # Raw (unnormalized) ReLU gate — ReMoE original design. Parameter-identical
    # to mc_relu_370M (the gate has no weights): checkpoints are interchangeable.
    # Ablation registered in the experiment protocol; promoted to primary
    # candidate after the normalized gate starved 8K reads during FT (gold-hit
    # fell 0.69->0.55 by 200M tokens while 8K scores stayed at 0).
    dict(
        org="LLM-OS-Models",
        name="mc_relu_raw_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=2048,
        mamba_init=True,
        mc=True,
        mc_variant="relu_raw",
        mc_chunk_size=256,
        mc_route_block_size=16,
        block_activation_checkpoint=False,
    ),
    # ---------------- Pure Transformer 370M-class (controlled baseline, plan a) ----------------
    # gdn2_per_layer=0 -> Block.use_gdn2 is False on every layer, so ALL layers
    # are CausalSelfAttention (full causal attention: local_window=-1 maps to an
    # unlimited left window in the flash-attn call). Same depth/width/tokenizer/
    # recipe as gdn2_370M for a controlled upper-reference arm. NOTE: rope cache
    # is built for block_size=4096; evaluation at 8192 extrapolates positions.
    dict(
        org="LLM-OS-Models",
        name="transformer_370M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=0,  # attention on every layer
        n_layer=16,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        local_window=-1,  # full attention (no sliding window)
        mamba_init=True,
    ),
]

name_to_config = {config["name"]: config for config in configs}
