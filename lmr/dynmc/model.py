# -*- coding: utf-8 -*-
"""DynMC model factory: GatedDeltaNetForCausalLM with DynMC mixer layers.

The upstream fla model classes already thread arbitrary **kwargs down to each
mixer, so `dynmc_segs` (see layer.py / segmenting.py) flows through
`model(input_ids, dynmc_segs=...)` unchanged. The no-cache anchor (0025) is the
same factory with `dynmc=False` — a stock GatedDeltaNetForCausalLM.
"""
from __future__ import annotations

from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

from .layer import DynMCGatedDeltaNet

# 실험 config 4종 (plan §4). vocab 32000 = Mistral tokenizer.
MODEL_CONFIGS = {
    "340m": dict(hidden_size=1024, num_hidden_layers=24, num_heads=4, head_dim=256,
                 expand_v=1, hidden_ratio=4, conv_size=4, tie_word_embeddings=True,
                 vocab_size=32000),
    "170m": dict(hidden_size=1024, num_hidden_layers=12, num_heads=4, head_dim=256,
                 expand_v=1, hidden_ratio=4, conv_size=4, tie_word_embeddings=True,
                 vocab_size=32000),
    "46m": dict(hidden_size=512, num_hidden_layers=12, num_heads=4, head_dim=128,
                expand_v=1, hidden_ratio=4, conv_size=4, tie_word_embeddings=True,
                vocab_size=32000),
}


def build_model(size: str = "340m", dynmc: bool = True, cache_budget: int = 32,
                cur_logit_init: float = 2.0, **overrides) -> GatedDeltaNetForCausalLM:
    kwargs = dict(MODEL_CONFIGS[size])
    kwargs.update(overrides)
    config = GatedDeltaNetConfig(**kwargs)
    config.fuse_cross_entropy = False  # loss는 학습 루프에서 doc-boundary 마스킹과 함께 계산
    model = GatedDeltaNetForCausalLM(config)
    if not dynmc:
        return model

    for block in model.model.layers:
        old = block.attn
        new = DynMCGatedDeltaNet(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_v_heads,
            use_gate=config.use_gate,
            use_short_conv=config.use_short_conv,
            allow_neg_eigval=config.allow_neg_eigval,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=old.layer_idx,
            cache_budget=cache_budget,
            cur_logit_init=cur_logit_init,
        )
        new = new.to(next(old.parameters()).dtype)
        block.attn = new
        # 표준 init 재적용 (Linear/conv/A_log/dt_bias 규약을 upstream과 동일하게)
        new.apply(model._init_weights)
    return model


def param_count(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    if not model.config.tie_word_embeddings:
        emb += model.lm_head.weight.numel()
    return total, total - emb
