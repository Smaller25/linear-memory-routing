# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""One ``--arch``-keyed entry point for loading a frozen backbone + its tokenizer.

- ``mamba2``: reuses :func:`lmr.converter.load_fla_mamba2` (official mamba_ssm checkpoint -> FLA
  ``Mamba2ForCausalLM``) with the gpt-neox tokenizer the experiments use.
- ``gdn``: loads an FLA-native ``GatedDeltaNetForCausalLM`` directly (no converter) plus its own
  tokenizer (which is *not* gpt-neox -- the passkey generator is tokenizer-driven, so eval/training
  must use the tokenizer this returns).
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from lmr.converter import load_fla_mamba2

MAMBA2_TOKENIZER = "EleutherAI/gpt-neox-20b"
GDN_DEFAULT_REPO = "linear-moe-hub/Gated-Deltanet-1.3B"


def load_gdn(repo: str = GDN_DEFAULT_REPO, device="cuda", dtype=torch.float32):
    """Load an FLA-native Gated-DeltaNet causal LM + its tokenizer. Returns ``(model, tokenizer)``."""
    from fla.models.gated_deltanet import GatedDeltaNetForCausalLM

    model = GatedDeltaNetForCausalLM.from_pretrained(repo, torch_dtype=dtype)
    model = model.to(device=device).eval()
    tok = AutoTokenizer.from_pretrained(repo)
    return model, tok


def load_backbone(arch: str, repo: str | None = None, tokenizer: str | None = None,
                  device="cuda", dtype=torch.float32):
    """Load ``(model, tokenizer)`` for ``arch in {"mamba2", "gdn"}``.

    For ``mamba2`` the tokenizer is loaded separately (gpt-neox by default); for ``gdn`` it ships
    with the checkpoint.
    """
    if arch == "mamba2":
        repo = repo or "state-spaces/mamba2-1.3b"
        model = load_fla_mamba2(repo, device=device, dtype=dtype)
        tok = AutoTokenizer.from_pretrained(tokenizer or MAMBA2_TOKENIZER)
        return model, tok
    if arch == "gdn":
        return load_gdn(repo or GDN_DEFAULT_REPO, device=device, dtype=dtype)
    raise ValueError(f"unknown arch: {arch!r} (expected 'mamba2' or 'gdn')")


__all__ = ["load_backbone", "load_gdn", "load_fla_mamba2", "MAMBA2_TOKENIZER", "GDN_DEFAULT_REPO"]
