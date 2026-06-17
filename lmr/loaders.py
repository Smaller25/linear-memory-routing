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
# The linear-moe-hub GDN checkpoint ships no tokenizer; it was trained (SlimPajama, MoM lineage)
# with the Mistral SentencePiece tokenizer (vocab 32000, bos=1/eos=2 — matches its config).
GDN_TOKENIZER = "mistralai/Mistral-7B-v0.1"


def load_gdn(repo: str = GDN_DEFAULT_REPO, device="cuda", dtype=torch.float32):
    """Load an FLA-native Gated-DeltaNet causal LM + its tokenizer. Returns ``(model, tokenizer)``.

    The public ``linear-moe-hub/Gated-Deltanet-1.3B`` checkpoint predates the current FLA layout
    (transformers 4.47-era), so a couple of keys are remapped before loading:
      - MLP: the old SwiGLU used ONE fused ``mlp.gate_proj`` of ``2*intermediate`` rows; current FLA
        splits it into ``gate_proj`` + ``up_proj`` (each ``intermediate``). Split first half = gate,
        second = up (fla SwiGLU does ``silu(gate)*up`` on ``chunk(2)``).
      - ``attn.D``: the old GatedDeltaNet had a per-head skip ``D`` that current FLA dropped; it is
        omitted here (small skip; the model stays a competent LM — sanity-checked by generation).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    config = GatedDeltaNetConfig.from_pretrained(repo)
    model = GatedDeltaNetForCausalLM(config).to(device=device, dtype=dtype)

    # expected (split) gate_proj width from the freshly-built model
    inter = model.model.layers[0].mlp.gate_proj.weight.shape[0]

    src = load_file(hf_hub_download(repo, "model.safetensors"))
    remapped, dropped = {}, []
    for k, v in src.items():
        if k.endswith("mlp.gate_proj.weight") and v.shape[0] == 2 * inter:
            remapped[k] = v[:inter]                                  # gate
            remapped[k[:-len("gate_proj.weight")] + "up_proj.weight"] = v[inter:]  # up
        elif k.endswith(".attn.D"):
            dropped.append(k)
        else:
            remapped[k] = v
    if "lm_head.weight" not in remapped and config.tie_word_embeddings:
        remapped["lm_head.weight"] = remapped["model.embeddings.weight"]

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    bad_missing = [m for m in missing if not m.startswith("lm_head")]
    if bad_missing:
        raise RuntimeError(f"GDN load: unexpected missing keys: {bad_missing[:8]}")
    if unexpected:
        raise RuntimeError(f"GDN load: unexpected keys: {unexpected[:8]}")
    print(f"[load_gdn] remapped MLP gate->gate+up, dropped {len(dropped)} attn.D skip params")

    model = model.to(device=device, dtype=dtype).eval()
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception:
        # repo ships no tokenizer files -> use the Mistral tokenizer it was trained with
        tok = AutoTokenizer.from_pretrained(GDN_TOKENIZER)
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
