# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert official ``state-spaces/mamba2-*`` checkpoints into FLA ``Mamba2ForCausalLM``.

The official checkpoint (mamba_ssm ``MambaLMHeadModel`` format) and FLA's modeling share almost
all parameter names inside the mixer (``in_proj``, ``conv1d``, ``dt_bias``, ``A_log``, ``D``,
``norm``, ``out_proj``). The known differences:

- ``backbone.embedding.weight``  (mamba_ssm, singular)  ->  ``backbone.embeddings.weight`` (FLA)
- ``lm_head.weight`` is tied to the embedding in the official model and may be absent.

This runs on the local A100 (it needs the checkpoint download + CUDA for the logit-match);
in the CPU-only planning container it is import-only. ``tests/lmr/test_converter.py`` gates the
logit-match on CUDA availability.
"""

from __future__ import annotations

import torch

# Source-key -> exact FLA-key renames. Anything not listed is assumed identical.
_RENAME = {
    "backbone.embedding.weight": "backbone.embeddings.weight",
}


def convert_state_dict(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap an official mamba2 state_dict to FLA ``Mamba2ForCausalLM`` keys."""
    out: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        out[_RENAME.get(k, k)] = v
    # Tie lm_head to the embedding when the source omits an untied head.
    if "lm_head.weight" not in out and "backbone.embeddings.weight" in out:
        out["lm_head.weight"] = out["backbone.embeddings.weight"]
    return out


def load_fla_mamba2(repo: str = "state-spaces/mamba2-1.3b", device="cuda", dtype=torch.float32):
    """Download the official checkpoint and load it into an FLA ``Mamba2ForCausalLM``.

    Returns the loaded FLA model. Builds the FLA config from the official ``config.json`` so the
    dims match. Requires network + the ``mamba_ssm`` config conventions.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from fla.models.mamba2 import Mamba2Config, Mamba2ForCausalLM

    # Official mamba2 configs use a flat json; map the fields FLA needs.
    import json
    cfg_path = hf_hub_download(repo, "config.json")
    with open(cfg_path) as f:
        raw = json.load(f)

    # Load the checkpoint first: mamba_ssm pads the vocab to a multiple of
    # pad_vocab_size_multiple (16), so the actual embedding rows (e.g. 50288) exceed
    # config.json's raw vocab_size (e.g. 50277). Build the FLA model at the *padded*
    # size from the embedding tensor so the weights load 1:1.
    try:
        ckpt = load_file(hf_hub_download(repo, "model.safetensors"))
    except Exception:
        ckpt = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu")
    converted = convert_state_dict(ckpt)
    vocab_size = converted["backbone.embeddings.weight"].shape[0]

    config = Mamba2Config(
        vocab_size=vocab_size,
        hidden_size=raw["d_model"],
        num_hidden_layers=raw["n_layer"],
        state_size=raw.get("d_state", 128),
        num_heads=raw.get("d_model", 0) * raw.get("expand", 2) // raw.get("headdim", 64),
        head_dim=raw.get("headdim", 64),
        n_groups=raw.get("ngroups", 1),
        chunk_size=raw.get("chunk_size", 256),
        tie_word_embeddings=raw.get("tie_embeddings", True),
    )
    model = Mamba2ForCausalLM(config).to(device=device, dtype=dtype)

    missing, unexpected = model.load_state_dict(converted, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading mamba2: {unexpected[:8]}")
    # Missing lm_head is fine when weights are tied.
    bad = [m for m in missing if not m.startswith("lm_head")]
    if bad:
        raise RuntimeError(f"missing keys when loading mamba2: {bad[:8]}")
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def logit_match(repo: str = "state-spaces/mamba2-1.3b", prompt_len: int = 32, atol: float = 1e-2):
    """Assert the converted FLA model matches the mamba_ssm reference on a fixed prompt.

    GPU + network only. Returns the max abs logit difference.
    """
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    fla_model = load_fla_mamba2(repo, device="cuda", dtype=torch.float32)
    ref = MambaLMHeadModel.from_pretrained(repo, device="cuda", dtype=torch.float32).eval()

    ids = torch.randint(0, fla_model.config.vocab_size, (1, prompt_len), device="cuda")
    fla_logits = fla_model(ids).logits.float()
    ref_logits = ref(ids).logits.float()
    max_diff = (fla_logits - ref_logits).abs().max().item()
    assert max_diff < atol, f"logit mismatch: max|Δ|={max_diff:.3e} >= {atol}"
    return max_diff
