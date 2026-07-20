# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model-level tests: the MC segment runner reproduces FLA's own Gated-DeltaNet forward.

These pin the two equivalences that justify GDN Memory-Caching:
1. A single GDN block with an empty checkpoint cache (RM) == the block's own forward.
2. A single full-length segment of ``run_segmented_lm`` with RM == the model's plain logits
   (``MC-RM`` reduces to vanilla when nothing is cached -- the N=1 case).

The GDN ops are Triton kernels with no CPU build, so these are GPU-gated (mirroring
``test_segment_runner.py`` for Mamba2). The CPU suite proves the underlying decomposition separately
in ``test_gdn_superposition.py``. A long-enough sequence (T>64) keeps the reference in ``chunk``
mode, matching the adapter's scan.
"""

import pytest
import torch

from lmr.adapters import get_adapter
from lmr.readout import ResidualMemory
from lmr.segment_runner import run_mixer_with_cache, run_segmented_lm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FLA Gated-DeltaNet ops need Triton -> CUDA only",
)

T = 128  # > 64 so the reference forward stays in chunk mode (not fused_recurrent)


def _tiny_model(seed=0, device="cuda"):
    torch.manual_seed(seed)
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    config = GatedDeltaNetConfig(
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        head_dim=64,
        num_heads=2,
        expand_v=1.0,
        conv_size=4,
    )
    return GatedDeltaNetForCausalLM(config).to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def test_single_block_matches_fla():
    model = _tiny_model()
    adapter = get_adapter("gdn")
    block = adapter.blocks(model)[0]
    x = torch.randn(2, T, model.config.hidden_size, dtype=torch.float32, device="cuda")

    out, final_state, aux = adapter.run_block(block, x, [], ResidualMemory(), backend="cuda")
    ref = block(x)[0]

    assert aux is None
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2)


@torch.no_grad()
def test_mc_rm_reduces_to_vanilla_single_segment():
    model = _tiny_model()
    adapter = get_adapter("gdn")
    ids = torch.randint(0, model.config.vocab_size, (2, T), device="cuda")
    readouts = [ResidualMemory() for _ in adapter.blocks(model)]

    logits, aux = run_segmented_lm(model, ids, readouts, chunk_size=T, backend="cuda", arch="gdn")
    ref = model(ids).logits

    assert float(aux) == 0.0
    assert torch.allclose(logits, ref, atol=1e-3, rtol=1e-2)


@torch.no_grad()
def test_multi_segment_runs_and_adds_memory():
    model = _tiny_model()
    adapter = get_adapter("gdn")
    ids = torch.randint(0, model.config.vocab_size, (2, 2 * T), device="cuda")
    readouts = [ResidualMemory() for _ in adapter.blocks(model)]

    mc_logits, _ = run_segmented_lm(model, ids, readouts, chunk_size=T, backend="cuda", arch="gdn")
    no_mem, _ = run_segmented_lm(model, ids, readouts, chunk_size=2 * T, backend="cuda", arch="gdn")

    assert mc_logits.shape == no_mem.shape == (2, 2 * T, model.config.vocab_size)
    assert not torch.allclose(mc_logits[:, T:], no_mem[:, T:], atol=1e-4)
