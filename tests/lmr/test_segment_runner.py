# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model-level tests: the MC segment runner reproduces FLA's own Mamba2 forward.

These pin two equivalences that justify the whole mechanism:
1. A single mixer segment with an empty checkpoint cache (RM) == FLA's mixer forward.
2. A single full-length segment of ``run_segmented_lm`` with RM == the model's plain logits
   (i.e. ``MC-RM`` reduces to vanilla when there is nothing cached -- the N=1 case).

They require a *forwardable* FLA Mamba2 model. FLA's mixer activation (Triton ``swish``) and
fused RMSNorm have no CPU kernel in this build, so these are GPU-gated; on the A100 they exercise
the real ``mamba_chunk_scan_combined`` path used in the experiments. The CPU suite proves the
underlying numerics separately in ``test_ssd_scan.py`` (scan linearity) and ``test_readout.py``.
"""

import pytest
import torch

from lmr.adapters import get_adapter
from lmr.readout import ResidualMemory
from lmr.segment_runner import run_mixer_with_cache, run_segmented_lm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FLA Mamba2 forward needs Triton (swish/RMSNorm) -> CUDA only",
)


def _tiny_model(seed=0, device="cuda"):
    torch.manual_seed(seed)
    from fla.models.mamba2 import Mamba2Config, Mamba2ForCausalLM
    config = Mamba2Config(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        state_size=16,
        head_dim=16,
        expand=2,
        n_groups=1,
        chunk_size=8,
    )
    return Mamba2ForCausalLM(config).to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def test_single_segment_mixer_matches_fla():
    model = _tiny_model()
    mixer = model.backbone.layers[0].mixer
    x = torch.randn(2, 8, mixer.in_proj.in_features, dtype=torch.float32, device="cuda")

    out, final_state, aux = run_mixer_with_cache(get_adapter("mamba2"), mixer, x, [],
                                                 ResidualMemory(), backend="cuda")
    ref, _, _ = mixer(x, use_cache=False)

    assert aux is None
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2)


@torch.no_grad()
def test_mc_rm_reduces_to_vanilla_single_segment():
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (2, 8), device="cuda")
    readouts = [ResidualMemory() for _ in model.backbone.layers]

    logits, aux = run_segmented_lm(model, ids, readouts, chunk_size=8, backend="cuda")
    ref = model(ids).logits

    assert float(aux) == 0.0
    assert torch.allclose(logits, ref, atol=1e-3, rtol=1e-2)


@torch.no_grad()
def test_multi_segment_runs_and_adds_memory():
    """Two segments: cached first-segment memory must change the second-segment logits."""
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (2, 16), device="cuda")
    readouts = [ResidualMemory() for _ in model.backbone.layers]

    mc_logits, _ = run_segmented_lm(model, ids, readouts, chunk_size=8, backend="cuda")
    no_mem, _ = run_segmented_lm(model, ids, readouts, chunk_size=16, backend="cuda")  # single segment

    assert mc_logits.shape == no_mem.shape == (2, 16, model.config.vocab_size)
    assert not torch.allclose(mc_logits[:, 8:], no_mem[:, 8:], atol=1e-4)
