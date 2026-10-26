"""CPU checks of the model: causality, padding exactness, oracle routing, registry."""

import pytest
import torch

from mslm.models import LM, MCConfig, ModelConfig, registry


def tiny(arch="mc", **mc):
    mc = {"chunk": 16, "topk": 2, **mc}
    return ModelConfig(arch=arch, size="tiny", n_layer=2, d_model=64, n_head=2, head_dim=32, ffn_dim=128,
                       vocab_size=97, gdn_backend="torch", mc=MCConfig(**mc))


@pytest.mark.parametrize("arch,connect", [("vanilla", "independent"), ("mc", "independent"), ("mc", "chained")])
def test_causal(arch, connect):
    torch.manual_seed(0)
    m = LM(tiny(arch, connect=connect)).eval()
    x = torch.randint(0, 97, (1, 64))
    y = x.clone()
    y[0, 40:] = (y[0, 40:] + 1) % 97
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :40], b[:, :40], atol=1e-5)


def test_non_multiple_length_equals_prefix():
    """Right-padding to a chunk multiple is exact for real positions."""
    torch.manual_seed(0)
    m = LM(tiny()).eval()
    x = torch.randint(0, 97, (1, 64))
    with torch.no_grad():
        full, part = m(x), m(x[:, :50])
    assert torch.allclose(full[:, :50], part, atol=1e-5)


def test_oracle_forces_gold_and_records():
    torch.manual_seed(0)
    m = LM(tiny()).eval()
    x = torch.randint(0, 97, (2, 64))
    m.set_routing(oracle_segments=torch.tensor([[1, -1], [0, 2]]), capture=True)
    with torch.no_grad():
        m(x)
    for rec in m.routing_records():
        last = rec.indices[:, -1]                       # position 63 -> segment 3, eligible 0..2
        assert 1 in last[0].tolist()
        assert {0, 2} <= set(last[1].tolist())
        assert torch.isinf(rec.scores[:, 0]).all()      # first segment has no eligible past
    m.set_routing()


def test_registry_plugins_and_aux_hook():
    @registry.register("gate", "_test_uniform")
    def uniform(online, sel, valid):
        w = torch.cat([torch.ones_like(online).unsqueeze(-1), valid.float()], -1)
        return w / w.sum(-1, keepdim=True)
    try:
        cfg = tiny(gate="_test_uniform", aux_loss="gold_ce", aux_weight=0.5)
        m = LM(cfg)
        x = torch.randint(0, 97, (2, 65))
        meta = {"gold_segments": torch.tensor([[1], [2]]), "query_pos": torch.tensor([63, 63])}
        out = m(x[:, :-1], x[:, 1:], batch_meta=meta)
        assert out["aux_loss"] > 0
        out["loss"].backward()
        assert m.blocks[0].mixer.router.connector.weight.grad.abs().sum() > 0
    finally:
        registry.GATES.pop("_test_uniform")


def test_unknown_part_lists_choices():
    with pytest.raises(KeyError, match="registered"):
        LM(tiny(descriptor="nope"))


def test_presets_param_counts():
    from mslm.models import PRESETS
    for size, (lo, hi) in {"50m": (50e6, 80e6), "370m": (330e6, 460e6)}.items():
        p = PRESETS[size]
        cfg = ModelConfig(arch="mc", size=size, gdn_backend="torch", **p)
        with torch.device("meta"):
            n = LM(cfg).num_params()
        assert lo < n < hi, (size, n)
