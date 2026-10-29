"""Span-copy labels, the routing CE, the decoupled gate and the sub-block router (CPU, tiny)."""

import math

import numpy as np
import pytest
import torch

from mslm.data.span_copy import batch_copy_gold, copy_gold
from mslm.eval.metrics import hypergeom_hit_at_k
from mslm.models import LM, MCConfig, ModelConfig
from mslm.routing_monitor import copy_hit_counts, random_hit


def tiny(**mc):
    mc = {"chunk": 16, "topk": 2, **mc}
    return ModelConfig(arch="mc", size="tiny", n_layer=2, d_model=64, n_head=2, head_dim=32, ffn_dim=128,
                       vocab_size=97, gdn_backend="torch", mc=MCConfig(**mc))


def planted(T=128, chunk=16, n=4, src=10, dst=100, seed=0):
    """Random tokens from a range that never repeats an n-gram, with x[src-n+1..src+1] copied to dst."""
    rng = np.random.default_rng(seed)
    w = rng.permutation(np.arange(1000, 1000 + T + 1))       # all distinct: no accidental repeats
    w[dst - n + 1: dst + 2] = w[src - n + 1: src + 2]
    return w[:-1], w[1:]


def test_copy_gold_finds_planted_span():
    x, y = planted()
    g = copy_gold(x, y, chunk=16, n=4)
    lab = np.flatnonzero(g[:, 0] >= 0)
    assert lab.tolist() == [100]                              # t=dst: same 4-gram, same next token
    assert g[100, 0] == 11 // 16 == 0                         # gold = segment of x[src+1]
    assert (g[100, 1:] == -1).all()


def test_copy_gold_min_gap_and_next_token():
    x, y = planted(src=90, dst=100)                           # source segment 5, target segment 6
    assert (copy_gold(x, y, chunk=16, n=4, min_gap=2)[:, 0] < 0).all()
    assert copy_gold(x, y, chunk=16, n=4, min_gap=1)[100, 0] == 91 // 16
    x, y = planted()
    x, y = x.copy(), y.copy()
    x[101] = y[100] = 5                                       # next token differs -> copying would not help
    assert (copy_gold(x, y, chunk=16, n=4)[:, 0] < 0).all()


def test_copy_gold_multiple_sources_most_recent_first():
    rng = np.random.default_rng(1)
    w = rng.permutation(np.arange(1000, 1000 + 161))
    span = w[5:10].copy()
    for s in (5, 37, 70, 150):                                # four occurrences of a 4-gram + next token
        w[s:s + 5] = span
    x, y = w[:-1], w[1:]
    g = copy_gold(x, y, chunk=16, n=4, max_gold=4)
    t = 150 + 3                                               # end of the 4-gram at the last occurrence
    assert g[t].tolist() == [(70 + 4) // 16, (37 + 4) // 16, (5 + 4) // 16, -1]


def test_batch_shape():
    x, y = planted()
    xb, yb = torch.tensor(np.stack([x, x])), torch.tensor(np.stack([y, y]))
    assert batch_copy_gold(xb, yb, 16, 4, 2, 3).shape == (2, 128, 3)


def test_random_hit_matches_hypergeom():
    n_cand = torch.tensor([31, 31, 3, 10, 5])
    n_gold = torch.tensor([4, 1, 1, 0, 4])
    got = random_hit(n_cand, n_gold, 2).tolist()
    want = [hypergeom_hit_at_k(int(a), int(b), 2) for a, b in zip(n_cand, n_gold)]
    assert got == pytest.approx(want, abs=1e-6)


def _span_batch():
    x, y = planted(T=128, src=10, dst=100)
    x = torch.tensor(x % 97).unsqueeze(0)                    # vocab 97 (labels computed before the modulo)
    return x, batch_copy_gold(torch.tensor(planted()[0]).unsqueeze(0), torch.tensor(planted()[1]).unsqueeze(0),
                              16, 4, 2, 4)


def test_ce_reaches_unselected_segments():
    torch.manual_seed(0)
    m = LM(tiny(aux_loss="span_copy_ce", aux_weight=1.0, topk=1))
    x, gold = _span_batch()
    m.set_routing(capture=True)
    out = m(x, x, batch_meta={"copy_gold": gold})
    assert out["aux_loss"] > 0
    scores = [r.scores for r in m.routing_records()]
    for s in scores:
        s.retain_grad()
    out["aux_loss"].backward()
    for s, r in zip(scores, m.routing_records()):
        g = s.grad[0, 100]                                    # labelled position, eligible segments 0..5
        chosen = set(r.indices[0, 100].tolist())
        unselected = [i for i in range(6) if i not in chosen]
        assert all(g[i].abs() > 0 for i in unselected)
    m.set_routing()


def test_no_labels_keeps_graph():
    m = LM(tiny(aux_loss="span_copy_ce", aux_weight=1.0))
    x = torch.randint(0, 97, (1, 64))
    out = m(x, x, batch_meta={"copy_gold": torch.full((1, 64, 4), -1)})
    assert out["aux_loss"].item() == 0 and out["aux_loss"].requires_grad


def test_sep_gate_isolates_router_and_gate():
    torch.manual_seed(0)
    m = LM(tiny(aux_loss="span_copy_ce", aux_weight=1.0, gate="sep_softmax"))
    x, gold = _span_batch()
    out = m(x, x, batch_meta={"copy_gold": gold})
    out["aux_loss"].backward(retain_graph=True)
    # The CE of an upper layer flows down the residual stream into lower layers
    # (gates included) — that is the backbone learning to route. The isolation
    # that matters is within a layer: the top layer's CE never reaches its gate.
    top = m.blocks[-1].mixer
    assert top.router.connector.weight.grad.abs().sum() > 0
    assert top.gate.proj.weight.grad is None or top.gate.proj.weight.grad.abs().sum() == 0
    m.zero_grad(set_to_none=True)
    out["lm_loss"].backward()
    for blk in m.blocks:                                      # no LM gradient reaches any router
        mix = blk.mixer
        assert mix.gate.proj.weight.grad.abs().sum() > 0
        assert mix.router.connector.weight.grad is None or mix.router.connector.weight.grad.abs().sum() == 0


def test_sep_gate_requires_aux():
    with pytest.raises(ValueError, match="MC_AUX_LOSS"):
        LM(tiny(gate="sep_softmax"))


def test_aux_weight_zero_is_identical_to_plain_mc():
    torch.manual_seed(0)
    a = LM(tiny())
    b = LM(tiny(aux_loss="span_copy_ce", aux_weight=0.0))
    b.load_state_dict(a.state_dict())
    x, gold = _span_batch()
    la = a(x, x)["loss"]
    lb = b(x, x, batch_meta={"copy_gold": gold})["loss"]
    assert torch.equal(la, lb)


def test_subblock_maxsim_matches_explicit_max():
    torch.manual_seed(0)
    m = LM(tiny(descriptor="subblock8", router="maxsim", aux_loss="span_copy_ce", aux_weight=1.0,
                gate="sep_softmax")).eval()
    x = torch.randint(0, 97, (1, 64))
    m.set_routing(capture=True)
    with torch.no_grad():
        m(x)
    rec = m.routing_records()[0]
    assert rec.scores.shape == (1, 64, 4) and torch.isinf(rec.scores[0, :16]).all()
    m.set_routing()
    with pytest.raises(ValueError, match="sub-block"):
        LM(tiny(router="maxsim"))(torch.randint(0, 97, (1, 32)))


def test_copy_hit_counts():
    torch.manual_seed(0)
    m = LM(tiny()).eval()
    x, gold = _span_batch()
    m.set_routing(capture=True)
    with torch.no_grad():
        m(x)
    h = copy_hit_counts(m.routing_records(), gold, 16)
    m.set_routing()
    assert h.shape == (2, 3) and (h[:, 2] == 1).all()
    assert h[0, 1].item() == pytest.approx(1 - math.comb(5, 2) / math.comb(6, 2))
