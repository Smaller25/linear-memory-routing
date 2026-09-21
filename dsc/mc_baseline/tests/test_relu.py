"""Correctness tests for ReLU Dynamic Selection (fluid multi-state routing).

CPU-only: the dense read falls back to the pure-PyTorch path of
``cached_memory_read.ssc_gather_read`` on CPU, which is the same numerical
contract as the Triton kernel.

Run:
    cd dsc && python -m pytest mc_baseline/tests/test_relu.py -q
"""

from __future__ import annotations

import torch

from dsc.mc_baseline.mc_grm import GatedResidualMemory
from dsc.mc_baseline.mc_relu import ReLUSelectiveCaching, ReLUSSCOutput
from dsc.mc_gdn2.dense_layer import DenseMemoryCachingGDN2Layer


def make_core_inputs(seed=0, batch=2, length=7, chunk=2):
    generator = torch.Generator().manual_seed(seed)
    d, h, k, v = 6, 2, 3, 4
    n_seg = (length + chunk - 1) // chunk
    hidden = torch.randn(batch, length, d, generator=generator)
    q = torch.randn(batch, length, h, k, generator=generator)
    keys = torch.randn(batch, length, h, k, generator=generator)
    online = torch.randn(batch, length, h, v, generator=generator)
    memories = torch.randn(batch, n_seg, h, k, v, generator=generator)
    return hidden, q, keys, online, memories, chunk


def test_shape_gradient_and_output_type():
    hidden, q, keys, online, memories, c = make_core_inputs()
    hidden.requires_grad_(True)
    q.requires_grad_(True)
    relu = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)
    result = relu(hidden, q, keys, online, memories)
    assert isinstance(result, ReLUSSCOutput)
    assert result.output.shape == online.shape
    assert result.route_weights.shape == (2, 7, memories.shape[1])
    assert result.active_counts.shape == (2, 7)
    result.output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert relu.connector.weight.grad is not None
    assert torch.isfinite(result.gate_l1)


def test_first_segment_is_exactly_online():
    """No completed segment exists inside chunk 0 — exact vanilla fallback."""
    hidden, q, keys, online, memories, c = make_core_inputs()
    result = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    assert torch.allclose(result.output[:, :c], online[:, :c], atol=1e-6)
    assert torch.equal(
        result.online_weight[:, :c],
        torch.ones_like(result.online_weight[:, :c]),
    )
    assert (result.active_counts[:, :c] == 0).all()


def test_causal_eligibility():
    """A token may never read its own or any future segment's memory."""
    hidden, q, keys, online, memories, c = make_core_inputs()
    result = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    length = q.shape[1]
    for t in range(length):
        cur_seg = t // c
        future = result.route_weights[:, t, cur_seg:]
        assert (future == 0).all(), f"token {t} reads segment >= {cur_seg}"


def test_active_count_is_fluid_and_matches_positive_scores():
    hidden, q, keys, online, memories, c = make_core_inputs(seed=3)
    relu = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)
    result = relu(hidden, q, keys, online, memories)
    expected = (result.route_scores > 0).sum(dim=-1)  # -inf masked ineligible
    assert torch.equal(result.active_counts, expected)
    # Fluidity: with random inputs the active count must not be a constant
    # (that would mean we reimplemented a fixed top-k).
    eligible_positions = result.active_counts[:, c:]
    assert eligible_positions.max() != eligible_positions.min()


def test_active_count_not_capped_at_topk():
    """Force every past score positive: the gate must read ALL completed
    segments, which fixed top-k (k=2) can never do."""
    hidden, q, keys, online, memories, c = make_core_inputs(seed=1)
    relu = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)
    with torch.no_grad():
        # Positive connector output x positive keys -> positive scores.
        relu.connector.weight.fill_(0.05)
    hidden = hidden.abs() + 0.1
    keys = keys.abs() + 0.1
    result = relu(hidden, q, keys, online, memories)
    length = q.shape[1]
    for t in range(length):
        n_eligible = t // c
        assert int(result.active_counts[0, t]) == n_eligible
    assert int(result.active_counts[:, -1].max()) > 2  # exceeds hard top-2


def test_gates_are_normalized():
    hidden, q, keys, online, memories, c = make_core_inputs(seed=2)
    result = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    total = result.online_weight.squeeze(-1) + result.route_weights.sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_score_path_matches_grm():
    """Everything upstream of the gate must be identical to GRM so the arm
    comparison isolates the gate."""
    hidden, q, keys, online, memories, c = make_core_inputs(seed=4)
    relu = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)
    grm = GatedResidualMemory(6, 2, 3, chunk_size=c)
    with torch.no_grad():
        grm.connector.weight.copy_(relu.connector.weight)
    out_relu = relu(hidden, q, keys, online, memories)
    out_grm = grm(hidden, q, keys, online, memories)
    assert torch.allclose(out_relu.route_scores, out_grm.route_scores)
    assert torch.equal(out_relu.route_indices, out_grm.route_indices)


def test_logging_hook():
    hidden, q, keys, online, memories, c = make_core_inputs(seed=5)
    relu = ReLUSelectiveCaching(6, 2, 3, chunk_size=c)
    assert relu.last_active_counts is None
    relu(hidden, q, keys, online, memories)
    assert relu.last_active_counts is None  # off by default
    relu.log_active_states = True
    result = relu(hidden, q, keys, online, memories)
    assert torch.equal(relu.last_active_counts, result.active_counts)
    assert relu.last_final_route_weights.shape == (2, memories.shape[1])
    assert relu.last_final_online_weight.shape == (2,)


def _stub_chunk_gdn2(q, k, v, g, b, w, *, initial_state, output_final_state,
                     use_qk_l2norm_in_kernel, use_gate_in_kernel, cu_seqlens):
    """Tiny deterministic stand-in for the Triton chunk_gdn2 kernel (CPU)."""
    bsz, t, h, key_dim = q.shape
    v_dim = v.shape[-1]
    out = torch.einsum("bthk,bthv->bthv", q.sigmoid(), v)
    state = torch.einsum("bthk,bthv->bhkv", k, v) / max(t, 1)
    return out, state


class _FakeGDN2Base(torch.nn.Module):
    """Minimal GDN-2 projection host for the dense wrapper (CPU smoke)."""

    def __init__(self, hidden_size=8, heads=2, key_dim=4, value_dim=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = heads
        self.num_v_heads = heads
        self.head_k_dim = key_dim
        self.head_v_dim = value_dim
        self.use_short_conv = False
        self.allow_neg_eigval = False
        self.q_proj = torch.nn.Linear(hidden_size, heads * key_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, heads * key_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, heads * value_dim, bias=False)
        self.f_proj = torch.nn.Linear(hidden_size, heads * key_dim, bias=False)
        self.b_proj = torch.nn.Linear(hidden_size, heads * key_dim, bias=False)
        self.w_proj = torch.nn.Linear(hidden_size, heads * value_dim, bias=False)
        self.g_proj = torch.nn.Linear(hidden_size, heads * value_dim, bias=False)
        self.o_proj = torch.nn.Linear(heads * value_dim, hidden_size, bias=False)
        self.A_log = torch.nn.Parameter(torch.zeros(heads))
        self.dt_bias = torch.nn.Parameter(torch.zeros(heads * key_dim))
        self.o_norm = lambda x, gate: x * torch.sigmoid(gate)


def test_dense_layer_relu_variant_end_to_end():
    base = _FakeGDN2Base()
    layer = DenseMemoryCachingGDN2Layer(
        base, variant="relu", chunk_size=2, chunk_gdn2_fn=_stub_chunk_gdn2
    )
    assert layer.variant == "relu"
    layer.enable_active_state_logging(True)
    hidden = torch.randn(2, 6, 8)
    out, result = layer.forward_with_diagnostics(hidden)
    assert out.shape == hidden.shape
    assert isinstance(result, ReLUSSCOutput)
    assert layer.aggregator.last_active_counts is not None
    out.square().mean().backward()
    assert base.q_proj.weight.grad is not None
    assert layer.aggregator.connector.weight.grad is not None


def test_dense_layer_rejects_unknown_variant():
    base = _FakeGDN2Base()
    try:
        DenseMemoryCachingGDN2Layer(base, variant="topk_relu")
    except ValueError as e:
        assert "relu" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown variant")


def test_active_state_logging_rejected_for_grm():
    base = _FakeGDN2Base()
    layer = DenseMemoryCachingGDN2Layer(
        base, variant="grm", chunk_size=2, chunk_gdn2_fn=_stub_chunk_gdn2
    )
    try:
        layer.enable_active_state_logging(True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-relu variant")
