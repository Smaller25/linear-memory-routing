"""Correctness tests for the RAW (unnormalized) ReLU gate — ReMoE original.

The raw variant differs from the normalized gate in exactly one way:
route weights are relu(scores) verbatim (no sum-to-1 division) and the
online weight is the constant 1. Everything else — score path, eligibility
masking, read kernel — is shared code.

Run:
    cd dsc && python -m pytest mc_baseline/tests/test_relu_raw.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dsc.mc_baseline.mc_relu import ReLUSelectiveCaching
from dsc.mc_gdn2.dense_layer import DenseMemoryCachingGDN2Layer
from dsc.mc_baseline.tests.test_relu import make_core_inputs


def make_pair(chunk):
    """Normalized and raw gates sharing identical connector weights."""
    torch.manual_seed(1)
    norm = ReLUSelectiveCaching(6, 2, 3, chunk_size=chunk)
    raw = ReLUSelectiveCaching(6, 2, 3, chunk_size=chunk, normalize_gate=False)
    raw.load_state_dict(norm.state_dict())
    return norm, raw


def test_raw_weights_are_unnormalized_relu_scores():
    hidden, q, keys, online, memories, c = make_core_inputs()
    norm, raw = make_pair(c)
    rn = norm(hidden, q, keys, online, memories)
    rr = raw(hidden, q, keys, online, memories)
    # Same activations pre-normalization: raw weights == relu(scores) exactly,
    # so re-normalizing the raw weights must reproduce a sum-to-1 simplex over
    # ACTIVE entries only where the normalized version puts nonzero mass.
    assert (rr.route_weights >= 0).all()
    assert torch.equal(rr.route_weights > 0, rn.route_weights > 0)
    # Online weight is exactly 1 in the raw gate.
    assert torch.equal(rr.online_weight, torch.ones_like(rr.online_weight))
    # Raw weights are NOT sum-normalized whenever anything is active.
    active_rows = rr.active_counts > 0
    if active_rows.any():
        sums = rr.route_weights.sum(dim=-1)[active_rows]
        assert not torch.allclose(sums, torch.ones_like(sums))


def test_raw_vanilla_fallback_exact():
    """All cached scores <= 0 -> output == online, bit-exact (no division)."""
    hidden, q, keys, online, memories, c = make_core_inputs()
    _, raw = make_pair(c)
    with torch.no_grad():
        raw.connector.weight.zero_()  # all scores 0 -> relu == 0 everywhere
    result = raw(hidden, q, keys, online, memories)
    assert torch.equal(result.output, online)
    assert int(result.active_counts.sum()) == 0


def test_raw_active_counts_match_normalized():
    """The gate SELECTION (which segments are active) is identical; only the
    read weighting differs."""
    hidden, q, keys, online, memories, c = make_core_inputs(seed=3)
    norm, raw = make_pair(c)
    rn = norm(hidden, q, keys, online, memories)
    rr = raw(hidden, q, keys, online, memories)
    assert torch.equal(rn.active_counts, rr.active_counts)


def test_raw_gradients_flow_and_finite():
    hidden, q, keys, online, memories, c = make_core_inputs(seed=5)
    hidden.requires_grad_(True)
    _, raw = make_pair(c)
    result = raw(hidden, q, keys, online, memories)
    result.output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert raw.connector.weight.grad is not None
    assert torch.isfinite(raw.connector.weight.grad).all()


def test_dense_layer_variant_plumbing():
    """variant='relu_raw' builds the same module with normalize_gate=False and
    the same parameter names (checkpoints interchangeable with 'relu')."""

    class FakeBase(torch.nn.Module):
        hidden_size, num_v_heads, head_k_dim = 6, 2, 3

    relu_layer = DenseMemoryCachingGDN2Layer(FakeBase(), variant="relu",
                                             chunk_size=2)
    raw_layer = DenseMemoryCachingGDN2Layer(FakeBase(), variant="relu_raw",
                                            chunk_size=2)
    assert raw_layer.variant == "relu_raw"
    assert raw_layer.aggregator.normalize_gate is False
    assert relu_layer.aggregator.normalize_gate is True
    assert set(raw_layer.state_dict()) == set(relu_layer.state_dict())
    raw_layer.enable_active_state_logging(True)
    assert raw_layer.aggregator.log_active_states is True
