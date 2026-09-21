"""Correctness tests for GDN-2 GRM and linear Memory Soup (paper-faithful v3).

v3 changes (2026-07-27):
    • GRM forward now requires ``keys`` (paper Eq. 10 MeanPooling uses Σ k_j).
    • Connector projects to per-head dim (num_heads * head_qk_dim), matching SSC.
    • Segment summaries / online summaries use SSC's segment_key_sums helpers.
"""

from __future__ import annotations

import torch

from dsc.mc_baseline.mc_grm import (
    GatedResidualMemory,
    LinearMemorySoup,
)
from dsc.mc_baseline.tests.test_shape import _FakeGDN2, _gdn2_mock
from dsc.mc_gdn2.dense_layer import DenseMemoryCachingGDN2Layer
from dsc.mc_gdn2.layer import MemoryCachingGDN2Layer


def _make_inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    batch, length, hidden_size = 2, 7, 6
    heads, key_dim, value_dim, chunk_size = 2, 3, 4, 2
    hidden = torch.randn(
        batch,
        length,
        hidden_size,
        generator=generator,
    )
    queries = torch.randn(
        batch,
        length,
        heads,
        key_dim,
        generator=generator,
    )
    keys = torch.randn(
        batch,
        length,
        heads,
        key_dim,
        generator=generator,
    )
    online = torch.randn(
        batch,
        length,
        heads,
        value_dim,
        generator=generator,
    )
    memories = torch.randn(
        batch,
        4,
        heads,
        key_dim,
        value_dim,
        generator=generator,
    )
    return hidden, queries, keys, online, memories, chunk_size


def test_grm_reads_every_completed_segment_and_is_causal():
    hidden, queries, keys, online, memories, chunk_size = _make_inputs()
    grm = GatedResidualMemory(
        6,
        2,
        3,
        chunk_size=chunk_size,
    )
    result = grm(hidden, queries, keys, online, memories)

    assert result.output.shape == online.shape
    assert result.route_indices.shape == (2, 7, 4)
    # Online-only at chunk 0 (no completed segments yet).
    assert torch.allclose(result.output[:, :chunk_size], online[:, :chunk_size], atol=1e-6)
    for position in range(queries.shape[1]):
        selected = result.route_indices[0, position]
        selected = selected[selected >= 0]
        assert torch.equal(
            selected,
            torch.arange(position // chunk_size),
        )

    # Causality: perturbing future tokens / segments must not change past output.
    cutoff = 5
    hidden_2 = hidden.clone()
    queries_2 = queries.clone()
    keys_2 = keys.clone()
    online_2 = online.clone()
    memories_2 = memories.clone()
    hidden_2[:, cutoff:] = torch.randn_like(hidden_2[:, cutoff:]) * 100
    queries_2[:, cutoff:] = torch.randn_like(queries_2[:, cutoff:]) * 100
    keys_2[:, cutoff:] = torch.randn_like(keys_2[:, cutoff:]) * 100
    online_2[:, cutoff:] = torch.randn_like(online_2[:, cutoff:]) * 100
    memories_2[:, cutoff // chunk_size :] = (
        torch.randn_like(memories_2[:, cutoff // chunk_size :]) * 100
    )
    second = grm(hidden_2, queries_2, keys_2, online_2, memories_2)
    assert torch.allclose(result.output[:, :cutoff], second.output[:, :cutoff], atol=1e-6)


def test_grm_matches_explicit_souped_state():
    generator = torch.Generator().manual_seed(7)
    batch, length, hidden_size = 1, 6, 3
    heads, key_dim, value_dim, chunk_size = 1, 2, 3, 2
    hidden = torch.randn(batch, length, hidden_size, generator=generator)
    queries = torch.randn(batch, length, heads, key_dim, generator=generator)
    keys = torch.randn(batch, length, heads, key_dim, generator=generator)
    online_states = torch.randn(
        batch, length, heads, key_dim, value_dim, generator=generator
    )
    memories = torch.randn(
        batch, 3, heads, key_dim, value_dim, generator=generator
    )
    online_output = torch.einsum("bthk,bthkv->bthv", queries, online_states)

    grm = GatedResidualMemory(
        hidden_size,
        heads,
        key_dim,
        chunk_size=chunk_size,
        read_scale=1.0,
    )
    result = grm(hidden, queries, keys, online_output, memories)

    safe_indices = result.route_indices.clamp_min(0)
    batch_indices = torch.arange(batch)[:, None, None]
    selected = memories[batch_indices.expand_as(safe_indices), safe_indices]
    soup_state = (
        result.online_weight[..., None, None] * online_states
        + torch.einsum(
            "btn,btnhkv->bthkv",
            result.route_weights,
            selected,
        )
    )
    soup_output = torch.einsum("bthk,bthkv->bthv", queries, soup_state)
    assert torch.allclose(result.output, soup_output, atol=1e-6, rtol=1e-6)


def test_grm_and_memory_soup_named_paths_are_identical():
    hidden, queries, keys, online, memories, chunk_size = _make_inputs(seed=11)
    grm = GatedResidualMemory(6, 2, 3, chunk_size=chunk_size)
    soup = LinearMemorySoup(6, 2, 3, chunk_size=chunk_size)
    soup.load_state_dict(grm.state_dict())

    first = grm(hidden, queries, keys, online, memories)
    second = soup(hidden, queries, keys, online, memories)
    assert torch.equal(first.route_indices, second.route_indices)
    assert torch.equal(first.route_weights, second.route_weights)
    assert torch.equal(first.output, second.output)


def test_dense_route_blocking_is_numerically_equivalent():
    generator = torch.Generator().manual_seed(17)
    batch, length, hidden_size = 1, 39, 3
    heads, key_dim, value_dim, chunk_size = 1, 2, 2, 2
    hidden = torch.randn(batch, length, hidden_size, generator=generator)
    queries = torch.randn(batch, length, heads, key_dim, generator=generator)
    keys = torch.randn(batch, length, heads, key_dim, generator=generator)
    online = torch.randn(batch, length, heads, value_dim, generator=generator)
    memories = torch.randn(
        batch, 20, heads, key_dim, value_dim, generator=generator
    )
    small_blocks = GatedResidualMemory(
        hidden_size, heads, key_dim, chunk_size=chunk_size, route_block_size=3
    )
    large_blocks = GatedResidualMemory(
        hidden_size, heads, key_dim, chunk_size=chunk_size, route_block_size=16
    )
    large_blocks.load_state_dict(small_blocks.state_dict())

    first = small_blocks(hidden, queries, keys, online, memories)
    second = large_blocks(hidden, queries, keys, online, memories)
    assert torch.equal(first.route_weights, second.route_weights)
    assert torch.allclose(first.output, second.output, atol=2e-6, rtol=2e-6)


def test_grm_gradients_reach_connector_query_and_memories():
    hidden, queries, keys, online, memories, chunk_size = _make_inputs(seed=13)
    hidden.requires_grad_(True)
    queries.requires_grad_(True)
    keys.requires_grad_(True)
    online.requires_grad_(True)
    memories.requires_grad_(True)
    grm = GatedResidualMemory(6, 2, 3, chunk_size=chunk_size)
    grm(hidden, queries, keys, online, memories).output.square().mean().backward()

    for gradient in (
        hidden.grad,
        queries.grad,
        keys.grad,
        online.grad,
        memories.grad,
        grm.connector.weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_gdn2_wrapper_variants_share_checkpoint_and_output():
    hidden = torch.randn(1, 5, 6)
    base_grm = _FakeGDN2()
    base_soup = _FakeGDN2()
    grm = DenseMemoryCachingGDN2Layer(
        base_grm,
        variant="grm",
        chunk_size=2,
        chunk_gdn2_fn=_gdn2_mock,
    )
    soup = DenseMemoryCachingGDN2Layer(
        base_soup,
        variant="memory_soup",
        chunk_size=2,
        chunk_gdn2_fn=_gdn2_mock,
    )
    soup.load_state_dict(grm.state_dict())

    grm_output, grm_diagnostics = grm.forward_with_diagnostics(hidden)
    soup_output, soup_diagnostics = soup.forward_with_diagnostics(hidden)
    assert torch.equal(grm_output, soup_output)
    assert torch.equal(
        grm_diagnostics.route_weights,
        soup_diagnostics.route_weights,
    )
    assert set(grm.state_dict()) == set(soup.state_dict())


def test_dense_wrapper_projection_matches_untouched_ssc_wrapper():
    hidden = torch.randn(2, 5, 6)
    base_ssc = _FakeGDN2()
    base_dense = _FakeGDN2()
    base_dense.load_state_dict(base_ssc.state_dict())
    ssc = MemoryCachingGDN2Layer(
        base_ssc,
        chunk_size=2,
        chunk_gdn2_fn=_gdn2_mock,
    )
    dense = DenseMemoryCachingGDN2Layer(
        base_dense,
        chunk_size=2,
        chunk_gdn2_fn=_gdn2_mock,
    )

    for ssc_projection, dense_projection in zip(
        ssc._project(hidden),
        dense._project(hidden),
    ):
        assert torch.equal(ssc_projection, dense_projection)


def test_invalid_variant_is_rejected():
    try:
        DenseMemoryCachingGDN2Layer(
            _FakeGDN2(),
            variant="not-a-paper-variant",
        )
    except ValueError as error:
        assert "expected 'grm', 'memory_soup', 'relu', or 'relu_raw'" in str(error)
    else:
        raise AssertionError("invalid Memory Caching variant was accepted")


def main():
    tests = [
        test_grm_reads_every_completed_segment_and_is_causal,
        test_grm_matches_explicit_souped_state,
        test_grm_and_memory_soup_named_paths_are_identical,
        test_dense_route_blocking_is_numerically_equivalent,
        test_grm_gradients_reach_connector_query_and_memories,
        test_gdn2_wrapper_variants_share_checkpoint_and_output,
        test_dense_wrapper_projection_matches_untouched_ssc_wrapper,
        test_invalid_variant_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
