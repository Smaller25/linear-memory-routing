"""Strict causality and routing tests for paper-faithful SSC."""

import torch

from dsc.mc_baseline.mc_ssc import SparseSelectiveCaching


def make_inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    b, t, d, h, k, v, c = 1, 12, 8, 2, 4, 5, 3
    hidden = torch.randn(b, t, d, generator=generator)
    q = torch.randn(b, t, h, k, generator=generator)
    keys = torch.randn(b, t, h, k, generator=generator)
    online = torch.randn(b, t, h, v, generator=generator)
    memories = torch.randn(b, t // c, h, k, v, generator=generator)
    return hidden, q, keys, online, memories, c


def test_future_perturbation_does_not_change_past():
    hidden, q, keys, online, memories, c = make_inputs()
    ssc = SparseSelectiveCaching(8, 2, 4, topk=2, chunk_size=c)
    cutoff = 7
    first = ssc(hidden, q, keys, online, memories)

    hidden2, q2, keys2, online2 = [tensor.clone() for tensor in (hidden, q, keys, online)]
    for tensor in (hidden2, q2, keys2, online2):
        tensor[:, cutoff:] = torch.randn_like(tensor[:, cutoff:]) * 100
    memories2 = memories.clone()
    # Only states for segments that end after cutoff may change.
    memories2[:, cutoff // c:] = torch.randn_like(memories2[:, cutoff // c:]) * 100
    second = ssc(hidden2, q2, keys2, online2, memories2)

    assert torch.equal(first.output[:, :cutoff], second.output[:, :cutoff])
    assert torch.equal(first.route_indices[:, :cutoff], second.route_indices[:, :cutoff])


def test_only_completed_segments_are_routable():
    hidden, q, keys, online, memories, c = make_inputs()
    result = SparseSelectiveCaching(8, 2, 4, topk=4, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    for pos in range(q.shape[1]):
        chosen = result.route_indices[0, pos]
        chosen = chosen[chosen >= 0]
        assert (chosen < pos // c).all(), (pos, chosen)


def test_equation_16_routes_by_connector_and_segment_key_sum():
    # D=H=K=1 makes W_u, key sums, and expected top-1 score transparent.
    hidden = torch.ones(1, 6, 1)
    q = torch.ones(1, 6, 1, 1)
    keys = torch.tensor([1.0, 1.0, 5.0, 5.0, -2.0, -2.0]).view(1, 6, 1, 1)
    online = torch.zeros(1, 6, 1, 1)
    memories = torch.tensor([2.0, 7.0, 11.0]).view(1, 3, 1, 1, 1)
    ssc = SparseSelectiveCaching(1, 1, 1, topk=1, chunk_size=2, read_scale=1.0)
    with torch.no_grad():
        ssc.connector.weight.fill_(1.0)
    result = ssc(hidden, q, keys, online, memories)
    # Descriptors are mean-pooled, so segment 2 sees completed segments
    # 0 (score mean(1, 1) = 1) and 1 (score mean(5, 5) = 5), and its own causal
    # prefix scores mean(-2) = -2 at t=4 and mean(-2, -2) = -2 at t=5.
    assert torch.equal(result.route_indices[0, 4:], torch.ones(2, 1, dtype=torch.long))
    assert torch.equal(result.route_scores[0, 4:], torch.full((2, 1), 5.0))
    expected_weights = torch.softmax(torch.tensor([[-2.0, 5.0], [-2.0, 5.0]]), dim=-1)[:, 1]
    assert torch.allclose(result.output[0, 4:, 0, 0], 7.0 * expected_weights)


def main():
    tests = [
        test_future_perturbation_does_not_change_past,
        test_only_completed_segments_are_routable,
        test_equation_16_routes_by_connector_and_segment_key_sum,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
