"""Equation, shape, gradient, and adapter tests for paper-faithful SSC."""

import torch
from torch import nn

from dsc.mc_baseline.mc_ssc import SparseSelectiveCaching, segment_key_sums
from dsc.mc_baseline.diagnostics import routing_metrics
from dsc.mc_baseline.segment_checkpoint import scan_segments
from dsc.mc_gdn1 import GDN1SSC, MemoryCachingGDN1Layer, gdn1_ssc_forward
from dsc.mc_gdn2 import GDN2SSC, MemoryCachingGDN2Layer, gdn2_ssc_forward


def make_core_inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    b, t, d, h, k, v, c = 2, 7, 6, 2, 3, 4, 2
    hidden = torch.randn(b, t, d, generator=generator)
    q = torch.randn(b, t, h, k, generator=generator)
    keys = torch.randn(b, t, h, k, generator=generator)
    online = torch.randn(b, t, h, v, generator=generator)
    memories = torch.randn(b, 4, h, k, v, generator=generator)
    return hidden, q, keys, online, memories, c


def test_segment_key_sums_use_keys_not_states():
    # segment_key_sums mean-pools (not raw-sums) so the descriptor magnitude
    # stays ~1 and softmax(gate_logits) does not collapse to one-hot at init.
    keys = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1, 1)
    got = segment_key_sums(keys, 2).flatten()
    assert torch.equal(got, torch.tensor([1.5, 3.5, 5.5]))


def test_shape_and_gradient():
    hidden, q, keys, online, memories, c = make_core_inputs()
    hidden.requires_grad_(True)
    q.requires_grad_(True)
    ssc = SparseSelectiveCaching(6, 2, 3, topk=2, chunk_size=c)
    result = ssc(hidden, q, keys, online, memories)
    assert result.output.shape == online.shape
    assert result.route_indices.shape == (2, 7, 2)
    result.output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert ssc.connector.weight.grad is not None


def test_first_segment_is_unmodified_online_memory():
    hidden, q, keys, online, memories, c = make_core_inputs()
    result = SparseSelectiveCaching(6, 2, 3, topk=2, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    assert torch.equal(result.output[:, :c], online[:, :c])
    assert (result.route_indices[:, :c] == -1).all()
    assert torch.equal(result.online_weight[:, :c], torch.ones_like(result.online_weight[:, :c]))


def test_diagnostics_are_json_scalars():
    hidden, q, keys, online, memories, c = make_core_inputs()
    result = SparseSelectiveCaching(6, 2, 3, topk=2, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    metrics = routing_metrics(result)
    assert len(metrics) == 6
    assert all(isinstance(value, float) for value in metrics.values())


def test_topk_zero_is_exact_online_path():
    hidden, q, keys, online, memories, c = make_core_inputs()
    result = SparseSelectiveCaching(6, 2, 3, topk=0, chunk_size=c)(
        hidden, q, keys, online, memories
    )
    assert torch.equal(result.output, online)
    assert result.route_indices.shape[-1] == 0


def test_read_blocking_is_numerically_identical():
    hidden, q, keys, online, memories, c = make_core_inputs()
    full = SparseSelectiveCaching(6, 2, 3, topk=2, chunk_size=c, read_block_size=256)
    blocked = SparseSelectiveCaching(6, 2, 3, topk=2, chunk_size=c, read_block_size=1)
    blocked.load_state_dict(full.state_dict())
    first = full(hidden, q, keys, online, memories)
    second = blocked(hidden, q, keys, online, memories)
    assert torch.equal(first.route_indices, second.route_indices)
    assert torch.equal(first.output, second.output)


def test_segment_scan_modes():
    source = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def scan(start, stop, initial):
        state = source[start:stop].sum() + (0.0 if initial is None else initial)
        return source[start:stop].view(1, -1, 1, 1), state.view(1, 1, 1)

    _, independent = scan_segments(4, 2, scan, checkpoint_mode="independent")
    _, checkpoints = scan_segments(4, 2, scan, checkpoint_mode="checkpoint")
    assert torch.equal(independent.flatten(), torch.tensor([3.0, 7.0]))
    assert torch.equal(checkpoints.flatten(), torch.tensor([3.0, 10.0]))


def _gdn2_mock(**kwargs):
    q, v = kwargs["q"], kwargs["v"]
    state = kwargs["initial_state"]
    if state is None:
        state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = []
    for pos in range(q.shape[1]):
        state = state + torch.einsum("bhk,bhv->bhkv", kwargs["k"][:, pos], v[:, pos])
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, pos], state))
    return torch.stack(outputs, dim=1), state


def _gdn1_mock(q, k, v, beta, g, initial_state=None, output_final_state=False):
    state = initial_state
    if state is None:
        state = torch.zeros(q.shape[0], q.shape[1], q.shape[3], v.shape[3])
    outputs = []
    for pos in range(q.shape[2]):
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, :, pos], v[:, :, pos])
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, pos], state))
    return torch.stack(outputs, dim=2), state


def test_gdn2_adapter():
    b, t, d, h, k, v = 1, 5, 6, 2, 3, 4
    hidden = torch.randn(b, t, d)
    tensors = [torch.randn(b, t, h, dim) for dim in (k, k, v, k, k, v)]
    result = gdn2_ssc_forward(
        GDN2SSC(d, h, k, topk=1, chunk_size=2), hidden, *tensors,
        chunk_gdn2_fn=_gdn2_mock,
    )
    assert result.output.shape == (b, t, h, v)


def test_gdn1_adapter():
    b, t, d, h, k, v = 1, 5, 6, 2, 3, 4
    hidden = torch.randn(b, t, d)
    q = torch.randn(b, h, t, k)
    keys = torch.randn_like(q)
    values = torch.randn(b, h, t, v)
    beta = torch.rand(b, h, t)
    gate = torch.randn(b, h, t)
    result = gdn1_ssc_forward(
        GDN1SSC(d, h, k, topk=1, chunk_size=2), hidden, q, keys, values, beta, gate,
        chunk_gated_delta_rule_fn=_gdn1_mock,
    )
    assert result.output.shape == (b, t, h, v)


class _NormGate(nn.Module):
    def forward(self, output, gate):
        return output


class _Conv(nn.Module):
    def forward(self, value, attention_mask=None, state=None):
        return value


class _FakeGDN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.num_heads, self.num_v_heads = 6, 2, 2
        self.head_k_dim, self.head_v_dim = 3, 4
        self.use_short_conv, self.allow_neg_eigval = False, False
        self.q_proj = nn.Linear(6, 6, bias=False)
        self.k_proj = nn.Linear(6, 6, bias=False)
        self.v_proj = nn.Linear(6, 8, bias=False)
        self.f_proj = nn.Linear(6, 6, bias=False)
        self.b_proj = nn.Linear(6, 6, bias=False)
        self.w_proj = nn.Linear(6, 8, bias=False)
        self.g_proj = nn.Linear(6, 8)
        self.A_log = nn.Parameter(torch.zeros(2))
        self.dt_bias = nn.Parameter(torch.zeros(6))
        self.o_norm = _NormGate()
        self.o_proj = nn.Linear(8, 6, bias=False)


class _FakeGDN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.num_heads, self.num_kv_heads = 6, 2, 2
        self.num_kv_groups, self.head_qk_dim, self.head_v_dim = 1, 3, 4
        self.key_dim, self.value_dim_per_group, self.value_dim = 6, 8, 8
        self.q_proj = nn.Linear(6, 6, bias=False)
        self.k_proj = nn.Linear(6, 6, bias=False)
        self.v_proj = nn.Linear(6, 8, bias=False)
        self.q_conv1d = self.k_conv1d = self.v_conv1d = _Conv()
        self.gk_proj = nn.Linear(6, 2)
        self.b_proj = nn.Linear(6, 2)
        self.g_proj = nn.Linear(6, 8)
        self.A_log = nn.Parameter(torch.zeros(2))
        self.dt_bias = nn.Parameter(torch.zeros(2))
        self.use_mamba_gate, self.use_input_gate, self.use_residual = True, False, False
        self.qk_norm, self.gate_logit_normalizer = "l2", 16
        self.fuse_norm_and_gate = True
        self.g_norm_swish_gate = _NormGate()
        self.o_proj = nn.Linear(8, 6, bias=False)


def test_full_layer_wrappers():
    hidden = torch.randn(1, 5, 6)
    gdn2 = MemoryCachingGDN2Layer(
        _FakeGDN2(), topk=1, chunk_size=2, chunk_gdn2_fn=_gdn2_mock
    )
    gdn1 = MemoryCachingGDN1Layer(
        _FakeGDN1(), topk=1, chunk_size=2, chunk_gated_delta_rule_fn=_gdn1_mock
    )
    for wrapper in (gdn1, gdn2):
        output, _, cache = wrapper(hidden)
        assert output.shape == hidden.shape
        assert cache is None
        output.square().mean().backward()
        assert wrapper.ssc.connector.weight.grad is not None


def main():
    tests = [
        test_segment_key_sums_use_keys_not_states,
        test_shape_and_gradient,
        test_first_segment_is_unmodified_online_memory,
        test_diagnostics_are_json_scalars,
        test_topk_zero_is_exact_online_path,
        test_read_blocking_is_numerically_identical,
        test_segment_scan_modes,
        test_gdn2_adapter,
        test_gdn1_adapter,
        test_full_layer_wrappers,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
