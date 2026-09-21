from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from dsc.log_linear_gdn2.core import (
    log_linear_gdn2_chunkwise,
    log_linear_gdn2_materialized,
    log_linear_gdn2_recurrent,
    required_num_levels,
    weak_level_index,
)
from dsc.log_linear_gdn2.layer import LogLinearGDN2Layer


def _inputs(length: int = 7, *, requires_grad: bool = False):
    torch.manual_seed(17)
    shape_k = (2, length, 2, 3)
    shape_v = (2, length, 2, 4)
    q = torch.randn(shape_k, dtype=torch.float64, requires_grad=requires_grad)
    k = torch.randn(shape_k, dtype=torch.float64, requires_grad=requires_grad)
    v = torch.randn(shape_v, dtype=torch.float64, requires_grad=requires_grad)
    g = (-0.2 * torch.rand(shape_k, dtype=torch.float64)).requires_grad_(requires_grad)
    b = (0.4 * torch.rand(shape_k, dtype=torch.float64)).requires_grad_(requires_grad)
    w = torch.sigmoid(torch.randn(shape_v, dtype=torch.float64)).requires_grad_(
        requires_grad
    )
    lambdas = torch.nn.functional.softplus(
        torch.randn(
            2,
            length,
            2,
            required_num_levels(length),
            dtype=torch.float64,
        )
    ).requires_grad_(requires_grad)
    return q, k, v, g, b, w, lambdas


def _fake_chunk_gdn2(
    *,
    q,
    k,
    v,
    g,
    b,
    w,
    scale,
    output_final_state,
    use_qk_l2norm_in_kernel,
    **kwargs,
):
    assert not output_final_state
    assert use_qk_l2norm_in_kernel
    levels = torch.ones(
        (*q.shape[:3], required_num_levels(q.shape[1])),
        dtype=q.dtype,
        device=q.device,
    )
    output, state = log_linear_gdn2_recurrent(
        q,
        k,
        v,
        g,
        b,
        w,
        levels,
        scale=scale,
    )
    # With every lambda equal to one, the hierarchy sums to one ordinary
    # recurrent state.  That is exactly the primitive required by the
    # level-decomposition test double.
    final = state.memories.sum(dim=-1)
    return output, final


def test_level_index_matches_weak_hierarchy():
    expected = [
        [0],
        [1, 0],
        [2, 2, 0],
        [2, 2, 1, 0],
        [3, 3, 3, 3, 0],
        [3, 3, 3, 3, 1, 0],
        [3, 3, 3, 3, 2, 2, 0],
        [3, 3, 3, 3, 2, 2, 1, 0],
    ]
    for target, row in enumerate(expected):
        assert [weak_level_index(target, source) for source in range(target + 1)] == row


@pytest.mark.parametrize("length", [1, 2, 3, 7, 8])
def test_recurrent_matches_independent_materialized_oracle(length: int):
    inputs = _inputs(length)
    recurrent, _ = log_linear_gdn2_recurrent(*inputs)
    materialized = log_linear_gdn2_materialized(*inputs)
    torch.testing.assert_close(recurrent, materialized, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("length", [3, 7, 8])
def test_chunkwise_decomposition_matches_recurrent_for_arbitrary_lengths(length: int):
    inputs = _inputs(length)
    chunkwise = log_linear_gdn2_chunkwise(
        *inputs,
        chunk_gdn2_fn=_fake_chunk_gdn2,
    )
    recurrent, _ = log_linear_gdn2_recurrent(*inputs)
    assert chunkwise.num_levels == required_num_levels(length)
    assert chunkwise.padded_length == 1 << (length - 1).bit_length()
    torch.testing.assert_close(
        chunkwise.output,
        recurrent,
        rtol=1e-10,
        atol=1e-10,
    )


def test_all_one_lambdas_collapse_to_vanilla_gdn2():
    q, k, v, g, b, w, _ = _inputs(8)
    lambdas = torch.ones(
        (*q.shape[:3], required_num_levels(q.shape[1])),
        dtype=q.dtype,
    )
    log_linear, _ = log_linear_gdn2_recurrent(q, k, v, g, b, w, lambdas)

    # A single occupied level at every step is an ordinary GDN-2 recurrence.
    q_norm = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k_norm = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=q.dtype)
    vanilla = []
    for token in range(q.shape[1]):
        decayed = state * torch.exp(g[:, token]).unsqueeze(-1)
        erase = torch.einsum(
            "bhk,bhkv->bhv",
            b[:, token] * k_norm[:, token],
            decayed,
        )
        state = decayed - k_norm[:, token].unsqueeze(-1) * erase.unsqueeze(-2)
        state = state + k_norm[:, token].unsqueeze(-1) * (
            w[:, token] * v[:, token]
        ).unsqueeze(-2)
        vanilla.append(
            torch.einsum("bhk,bhkv->bhv", q_norm[:, token], state)
            / math.sqrt(q.shape[-1])
        )
    vanilla = torch.stack(vanilla, dim=1)
    torch.testing.assert_close(log_linear, vanilla, rtol=1e-10, atol=1e-10)


def test_chunkwise_backward_reaches_every_input():
    inputs = _inputs(7, requires_grad=True)
    result = log_linear_gdn2_chunkwise(
        *inputs,
        chunk_gdn2_fn=_fake_chunk_gdn2,
    )
    result.output.square().mean().backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_level_checkpoint_path_matches_and_backpropagates():
    inputs = _inputs(7, requires_grad=True)
    plain = log_linear_gdn2_chunkwise(
        *inputs,
        chunk_gdn2_fn=_fake_chunk_gdn2,
    )
    checkpointed = log_linear_gdn2_chunkwise(
        *inputs,
        chunk_gdn2_fn=_fake_chunk_gdn2,
        checkpoint_levels=True,
    )
    torch.testing.assert_close(checkpointed.output, plain.output)
    checkpointed.output.sum().backward()
    assert all(tensor.grad is not None for tensor in inputs)


def test_future_tokens_do_not_change_prefix():
    inputs = list(_inputs(7))
    baseline, _ = log_linear_gdn2_recurrent(*inputs)
    for index in range(6):
        changed = inputs[index].detach().clone()
        changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 10
        modified = inputs.copy()
        modified[index] = changed
        output, _ = log_linear_gdn2_recurrent(*modified)
        torch.testing.assert_close(output[:, :5], baseline[:, :5])


class _GateNorm(nn.Module):
    def forward(self, output, gate):
        return output * torch.sigmoid(gate)


class _FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 6
        self.num_heads = 2
        self.num_v_heads = 2
        self.head_k_dim = 3
        self.head_v_dim = 4
        self.use_short_conv = False
        self.allow_neg_eigval = False
        self.q_proj = nn.Linear(6, 6, bias=False, dtype=torch.float64)
        self.k_proj = nn.Linear(6, 6, bias=False, dtype=torch.float64)
        self.v_proj = nn.Linear(6, 8, bias=False, dtype=torch.float64)
        self.f_proj = nn.Linear(6, 6, bias=False, dtype=torch.float64)
        self.b_proj = nn.Linear(6, 6, bias=False, dtype=torch.float64)
        self.w_proj = nn.Linear(6, 8, bias=False, dtype=torch.float64)
        self.g_proj = nn.Linear(6, 8, bias=False, dtype=torch.float64)
        self.o_norm = _GateNorm()
        self.o_proj = nn.Linear(8, 6, bias=False, dtype=torch.float64)
        self.A_log = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.dt_bias = nn.Parameter(torch.zeros(6, dtype=torch.float64))


def test_layer_preserves_gdn2_path_and_adds_positive_lambdas():
    torch.manual_seed(29)
    layer = LogLinearGDN2Layer(
        _FakeBase(),
        max_sequence_length=8,
        checkpoint_levels=False,
        chunk_gdn2_fn=_fake_chunk_gdn2,
    ).to(dtype=torch.float64)
    hidden = torch.randn(2, 7, 6, dtype=torch.float64, requires_grad=True)
    output, result, lambdas = layer.forward_with_diagnostics(hidden)
    assert output.shape == hidden.shape
    assert result.num_levels == 4
    assert lambdas.shape == (2, 7, 2, 4)
    assert (lambdas > 0).all()
    output.square().mean().backward()
    assert hidden.grad is not None
    assert layer.l_proj.weight.grad is not None
    assert layer.L.grad is not None
