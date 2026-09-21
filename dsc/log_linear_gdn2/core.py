"""Exact Log-Linear Attention algorithms for GDN-2.

The temporal hierarchy follows the weak/base-2 Fenwick partition used by
Guo et al., "Log-Linear Attention" (arXiv:2506.04761).  The state transition
is the GDN-2 transition, rather than the scalar-beta Gated DeltaNet transition
used by the paper's original case study:

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1}
          + k_t (w_t * v_t)^T.

``log_linear_gdn2_chunkwise`` is the pretraining path.  It decomposes the
hierarchical mask into disjoint level blocks and invokes the already validated
GDN-2 chunk primitive once per level.  ``log_linear_gdn2_recurrent`` and
``log_linear_gdn2_materialized`` are independent PyTorch reference paths used
for numerical tests and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


Tensor = torch.Tensor


@dataclass(frozen=True)
class LogLinearGDN2State:
    """Final Fenwick memory produced by the recurrent reference path.

    ``memories`` has shape ``[B, H, K, V, L]``.  ``occupied`` identifies the
    levels that contain a live bucket after ``tokens_seen`` tokens.
    """

    memories: Tensor
    occupied: tuple[bool, ...]
    tokens_seen: int


@dataclass(frozen=True)
class LogLinearGDN2Result:
    """Output and small diagnostics returned by the chunkwise path."""

    output: Tensor
    num_levels: int
    padded_length: int


def required_num_levels(length: int) -> int:
    """Return the number of weak/base-2 levels needed for ``length`` tokens."""
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    return (length - 1).bit_length() + 1


def weak_level_index(target: int, source: int) -> int:
    """Fenwick/HODLR level for a causal ``(target, source)`` pair.

    The diagonal is level zero.  For a strict lower-triangular pair, the level
    is the position of the most-significant differing bit, counted from one.
    This is equivalent to the recursive weak hierarchy in the paper's code.
    """
    if source < 0 or target < 0 or source > target:
        raise ValueError(
            f"expected 0 <= source <= target, got target={target}, source={source}"
        )
    if source == target:
        return 0
    return (target ^ source).bit_length()


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    lambdas: Tensor,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g, and b must have the same [B,T,H,K] shape")
    if v.ndim != 4 or w.shape != v.shape:
        raise ValueError("v and w must have the same [B,T,H,V] shape")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k and v/w must share batch, time, and head axes")
    if lambdas.ndim != 4 or lambdas.shape[:3] != q.shape[:3]:
        raise ValueError("lambdas must have shape [B,T,H,L]")

    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    levels = required_num_levels(length)
    if lambdas.shape[-1] < levels:
        raise ValueError(
            f"sequence length {length} needs {levels} lambda levels, "
            f"but only {lambdas.shape[-1]} were supplied"
        )
    return batch, length, heads, key_dim, value_dim


def _normalize_qk(q: Tensor, k: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    # FLA's Triton l2norm accumulates the squared norm in fp32 and returns the
    # normalized vector in the input dtype.  Match it here for level zero and
    # for both independent reference implementations.
    def normalize(x: Tensor) -> Tensor:
        compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = compute * torch.rsqrt(
            compute.square().sum(dim=-1, keepdim=True) + eps
        )
        return normalized.to(dtype=x.dtype)

    return normalize(q), normalize(k)


def _apply_transition(state: Tensor, k: Tensor, g: Tensor, b: Tensor) -> Tensor:
    """Apply one GDN-2 transition to states ending in ``[..., K, V]``."""
    decayed = state * torch.exp(g).unsqueeze(-1)
    erase_projection = torch.einsum("bhk,bhkv->bhv", b * k, decayed)
    return decayed - k.unsqueeze(-1) * erase_projection.unsqueeze(-2)


def _apply_transition_levels(
    states: Tensor,
    k: Tensor,
    g: Tensor,
    b: Tensor,
) -> Tensor:
    """Apply one transition to ``[B,H,K,V,L]`` hierarchical states."""
    decayed = states * torch.exp(g).unsqueeze(-1).unsqueeze(-1)
    erase_projection = torch.einsum("bhk,bhkvl->bhvl", b * k, decayed)
    return decayed - k.unsqueeze(-1).unsqueeze(-1) * erase_projection.unsqueeze(2)


def _diagonal_output(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    *,
    scale: float,
    eps: float,
) -> Tensor:
    q_norm, k_norm = _normalize_qk(q, k, eps)
    compute_q = (
        q_norm.float() if q_norm.dtype in (torch.float16, torch.bfloat16) else q_norm
    )
    compute_k = (
        k_norm.float() if k_norm.dtype in (torch.float16, torch.bfloat16) else k_norm
    )
    score = (compute_q * compute_k).sum(dim=-1, keepdim=True) * scale
    return (score * (w * v)).to(dtype=v.dtype)


def _pad_time(x: Tensor, padded_length: int) -> Tensor:
    if x.shape[1] == padded_length:
        return x
    padding_shape = (x.shape[0], padded_length - x.shape[1], *x.shape[2:])
    return torch.cat((x, x.new_zeros(padding_shape)), dim=1)


def _direct_low_level_compute(
    q_pad: Tensor,
    k_pad: Tensor,
    v_pad: Tensor,
    g_pad: Tensor,
    b_pad: Tensor,
    w_pad: Tensor,
    level: int,
    padded_length: int,
    batch: int,
    *,
    scale: float,
    eps: float,
    max_blocks_per_subbatch: int = 64,
) -> Tensor:
    """Direct PyTorch GDN-2 transition for levels that overflow the Triton grid Z limit.

    chunk_gdn2's primitive has chunk_size=64 hardcoded; its chunk_local_cumsum_vector_kernel
    uses grid Z = B*H*num_blocks, capped by CUDA at 65535. Low levels have many tiny
    blocks and overflow this. This helper vectorizes the per-block GDN-2 transition
    in PyTorch with sub-batching to bound peak memory.
    """
    block_length = 1 << level
    half = block_length >> 1
    num_blocks = padded_length // block_length
    heads = q_pad.shape[2]
    key_dim = q_pad.shape[3]
    value_dim = v_pad.shape[3]

    def blockify(x: Tensor) -> Tensor:
        return x.reshape(batch * num_blocks, block_length, *x.shape[2:])

    q_block, k_block, v_block, g_block, b_block, w_block = (
        blockify(x) for x in (q_pad, k_pad, v_pad, g_pad, b_pad, w_pad)
    )
    q_n, k_n = _normalize_qk(q_block, k_block, eps)

    in_dtype = q_n.dtype
    compute_dtype = (
        torch.float32 if in_dtype in (torch.float16, torch.bfloat16) else in_dtype
    )
    q_c, k_c, v_c, g_c, b_c, w_c = (
        x.to(dtype=compute_dtype) for x in (q_n, k_n, v_block, g_block, b_block, w_block)
    )

    total_blocks = q_c.shape[0]
    full_output = q_c.new_zeros(
        (total_blocks, block_length, heads, value_dim), dtype=in_dtype
    )

    for start in range(0, total_blocks, max_blocks_per_subbatch):
        end = min(start + max_blocks_per_subbatch, total_blocks)
        qs = q_c[start:end]
        ks = k_c[start:end]
        vs = v_c[start:end]
        gs = g_c[start:end]
        bs = b_c[start:end]
        ws = w_c[start:end]
        n = end - start

        state = qs.new_zeros((n, heads, key_dim, value_dim))
        for t in range(block_length):
            if t > 0:
                decayed = state * torch.exp(gs[:, t]).unsqueeze(-1)
                erase = torch.einsum("nhk,nhkv->nhv", bs[:, t] * ks[:, t], decayed)
                state = decayed - ks[:, t].unsqueeze(-1) * erase.unsqueeze(-2)
            if t < half:
                state = state + ks[:, t].unsqueeze(-1) * (ws[:, t] * vs[:, t]).unsqueeze(-2)
            if t >= half:
                read = torch.einsum("nhk,nhkv->nhv", qs[:, t], state)
                full_output[start:end, t] = (read * scale).to(in_dtype)

    return full_output.reshape(batch, padded_length, heads, value_dim)


def log_linear_gdn2_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    lambdas: Tensor,
    *,
    chunk_gdn2_fn: Callable | None = None,
    scale: float | None = None,
    l2norm_eps: float = 1e-6,
    checkpoint_levels: bool = False,
) -> LogLinearGDN2Result:
    """Compute exact weak Log-Linear GDN-2 with repeated GDN-2 primitives.

    At level ``ell >= 1``, each block of length ``2**ell`` is an independent
    GDN-2 sequence.  Writes are enabled only in its left half and outputs are
    kept only in its right half.  These rectangles are disjoint across levels
    and, together with the diagonal, partition every causal source/target pair.

    Arbitrary positive lengths are supported by right-padding to a power of
    two and trimming the result.  Padding is an identity transition and cannot
    influence any real-token output.
    """
    batch, length, heads, key_dim, value_dim = _validate_inputs(
        q, k, v, g, b, w, lambdas
    )
    del value_dim
    if scale is None:
        scale = key_dim**-0.5
    if chunk_gdn2_fn is None:
        from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2 as chunk_gdn2_fn

    num_levels = required_num_levels(length)
    padded_length = 1 << (length - 1).bit_length()
    q_pad, k_pad, v_pad, g_pad, b_pad, w_pad, lambda_pad = (
        _pad_time(x, padded_length) for x in (q, k, v, g, b, w, lambdas)
    )

    diagonal = _diagonal_output(
        q_pad,
        k_pad,
        v_pad,
        w_pad,
        scale=scale,
        eps=l2norm_eps,
    )
    output = diagonal * lambda_pad[..., 0].unsqueeze(-1).to(diagonal.dtype)

    # chunk_gdn2's chunk_local_cumsum_vector_kernel uses grid Z = B*H*num_blocks,
    # capped by CUDA at 65535. Low levels have many tiny blocks and overflow this
    # (seq=4096, B=8, H=16: level 1 -> 262144, level 2 -> 131072, level 3 -> 65536).
    # For those levels we compute the GDN-2 transition directly in PyTorch.
    MIN_BLOCK = 64
    MAX_GRID_Z = 65535

    for level in range(1, num_levels):
        block_length = 1 << level
        half = block_length >> 1
        num_blocks = padded_length // block_length

        use_direct = (batch * num_blocks * heads) > MAX_GRID_Z

        if use_direct:
            def run_direct_level(
                q_arg: Tensor,
                k_arg: Tensor,
                v_arg: Tensor,
                g_arg: Tensor,
                b_arg: Tensor,
                w_arg: Tensor,
                _level: int = level,
                _padded_length: int = padded_length,
                _batch: int = batch,
                _scale: float = scale,
                _eps: float = l2norm_eps,
            ) -> Tensor:
                return _direct_low_level_compute(
                    q_arg, k_arg, v_arg, g_arg, b_arg, w_arg,
                    _level, _padded_length, _batch,
                    scale=_scale, eps=_eps,
                )

            if checkpoint_levels and torch.is_grad_enabled():
                level_output = checkpoint(
                    run_direct_level,
                    q_pad, k_pad, v_pad, g_pad, b_pad, w_pad,
                    use_reentrant=False,
                )
            else:
                level_output = run_direct_level(
                    q_pad, k_pad, v_pad, g_pad, b_pad, w_pad,
                )
        else:
            effective_block = max(MIN_BLOCK, block_length)
            pad_amount = effective_block - block_length

            def as_blocks(x: Tensor) -> Tensor:
                blocked = x.reshape(
                    x.shape[0] * num_blocks,
                    block_length,
                    *x.shape[2:],
                )
                if pad_amount == 0:
                    return blocked
                pad_spec = (0, 0) * (blocked.ndim - 2) + (0, pad_amount)
                return torch.nn.functional.pad(blocked, pad_spec)

            q_block, k_block, v_block, g_block, b_block, w_block = (
                as_blocks(x) for x in (q_pad, k_pad, v_pad, g_pad, b_pad, w_pad)
            )
            source_mask = w_block.new_zeros((1, effective_block, 1, 1))
            source_mask[:, :half] = 1

            def run_level(
                q_arg: Tensor,
                k_arg: Tensor,
                v_arg: Tensor,
                g_arg: Tensor,
                b_arg: Tensor,
                w_arg: Tensor,
                source_mask_arg: Tensor = source_mask,
                chunk_fn: Callable = chunk_gdn2_fn,
                level_scale: float = scale,
                block_length_arg: int = block_length,
            ) -> Tensor:
                level_result, _ = chunk_fn(
                    q=q_arg.contiguous(),
                    k=k_arg.contiguous(),
                    v=v_arg.contiguous(),
                    g=g_arg.contiguous(),
                    b=b_arg.contiguous(),
                    w=(w_arg * source_mask_arg).contiguous(),
                    scale=level_scale,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    cu_seqlens=None,
                )
                if pad_amount > 0:
                    level_result = level_result[:, :block_length_arg]
                return level_result

            if checkpoint_levels and torch.is_grad_enabled():
                level_output = checkpoint(
                    run_level,
                    q_block,
                    k_block,
                    v_block,
                    g_block,
                    b_block,
                    w_block,
                    use_reentrant=False,
                )
            else:
                level_output = run_level(
                    q_block,
                    k_block,
                    v_block,
                    g_block,
                    b_block,
                    w_block,
                )
            level_output = level_output.reshape(
                q_pad.shape[0],
                padded_length,
                q_pad.shape[2],
                v_pad.shape[-1],
            )
        query_mask = level_output.new_zeros((1, block_length, 1, 1))
        query_mask[:, half:] = 1
        query_mask = query_mask.repeat(1, num_blocks, 1, 1).reshape(
            1, padded_length, 1, 1
        )
        level_lambda = lambda_pad[..., level].unsqueeze(-1).to(level_output.dtype)
        output = output + level_output * query_mask * level_lambda

    return LogLinearGDN2Result(
        output=output[:, :length],
        num_levels=num_levels,
        padded_length=padded_length,
    )


def log_linear_gdn2_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    lambdas: Tensor,
    *,
    scale: float | None = None,
    l2norm_eps: float = 1e-6,
) -> tuple[Tensor, LogLinearGDN2State]:
    """Token-recurrent PyTorch reference with O(T log T) work and O(log T) state."""
    batch, length, heads, key_dim, value_dim = _validate_inputs(
        q, k, v, g, b, w, lambdas
    )
    if scale is None:
        scale = key_dim**-0.5
    q, k = _normalize_qk(q, k, l2norm_eps)
    compute_dtype = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    )
    q_c, k_c, v_c, g_c, b_c, w_c = (
        x.to(dtype=compute_dtype) for x in (q, k, v, g, b, w)
    )
    lambda_c = lambdas.to(dtype=compute_dtype)

    num_levels = required_num_levels(length)
    states: list[Tensor | None] = [None] * num_levels
    outputs: list[Tensor] = []
    zero_state = q_c.new_zeros((batch, heads, key_dim, value_dim))

    for token in range(length):
        # Binary carry before insertion is the paper's weak Fenwick cascade.
        carry = states[0]
        states[0] = None
        level = 1
        while carry is not None:
            if level >= num_levels:
                raise RuntimeError("insufficient Fenwick levels")
            if states[level] is None:
                states[level] = carry
                carry = None
            else:
                carry = states[level] + carry
                states[level] = None
                level += 1

        stacked = torch.stack(
            [state if state is not None else zero_state for state in states],
            dim=-1,
        )
        stacked = _apply_transition_levels(
            stacked,
            k_c[:, token],
            g_c[:, token],
            b_c[:, token],
        )
        write = k_c[:, token].unsqueeze(-1) * (w_c[:, token] * v_c[:, token]).unsqueeze(
            -2
        )
        stacked = torch.cat((write.unsqueeze(-1), stacked[..., 1:]), dim=-1)

        occupied = [True]
        for old_state in states[1:]:
            occupied.append(old_state is not None)
        states = [
            stacked[..., index] if occupied[index] else None
            for index in range(num_levels)
        ]

        weighted_state = torch.einsum(
            "bhkvl,bhl->bhkv",
            stacked,
            lambda_c[:, token, :, :num_levels],
        )
        token_output = torch.einsum(
            "bhk,bhkv->bhv",
            q_c[:, token],
            weighted_state,
        )
        outputs.append(token_output * scale)

    final_occupied = tuple(state is not None for state in states)
    final_memories = torch.stack(
        [state if state is not None else zero_state for state in states],
        dim=-1,
    )
    output = torch.stack(outputs, dim=1).to(dtype=v.dtype)
    return output, LogLinearGDN2State(
        memories=final_memories,
        occupied=final_occupied,
        tokens_seen=length,
    )


def log_linear_gdn2_materialized(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    lambdas: Tensor,
    *,
    scale: float | None = None,
    l2norm_eps: float = 1e-6,
) -> Tensor:
    """Independent O(T^3) small-shape oracle that materializes every path."""
    batch, length, heads, key_dim, value_dim = _validate_inputs(
        q, k, v, g, b, w, lambdas
    )
    del batch, heads, value_dim
    if scale is None:
        scale = key_dim**-0.5
    q, k = _normalize_qk(q, k, l2norm_eps)
    compute_dtype = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    )
    q_c, k_c, v_c, g_c, b_c, w_c = (
        x.to(dtype=compute_dtype) for x in (q, k, v, g, b, w)
    )
    lambda_c = lambdas.to(dtype=compute_dtype)

    outputs: list[Tensor] = []
    for target in range(length):
        target_output = torch.zeros_like(v_c[:, target])
        for source in range(target + 1):
            contribution = k_c[:, source].unsqueeze(-1) * (
                w_c[:, source] * v_c[:, source]
            ).unsqueeze(-2)
            for step in range(source + 1, target + 1):
                contribution = _apply_transition(
                    contribution,
                    k_c[:, step],
                    g_c[:, step],
                    b_c[:, step],
                )
            level = weak_level_index(target, source)
            read = torch.einsum(
                "bhk,bhkv->bhv",
                q_c[:, target],
                contribution,
            )
            target_output = (
                target_output + lambda_c[:, target, :, level].unsqueeze(-1) * read
            )
        outputs.append(target_output * scale)
    return torch.stack(outputs, dim=1).to(dtype=v.dtype)
