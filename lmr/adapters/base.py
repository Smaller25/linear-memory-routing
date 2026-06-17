# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-architecture adapter interface for the Memory-Caching segment runner.

The segment runner (:mod:`lmr.segment_runner`) is otherwise architecture-agnostic: it segments the
input, caches each segment's final recurrent state, and asks a read-out head (:mod:`lmr.readout`) to
combine the online output with the frozen checkpoints. Everything that *is* architecture-specific
lives behind an :class:`Adapter`:

- ``project``/``scan``/``finalize`` -- the per-mixer math (projection, the linear-recurrent scan
  that accepts an injected ``initial_state``, and the output gate / projection).
- ``descriptor``/``descriptor_dim`` -- how a frozen recurrent state is pooled into the routing
  descriptor consumed by GRM/SSC/MoM/AoM.
- ``blocks``/``embed``/``final_norm``/``lm_head``/``run_block`` -- the surrounding model structure.
  Mamba2 blocks are mixer-only; Gated-DeltaNet blocks add an MLP sublayer, so the boundary the
  runner delegates to is the *block*, not just the scan.

``run_mixer_with_cache`` is the one generic routine shared by every adapter: it runs the online scan
once, then re-runs the scan per cached checkpoint in *memory-only* mode (write/value path zeroed,
checkpoint injected as the initial state) to isolate that checkpoint's contribution -- exact by the
linearity of the recurrence in its initial state.
"""

from __future__ import annotations

import abc

import torch


class Adapter(abc.ABC):
    """Architecture-specific hooks for the segment runner. Stateless; instances are singletons."""

    name: str

    # ---- mixer-level (per recurrent layer) -----------------------------------------------------
    @abc.abstractmethod
    def project(self, mixer, x_in: torch.Tensor):
        """Run input projection + causal conv; return an arch ``proj`` bundle (carries the gate)."""

    @abc.abstractmethod
    def scan(self, mixer, proj, initial_state, backend: str, *, memory_only: bool):
        """One recurrent scan. Returns ``(y:[b,l,h,p], final_state)``.

        ``memory_only=True`` zeros the write/value path so the output is purely the contribution of
        ``initial_state`` (the cached checkpoint), keeping the read-out/transition path intact.
        """

    @abc.abstractmethod
    def descriptor(self, state: torch.Tensor) -> torch.Tensor:
        """Pool a frozen recurrent state into a ``[b, descriptor_dim]`` routing descriptor."""

    @abc.abstractmethod
    def descriptor_dim(self, mixer) -> int:
        """Dimension of :meth:`descriptor` for this mixer."""

    @abc.abstractmethod
    def finalize(self, mixer, proj, y: torch.Tensor, dtype) -> torch.Tensor:
        """Apply the output gate / norm and final projection. Returns ``[b, l, hidden]``."""

    # ---- block / model level -------------------------------------------------------------------
    @abc.abstractmethod
    def blocks(self, model):
        """The list of decoder blocks (one recurrent mixer each)."""

    @abc.abstractmethod
    def mixer_of(self, block):
        """The recurrent mixer module inside a block (mamba2: ``block.mixer``; gdn: ``block.attn``)."""

    @abc.abstractmethod
    def embed(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding lookup."""

    @abc.abstractmethod
    def final_norm(self, model, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm before the LM head."""

    @abc.abstractmethod
    def vanilla_hidden(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """The model's own (non-segmented) post-final-norm hidden states -- the vanilla baseline."""

    @abc.abstractmethod
    def lm_head(self, model):
        """The LM head module."""

    @abc.abstractmethod
    def run_block(self, block, hidden, cached_states, readout, backend: str):
        """Run one decoder block over a segment. Returns ``(hidden, final_state, aux)``.

        Wraps the block's norm/residual structure (and, for archs that have one, the MLP sublayer,
        which is recurrence-free and runs per-segment unchanged) around :func:`run_mixer_with_cache`.
        """


def run_mixer_with_cache(adapter: Adapter, mixer, x_in, cached_states, readout, backend="naive"):
    """Run one recurrent mixer over a segment, augmenting with cached checkpoints.

    ``cached_states``: list of frozen recurrent-state checkpoints from earlier segments.
    ``readout``: an :mod:`lmr.readout` head combining the online + cached contributions.
    Returns ``(out:[b,l,hidden], final_state, aux)``.
    """
    dtype = x_in.dtype
    proj = adapter.project(mixer, x_in)
    y_main, final_state = adapter.scan(mixer, proj, None, backend, memory_only=False)

    y_cached, descriptors = [], []
    for h_i in cached_states:
        y_i, _ = adapter.scan(mixer, proj, h_i, backend, memory_only=True)
        y_cached.append(y_i)
        descriptors.append(adapter.descriptor(h_i))
    # The scan computes states in fp32; cast routing descriptors to the read-out dtype (e.g. bf16
    # for large models) so the read-out heads' einsums don't hit a dtype mismatch.
    descriptors = torch.stack(descriptors, dim=1).to(x_in.dtype) if descriptors else None

    y, aux = readout(y_main, y_cached, x_in, descriptors)
    out = adapter.finalize(mixer, proj, y, dtype)
    return out, final_state, aux
