# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-architecture adapters for the Memory-Caching segment runner.

``get_adapter("mamba2" | "gdn")`` returns the singleton adapter used by
:func:`lmr.segment_runner.run_segmented_lm` to run a frozen backbone segment-by-segment.
"""

from __future__ import annotations

from lmr.adapters.base import Adapter, run_mixer_with_cache
from lmr.adapters.gdn import GDNAdapter
from lmr.adapters.mamba2 import Mamba2Adapter

ADAPTERS: dict[str, Adapter] = {
    "mamba2": Mamba2Adapter(),
    "gdn": GDNAdapter(),
}


def get_adapter(arch: str) -> Adapter:
    try:
        return ADAPTERS[arch]
    except KeyError:
        raise ValueError(f"unknown arch: {arch!r} (expected one of {sorted(ADAPTERS)})") from None


def descriptor_dim_for(model, arch: str) -> int:
    """The read-out descriptor dimension for ``model`` under ``arch`` (for building heads)."""
    adapter = get_adapter(arch)
    first = adapter.blocks(model)[0]
    return adapter.descriptor_dim(adapter.mixer_of(first))


__all__ = ["Adapter", "run_mixer_with_cache", "get_adapter", "descriptor_dim_for", "ADAPTERS",
           "Mamba2Adapter", "GDNAdapter"]
