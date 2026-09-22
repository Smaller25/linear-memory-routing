#!/usr/bin/env python3
"""CPU checks for the chained compressor mode, with a fake recurrence.

The real scan is a Triton kernel and needs a GPU, so these run against a
stand-in that has the one property under test: a state carried from one call
to the next. What is being checked is the *plumbing* — that chaining actually
threads the state, that independent does not, and that the two agree exactly
when there is only one segment, which is the boundary where they must.

That last one matters because an implementation that silently ignored
`initial_state` would still produce output, still train, and still report a
routing number.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (W, os.path.join(W, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location(
    "_gssc", os.path.join(W, "dsc/mc_gdn2/ssc.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["_gssc"] = mod
spec.loader.exec_module(mod)


def fake_scan(*, q, k, v, g, b, w, initial_state=None, **kw):
    """A cumulative-sum recurrence: out[t] = state + sum_{s<=t} v[s]."""
    B, T, H, V = v.shape
    s0 = initial_state if initial_state is not None else v.new_zeros(B, H, V, V)
    running = v.cumsum(dim=1)
    carried = s0[:, :, 0, :].unsqueeze(1)
    out = running + carried
    final = s0.clone()
    final[:, :, 0, :] = s0[:, :, 0, :] + v.sum(dim=1)
    return out, final


B, T, H, K, V, CH = 2, 12, 2, 4, 4, 4
torch.manual_seed(0)
mk = lambda *s: torch.randn(*s)
q = k = mk(B, T, H, K)
v = mk(B, T, H, V)
g = b = mk(B, T, H, K)
w = mk(B, T, H, V)

ind_o, ind_m = mod._segment_gdn2_batched(
    q, k, v, g, b, w, chunk_size=CH, chunk_gdn2_fn=fake_scan)
chn_o, chn_m = mod._segment_gdn2_chained(
    q, k, v, g, b, w, chunk_size=CH, chunk_gdn2_fn=fake_scan)
assert ind_o.shape == chn_o.shape == (B, T, H, V)
assert ind_m.shape == chn_m.shape == (B, T // CH, H, K, V)
print(f"[c] shapes agree: out {tuple(ind_o.shape)}, memories {tuple(ind_m.shape)}")

# segment 0 is identical: nothing has been carried into it yet
assert torch.allclose(ind_o[:, :CH], chn_o[:, :CH], atol=1e-6)
print("[c] the first segment is identical under both modes")

# later segments are NOT, or chaining is a no-op that would still train
d = (ind_o[:, CH:] - chn_o[:, CH:]).abs().max().item()
assert d > 1e-3, "chaining changed nothing — initial_state is being ignored"
print(f"[c] later segments differ by {d:.3f} — the state is really threaded")

# one segment: the two modes must agree exactly
o1, m1 = mod._segment_gdn2_batched(
    q, k, v, g, b, w, chunk_size=T, chunk_gdn2_fn=fake_scan)
o2, m2 = mod._segment_gdn2_chained(
    q, k, v, g, b, w, chunk_size=T, chunk_gdn2_fn=fake_scan)
assert torch.equal(o1, o2) and torch.equal(m1, m2)
print("[c] with a single segment the two modes are bit-identical")

# chained must equal an unfragmented scan of the whole sequence
full_o, _ = fake_scan(q=q, k=k, v=v, g=g, b=b, w=w, initial_state=None)
assert torch.allclose(chn_o, full_o, atol=1e-5), \
    "chained must reproduce the unfragmented recurrence"
print("[c] chained reproduces a single scan over the whole sequence")

# independent must NOT, which is the cost being measured
gap = (ind_o - full_o).abs().max().item()
assert gap > 1e-3
print(f"[c] independent departs from it by {gap:.3f} — that gap is the "
      "fragmentation")

# the dispatcher rejects anything else rather than falling back silently
class S:
    chunk_size = CH
for bad in ("checkpoint", "", None):
    try:
        mod.gdn2_ssc_forward(S(), mk(B, T, 8), q, k, v, g, b, w,
                             checkpoint_mode=bad, chunk_gdn2_fn=fake_scan)
    except ValueError:
        pass
    else:
        raise AssertionError(f"checkpoint_mode={bad!r} must be rejected")
print("[c] unknown modes are rejected, not silently treated as independent")

print("ALL CHAINED MODE CHECKS PASS")
