#!/usr/bin/env python3
"""CPU-only checks for the segment soft prompt (track 2 of report 0027).

Runs without a GPU, a checkpoint or Triton. The properties that matter here
are layout arithmetic and gradient scope: a soft prompt that lands one
position off, or that quietly unfreezes the 33M-parameter embedding table,
would still train and still produce a number.
"""
import importlib.util, os, sys

import torch
import torch.nn as nn

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "dsc"))
spec = importlib.util.spec_from_file_location(
    "_sp", os.path.join(W, "dsc/mc_baseline/mc_ssc_soft_prompt.py"))
sp = importlib.util.module_from_spec(spec)
sys.modules["_sp"] = sp          # dataclass() resolves cls.__module__ here
spec.loader.exec_module(sp)

# --- plan arithmetic ---
T, CH, P, S = 300, 128, 4, 0          # 3 segments, last one short (44 tokens)
pm = sp.plan_expansion(T, CH, P, S)
assert pm.chunk_out == CH + P
# the last segment holds only 44 real tokens and is NOT padded out
assert pm.length_out == 2 * (CH + P) + P + 44, pm.length_out
assert int(pm.slot_mask.sum()) == 3 * P, int(pm.slot_mask.sum())
assert int(pm.slot_mask.sum()) + T == pm.length_out
print(f"[sp] plan: T {T} -> {pm.length_out}, chunk {CH} -> {pm.chunk_out}, "
      f"{int(pm.slot_mask.sum())} slots")

# every original token keeps its segment
for pos in (0, CH - 1, CH, 2 * CH, T - 1):
    assert pm.segment_of_old(pos) == pos // CH, (pos, pm.segment_of_old(pos))
print("[sp] segment membership preserved for every original position")

# the map is injective and never lands on a slot
assert len(set(pm.new_of_old.tolist())) == T
assert not pm.slot_mask[pm.new_of_old].any()
print("[sp] position map is injective and avoids all slots")

# --- id expansion ---
ids = torch.arange(1, T + 1)[None]
out = sp.expand_ids(ids, pm, fill_id=0)
assert out.shape == (1, pm.length_out)
assert torch.equal(out[0, pm.new_of_old], ids[0])
assert (out[0, pm.slot_mask] == 0).all()
print("[sp] expand_ids places every real token and parks filler in slots")

# --- p=0 is the only exact control ---
D, V = 16, 50
wte = nn.Embedding(V, D)
torch.manual_seed(0)
m0 = sp.SoftPromptEmbedding(wte, 0, 0)
base = torch.randint(0, V, (2, 9))
assert torch.equal(m0(base), wte(base))
print("[sp] p=0 reproduces the embedding exactly (the only exact control)")

# --- warm start copies real vocabulary rows ---
ws = [7, 11]
m1 = sp.SoftPromptEmbedding(wte, 4, 0, warm_start_ids=ws)
want = wte.weight.detach()[torch.tensor([7, 11, 7, 11])]
assert torch.allclose(m1.prefix.float(), want.float(), atol=1e-6)
print("[sp] warm start tiles the requested vocabulary rows")

# --- slots actually receive the trained vectors, in every segment ---
pm2 = sp.plan_expansion(8, 4, 2, 0)          # 2 segments of 4 -> chunk_out 6
m2 = sp.SoftPromptEmbedding(wte, 2, 0)
with torch.no_grad():
    m2.prefix.copy_(torch.arange(2 * D, dtype=torch.float).view(2, D))
m2.set_plan(pm2)
ids2 = sp.expand_ids(torch.arange(1, 9)[None], pm2)
emb = m2(ids2)
for seg in range(2):
    for j in range(2):
        got = emb[0, seg * 6 + j]
        assert torch.allclose(got, m2.prefix[j].to(got.dtype), atol=1e-6), (seg, j)
print("[sp] both segments receive the same trained prefix vectors")

# real tokens are untouched
assert torch.allclose(emb[0, pm2.new_of_old], wte(torch.arange(1, 9)), atol=1e-6)
print("[sp] real token embeddings pass through unchanged")

# --- forgetting set_plan must fail loudly, not silently ignore training ---
m3 = sp.SoftPromptEmbedding(wte, 2, 0)
try:
    m3(base)
except RuntimeError as e:
    print(f"[sp] missing slot_mask rejected: {str(e)[:52]}")
else:
    raise AssertionError("a missing plan must raise")

# --- wrong-length mask rejected ---
m3.set_plan(pm2)
try:
    m3(torch.randint(0, V, (1, 7)))
except ValueError as e:
    print(f"[sp] mismatched T rejected: {str(e)[:52]}")
else:
    raise AssertionError("a length mismatch must raise")

# --- suffix / sandwich layout ---
pm3 = sp.plan_expansion(8, 4, 2, 2)
assert pm3.chunk_out == 8 and pm3.length_out == 16, pm3.length_out
assert int(pm3.slot_mask.sum()) == 8
m4 = sp.SoftPromptEmbedding(wte, 2, 2)
m4.set_plan(pm3)
e4 = m4(sp.expand_ids(torch.arange(1, 9)[None], pm3))
# layout per segment: [pre pre tok tok tok tok suf suf]
assert torch.allclose(e4[0, 0], m4.prefix[0].to(e4.dtype), atol=1e-6)
assert torch.allclose(e4[0, 6], m4.suffix[0].to(e4.dtype), atol=1e-6)
print("[sp] sandwich layout puts prefix at the head and suffix at the tail")

# --- attach freezes the backbone and reports the budget ---
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(V, D)})
        self.other = nn.Linear(D, D)
tm = Tiny()
s2 = sp.attach_soft_prompt(tm, 8, 0, warm_start_ids=[3])
assert not tm.other.weight.requires_grad
assert s2.prefix.requires_grad and not s2.wte.weight.requires_grad
assert sum(p.numel() for p in tm.parameters() if p.requires_grad) == 8 * D, \
    "only the prompt may be trainable"
ps = sp.prompt_state_dict(s2)
assert set(ps) == {"prefix"} and ps["prefix"].shape == (8, D)
print("[sp] " + sp.trainable_report(s2))
print(f"[sp] prompt_state_dict carries only {list(ps)} — no backbone")
try:
    sp.attach_soft_prompt(tm, 4)
except RuntimeError as e:
    print(f"[sp] double attach rejected: {str(e)[:40]}")
else:
    raise AssertionError("double attach must raise")

# --- gradients reach the prompt and nothing else ---
tm.transformer.wte.set_plan(pm3)
e5 = tm.transformer.wte(sp.expand_ids(torch.arange(1, 9)[None], pm3))
e5.sum().backward()
assert s2.prefix.grad is not None and s2.prefix.grad.abs().sum() > 0
assert s2.wte.weight.grad is None, "the backbone embedding must get no gradient"
print("[sp] gradient reaches the prompt; the embedding table gets none")
# --- the perplexity gate's target masking -----------------------------------
# A prompt makes the LM objective ill-defined at segment boundaries: the
# position that used to predict the next token now predicts a trained vector
# with no token id. Scoring those against an arbitrary target would move the
# ppl number for a reason that has nothing to do with damage to the backbone,
# which is exactly what the gate is supposed to detect.
for (T4, C4, P4, S4) in ((12, 4, 2, 0), (12, 4, 1, 1), (10, 4, 3, 0)):
    pm4 = sp.plan_expansion(T4, C4, P4, S4)
    src4 = pm4.new_of_old
    keep = (src4[1:] - src4[:-1]) == 1
    # every dropped i must be the last real token of its segment
    dropped = [i for i in range(T4 - 1) if not keep[i]]
    want = [i for i in range(T4 - 1) if (i + 1) % C4 == 0]
    assert dropped == want, (T4, C4, P4, S4, dropped, want)
    # and the kept count is T-1 minus one per interior boundary
    assert int(keep.sum()) == (T4 - 1) - len(want)
print("[sp] ppl masking drops exactly the segment-final targets, "
      "for prefix / sandwich / uneven layouts")

# the masking must be identical with and without a prompt, or the two ppl
# numbers are not comparable and the gate compares apples to oranges
pm5 = sp.plan_expansion(12, 4, 0, 0)
src5 = pm5.new_of_old
assert bool(((src5[1:] - src5[:-1]) == 1).all()), \
    "with p=0 nothing may be dropped"
print("[sp] p=0 drops nothing, so the gate's baseline uses every target")

# --- per-row masks, which is how the evaluation batches ------------------------
# Rows of different lengths are right-padded into one batch, so each row has
# its own layout and its own slot count. A single 1-D mask reused across rows
# would drop one row's prefix vectors into another row's text, and the run
# would still produce a routing number.
m6 = sp.SoftPromptEmbedding(wte, 2, 0)
with torch.no_grad():
    m6.prefix.copy_(torch.arange(2 * D, dtype=torch.float).view(2, D) + 100)
lens = [8, 4]                                   # 2 segments and 1 segment
pms = [sp.plan_expansion(L, 4, 2, 0) for L in lens]
T6 = max(p.length_out for p in pms)
mask = torch.zeros(2, T6, dtype=torch.bool)
ids6 = torch.zeros(2, T6, dtype=torch.long)
for i, pm_i in enumerate(pms):
    mask[i, :pm_i.length_out] = pm_i.slot_mask
    ids6[i, :pm_i.length_out] = sp.expand_ids(
        torch.arange(1, lens[i] + 1)[None], pm_i)[0]
m6.set_plan(mask)
e6 = m6(ids6)
assert int(mask[0].sum()) == 4 and int(mask[1].sum()) == 2
for i, pm_i in enumerate(pms):
    for j, pos in enumerate(torch.nonzero(pm_i.slot_mask).flatten().tolist()):
        want = m6.prefix[j % 2].to(e6.dtype)
        assert torch.allclose(e6[i, pos], want, atol=1e-6), (i, pos)
    # and this row's real tokens are untouched
    assert torch.allclose(e6[i, pm_i.new_of_old],
                          wte(torch.arange(1, lens[i] + 1)), atol=1e-6)
print("[sp] per-row masks: rows with 4 and 2 slots each get their own layout")

# a shorter mask is padded with non-slots; a longer one with live slots is an error
m6.set_plan(mask[:, :T6 - 1])
m6(ids6)
print("[sp] a short mask is padded with non-slots")
bad = torch.zeros(2, T6 + 2, dtype=torch.bool); bad[:, -1] = True
m6.set_plan(bad)
try:
    m6(ids6)
except ValueError as e:
    print(f"[sp] slots past the batch rejected: {str(e)[:46]}")
else:
    raise AssertionError("slots beyond T must raise")

print("ALL SOFT PROMPT CHECKS PASS")
