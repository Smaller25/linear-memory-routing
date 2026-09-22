"""CPU micro-test: train-time and inject-time scoring must be the same
function, and layer indexing must map router_L{i}.pt onto MC layer i."""
import importlib.util, json, os, sys, types
import torch, torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    sys.path.insert(0, p)


def load_isolated(name, path):
    """Load a module file without executing its package __init__ (avoids
    pulling in the triton kernels, which need CUDA)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# stub out the triton read kernel + mc_ssc imports the injection module needs
pkg = types.ModuleType("dsc.mc_baseline.cached_memory_read")
pkg.ssc_gather_read = lambda *a, **k: None
sys.modules["dsc.mc_baseline.cached_memory_read"] = pkg
mc_ssc = load_isolated("_mc_ssc", os.path.join(REPO, "dsc/mc_baseline/mc_ssc.py"))
shim = types.ModuleType("dsc.mc_baseline.mc_ssc")
for k in ("SSCOutput", "causal_online_key_sums", "segment_key_sums"):
    setattr(shim, k, getattr(mc_ssc, k))
sys.modules["dsc.mc_baseline.mc_ssc"] = shim

inj = load_isolated("_inj", os.path.join(REPO, "dsc/mc_baseline/mc_ssc_mlp_router.py"))
trn = load_isolated("_trn", os.path.join(REPO, "dsc/scripts/train_mlp_router.py"))

torch.manual_seed(0)
D, H, Kd, Nseg = 64, 4, 8, 6
train_router = trn.MLPRouter(D, H, Kd)
inject_head = inj.MLPRouterHead(D, H, Kd)
inject_head.load_state_dict(train_router.state_dict(), strict=True)

h = torch.randn(1, D)
gam = torch.randn(1, Nseg, H, Kd)
with torch.no_grad():
    a = train_router.scores(h, gam)[0]
    b = inject_head.scores(h[:, None], gam)[0, 0]
d = (a - b).abs().max().item()
print(f"[1] train vs inject score max|diff| = {d:.2e}")
assert d < 1e-6, "scoring formula diverged between train and inject"

# a broadcast bug would still match on B=T=1; check a real [B,T] grid too
h2 = torch.randn(2, 5, D)
gam2 = torch.randn(2, Nseg, H, Kd)
with torch.no_grad():
    grid = inject_head.scores(h2, gam2)
    ref = torch.stack([torch.stack([train_router.scores(h2[b, t][None], gam2[b][None])[0]
                                    for t in range(5)]) for b in range(2)])
d2 = (grid - ref).abs().max().item()
print(f"[2] batched vs per-position max|diff| = {d2:.2e}  shape {tuple(grid.shape)}")
assert d2 < 1e-6 and grid.shape == (2, 5, Nseg)


class FakeSSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.num_heads, self.head_qk_dim = D, H, Kd
        self.connector = nn.Linear(D, H * Kd, bias=False)


class FakeLayer(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.ssc = FakeSSC()
        self.tag = tag


FakeSSC.__name__ = "GDN2SSC"
FakeLayer.__name__ = "MemoryCachingGDN2Layer"


class FakeModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(i) for i in range(n)])


tmp = "/tmp/_fake_routers"
os.makedirs(tmp, exist_ok=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
marks = {}
for L in (0, 3, 9):
    r = inj.MLPRouterHead(D, H, Kd)
    with torch.no_grad():
        r.net[0].weight.fill_(float(L) + 1.0)   # per-layer fingerprint
    marks[L] = float(L) + 1.0
    torch.save(r.state_dict(), os.path.join(tmp, f"router_L{L}.pt"))

model = FakeModel(16)
attached = inj.enable_mlp_router(model, tmp, None, "select", device="cpu")
print(f"[3] attached {attached}")
assert attached == [0, 3, 9]
for L in (0, 3, 9):
    got = float(model.layers[L].ssc.mlp_router.net[0].weight[0, 0])
    assert abs(got - marks[L]) < 1e-6, f"L{L} got fingerprint {got}"
print("[3] each router_L{i}.pt landed on MC layer i")
for L in range(16):
    if L not in (0, 3, 9):
        assert getattr(model.layers[L].ssc, "mlp_router", None) is None

for bad in (dict(layers=[0, 4]), dict(mode="bogus")):
    try:
        inj.enable_mlp_router(FakeModel(16), tmp, device="cpu", **bad)
    except (RuntimeError, ValueError) as e:
        print(f"[4] rejected {bad}: {type(e).__name__}")
    else:
        raise AssertionError(f"{bad} should have been rejected")

print("ALL CPU CHECKS PASS")

# The dot scorer has to survive the same round trip, and the inject side must
# infer it from the checkpoint: loading a dot head as cos would silently drop
# the learned temperature and renormalize, scoring with a formula the head was
# never fitted under.
train_dot = trn.MLPRouter(D, H, Kd, scorer="dot")
inject_dot = inj.MLPRouterHead(D, H, Kd, scorer="dot")
inject_dot.load_state_dict(train_dot.state_dict(), strict=True)
with torch.no_grad():
    a = train_dot.scores(h, gam)[0]
    b = inject_dot.scores(h[:, None], gam)[0, 0]
d = (a - b).abs().max().item()
print(f"[8] dot: train vs inject max|diff| = {d:.2e}")
assert d < 1e-6, "dot scoring diverged between train and inject"

sd_dot = train_dot.state_dict()
sd_cos = trn.MLPRouter(D, H, Kd, scorer="cos").state_dict()
assert "logit_scale" in sd_dot and "logit_scale" not in sd_cos
print("[8] logit_scale marks a dot checkpoint and is absent from a cos one")

try:
    inj.MLPRouterHead(D, H, Kd, scorer="cos").load_state_dict(sd_dot,
                                                              strict=True)
except RuntimeError as e:
    print(f"[8] cos head rejects a dot checkpoint: {str(e)[:52]}")
else:
    raise AssertionError("a cos head must not accept a dot checkpoint")

tmp3 = "/tmp/_fake_routers_dot"
os.makedirs(tmp3, exist_ok=True)
for f in os.listdir(tmp3):
    os.remove(os.path.join(tmp3, f))
torch.save(sd_dot, os.path.join(tmp3, "router_L0.pt"))
m4 = FakeModel(16)
inj.enable_broadcast_routing(m4, 0, "mlp", tmp3, device="cpu")
assert m4.layers[0].ssc.mlp_router.scorer == "dot"
print("[8] enable_broadcast_routing infers scorer=dot from the file")

m5 = FakeModel(16)
inj.enable_mlp_router(m5, tmp3, [0], "select", device="cpu")
assert m5.layers[0].ssc.mlp_router.scorer == "dot"
print("[8] enable_mlp_router infers scorer=dot from the file")

# RouteBus: a consumer must never silently accept a previous forward's picks.
bus = inj.RouteBus()
idx = torch.zeros(2, 5, 2, dtype=torch.long)
bus.publish(idx)
assert bus.take(2, 5, 2, last_seq=0) is idx
print("[5] fresh publish accepted")

for name, call in (
    ("stale seq", lambda: bus.take(2, 5, 2, last_seq=bus.seq)),
    ("wrong batch", lambda: bus.take(3, 5, 2, last_seq=0)),
    ("wrong length", lambda: bus.take(2, 6, 2, last_seq=0)),
    ("wrong k", lambda: bus.take(2, 5, 4, last_seq=0)),
):
    try:
        call()
    except RuntimeError as e:
        print(f"[5] rejected {name}: {str(e)[:60]}")
    else:
        raise AssertionError(f"{name} should have been rejected")

try:
    inj.RouteBus().take(2, 5, 2, last_seq=0)
except RuntimeError as e:
    print(f"[5] rejected empty bus: {str(e)[:60]}")
else:
    raise AssertionError("empty bus should have been rejected")

bus.publish(torch.ones(2, 5, 2, dtype=torch.long))
assert bus.seq == 2 and bus.take(2, 5, 2, last_seq=1).sum() == 20
print("[5] second publish advances seq")

for bad in (dict(source="bogus"),
            dict(source_layer=5, source="mlp", router_dir=tmp),
            dict(source_layer=15, source="mlp", router_dir=tmp),
            dict(source_layer=1, source="native")):
    try:
        inj.enable_broadcast_routing(FakeModel(16), device="cpu", **bad)
    except (RuntimeError, ValueError) as e:
        print(f"[6] rejected {bad}: {type(e).__name__}")
    else:
        raise AssertionError(f"{bad} should have been rejected")

m = FakeModel(16)
on = inj.enable_broadcast_routing(m, 0, "native", device="cpu")
assert on == list(range(16))
assert m.layers[0].ssc.is_bcast_source
assert not any(m.layers[i].ssc.is_bcast_source for i in range(1, 16))
shared = {id(m.layers[i].ssc.bus) for i in range(16)}
assert len(shared) == 1, "layers must share one bus"
print("[6] native broadcast: layer 0 is the only source, one shared bus")

m2 = FakeModel(16)
inj.enable_broadcast_routing(m2, 0, "mlp", tmp, device="cpu")
got = float(m2.layers[0].ssc.mlp_router.net[0].weight[0, 0])
assert abs(got - marks[0]) < 1e-6, f"source got fingerprint {got}"
assert all(getattr(m2.layers[i].ssc, "mlp_router", None) is None
           for i in range(1, 16))
print("[6] mlp broadcast: only layer 0 carries a router")

# a source deeper than 0 needs its own router for every pre-source layer,
# which must be the layer's OWN weights and not the source's.
tmp2 = "/tmp/_fake_routers_012"
os.makedirs(tmp2, exist_ok=True)
for f in os.listdir(tmp2):
    os.remove(os.path.join(tmp2, f))
marks2 = {}
for L in (0, 1, 2):
    r = inj.MLPRouterHead(D, H, Kd)
    with torch.no_grad():
        r.net[0].weight.fill_(10.0 + L)
    marks2[L] = 10.0 + L
    torch.save(r.state_dict(), os.path.join(tmp2, f"router_L{L}.pt"))

m3 = FakeModel(16)
on3 = inj.enable_broadcast_routing(m3, 2, "mlp", tmp2, device="cpu")
assert on3 == list(range(2, 16)), on3
assert m3.layers[2].ssc.is_bcast_source
for L in (0, 1):
    a = m3.layers[L].ssc
    assert a.mlp_router_mode == "select", f"L{L} should route on its own"
    assert abs(float(a.mlp_router.net[0].weight[0, 0]) - marks2[L]) < 1e-6
    assert not getattr(a, "is_bcast_source", False)
assert abs(float(m3.layers[2].ssc.mlp_router.net[0].weight[0, 0])
           - marks2[2]) < 1e-6
for L in range(3, 16):
    assert getattr(m3.layers[L].ssc, "mlp_router", None) is None
    assert m3.layers[L].ssc.bus is m3.layers[2].ssc.bus
print("[7] source=L2: L0/L1 keep their OWN routers in select mode, "
      "L2 publishes, L3-15 consume one bus")

# gate_mode. "order" relies on -inf sorting last, so that the finite entries
# keep occupying the leading slots and the valid mask still lines up with
# top_indices. Pin that assumption rather than trusting it.
x = torch.tensor([[[0.7, 2.5, float("-inf"), 1.1]]])
srt = torch.sort(x, dim=-1, descending=True).values
assert torch.isinf(srt[0, 0, -1]) and srt[0, 0, -1] < 0
assert srt[0, 0, 0] == 2.5 and srt[0, 0, 1] == 1.1 and srt[0, 0, 2] == 0.7
fin_before = sorted(v for v in x.flatten().tolist() if v != float("-inf"))
fin_after = sorted(v for v in srt.flatten().tolist() if v != float("-inf"))
assert fin_before == fin_after, "order must not change the multiset of logits"
print("[9] order: -inf sorts last, finite logits are a permutation")

for bad in ("bogus", "", "Native"):
    try:
        inj.enable_broadcast_routing(FakeModel(16), 0, "native", device="cpu",
                                     gate_mode=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"gate_mode={bad!r} should have been rejected")
print("[9] unknown gate_mode rejected")

for gm in inj.GATE_MODES:
    m = FakeModel(16)
    inj.enable_broadcast_routing(m, 0, "native", device="cpu", gate_mode=gm,
                                 gate_margin=2.0)
    assert all(m.layers[i].ssc.gate_mode == gm for i in range(16))
    assert all(m.layers[i].ssc.gate_margin == 2.0 for i in range(16))
print(f"[9] all {len(inj.GATE_MODES)} gate modes propagate to every layer")

# Sub-block descriptors: blocks=1 must reproduce the segment mean exactly,
# and a head fitted at m=8 must refuse the m=1 layout rather than scoring a
# formula it never saw.
keys = torch.randn(1, 512, H, Kd)
b1 = inj.block_summaries(keys, 256, 1)
seg_mean = keys.view(1, 2, 256, H, Kd).mean(dim=2)
assert torch.allclose(b1[:, :, 0], seg_mean, atol=1e-6)
b8 = inj.block_summaries(keys, 256, 8)
assert b8.shape == (1, 2, 8, H, Kd)
assert torch.allclose(b8.mean(dim=2), seg_mean, atol=1e-6), \
    "averaging the 8 blocks must return the segment mean"
print("[10] blocks=1 is the segment mean; 8 blocks average back to it")

h8 = inj.MLPRouterHead(D, H, Kd, scorer="dot", blocks=8)
g8 = torch.randn(1, 5, 8, H, Kd)
with torch.no_grad():
    per_block = torch.einsum("bthk,bnmhk->btnm",
                             h8.u(torch.randn(1, 1, D), normalize=False), g8)
assert h8.scores(torch.zeros(1, 1, D), g8).shape == (1, 1, 5)
for bad_layout in (torch.randn(1, 5, H, Kd),):
    try:
        h8.scores(torch.zeros(1, 1, D), bad_layout)
    except RuntimeError as e:
        print(f"[10] m=8 head rejects the m=1 layout: {str(e)[:50]}")
    else:
        raise AssertionError("layout mismatch must be rejected")
h1 = inj.MLPRouterHead(D, H, Kd, scorer="dot", blocks=1)
try:
    h1.scores(torch.zeros(1, 1, D), g8)
except RuntimeError:
    print("[10] m=1 head rejects the m=8 layout")
else:
    raise AssertionError("layout mismatch must be rejected")

# max over one block is the plain dot, so m=8 weights loaded as m=1 would be
# silently wrong rather than loudly wrong -- meta.json is what prevents it.
tmp4 = "/tmp/_fake_routers_m8"
os.makedirs(tmp4, exist_ok=True)
for f in os.listdir(tmp4):
    os.remove(os.path.join(tmp4, f))
torch.save(h8.state_dict(), os.path.join(tmp4, "router_L0.pt"))
json.dump({"blocks": 8}, open(os.path.join(tmp4, "meta.json"), "w"))
m6 = FakeModel(16)
inj.enable_broadcast_routing(m6, 0, "mlp", tmp4, device="cpu")
assert m6.layers[0].ssc.mlp_router.blocks == 8
print("[10] meta.json carries blocks=8 through injection")
try:
    inj.enable_broadcast_routing(FakeModel(16), 0, "mlp", tmp4, device="cpu",
                                 blocks=2)
except RuntimeError as e:
    print(f"[10] a conflicting --blocks is rejected: {str(e)[:46]}")
else:
    raise AssertionError("blocks conflict must be rejected")

# ---- 11. surprisal-weighted descriptors ------------------------------------
# The whole idea rests on tau=0 being the CURRENT descriptor, not something
# close to it. If that identity does not hold, a "gain" at tau>0 could just be
# a different descriptor scale and the dial would have no known zero.
torch.manual_seed(0)
B, T, H, Kd, CH = 2, 512, 4, 8, 256
keys = torch.randn(B, T, H, Kd)

plain = inj.block_summaries(keys, CH, 8)
w_none = inj.weighted_block_summaries(keys, CH, 8, None)
assert torch.equal(plain, w_none), "weights=None must be bit-identical"
print("[11] weights=None is bit-identical to block_summaries")

ids = torch.randint(0, 97, (B, T))
logits = torch.randn(B, T, 97)
sur = inj.token_surprisal(logits, ids)
assert sur.shape == (B, T) and torch.isfinite(sur).all()
assert (sur[:, 1:] > 0).all(), "surprisal in nats is positive"
print(f"[11] surprisal shape {tuple(sur.shape)}, "
      f"range {sur.min():.2f}-{sur.max():.2f} nats")

assert inj.surprisal_weights(sur, 0.0) is None, "tau=0 must take the None path"
w1 = inj.surprisal_weights(sur, 1.0)
uniform = torch.ones(B, T)
got = inj.weighted_block_summaries(keys, CH, 8, uniform)
assert torch.allclose(plain, got, atol=1e-6), "uniform weights == plain mean"
print("[11] tau=0 returns None; uniform weights reproduce the mean to 1e-6")

# a weight vector that keeps one token per block must return that token's key
onehot = torch.zeros(B, T)
onehot[:, ::(CH // 8)] = 1.0
picked = inj.weighted_block_summaries(keys, CH, 8, onehot)
want = keys[:, ::(CH // 8)].view(B, T // CH, 8, H, Kd)
assert torch.allclose(picked, want, atol=1e-6), "one-hot weights must select"
print("[11] one-hot weights select a single token per block")

# blocks=1 must still reduce to the deployed [B,N,H,Kd] segment mean
seg = mc_ssc.segment_key_sums(keys, CH)
one = inj.weighted_block_summaries(keys, CH, 1, None)[:, :, 0]
assert torch.allclose(seg, one, atol=1e-6), "blocks=1 == segment_key_sums"
print("[11] blocks=1 reduces to the deployed segment descriptor")

# padding must not vote: a short sequence weighted uniformly equals the mean
# over real tokens only
Tp = 300  # 2 segments, second one 44/256 full
kp = torch.randn(B, Tp, H, Kd)
wp = torch.ones(B, Tp)
got = inj.weighted_block_summaries(kp, CH, 1, wp)[:, 1, 0]
want = kp[:, CH:].mean(dim=1)
assert torch.allclose(got, want, atol=1e-5), "padded positions must not vote"
print("[11] padded tail does not dilute the last segment")

m7 = FakeModel(16)
on = inj.enable_broadcast_routing(m7, 0, "native")
n = inj.enable_surprisal_weights([l.ssc for l in m7.layers], w1)
assert n == 16 and m7.layers[0].ssc.desc_weights is w1
inj.enable_surprisal_weights([l.ssc for l in m7.layers], None)
assert m7.layers[3].ssc.desc_weights is None
print(f"[11] enable_surprisal_weights reached {n} layers and clears back to None")

try:
    inj.enable_surprisal_weights([object()], w1)
except RuntimeError as e:
    print(f"[11] unpatched layer rejected: {str(e)[:44]}")
else:
    raise AssertionError("an unpatched layer must be rejected")

try:
    inj.weighted_block_summaries(keys, CH, 8, torch.ones(B, T + 1))
except ValueError as e:
    print(f"[11] wrong weight length rejected: {str(e)[:44]}")
else:
    raise AssertionError("a mismatched weight length must be rejected")

# The offline sweep coarsens stored SUMS with a mean, on both numerator and
# denominator. That is only correct because mean(num)/mean(den) equals
# sum(num)/sum(den) — if it were applied to stored means instead, every block
# would get equal say and the weighting would silently vanish.
w2 = inj.surprisal_weights(sur, 1.0)
num, den = inj.block_weighted_sums(keys, CH, 8, w2)
direct = inj.weighted_block_summaries(keys, CH, 2, w2)
cn = num.view(B, T // CH, 2, 4, H, Kd).mean(dim=3)
cd = den.view(B, T // CH, 2, 4, 1, 1).mean(dim=3)
assert torch.allclose(direct.float(), cn / cd, atol=1e-5), \
    "coarsening sums must reproduce a direct weighted mean at the coarser m"
print("[11] coarsening stored sums reproduces the direct weighted mean (m 8->2)")

# and the dangerous alternative is measurably different, so the check has teeth
wrong = inj.weighted_block_summaries(keys, CH, 8, w2).view(
    B, T // CH, 2, 4, H, Kd).mean(dim=3)
gap = (wrong.float() - direct.float()).abs().max().item()
assert gap > 1e-4, "averaging per-block means should differ; check is vacuous"
print(f"[11] averaging per-block means instead differs by {gap:.3f} — not vacuous")

print("ALL BROADCAST CHECKS PASS")
