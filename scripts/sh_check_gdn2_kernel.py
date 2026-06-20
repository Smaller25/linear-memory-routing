"""Decisive kernel-correctness test: does GDN-2's Triton chunk op match the naive reference on this
GPU, in BOTH forward and backward? A forward match but backward mismatch == miscomputed gradients
(the fla #640 class of bug), which would silently prevent pure-recurrent training from learning."""
import math

import torch

from fla.ops.gdn2 import chunk_gdn2, naive_recurrent_gdn2

torch.manual_seed(0)
DEV = "cuda"
B, T, H, K, V = 2, 128, 4, 32, 32


def mk():
    q = torch.randn(B, T, H, K, device=DEV, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=DEV, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=DEV, dtype=torch.float32)
    g = -torch.nn.functional.softplus(torch.randn(B, T, H, K, device=DEV)) * 0.1   # log-decay <0
    b = torch.sigmoid(torch.randn(B, T, H, K, device=DEV))                          # erase [0,1]
    w = torch.sigmoid(torch.randn(B, T, H, V, device=DEV))                          # write [0,1]
    return [x.clone().requires_grad_(True) for x in (q, k, v, g, b, w)]


def run(fn, inp, **kw):
    q, k, v, g, b, w = inp
    o = fn(q, k, v, g, b, w, **kw)
    o = o[0] if isinstance(o, tuple) else o
    loss = o.float().pow(2).sum()
    grads = torch.autograd.grad(loss, inp, retain_graph=False)
    return o.detach(), grads


def maxrel(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item() + 1e-8
    return d, d / s


inp_c = mk()
inp_n = mk()  # identical seed -> identical values
o_c, g_c = run(chunk_gdn2, inp_c, use_gate_in_kernel=False, use_qk_l2norm_in_kernel=False)
o_n, g_n = run(naive_recurrent_gdn2, inp_n)

print(f"device {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}")
fd, fr = maxrel(o_c, o_n)
print(f"FORWARD   out: max_abs={fd:.3e}  max_rel={fr:.3e}   {'OK' if fr < 1e-2 else 'MISMATCH'}")
names = ["q", "k", "v", "g", "b", "w"]
worst = 0.0
for n, gc, gn in zip(names, g_c, g_n):
    d, r = maxrel(gc, gn)
    worst = r if math.isnan(r) else max(worst, r)  # nan must NOT be swallowed by max()
    print(f"BACKWARD d{n}: max_abs={d:.3e}  max_rel={r:.3e}   {'OK' if r < 5e-2 else 'MISMATCH'}")
ok = (not math.isnan(worst)) and (not math.isnan(fr)) and worst < 5e-2 and fr < 1e-2
print(f"\nVERDICT: {'chunk kernel matches naive (gradients OK)' if ok else 'MISMATCH/NaN -> inconclusive or kernel issue (check inputs first; this hand test is nan-prone)'}")
