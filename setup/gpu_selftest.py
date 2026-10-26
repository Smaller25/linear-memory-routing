"""GPU self-test: vendored GDN-2 kernel vs the PyTorch reference, Triton gather-read
vs the PyTorch read, and one fwd/bwd of a small MC model. Run on a GPU node only."""

import sys

import torch

from mslm.kernels.gather_read import _ssc_gather_read_torch, ssc_gather_read
from mslm.kernels.gdn2_ops.chunk_gdn2 import chunk_gdn2
from mslm.kernels.naive_gdn2 import naive_gdn2
from mslm.models import LM, MCConfig, ModelConfig


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print("  device:", torch.cuda.get_device_name(), "cap", torch.cuda.get_device_capability())
    B, T, H, K = 2, 256, 2, 128
    q, k = torch.randn(B, T, H, K, device=dev), torch.randn(B, T, H, K, device=dev)
    v = torch.randn(B, T, H, K, device=dev)
    g = -torch.rand(B, T, H, K, device=dev) * 0.1
    b, w = torch.rand(B, T, H, K, device=dev), torch.rand(B, T, H, K, device=dev)
    kw = dict(initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o1, s1 = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, use_gate_in_kernel=False, cu_seqlens=None, **kw)
    o2, s2 = naive_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **kw)
    eo, es = rel(o1, o2), rel(s1, s2)
    print(f"  chunk_gdn2 vs naive: out rel err {eo:.2e}, state rel err {es:.2e}")
    ok = eo < 1e-2 and es < 1e-2

    mems = torch.randn(B, 8, H, K, K, device=dev)
    idx = torch.randint(0, 8, (B, T, 2), device=dev)
    wts = torch.rand(B, T, 2, device=dev)
    r1 = ssc_gather_read(q, mems, idx, wts, scale=K ** -0.5, normalize_queries=True)
    r2 = _ssc_gather_read_torch(q, mems, idx, wts, K ** -0.5, True)
    er = rel(r1, r2)
    print(f"  gather_read triton vs torch: rel err {er:.2e}")
    ok &= er < 1e-2

    cfg = ModelConfig(arch="mc", size="tiny", n_layer=2, d_model=256, n_head=2, head_dim=128, ffn_dim=512,
                      vocab_size=1000, mc=MCConfig(chunk=64, topk=2))
    m = LM(cfg).to(dev)
    x = torch.randint(0, 1000, (2, 513), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x[:, :-1], x[:, 1:])
    out["loss"].backward()
    gn = sum(p.grad.float().norm() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    print(f"  MC fwd/bwd bf16: loss {out['loss'].item():.4f} grad norm {gn.item():.3f}")
    ok &= bool(torch.isfinite(out["loss"])) and bool(torch.isfinite(gn))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
