"""From-scratch backbone readiness check (run via Slurm: sbatch scripts/sh_slurm_run.sh
python scripts/sh_check_backbones.py). Builds small FLA backbones (head_dim=64, 2 layers) and runs
forward + backward on GPU to confirm they TRAIN on this hardware with FLA Triton alone.

Verified 2026-06-19 on RTX PRO 6000 Blackwell (sm_120), sh_routing env (torch 2.11+cu128):
  KDA, GatedDeltaNet -> OK (Triton-only).  Mamba3 -> needs mamba_ssm w/ Mamba-3 SISO kernels.
"""
import torch

DEV = "cuda"
B, T, V = 2, 512, 1000


def trial(name, build):
    try:
        model = build().to(DEV).train()
        n = sum(p.numel() for p in model.parameters())
        ids = torch.randint(0, V, (B, T), device=DEV)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss if getattr(out, "loss", None) is not None else out.logits.float().mean()
        loss.backward()
        gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        finite = torch.isfinite(loss).item() and all(
            torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        print(f"[OK]   {name:14s} params={n/1e6:5.1f}M  loss={loss.item():.4f}  grad_norm={gnorm:.3f}  finite={finite}")
    except Exception as e:
        print(f"[FAIL] {name:14s} {type(e).__name__}: {e}")


def mamba3():
    from fla.models.mamba3 import Mamba3Config
    from fla.models.mamba3.modeling_mamba3 import Mamba3ForCausalLM
    return Mamba3ForCausalLM(Mamba3Config(hidden_size=256, num_hidden_layers=2, vocab_size=V,
                                          head_dim=64, expand=2, state_size=64, n_groups=1))


def gdn():
    from fla.models.gated_deltanet import GatedDeltaNetConfig
    from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM
    return GatedDeltaNetForCausalLM(GatedDeltaNetConfig(hidden_size=256, num_hidden_layers=2,
                                    vocab_size=V, head_dim=64, num_heads=4, expand_v=2.0))


def kda():
    from fla.models.kda import KDAConfig
    from fla.models.kda.modeling_kda import KDAForCausalLM
    return KDAForCausalLM(KDAConfig(hidden_size=256, num_hidden_layers=2, vocab_size=V,
                          head_dim=64, num_heads=4, expand_v=1.0))


def trial_layer(name, build, D=256):
    """GDN2 ships as a token-mixing layer (no packaged ForCausalLM) — exercise the Triton kernel
    directly: forward+backward on a random hidden-state tensor."""
    try:
        layer = build().to(DEV).train()
        n = sum(p.numel() for p in layer.parameters())
        x = torch.randn(B, T, D, device=DEV, requires_grad=True)
        out = layer(hidden_states=x)
        out = out[0] if isinstance(out, tuple) else out
        loss = out.float().pow(2).mean()
        loss.backward()
        gnorm = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None) ** 0.5
        finite = torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
        print(f"[OK]   {name:14s} params={n/1e6:5.1f}M  loss={loss.item():.4f}  grad_norm={gnorm:.3f}  finite={finite}")
    except Exception as e:
        print(f"[FAIL] {name:14s} {type(e).__name__}: {e}")


def gdn2_layer():
    from fla.layers.gdn2 import GatedDeltaNet2
    return GatedDeltaNet2(hidden_size=256, head_dim=64, num_heads=4, expand_v=1.0)


if __name__ == "__main__":
    print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0),
          "| cap", torch.cuda.get_device_capability(0))
    for name, fn in [("Mamba3", mamba3), ("GatedDeltaNet", gdn), ("KDA", kda)]:
        trial(name, fn)
    trial_layer("GDN2(layer)", gdn2_layer)
