"""Model configuration and size presets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Sizes follow long-gdn/dsc configs (gdn2_370M / mc_370M, worktree mc_50M).
PRESETS: dict[str, dict] = {
    # tiny: CPU / unit tests only
    "tiny": dict(n_layer=2, d_model=64, n_head=2, head_dim=32, ffn_dim=128),
    "50m": dict(n_layer=16, d_model=384, n_head=6, head_dim=128, ffn_dim=768),
    "370m": dict(n_layer=16, d_model=1024, n_head=16, head_dim=128, ffn_dim=2048),
}


@dataclass
class MCConfig:
    chunk: int = 256
    topk: int = 4
    descriptor: str = "meanpool"
    router: str = "dot"
    gate: str = "softmax"
    connect: str = "independent"
    aux_loss: str = "none"
    aux_weight: float = 0.0
    read_kernel: str = "auto"


@dataclass
class ModelConfig:
    arch: str = "mc"                 # vanilla | mc
    size: str = "50m"
    n_layer: int = 16
    d_model: int = 384
    n_head: int = 6
    head_dim: int = 128
    ffn_dim: int = 768
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    conv_size: int = 4
    gdn_backend: str = "auto"        # auto | fla | torch
    act_ckpt: bool = False
    mc: MCConfig = field(default_factory=MCConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        mc = MCConfig(**d.pop("mc", {}))
        return cls(mc=mc, **d)

    @classmethod
    def from_settings(cls, s) -> "ModelConfig":
        """Build from resolved env settings (``mslm.settings.Settings``)."""
        size = s.str("MODEL_SIZE")
        if size not in PRESETS:
            raise ValueError(f"MODEL_SIZE={size!r}; choose from {sorted(PRESETS)}")
        p = dict(PRESETS[size])
        for key, var in [("n_layer", "N_LAYER"), ("d_model", "D_MODEL"), ("n_head", "N_HEAD"),
                         ("head_dim", "HEAD_DIM"), ("ffn_dim", "FFN_DIM")]:
            if s.str(var):
                p[key] = s.int(var)
        arch = s.str("MODEL_ARCH")
        if arch not in ("vanilla", "mc"):
            raise ValueError(f"MODEL_ARCH={arch!r}; choose vanilla | mc")
        return cls(
            arch=arch, size=size, vocab_size=s.int("VOCAB_SIZE"),
            gdn_backend=s.str("GDN_BACKEND"), act_ckpt=s.bool("ACT_CKPT"),
            mc=MCConfig(
                chunk=s.int("MC_CHUNK"), topk=s.int("MC_TOPK"),
                descriptor=s.str("MC_DESCRIPTOR"), router=s.str("MC_ROUTER"),
                gate=s.str("MC_GATE"), connect=s.str("MC_CONNECT"),
                aux_loss=s.str("MC_AUX_LOSS"), aux_weight=s.float("MC_AUX_WEIGHT"),
                read_kernel=s.str("MC_READ_KERNEL"),
            ),
            **p,
        )
