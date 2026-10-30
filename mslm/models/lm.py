"""Decoder LM: embedding -> N x [RMSNorm, mixer, RMSNorm, SwiGLU] -> RMSNorm -> head.

Initialisation follows long-gdn/dsc lit_gpt ``mamba_init``: embeddings N(0, 0.02),
biases zero, MLP output projection kaiming-uniform / sqrt(2 * n_layer), GDN-2
linears keep their own xavier(gain 2^-2.5) init, everything else is the
PyTorch default.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mslm.models import registry
from mslm.models.config import ModelConfig
from mslm.models.gdn2 import GDN2Mixer
from mslm.models.mc import MCMixer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, ffn: int):
        super().__init__()
        self.fc_1 = nn.Linear(d, ffn, bias=False)
        self.fc_2 = nn.Linear(d, ffn, bias=False)
        self.proj = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.fc_1(x)) * self.fc_2(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        kw = dict(conv_size=cfg.conv_size, norm_eps=cfg.norm_eps, backend=cfg.gdn_backend)
        self.norm_1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.arch == "mc":
            self.mixer = MCMixer(cfg.d_model, cfg.n_head, cfg.head_dim, cfg.mc, **kw)
        else:
            self.mixer = GDN2Mixer(cfg.d_model, cfg.n_head, cfg.head_dim, **kw)
        self.norm_2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(self, x):
        x = x + self.mixer(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.arch == "mc" and cfg.mc.share_from >= 0:
            if cfg.mc.share_from >= cfg.n_layer:
                raise ValueError(f"MC_SHARE_FROM={cfg.mc.share_from} but the model has {cfg.n_layer} layers")
            share: dict = {}
            for i, m in enumerate(self.mc_layers()):
                m.layer_id, m.share = i, share
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.kaiming_uniform_(blk.mlp.proj.weight, a=math.sqrt(5))
            with torch.no_grad():
                blk.mlp.proj.weight /= math.sqrt(2 * self.cfg.n_layer)

    # -- MC helpers ---------------------------------------------------------
    def mc_layers(self) -> list[MCMixer]:
        return [b.mixer for b in self.blocks if isinstance(b.mixer, MCMixer)]

    def set_routing(self, *, oracle_segments=None, capture=False):
        for m in self.mc_layers():
            m.oracle_segments = oracle_segments
            m.capture = capture
            m.record = None

    def routing_records(self):
        return [m.record for m in self.mc_layers()]

    # -- forward ------------------------------------------------------------
    def hidden(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            if self.cfg.act_ckpt and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm_f(x)

    def forward(self, idx, targets=None, batch_meta: dict | None = None):
        """Returns logits if ``targets`` is None, else dict(loss, lm_loss, aux_loss)."""
        h = self.hidden(idx)
        if targets is None:
            return self.lm_head(h)
        lm_loss = self._lm_loss(h, targets)
        aux = lm_loss.new_zeros(())
        mc = self.cfg.mc
        if self.cfg.arch == "mc" and mc.aux_loss != "none" and mc.aux_weight > 0:
            fn = registry.get("aux_loss", mc.aux_loss)
            recs = self.routing_records()
            aux = sum(fn(r, batch_meta or {}) for r in recs) / max(len(recs), 1)
        return {"loss": lm_loss + mc.aux_weight * aux, "lm_loss": lm_loss, "aux_loss": aux}

    def _lm_loss(self, h, targets):
        if h.is_cuda:
            try:
                from fla.modules import FusedLinearCrossEntropyLoss
            except ImportError:
                FusedLinearCrossEntropyLoss = None
            if FusedLinearCrossEntropyLoss is not None:
                return FusedLinearCrossEntropyLoss()(h.contiguous(), targets.contiguous(), self.lm_head.weight)
        logits = self.lm_head(h).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))

    def num_params(self, exclude_embed: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if exclude_embed else n
