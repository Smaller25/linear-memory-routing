"""Pinned long-gdn worktree에서 MC/vanilla GDN2 370M litgpt 모델 로드."""
import os, sys
import torch

WORKTREE = os.environ.get("MC_LONGGDN_WORKTREE", "/data2/sohyung/worktrees/long-gdn-e71713e")
CHUNK, TOPK = 256, 2
CKPTS = {
    "mc-30B": ("LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool",
               "checkpoint-30B-model-ckpt.pth", "mc_370M"),
    "mc-5B": ("LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool",
              "checkpoint-5B-model-ckpt.pth", "mc_370M"),
    "vanilla-5B": ("LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla",
                   "checkpoint-5B-model-ckpt.pth", "gdn2_370M"),
}


def bootstrap():
    for p in (WORKTREE, os.path.join(WORKTREE, "dsc")):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_model(kind, device="cuda", dtype=torch.bfloat16):
    bootstrap()
    from huggingface_hub import hf_hub_download
    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    repo, fname, cfg_name = CKPTS[kind]
    path = hf_hub_download(repo, fname)
    model = GPT(Config.from_name(cfg_name))
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


def mc_layers(model):
    """[(layer_idx, MemoryCachingGDN2Layer)] — vanilla 모델이면 빈 리스트."""
    out = []
    for i, blk in enumerate(model.transformer.h):
        if type(blk.attn).__name__ == "MemoryCachingGDN2Layer":
            out.append((i, blk.attn))
    return out
