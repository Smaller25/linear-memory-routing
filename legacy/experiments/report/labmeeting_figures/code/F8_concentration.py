import sys; sys.path.insert(0,"/root/dscpkg")
import torch, torch.nn.functional as F, warnings, math; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lit_gpt.config import Config; from lit_gpt.model import GPT; from lit_gpt.gdn2 import GatedDeltaNet2
from huggingface_hub import hf_hub_download; from fla.models.utils import Cache
from transformers import AutoTokenizer
cfg=Config.from_name("gdn2_370M"); m=GPT(cfg).cuda().to(torch.bfloat16).eval()
ck=torch.load(hf_hub_download("gyung/gdn2-370m-fineweb-edu-100b","checkpoint-6B-model-ckpt.pth"),map_location="cpu",weights_only=False)
m.load_state_dict(ck["model"],strict=False)
mods=[blk.attn for blk in m.transformer.h if getattr(blk,"use_gdn2",False)]
cache=Cache.from_legacy_cache(None); kstash={}
def wrap(mod,orig,idx):
    def f(hidden_states,attention_mask=None,**kw):
        return orig(hidden_states,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
def khook(idx):
    def h(module,inp,out):
        kstash[idx]=(out[0] if isinstance(out,tuple) else out).detach().float().cpu()  # [1,T,key_dim]
    return h
for i,mod in enumerate(mods):
    mod.layer_idx=i; mod.mode="fused_recurrent"; mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod),i)
    mod.k_conv1d.register_forward_hook(khook(i))
txt="The history of science is a long and winding road. "*80
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
ids=tok(txt,return_tensors="pt").input_ids[:,:512].cuda()
with torch.no_grad(): m(ids)
H=mods[0].num_heads; dk=mods[0].head_k_dim
# pick layer 8 (mid), a few heads with highest erank (from F6) — just use heads 0..3 of layer 8
Li=8; S=cache[Li]["recurrent_state"][0]  # [H,dk,dv]
k=kstash[Li].reshape(-1,H,dk)            # [T,H,dk]
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
# (1) SV spectrum: actual heads vs random Gaussian state
rng=np.random.default_rng(0)
for h in [0,1,2,3]:
    sv=torch.linalg.svdvals(S[h].float().cpu()).numpy(); sv=sv/sv.sum()
    ax[0].plot(np.sort(sv)[::-1],alpha=.8,label=f"head {h}")
Rn=rng.standard_normal((dk,S.shape[2])); svr=np.linalg.svd(Rn,compute_uv=False); svr=svr/svr.sum()
ax[0].plot(np.sort(svr)[::-1],"k--",lw=1.3,label="random Gaussian state")
ax[0].set_yscale("log"); ax[0].set_xlabel("singular value index"); ax[0].set_ylabel("normalized σ (log)")
ax[0].set_title(f"State SV spectrum (layer {Li})\nsteep = concentrated (뭉침)"); ax[0].legend(fontsize=7,frameon=False)
# (2) key cosine-sim: actual vs isotropic null
def cospairs(V,n=4000):
    V=V/ (V.norm(dim=-1,keepdim=True)+1e-8); idx=torch.randint(0,V.shape[0],(n,2))
    return (V[idx[:,0]]*V[idx[:,1]]).sum(-1).numpy()
allc=np.concatenate([cospairs(k[:,h,:]) for h in range(H)])
null=np.concatenate([cospairs(torch.randn(k.shape[0],dk)) for _ in range(2)])
ax[1].hist(null,bins=60,density=True,alpha=.5,color="gray",label="isotropic null")
ax[1].hist(allc,bins=60,density=True,alpha=.6,color="#c0392b",label="actual keys")
ax[1].set_xlabel("cosine(k_i, k_j)"); ax[1].set_ylabel("density")
ax[1].set_title(f"Key pairwise cosine (layer {Li})\nright-shift = keys cluster (뭉침 cause)"); ax[1].legend(fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig("/root/F8_concentration.png",dpi=200)
print("key cosine: actual mean %.3f | null mean %.3f"%(allc.mean(),null.mean()))
print("state SV top-1 share (head0): %.3f (isotropic %.3f)"%(np.sort(torch.linalg.svdvals(S[0].float().cpu()).numpy())[::-1][0]/torch.linalg.svdvals(S[0].float().cpu()).numpy().sum(), np.sort(svr)[::-1][0]))
print("saved F8_concentration.png")
