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
print("gdn2 layers:",len(mods),"| head_k",mods[0].head_k_dim,"num_heads",mods[0].num_heads)
cache=Cache.from_legacy_cache(None); gstash={}
def wrap(mod,orig,idx):
    def f(hidden_states,attention_mask=None,**kw):
        g=(-mod.A_log.float().exp().repeat_interleave(mod.head_k_dim)*F.softplus(mod.f_proj(hidden_states).float()+mod.dt_bias))
        gstash[idx]=g.detach().float().cpu(); return orig(hidden_states,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
for i,mod in enumerate(mods): mod.layer_idx=i; mod.mode="fused_recurrent"; mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod),i)
txt="The history of science is a long and winding road. "*80
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
ids=tok(txt,return_tensors="pt").input_ids[:,:512].cuda(); print("seq len",ids.shape[1])
with torch.no_grad(): m(ids)
def erank(M):
    s=torch.linalg.svdvals(M.float().cpu()); s=s/(s.sum()+1e-12); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
d=mods[0].head_k_dim; rows=[]
for i,mod in enumerate(mods):
    S=cache[i]["recurrent_state"][0]; g=gstash[i].reshape(-1,mod.num_heads,mod.head_k_dim)
    for h in range(S.shape[0]):
        rbar=float(torch.exp(g[:,h,:].mean())); rows.append((i,h,rbar,erank(S[h])))
rows=np.array(rows); rb=rows[:,2]; er=rows[:,3]
print("rbar range %.3f-%.3f | erank range %.2f-%.2f (cap %d)"%(rb.min(),rb.max(),er.min(),er.max(),d))
np.save("/root/F6_data.npy",rows)
fig,ax=plt.subplots(figsize=(5,4))
col=np.where(rb>=0.99,"#c0392b",np.where(rb>=0.9,"#e67e22","#2471a3"))
ax.scatter(rb,er,c=col,s=18,alpha=0.7)
xs=np.linspace(rb.min(),min(rb.max(),0.999),200); ax.plot(xs,np.minimum(d,math.e/(1-xs)),"k--",lw=1,label="theory e/(1-r̄), cap %d"%d)
ax.set_xlabel("r̄ = exp(E[log a_t])  (per head)"); ax.set_ylabel("effective rank of state"); ax.set_title("GDN2-370m (6B): decay vs erank per head")
ax.legend(fontsize=8,frameon=False); plt.tight_layout(); plt.savefig("/root/F6_erank_vs_decay.png",dpi=200)
print("Type A(>=.99):",int((rb>=0.99).sum()),"B(.9-.99):",int(((rb>=.9)&(rb<.99)).sum()),"C(<.9):",int((rb<.9).sum()))
print("saved F6_erank_vs_decay.png")
