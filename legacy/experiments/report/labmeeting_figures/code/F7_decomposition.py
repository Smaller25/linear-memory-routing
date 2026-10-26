import sys; sys.path.insert(0,"/root/dscpkg")
import torch, torch.nn.functional as F, warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lit_gpt.gdn2 as G
from lit_gpt.config import Config; from lit_gpt.model import GPT; from lit_gpt.gdn2 import GatedDeltaNet2
from huggingface_hub import hf_hub_download; from fla.models.utils import Cache
from transformers import AutoTokenizer
cfg=Config.from_name("gdn2_370M"); m=GPT(cfg).cuda().to(torch.bfloat16).eval()
ck=torch.load(hf_hub_download("gyung/gdn2-370m-fineweb-edu-100b","checkpoint-6B-model-ckpt.pth"),map_location="cpu",weights_only=False)
m.load_state_dict(ck["model"],strict=False)
mods=[blk.attn for blk in m.transformer.h if getattr(blk,"use_gdn2",False)]
for i,mod in enumerate(mods): mod.layer_idx=i; mod.mode="fused_recurrent"
# capture kernel inputs per layer by wrapping the module-level fused_recurrent_gdn2
orig_fr=G.fused_recurrent_gdn2; stash={}; cnt={"i":0}
def cap(**kw):
    li=cnt["i"]%len(mods); cnt["i"]+=1
    stash[li]={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kw.items()}
    return orig_fr(**kw)
G.fused_recurrent_gdn2=cap
cache=Cache.from_legacy_cache(None)
def wrap(mod,orig,idx):
    def f(hs,attention_mask=None,**kw): return orig(hs,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
for i,mod in enumerate(mods): mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod),i)
txt="The history of science is a long and winding road. "*80
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
ids=tok(txt,return_tensors="pt").input_ids[:,:512].cuda()
with torch.no_grad(): m(ids)
G.fused_recurrent_gdn2=orig_fr  # restore
def erank(M):
    s=torch.linalg.svdvals(M.float().cpu()); s=s/(s.sum()+1e-12); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
def run_state(kw):  # call kernel, return final state [H,dk,dv]
    o,st=orig_fr(output_final_state=True,**{k:kw[k] for k in ["q","k","v","g","b","w","A_log","dt_bias"] if k in kw},
                 use_qk_l2norm_in_kernel=kw.get("use_qk_l2norm_in_kernel",True),use_gate_in_kernel=False)
    return st[0]
torch.manual_seed(0)
conds={"real_g+real_k":(0,0),"g=1+real_k":(1,0),"real_g+iso_k":(0,1),"g=1+iso_k":(1,1)}
res={c:[] for c in conds}
val_real=[]
for li in [4,6,8,10,12]:
    kw=stash[li]
    for c,(zg,rk) in conds.items():
        kw2=dict(kw)
        if zg: kw2["g"]=torch.zeros_like(kw["g"])
        if rk: kw2["k"]=torch.randn_like(kw["k"])
        st=run_state(kw2)
        er=np.mean([erank(st[h]) for h in range(st.shape[0])])
        res[c].append(er)
    val_real.append(res["real_g+real_k"][-1])
print("VALIDATION real_g+real_k erank mean %.2f (F6 layer-avg ~8-9 expected)"%np.mean(res["real_g+real_k"]))
means={c:np.mean(v) for c,v in res.items()}
for c in conds: print("  %-14s erank %.2f"%(c,means[c]))
fig,ax=plt.subplots(figsize=(5,3.6))
labels=list(conds); vals=[means[c] for c in labels]
ax.bar(range(4),vals,color=["#2471a3","#e67e22","#c0392b","#7f8c8d"])
ax.set_xticks(range(4)); ax.set_xticklabels(["real g\nreal k","g=1\nreal k","real g\niso k","g=1\niso k"],fontsize=8)
ax.set_ylabel("effective rank of state"); ax.set_title("F7: decay vs key-anisotropy decomposition\n(GDN2-370m 6B, layer-avg)")
for i,v in enumerate(vals): ax.text(i,v+0.2,"%.1f"%v,ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("/root/F7_decomposition.png",dpi=200)
np.save("/root/F7_data.npy",res); print("saved F7_decomposition.png")
