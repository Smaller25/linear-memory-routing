import sys; sys.path.insert(0,"/root/dscpkg")
import torch, torch.nn.functional as F, warnings, os; warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_XET"]="1"
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lit_gpt.gdn2 as G
from lit_gpt.config import Config; from lit_gpt.model import GPT; from lit_gpt.gdn2 import GatedDeltaNet2
from huggingface_hub import hf_hub_download; from fla.models.utils import Cache
from transformers import AutoTokenizer; from datasets import load_dataset
cfg=Config.from_name("gdn2_370M"); m=GPT(cfg).cuda().to(torch.bfloat16).eval()
ck=torch.load(hf_hub_download("gyung/gdn2-370m-fineweb-edu-100b","checkpoint-6B-model-ckpt.pth"),map_location="cpu",weights_only=False)
m.load_state_dict(ck["model"],strict=False)
mods=[blk.attn for blk in m.transformer.h if getattr(blk,"use_gdn2",False)]
for i,mod in enumerate(mods): mod.layer_idx=i; mod.mode="fused_recurrent"
orig_fr=G.fused_recurrent_gdn2; stash={}; cnt={"i":0}
def cap(**kw):
    li=cnt["i"]%len(mods); cnt["i"]+=1; stash[li]={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kw.items()}; return orig_fr(**kw)
G.fused_recurrent_gdn2=cap
cache=Cache.from_legacy_cache(None)
def wrap(mod,orig,idx):
    def f(hs,attention_mask=None,**kw): return orig(hs,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
for i,mod in enumerate(mods): mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod),i)
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
def erank(M):
    s=torch.linalg.svdvals(M.float().cpu()); s=s/(s.sum()+1e-12); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
def state(kw):
    o,st=orig_fr(q=kw["q"],k=kw["k"],v=kw["v"],g=kw["g"],b=kw["b"],w=kw["w"],A_log=kw["A_log"],dt_bias=kw["dt_bias"],output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False); return st[0]
def load_texts():
    T={}
    try: T["natural(wiki)"]=[x for x in load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"] if len(x)>200]
    except Exception as e: print("wiki fail",e)
    try:
        d=load_dataset("openai/gsm8k","main",split="test"); T["math(gsm8k)"]=[q+" "+a for q,a in zip(d["question"][:400],d["answer"][:400])]
    except Exception as e: print("gsm8k fail",e)
    try:
        d=load_dataset("rajpurkar/squad",split="validation"); T["knowledge(squad)"]=list(dict.fromkeys(d["context"]))[:400]
    except Exception as e: print("squad fail",e)
    try:
        d=load_dataset("google-research-datasets/mbpp",split="test"); T["code(mbpp)"]=[t+"\n"+c for t,c in zip(d["text"][:400],d["code"][:400])]
    except Exception as e: print("mbpp fail",e)
    T["repetitive"]=["The history of science is a long and winding road. "]*400
    return T
def windows(texts,n=5,L=512):
    ids=tok("\n\n".join(texts),return_tensors="pt").input_ids[0]
    return [ids[i*L:(i+1)*L].unsqueeze(0).cuda() for i in range(min(n, ids.shape[0]//L))]
conds=["real","g=1","iso_k","iso_v","all_off"]; rows={}; meta={}
for name,texts in load_texts().items():
    wins=windows(texts); 
    if not wins: print(name,"no windows"); continue
    acc={c:[] for c in conds}; kc=[];vc=[];rb=[]
    for ids in wins:
        stash.clear(); cnt["i"]=0
        with torch.no_grad(): m(ids)
        for li in [4,6,8,10,12]:
            kw=stash[li]
            for c in conds:
                kw2=dict(kw)
                if c in("g=1","all_off"): kw2["g"]=torch.zeros_like(kw["g"])
                if c in("iso_k","all_off"): kw2["k"]=torch.randn_like(kw["k"])
                if c in("iso_v","all_off"): kw2["v"]=torch.randn_like(kw["v"])
                st=state(kw2); acc[c].append(np.mean([erank(st[h]) for h in range(st.shape[0])]))
            def cos(V):
                V=V.reshape(-1,V.shape[-2],V.shape[-1]).float(); V=V/(V.norm(dim=-1,keepdim=True)+1e-8); idx=torch.randint(0,V.shape[0],(2000,2)); return (V[idx[:,0]]*V[idx[:,1]]).sum(-1).mean().item()
            kc.append(cos(kw["k"].cpu())); vc.append(cos(kw["v"].cpu())); rb.append(float(torch.exp(kw["g"].float().mean())))
    rows[name]={c:float(np.mean(acc[c])) for c in conds}; meta[name]={"key_cos":float(np.mean(kc)),"val_cos":float(np.mean(vc)),"rbar":float(np.mean(rb)),"n_win":len(wins)}
    print("%-17s erank "%name+" ".join("%s=%.1f"%(c,rows[name][c]) for c in conds)+" | kcos %.2f vcos %.2f rbar %.2f (n=%d)"%(meta[name]["key_cos"],meta[name]["val_cos"],meta[name]["rbar"],len(wins)))
np.save("/root/F10_data.npy",{"rows":rows,"meta":meta},allow_pickle=True)
dn=list(rows); fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(dn)); w=0.16
for j,c in enumerate(conds): ax.bar(x+(j-2)*w,[rows[d][c] for d in dn],w,label=c)
ax.set_xticks(x); ax.set_xticklabels(dn,rotation=12,fontsize=8); ax.axhline(128,ls=":",c="gray",lw=.7)
ax.set_ylabel("effective rank (layer-avg)"); ax.set_title("erank by REAL benchmark x intervention (GDN2-370m 6B)"); ax.legend(fontsize=8,ncol=5,frameon=False)
plt.tight_layout(); plt.savefig("/root/F10_benchmarks.png",dpi=200); print("saved F10_benchmarks.png")
