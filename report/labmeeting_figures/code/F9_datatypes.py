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
orig_fr=G.fused_recurrent_gdn2; stash={}; cnt={"i":0}
def cap(**kw):
    li=cnt["i"]%len(mods); cnt["i"]+=1
    stash[li]={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kw.items()}; return orig_fr(**kw)
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
    o,st=orig_fr(q=kw["q"],k=kw["k"],v=kw["v"],g=kw["g"],b=kw["b"],w=kw["w"],A_log=kw["A_log"],dt_bias=kw["dt_bias"],
                 output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False); return st[0]
NAT=("Rivers shape the land over millions of years, carving valleys and depositing sediment far downstream. "
"The economy of a coastal town often depends on fishing, tourism, and trade, each rising and falling with the seasons. "
"A good teacher notices when a student is confused before the student says a word. "
"In the north, winters are long and the light is thin; people learn patience from the weather. "
"Music can carry a memory more vividly than a photograph, folding years into a single chord. ")
MATH=("Let x be a positive integer. If 3x + 7 = 22, then x = 5. Consider the sum 1 + 2 + ... + n = n(n+1)/2. "
"The derivative of x^3 is 3x^2, and the integral of 2x dx is x^2 + C. If a triangle has sides 3, 4, 5 it is right-angled since 9 + 16 = 25. "
"Solve 2y - 4 = 10: y = 7. The probability of two independent events is the product of their probabilities. ")
CODE=("def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
"class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop()\n"
"for i in range(10):\n    if i % 2 == 0:\n        print(i)\n")
KNOW=("The capital of France is Paris. Water boils at 100 degrees Celsius at sea level. "
"Mount Everest is the highest mountain above sea level. The human heart has four chambers. "
"The speed of light is about 299792 kilometers per second. Photosynthesis converts sunlight into chemical energy. "
"The Great Wall of China was built over many centuries. DNA carries genetic information in living organisms. ")
REP="The history of science is a long and winding road. "
data={"natural":NAT*4,"math":MATH*4,"code":CODE*6,"knowledge":KNOW*4,"repetitive":REP*80}
conds=["real","g=1","iso_k","iso_v","all_off"]
rows={}; meta={}
for name,txt in data.items():
    ids=tok(txt,return_tensors="pt").input_ids[:,:512].cuda()
    if ids.shape[1]<128: ids=tok(txt*3,return_tensors="pt").input_ids[:,:512].cuda()
    stash.clear(); cnt["i"]=0
    with torch.no_grad(): m(ids)
    er={c:[] for c in conds}; kcos=[]; vcos=[]; rbar=[]
    for li in [4,6,8,10,12]:
        kw=stash[li]; H=kw["k"].shape[-2]; dk=kw["k"].shape[-1]
        for c in conds:
            kw2=dict(kw)
            if c in("g=1","all_off"): kw2["g"]=torch.zeros_like(kw["g"])
            if c in("iso_k","all_off"): kw2["k"]=torch.randn_like(kw["k"])
            if c in("iso_v","all_off"): kw2["v"]=torch.randn_like(kw["v"])
            st=state(kw2); er[c].append(np.mean([erank(st[h]) for h in range(st.shape[0])]))
        def cos(V):
            V=V.reshape(-1,V.shape[-2],V.shape[-1]).float(); V=V/(V.norm(dim=-1,keepdim=True)+1e-8)
            idx=torch.randint(0,V.shape[0],(2000,2)); return (V[idx[:,0]]*V[idx[:,1]]).sum(-1).mean().item()
        kcos.append(cos(kw["k"].cpu())); vcos.append(cos(kw["v"].cpu()))
        rbar.append(float(torch.exp(kw["g"].float().mean())))
    rows[name]={c:float(np.mean(er[c])) for c in conds}
    meta[name]={"key_cos":float(np.mean(kcos)),"val_cos":float(np.mean(vcos)),"rbar":float(np.mean(rbar)),"seqlen":int(ids.shape[1])}
    print("%-11s erank: "%name+" ".join("%s=%.1f"%(c,rows[name][c]) for c in conds)+" | key_cos %.2f val_cos %.2f rbar %.2f"%(meta[name]["key_cos"],meta[name]["val_cos"],meta[name]["rbar"]))
np.save("/root/F9_data.npy",{"rows":rows,"meta":meta},allow_pickle=True)
# plot: erank by datatype x condition
fig,ax=plt.subplots(figsize=(8,4)); dn=list(data); x=np.arange(len(dn)); w=0.16
for j,c in enumerate(conds): ax.bar(x+(j-2)*w,[rows[d][c] for d in dn],w,label=c)
ax.set_xticks(x); ax.set_xticklabels(dn); ax.set_ylabel("effective rank (layer-avg)"); ax.axhline(128,ls=":",c="gray",lw=.7)
ax.set_title("erank by input type x intervention (GDN2-370m 6B)"); ax.legend(fontsize=8,ncol=5,frameon=False)
plt.tight_layout(); plt.savefig("/root/F9_datatypes.png",dpi=200); print("saved F9_datatypes.png")
