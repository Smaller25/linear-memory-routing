import sys; sys.path.insert(0,"/root/vfla")
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, argparse, warnings
warnings.filterwarnings("ignore")
from fla.layers.gdn2 import GatedDeltaNet2
from fla.models.utils import Cache
IGNORE=-100
def make_mqar(n,vocab,kv,seqlen,seed):
    rng=np.random.default_rng(seed); half=vocab//2
    kc=np.arange(1,half); vc=np.arange(half,vocab-1); ctx=2*kv; assert 4*kv<=seqlen
    inp=np.zeros((n,seqlen),dtype=np.int64); lab=np.full((n,seqlen+1),IGNORE,dtype=np.int64)
    for i in range(n):
        ks=rng.choice(kc,kv,replace=False); vs=rng.choice(vc,kv,replace=False)
        inp[i,0:ctx:2]=ks; inp[i,1:ctx:2]=vs
        gaps=np.sort(rng.choice((seqlen-ctx)//2,kv,replace=False))*2; qpos=ctx+gaps
        for j in range(kv): inp[i,qpos[j]]=ks[j]; lab[i,qpos[j]]=vs[j]
    return torch.tensor(inp),torch.tensor(lab[:,:seqlen])
class GDN2LM(nn.Module):
    def __init__(self,vocab,d=128,L=2,hd=32,H=1,use_c4=False):
        super().__init__(); self.H=H; self.hd=hd; self.use_c4=use_c4; self._kbuf=[]
        self.embed=nn.Embedding(vocab,d)
        self.norms=nn.ModuleList([nn.RMSNorm(d) for _ in range(L)])
        self.mixers=nn.ModuleList([GatedDeltaNet2(hidden_size=d,head_dim=hd,num_heads=H,expand_v=1.0) for _ in range(L)])
        for i,m in enumerate(self.mixers):
            m.layer_idx=i
            if use_c4: m.k_proj.register_forward_hook(lambda mod,inp,out: self._kbuf.append(out))
        self.normf=nn.RMSNorm(d); self.head=nn.Linear(d,vocab,False)
    def forward(self,ids):
        self._kbuf=[]; h=self.embed(ids)
        for norm,mix in zip(self.norms,self.mixers): h=h+mix(hidden_states=norm(h))[0]
        return self.head(self.normf(h))
    def kreg(self):
        if not self._kbuf: return torch.zeros((),device=self.head.weight.device)
        reg=0.0
        for k in self._kbuf:
            k=k.reshape(k.shape[0]*k.shape[1],self.H,self.hd); k=k/(k.norm(dim=-1,keepdim=True)+1e-6)
            G=torch.einsum('nhi,nhj->hij',k,k)/k.shape[0]
            G=G/(torch.diagonal(G,dim1=-2,dim2=-1).mean(-1,keepdim=True)[...,None]+1e-6)
            reg=reg+((G-torch.eye(self.hd,device=k.device))**2).mean()
        return reg/len(self._kbuf)
    @torch.no_grad()
    def erank(self,ids):
        cache=Cache.from_legacy_cache(None); h=self.embed(ids)
        for i,(norm,mix) in enumerate(zip(self.norms,self.mixers)):
            o,_,cache=mix(hidden_states=norm(h),past_key_values=cache,use_cache=True); h=h+o
        st=cache[len(self.mixers)-1]["recurrent_state"]  # [B,H,K,V]
        ers=[]
        for hh in range(st.shape[1]):
            s=torch.linalg.svdvals(st[0,hh].float()); s=s/(s.sum()+1e-9); ers.append(float(torch.exp(-(s*(s+1e-12).log()).sum())))
        return float(np.mean(ers))
def run(use_c4,steps,seed,V=512,d=128,L=2,hd=32,H=1,lam=0.5,HD=None,LAM=None):
    hd = HD if HD else hd
    lam = LAM if LAM is not None else lam
    torch.manual_seed(seed); dev="cuda"
    m=GDN2LM(V,d,L,hd,H,use_c4).to(dev).to(torch.bfloat16)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.1,betas=(0.9,0.95)); tk=[16,32,48]; t0=time.time(); tag="C4" if use_c4 else "base"
    for step in range(1,steps+1):
        kv=tk[step%len(tk)]; ids,lab=make_mqar(32,V,kv,256,step); ids,lab=ids.to(dev),lab.to(dev)
        lg=m(ids).float(); loss=F.cross_entropy(lg.reshape(-1,V),lab.reshape(-1),ignore_index=IGNORE)
        tot=loss+ (lam*m.kreg() if use_c4 else 0.0)
        tot.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad()
        if step%1000==0: print("  [%s] step %d loss %.3f (%.0fs)"%(tag,step,loss.item(),time.time()-t0),flush=True)
    m.eval(); res={}
    with torch.no_grad():
        for kv in [32,48,64]:
            ids,lab=make_mqar(192,V,kv,256,9000+kv); ids,lab=ids.to(dev),lab.to(dev); c=t=0
            for i in range(0,192,64):
                p=m(ids[i:i+64]).float().argmax(-1); msk=lab[i:i+64]!=IGNORE
                c+=(p[msk]==lab[i:i+64][msk]).sum().item(); t+=msk.sum().item()
            res[kv]=c/max(t,1)
        er=m.erank(make_mqar(8,V,48,256,7)[0].to(dev))
    return res,er
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=4000); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--hd",type=int,default=32); ap.add_argument("--lam",type=float,default=0.5); a=ap.parse_args()
    if a.smoke:
        print("SMOKE start",flush=True); r,e=run(False,30,0); print("SMOKE base",r,e,flush=True); r,e=run(True,30,0); print("SMOKE C4",r,e,flush=True); print("SMOKE_OK",flush=True); sys.exit()
    print("RESULT_HEADER hd=%d lam=%.2f | recall kv32/48/64 | erank"%(a.hd,a.lam),flush=True)
    for c4 in [False,True]:
        res,er=run(c4,a.steps,0,HD=a.hd,LAM=a.lam); print("RESULT %-9s | %.3f / %.3f / %.3f | %.2f"%("C4" if c4 else "baseline",res[32],res[48],res[64],er),flush=True)
    print("ALL_DONE",flush=True)
