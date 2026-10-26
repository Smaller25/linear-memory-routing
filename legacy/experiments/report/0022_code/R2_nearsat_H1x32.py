import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, argparse, warnings, time
warnings.filterwarnings("ignore")
IGNORE=-100
def make_mqar(n,vocab,kv,seqlen,seed):
    rng=np.random.default_rng(seed); half=vocab//2
    kc=np.arange(1,half); vc=np.arange(half,vocab-1)
    ctx=2*kv; assert 4*kv<=seqlen
    inp=np.zeros((n,seqlen),dtype=np.int64); lab=np.full((n,seqlen+1),IGNORE,dtype=np.int64)
    for i in range(n):
        ks=rng.choice(kc,kv,replace=False); vs=rng.choice(vc,kv,replace=False)
        inp[i,0:ctx:2]=ks; inp[i,1:ctx:2]=vs
        space=(seqlen-ctx); gaps=np.sort(rng.choice(space//2,kv,replace=False))*2; qpos=ctx+gaps
        for j in range(kv): inp[i,qpos[j]]=ks[j]; lab[i,qpos[j]]=vs[j]
    return torch.tensor(inp),torch.tensor(lab[:,:seqlen])
def l2n(x): return x/(x.norm(dim=-1,keepdim=True)+1e-6)
class DeltaLayer(nn.Module):
    def __init__(self,d,H,hd,variant):
        super().__init__(); self.H=H; self.hd=hd; self.variant=variant
        self.qp=nn.Linear(d,H*hd,False); self.kp=nn.Linear(d,H*hd,False); self.vp=nn.Linear(d,H*hd,False); self.bp=nn.Linear(d,H,True)
        self.oc=nn.Linear(H*hd,d,False); self.conv=nn.Conv1d(3*H*hd,3*H*hd,4,groups=3*H*hd,padding=3)
        if variant=="C2": self.U=nn.Parameter(torch.eye(hd).unsqueeze(0).repeat(H,1,1))
    def transform(self,k,q):
        if self.variant=="C1":
            K=k.reshape(-1,self.H,self.hd); C=torch.einsum('nhi,nhj->hij',K,K)/K.shape[0]
            C=C+1e-3*torch.eye(self.hd,device=k.device)
            ev,eV=torch.linalg.eigh(C.float()); W=(eV@torch.diag_embed(ev.clamp_min(1e-6).rsqrt())@eV.transpose(-1,-2)).to(k.dtype)
            k=torch.einsum('bthi,hij->bthj',k,W); q=torch.einsum('bthi,hij->bthj',q,W)
        elif self.variant=="C2":
            k=torch.einsum('bthi,hij->bthj',k,self.U); q=torch.einsum('bthi,hij->bthj',q,self.U)
        return l2n(k),l2n(q)
    def forward(self,x):
        B,T,_=x.shape
        qkv=torch.cat([self.qp(x),self.kp(x),self.vp(x)],-1).transpose(1,2)
        qkv=self.conv(qkv)[...,:T].transpose(1,2); q,k,v=qkv.chunk(3,-1)
        q=q.reshape(B,T,self.H,self.hd); k=k.reshape(B,T,self.H,self.hd); v=v.reshape(B,T,self.H,self.hd)
        beta=self.bp(x).sigmoid(); k=l2n(k); q=l2n(q); k,q=self.transform(k,q)
        S=torch.zeros(B,self.H,self.hd,self.hd,device=x.device,dtype=x.dtype); outs=[]
        for t in range(T):
            kt,vt,qt,bt=k[:,t],v[:,t],q[:,t],beta[:,t]
            pred=torch.einsum('bhij,bhj->bhi',S,kt)
            S=S+bt[...,None,None]*torch.einsum('bhi,bhj->bhij',(vt-pred),kt)
            outs.append(torch.einsum('bhij,bhj->bhi',S,qt))
        o=torch.stack(outs,1).reshape(B,T,-1); self.lastS=S.detach(); self.kreg=None
        if self.variant=="C4":
            K=k.reshape(-1,self.H,self.hd); G=torch.einsum('nhi,nhj->hij',K,K)/K.shape[0]
            G=G/(torch.diagonal(G,dim1=-2,dim2=-1).mean(-1,keepdim=True)[...,None]+1e-6)
            self.kreg=((G-torch.eye(self.hd,device=x.device))**2).mean()
        return self.oc(o)
class Model(nn.Module):
    def __init__(self,vocab,d=96,H=1,hd=32,L=2,variant="baseline"):
        super().__init__(); self.emb=nn.Embedding(vocab,d)
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.layers=nn.ModuleList([DeltaLayer(d,H,hd,variant) for _ in range(L)])
        self.nf=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,False)
    def forward(self,x):
        h=self.emb(x)
        for n,l in zip(self.norms,self.layers): h=h+l(n(h))
        return self.head(self.nf(h))
def erank(M):
    s=torch.linalg.svdvals(M.float()); s=s/(s.sum()+1e-9); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
def run(variant,steps=2000,seed=0):
    torch.manual_seed(seed); dev="cuda"; V=512
    m=Model(V,variant=variant).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.1,betas=(0.9,0.95)); tk=[16,32,48]; t0=time.time()
    for step in range(1,steps+1):
        kv=tk[step%len(tk)]; ids,lab=make_mqar(64,V,kv,256,step); ids,lab=ids.to(dev),lab.to(dev)
        lg=m(ids); loss=F.cross_entropy(lg.reshape(-1,V),lab.reshape(-1),ignore_index=IGNORE)
        reg=sum((l.kreg for l in m.layers if l.kreg is not None),torch.zeros((),device=dev))
        (loss+0.5*reg).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad()
        if step%500==0: print("  [%s] step %d loss %.3f (%.0fs)"%(variant,step,loss.item(),time.time()-t0),flush=True)
    m.eval(); res={}
    with torch.no_grad():
        for kv in [32,48,64]:
            ids,lab=make_mqar(192,V,kv,256,9000+kv); ids,lab=ids.to(dev),lab.to(dev); c=t=0
            for i in range(0,192,64):
                lg=m(ids[i:i+64]); p=lg.argmax(-1); msk=lab[i:i+64]!=IGNORE
                c+=(p[msk]==lab[i:i+64][msk]).sum().item(); t+=msk.sum().item()
            res[kv]=c/max(t,1)
        er=float(np.mean([erank(l.lastS[0,h]) for l in m.layers for h in range(l.H)]))
    return res,er
if __name__=="__main__":
    print("RESULT_HEADER variant | recall kv32/48/64 | erank",flush=True)
    for var in ["baseline","C4","C1","C2"]:
        res,er=run(var); print("RESULT %-9s | %.3f / %.3f / %.3f | %.2f"%(var,res[32],res[48],res[64],er),flush=True)
    print("ALL_DONE",flush=True)
