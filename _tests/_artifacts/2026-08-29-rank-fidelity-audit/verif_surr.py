import sys, json, numpy as np, torch
sys.path.insert(0,'/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7/_code')
import forecast_gen as FG, norne_bulk as NB
from fc_mech import PosNet
from pathlib import Path
ROOT=Path('/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7')
OUT=ROOT/'_pipelines/fc_rank_fidelity'
CONFIRM=Path('/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_fc_confirm')
NPH,NT=2,FG.FC_N; DAYS=FG.FC_GRID
WT=np.gradient(DAYS).astype(float); WT[0]*=.5; WT[-1]*=.5
MS=((DAYS-DAYS[0])<=3*365.25).astype(float)
def load_dir(d,max_n=0):
    sh=sorted((d/'shards').glob('*.npz'))
    if max_n: sh=sh[:max_n]
    TH,Y=[],[]
    for p in sh:
        z=np.load(p); ob=z['obs']
        TH.append(z['theta'].ravel())
        Y.append(np.stack([[ob[(i*3+k)*NT:(i*3+k+1)*NT] for k in (0,1)] for i in range(len(NB.PRODUCERS))]))
    return np.stack(TH).astype(np.float32),np.stack(Y).astype(np.float32)
TH,Y=load_dir(FG.OUT,8000)
print('n_train_pool shards:',len(TH))
live=~(np.abs(Y).max(axis=(0,3))<1e-9).all(1); Y=Y[:,live]; nw=int(live.sum()); n=len(TH)
idx=np.random.default_rng(0).permutation(n); te,pool=idx[:400],idx[600:]; tr=pool[:7400]
Yf=Y.reshape(n,-1); NCH=nw*NPH
ym,ys=Yf[tr].mean(0),Yf[tr].std(0)+1e-8
xm,xs=TH[tr].mean(0),TH[tr].std(0)+1e-8
dev='cpu'
torch.manual_seed(0)
net=PosNet(TH.shape[1],NCH,NT,'fourier',n_bands=16).to(dev)
net.load_state_dict(torch.load(OUT/'surrogate.pt',map_location=dev)); net.eval()
def predict(THq):
    o=[]
    with torch.no_grad():
        for i in range(0,len(THq),256):
            z=torch.tensor(((THq[i:i+256]-xm)/xs).astype(np.float32),device=dev)
            p=net(z).cpu().numpy().reshape(len(z),-1)*ys+ym
            o.append(p.reshape(len(z),nw,NPH,NT)[:,:,0,:].sum(1))
    return np.concatenate(o)
def obj(c,w): return (c*WT*(MS if w=='short' else 1.0)).sum(-1)
ct=obj(Yf[te].reshape(-1,nw,NPH,NT)[:,:,0,:].sum(1),'long'); cp=obj(predict(TH[te]),'long')
print('heldout test relerr: %.6f  reported %.6f'%(np.mean(np.abs(cp-ct)/ct),0.0034850328748575582))
THc,Yc=load_dir(CONFIRM); Yc=Yc[:,live]
print('confirm n =',len(THc))
ctl=obj(Yc[:,:,0,:].sum(1),'long'); cts=obj(Yc[:,:,0,:].sum(1),'short')
Pc=predict(THc); cpl=obj(Pc,'long'); cps=obj(Pc,'short')
rel=np.abs(cpl-ctl)/ctl
print('confirm500 relerr(long): %.6f  reported %.6f'%(rel.mean(),0.003691762247084642))
print('confirm relerr percentiles:',np.round(np.percentile(rel,[50,90,95,99,100])*100,3),'%')
print('frac of confirm with relerr >= 1.435%%: %.4f'%np.mean(rel>=0.014351))
rels=np.abs(cps-cts)/cts
print('confirm relerr(short) mean %.4f%% pcts'%(rels.mean()*100),np.round(np.percentile(rels,[50,90,95,99,100])*100,3))
print('frac short confirm relerr >= 1.724%%: %.4f'%np.mean(rels>=0.017243))
# candidate surrogate predictions
R=json.load(open(OUT/'rank.json'))
for tgt in ('long','short'):
    rows=R['results'][tgt]['candidates']
    thc=np.array([r['theta_cal'] for r in rows],np.float32)
    thr=np.array([r['theta_raw'] for r in rows],np.float32)
    sc=obj(predict(thc),tgt); sr=obj(predict(thr),tgt)
    rep_c=np.array([r['surr_cal'] for r in rows]); rep_r=np.array([r['surr_raw'] for r in rows])
    print(f'{tgt}: surr_cal max reldiff {np.abs(sc-rep_c).max()/np.abs(rep_c).max():.3e}; surr_raw {np.abs(sr-rep_r).max()/np.abs(rep_r).max():.3e}')
    base=float(obj(predict(np.zeros((1,24),np.float32)),tgt)[0])
    print(f'   base_surr={base:.1f}')
    # theta norm vs training distribution
    print(f'   cand |theta| L2 mean {np.linalg.norm(thc,axis=1).mean():.3f}  training |theta| L2 mean {np.linalg.norm(TH[tr],axis=1).mean():.3f} p99 {np.percentile(np.linalg.norm(TH[tr],axis=1),99):.3f} max {np.linalg.norm(TH[tr],axis=1).max():.3f}')
    print(f'   cand theta min {thc.min():.3f} max {thc.max():.3f} | training min {TH[tr].min():.3f} max {TH[tr].max():.3f}')
# a-priori SNR check
for tgt in ('long','short'):
    ss=R['results'][tgt]['surr_spread_pct']; ts=R['results'][tgt]['true_spread_pct']
    pe=R['results'][tgt]['point_accuracy_at_candidates']['relerr_mean']*100
    print(f'{tgt}: SNR_true={ts/pe:.2f}  SNR_apriori(surr_spread/confirm_err)={ss/0.3692:.2f}  SNR_apriori2(surr_spread/point_err)={ss/pe:.2f}')
