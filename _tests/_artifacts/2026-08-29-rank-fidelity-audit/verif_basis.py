import json, numpy as np, sys, datetime as dt
from scipy import stats
sys.path.insert(0,'/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7/_code')
import norne_bulk as NB
from fc_npv import BRENT, BBL_PER_M3, T_START
ROOT='/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7'
SIM=ROOT+'/_pipelines/fc_decide/sim/'
NT=40; DAYS=np.linspace(3312.,8091.,40); n=len(NB.PRODUCERS)
W=np.gradient(DAYS).astype(float); W[0]*=.5; W[-1]*=.5
ms=(DAYS-DAYS[0])<=3*365.25; t_cut=DAYS[0]+3*365.25
def M(key):
    d=np.load(SIM+key+'.npz'); ob=d['obs']
    oil=np.stack([ob[(i*3)*NT:(i*3+1)*NT] for i in range(n)]).sum(0)
    fc=d['field_cum'].astype(np.float64); fopt,fwit=fc[0],fc[1]
    dq=np.diff(fopt); tm=.5*(DAYS[:-1]+DAYS[1:])
    yr=np.array([(T_START+dt.timedelta(days=float(x))).year for x in tm]); pr=np.array([BRENT[y] for y in yr])
    t=(tm-DAYS[0])/365.25; rev=dq*BBL_PER_M3*pr
    return dict(water=float(fwit[-1]-fwit[0]), oil=float(fopt[-1]-fopt[0]),
                short=float(np.interp(t_cut,DAYS,fopt)-fopt[0]),
                water_trap=float(np.trapezoid(d['inj_actual'].sum(0),DAYS)),
                oil_trap=float((oil*W).sum()), short_trap=float((oil*W*ms).sum()),
                npv8=float((rev/1.08**t).sum()), npv0=float(rev.sum()), npv15=float((rev/1.15**t).sum()))
R=json.load(open(ROOT+'/_pipelines/fc_rank_fidelity/rank.json'))
b=M('ms_base')
print('BASELINE  oil FOPT %.0f vs trap %.0f (%+.3f%%)'%(b['oil'],b['oil_trap'],(b['oil_trap']/b['oil']-1)*100))
print('          short FOPT %.0f vs trap %.0f (%+.3f%%)'%(b['short'],b['short_trap'],(b['short_trap']/b['short']-1)*100))
print('          water FWIT %.0f vs trap %.0f (%+.3f%%)'%(b['water'],b['water_trap'],(b['water_trap']/b['water']-1)*100))
print()
print('%-9s %12s %12s | %10s %10s'%('cand','dev_TRAP(used)','dev_FWIT(correct)','','' ))
nbad=0
for tgt in ('long','short'):
    print('===',tgt)
    for c in R['results'][tgt]['candidates']:
        m=M(c['sim_key'])
        dt_=m['water_trap']/b['water_trap']-1; df=m['water']/b['water']-1
        ok='OK ' if abs(df)<=5e-4 else 'FAIL'
        if abs(df)>5e-4: nbad+=1
        print(f'  {tgt}#{c["i"]:<2} {c["sim_key"]:<17} trap {dt_:+.5%}  FWIT {df:+.5%}  {ok}  ratio {abs(df)/max(abs(dt_),1e-12):.1f}x')
print(f'\n>>> Candidates FAILING the +-0.05% rule on FWIT (the correct measure): {nbad}/20')
print()
for tgt in ('long','short'):
    rows=R['results'][tgt]['candidates']
    surr=np.array([c['surr_cal'] for c in rows])
    for basis in ('trap','FOPT'):
        k=('oil_trap' if tgt=='long' else 'short_trap') if basis=='trap' else ('oil' if tgt=='long' else 'short')
        true=np.array([M(c['sim_key'])[k] for c in rows])
        rho,_=stats.spearmanr(surr,true); tau,_=stats.kendalltau(surr,true)
        os_,ot=np.argsort(-surr),np.argsort(-true)
        pick,best=int(os_[0]),int(ot[0])
        reg=(true[best]-true[pick])/true[best]*100
        hit1=float(pick==best); hit3=len(set(os_[:3].tolist())&set(ot[:3].tolist()))/3
        sp=(true.max()/true.min()-1)*100
        print(f'{tgt:5s} basis={basis:5s} rho={rho:+.4f} tau={tau:+.4f} hit1={hit1:.2f} hit3={hit3:.2f} '
              f'pick=#{rows[pick]["i"]} best=#{rows[best]["i"]} regret={reg:.4f}% spread={sp:.4f}%')
    # NPV regret on FOPT basis
    k='oil' if tgt=='long' else 'short'
    true=np.array([M(c['sim_key'])[k] for c in rows]); os_,ot=np.argsort(-surr),np.argsort(-true)
    p,bb=int(os_[0]),int(ot[0])
    print('      NPV regret (FOPT-based, exact):', {kk: round((M(rows[bb]['sim_key'])[kk]-M(rows[p]['sim_key'])[kk])/1e6,2) for kk in ('npv0','npv8','npv15')},
          ' | reported(trap-based):', {kk: round(v,2) for kk,v in R['results'][tgt]['top1_regret']['musd'].items()})
