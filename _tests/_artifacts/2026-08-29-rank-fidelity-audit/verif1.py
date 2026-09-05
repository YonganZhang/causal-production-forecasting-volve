import json, datetime as dt, numpy as np, sys
sys.path.insert(0,'_code')
ROOT='/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7'
SIM=ROOT+'/_pipelines/fc_decide/sim/'
# --- independent constants (hard-coded, not imported) ---
FC_T0,FC_T1,FC_N=3312.0,8091.0,40
DAYS=np.linspace(FC_T0,FC_T1,FC_N)
T_START=dt.date(1997,11,6); BBL=6.2898
BRENT={2006:65.16,2007:72.44,2008:96.94,2009:61.74,2010:79.61,2011:111.26,2012:111.63,
       2013:108.56,2014:98.97,2015:52.32,2016:43.64,2017:54.13,2018:71.31,2019:64.21,2020:41.96}
import fc_npv
print('BRENT match fc_npv:', all(fc_npv.BRENT.get(k)==v for k,v in BRENT.items()), 'len',len(fc_npv.BRENT))
BRENT=fc_npv.BRENT
import norne_bulk as NB
NPROD=len(NB.PRODUCERS)
W=np.gradient(DAYS).astype(float); W[0]*=.5; W[-1]*=.5
yrs=np.array([(T_START+dt.timedelta(days=float(x))).year for x in DAYS])
price=np.array([BRENT[y] for y in yrs])
t=(DAYS-DAYS[0])/365.25
MS=((DAYS-DAYS[0])<=3*365.25)

def met(key):
    d=np.load(SIM+key+'.npz')
    ob=d['obs']
    oil=np.stack([ob[(i*3)*FC_N:(i*3+1)*FC_N] for i in range(NPROD)]).sum(0)
    rev=oil*W*BBL*price
    return dict(oil=float((oil*W).sum()), short=float((oil*W*MS).sum()),
                water=float(np.trapezoid(d['inj_actual'].sum(0),DAYS)),
                npv0=float(rev.sum()), npv8=float((rev/1.08**t).sum()),
                npv15=float((rev/1.15**t).sum()), theta=d['theta'].astype(np.float64).ravel(),
                end=float(d['sim_days_end']), nan=int(np.isnan(ob).sum()))

R=json.load(open(ROOT+'/_pipelines/fc_rank_fidelity/rank.json'))
mb=met('ms_base'); print('ms_base recomputed:', {k:round(v,3) for k,v in mb.items() if k not in('theta',)})
print('rank.json baseline:', R['baseline_metrics'])
W0=mb['water']
print('W0 vs reported baseline_water:', W0, R['water_constraint']['baseline_water'], W0-R['water_constraint']['baseline_water'])
print()
bad=0
for tgt in ('long','short'):
    print('=== ',tgt)
    for c in R['results'][tgt]['candidates']:
        m=met(c['sim_key'])
        dth=np.abs(m['theta']-np.array(c['theta_cal'])).max()
        errs=[]
        for k in ('oil','short','npv0','npv8','npv15','water'):
            rel=abs(m[k]-c[k])/abs(c[k])
            if rel>1e-9: errs.append(f'{k} rel={rel:.2e}')
        wd=m['water']/W0-1
        if abs(wd-c['water_dev'])>1e-7: errs.append(f'water_dev {wd:.6f} vs {c["water_dev"]:.6f}')
        if dth>2e-7: errs.append(f'theta_max_diff={dth:.2e}')
        if m['nan']: errs.append(f'NaN in obs={m["nan"]}')
        if m['end']<FC_T1-1: errs.append(f'sim_days_end={m["end"]}')
        st='OK' if not errs else 'MISMATCH: '+'; '.join(errs)
        bad+= (0 if not errs else 1)
        print(f'{tgt}#{c["i"]:<2} {c["sim_key"]:<18} water_dev={wd:+.5%} end={m["end"]:.0f} {st}')
print('\nTOTAL MISMATCH CANDIDATES:',bad)
