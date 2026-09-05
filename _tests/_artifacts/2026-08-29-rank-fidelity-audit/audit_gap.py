import sys, json, numpy as np
sys.path.insert(0,'/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7/_code')
import fc_decide as FD, fc_agent as FA
from concurrent.futures import ThreadPoolExecutor
ROOT='/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7'
SIM=ROOT+'/_pipelines/fc_decide/sim/'
class A: threads=8
R=json.load(open(ROOT+'/_pipelines/fc_rank_fidelity/rank.json'))
W0=FA._metrics('ms_base')['water']
jobs=[]
# short#9 bracket: actual simulated s endpoints from npz
c9=[x for x in R['results']['short']['candidates'] if x['i']==9][0]
th9=np.array(c9['theta_raw'])
lo9,hi9=0.053877432055,0.053878734664
for j,f in enumerate([0.2,0.4,0.6,0.8]):
    jobs.append(('s9',f'audit_s9_{j}',th9,lo9+f*(hi9-lo9)))
c6=[x for x in R['results']['short']['candidates'] if x['i']==6][0]
th6=np.array(c6['theta_raw'])
a=np.load(SIM+'rf_short_6_c19.npz')['theta'].astype(np.float64).ravel()-th6
b=np.load(SIM+'rf_short_6_c17.npz')['theta'].astype(np.float64).ravel()-th6
lo6,hi6=a.mean(),b.mean()
print('short6 bracket',lo6,hi6,flush=True)
for j,f in enumerate([0.15,0.3,0.45,0.6,0.75,0.9]):
    jobs.append(('s6',f'audit_s6_{j}',th6,lo6+f*(hi6-lo6)))
def go(t):
    tag,key,th,s=t
    FD._run_sim(key, np.asarray(th+s,np.float32).reshape(6,4), A())
    m=FA._metrics(key)
    return (tag,key,s,m['water']/W0-1)
with ThreadPoolExecutor(max_workers=4) as ex:
    res=list(ex.map(go,jobs))
for tag,key,s,d in sorted(res):
    print(f'{tag} {key} s={s!r:24} dev={d:+.5%} {"WITHIN +-0.05%" if abs(d)<=5e-4 else ""}',flush=True)
