import sys, json, numpy as np
sys.path.insert(0,'/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7/_code')
import fc_decide as FD
from concurrent.futures import ThreadPoolExecutor
ROOT='/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7'
SIM=ROOT+'/_pipelines/fc_decide/sim/'
class A: threads=8
pairs=[('rf_long_0_c5','audit_long0'),('rf_long_4_c4','audit_long4'),('rf_short_6_c16','audit_short6')]
def go(p):
    src,dst=p
    th=np.load(SIM+src+'.npz')['theta']
    FD._run_sim(dst, th, A())
    print('done',dst,flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:
    list(ex.map(go,pairs))
for src,dst in pairs:
    a=np.load(SIM+src+'.npz'); b=np.load(SIM+dst+'.npz')
    for k in ('obs','inj_actual','field_cum'):
        d=np.abs(a[k].astype(float)-b[k].astype(float)).max()
        rel=d/ (np.abs(a[k]).max()+1e-12)
        print(f'{src} vs {dst}  {k}: maxabsdiff={d:.6g} rel={rel:.3e}')
