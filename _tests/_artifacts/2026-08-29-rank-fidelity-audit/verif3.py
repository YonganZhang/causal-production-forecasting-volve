import json, numpy as np, sys
sys.path.insert(0,'_code')
exec(open('/tmp/verif1.py').read().split("R=json.load")[0])
ROOT='/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7'
R=json.load(open(ROOT+'/_pipelines/fc_rank_fidelity/rank.json'))
mb=met('ms_base'); W0=mb['water']
for tgt in ('long','short'):
    rows=R['results'][tgt]['candidates']
    key='oil' if tgt=='long' else 'short'
    v=np.array([met(c['sim_key'])[key] for c in rows])
    wd=np.array([c['water_dev'] for c in rows])
    o=np.argsort(-v)
    print('===',tgt,' elasticity',R['results'][tgt]['water_to_objective_elasticity']['value'],
          'n_pairs',R['results'][tgt]['water_to_objective_elasticity']['n_pairs'])
    el=R['results'][tgt]['water_to_objective_elasticity']['value']
    print(' rank  cand   true        gap_to_next_pct   water_dev%   max_water_effect_on_gap%')
    for a,b in zip(o,o[1:]):
        gap=(v[a]-v[b])/v[b]*100
        eff=abs(wd[a]-wd[b])*100*el
        flag='  <== FLIPPABLE BY WATER' if eff>gap else ''
        print(f'  {tgt}#{a} > {tgt}#{b}: gap {gap:.4f}%  wdev {wd[a]*100:+.4f}/{wd[b]*100:+.4f}  water_effect {eff:.4f}%{flag}')
    # independent elasticity: pool all calib points across candidates, within-candidate differencing
    dx,dy=[],[]
    for c in rows:
        tp=[(met(k)['water'],met(k)[key]) for _,_,k,_ in c['calib_trace']]
        for u in range(len(tp)):
            for w2 in range(u+1,len(tp)):
                lw=np.log(tp[w2][0]/tp[u][0]); lo=np.log(tp[w2][1]/tp[u][1])
                if 1e-6<abs(lw)<0.02: dx.append(lw); dy.append(lo)
    dx,dy=np.array(dx),np.array(dy)
    print(f'  my elasticity {dx@dy/(dx@dx):.4f} n={len(dx)}')
    # widen window
    dx2,dy2=[],[]
    for c in rows:
        tp=[(met(k)['water'],met(k)[key]) for _,_,k,_ in c['calib_trace']]
        for u in range(len(tp)):
            for w2 in range(u+1,len(tp)):
                lw=np.log(tp[w2][0]/tp[u][0]); lo=np.log(tp[w2][1]/tp[u][1])
                if 1e-8<abs(lw)<0.2: dx2.append(lw); dy2.append(lo)
    dx2,dy2=np.array(dx2),np.array(dy2)
    print(f'  elasticity wide window {dx2@dy2/(dx2@dx2):.4f} n={len(dx2)}')
