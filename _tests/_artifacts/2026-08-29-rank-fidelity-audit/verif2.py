import json, numpy as np, sys
from scipy import stats
sys.path.insert(0,'_code')
ROOT='/mnt/data/yongan-admin-2/projects/自己-产能预测-04a0c7'
R=json.load(open(ROOT+'/_pipelines/fc_rank_fidelity/rank.json'))
sys.path.insert(0,'/tmp')
exec(open('/tmp/verif1.py').read().split("R=json.load")[0])
mb=met('ms_base')
for tgt in ('long','short'):
    rows=R['results'][tgt]['candidates']
    true=np.array([met(c['sim_key'])['oil' if tgt=='long' else 'short'] for c in rows])
    surr=np.array([c['surr_cal'] for c in rows])
    surr_raw=np.array([c['surr_raw'] for c in rows])
    rho,_=stats.spearmanr(surr,true); tau,_=stats.kendalltau(surr,true)
    os_,ot=np.argsort(-surr),np.argsort(-true)
    hits={k: len(set(os_[:k].tolist())&set(ot[:k].tolist()))/k for k in (1,3,5)}
    pick,best=int(os_[0]),int(ot[0])
    reg=(true[best]-true[pick])/true[best]*100
    m=R['results'][tgt]['rank_fidelity_on_simulated_theta']
    print(f'--- {tgt}')
    print(f'  rho  recomputed {rho:+.4f}  reported {m["spearman_rho"]:+.4f}')
    print(f'  tau  recomputed {tau:+.4f}  reported {m["kendall_tau"]:+.4f}')
    print(f'  hit  recomputed {hits}  reported {m["hit_rate"]}')
    print(f'  pick {tgt}#{pick} true_rank {int(np.where(ot==pick)[0][0])+1}/{len(rows)}  reported {m["surr_pick_true_rank"]}')
    print(f'  regret recomputed {reg:.4f}%  reported {m["top1_regret_pct"]:.4f}%')
    spread=(true.max()/true.min()-1)*100
    print(f'  true spread {spread:.4f}%  reported {R["results"][tgt]["true_spread_pct"]:.4f}%')
    # npv regret
    for k in ('npv0','npv8','npv15'):
        v=(met(rows[best]['sim_key'])[k]-met(rows[pick]['sim_key'])[k])/1e6
        print(f'  regret {k}: recomputed {v:+.3f} M$  reported {R["results"][tgt]["top1_regret"]["musd"][k]:+.3f}')
    # permutation p
    rng=np.random.default_rng(0)
    perm=np.array([stats.spearmanr(surr,rng.permutation(true)).statistic for _ in range(20000)])
    print(f'  perm p (my seed) {np.mean(np.abs(perm)>=abs(rho)):.4f}  reported {m["uncertainty"]["perm_p_two_sided"]:.4f}')
    # point accuracy
    relerr=np.abs(surr-true)/true
    print(f'  point relerr mean {relerr.mean():.4%} max {relerr.max():.4%}  reported {R["results"][tgt]["point_accuracy_at_candidates"]}')
    print(f'  SNR = {spread/ (relerr.mean()*100):.3f}  reported {R["results"][tgt]["signal_to_error_ratio"]:.3f}')
    # raw theta ranking
    rr,_=stats.spearmanr(surr_raw,true); rt,_=stats.kendalltau(surr_raw,true)
    mr=R['results'][tgt]['rank_fidelity_on_optimizer_raw_theta']
    print(f'  raw-theta rho {rr:+.4f}/{mr["spearman_rho"]:+.4f}  tau {rt:+.4f}/{mr["kendall_tau"]:+.4f}')
    # distinct sims
    keys=set(); nev=0
    for c in rows:
        nev+=c['n_eval']
        for e in c['calib_trace']: keys.add(e[2])
    print(f'  n_eval sum {nev} (reported {R["results"][tgt]["n_opm_evals"]}); DISTINCT sim keys {len(keys)}')
