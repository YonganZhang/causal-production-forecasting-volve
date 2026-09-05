"""含水成本的经济重评价 —— 不跑新模拟，复用已有算例的 FOPT/FWIT/FWPT。

🔴 动机（2026-08-29，用户指出）：此前 NPV 只算油收入（fc_fix.py:61 `rev = dq*BBL*price`），
   注水与采出水处理**零成本**。后果是"要不要多注水"这个问题在数学上退化——
   水免费且能换油，最优解必然是注到边界。之前用 softmax 把总水量**结构性锁死**，
   实际上是在用一个硬约束替代缺失的成本项。

   本脚本把水的成本显式放进目标，于是"注多少水"成为**内生**决策，而不是外加约束。

成本口径（参数化，不写死单一值 —— 与折现率一样做敏感性扫描）：
   注水成本   c_inj  USD/bbl   （举升、增压、处理、泵功）
   采出水处理 c_prod USD/bbl   （分离、处理、回注或排放）
   北海文献常见区间各约 0.5~4 USD/bbl，故扫 0~4。
"""
import numpy as np, datetime as dt, json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fc_fix import BRENT, T_START, BBL      # 🔴 复用已验证常量，不再自建
# 教训：本脚本第一版自建 T_START=2006-01-01，但 Norne 1997 投产，
#      FC_GRID 的 3312~8091 天应落在 2006~2019。自建常量把油价整体错位，
#      基准 NPV8 算成 1798M(真值 2857M)。凡已有真源的常量一律 import。

def econ(npz, c_inj=0.0, c_prod=0.0, rates=(0.0,0.02,0.08,0.15)):
    """一条算例在给定水价下的 NPV。收入-成本都按格点差分后逐点折现。"""
    d = np.load(npz)
    fc, days = d["field_cum"], d["grid_days"]
    fopt, fwit, fwpt = fc[0]-fc[0][0], fc[1]-fc[1][0], fc[2]-fc[2][0]
    yrs = np.array([(T_START+dt.timedelta(days=float(x))).year for x in days])
    price = np.array([BRENT[y] for y in yrs])
    t = (days-days[0])/365.25
    d_oil  = np.diff(fopt, prepend=fopt[0])
    d_winj = np.diff(fwit, prepend=fwit[0])
    d_wprd = np.diff(fwpt, prepend=fwpt[0])
    cash = d_oil*BBL*price - d_winj*BBL*c_inj - d_wprd*BBL*c_prod
    out = {f"npv{int(r*100)}": float((cash/(1+r)**t).sum()) for r in rates}
    # 前 3 年用插值到精确 3.0 年，消除 fc_fix 的格点 off-by-one(约 155k Sm3)
    out.update(oil=float(fopt[-1]), oil_3y=float(np.interp(3.0, t, fopt)),
               winj=float(fwit[-1]), wprd=float(fwpt[-1]))
    return out

def main():
    sims = sorted(glob.glob("_pipelines/*/sim/*.npz"))
    base = [s for s in sims if Path(s).stem in ("baseline","ms_base","baseline_cal")]
    if not base:
        print("找不到基准算例"); return
    print(f"算例总数 {len(sims)}，基准 {Path(base[0]).name}\n")
    grid = [(0.0,0.0),(1.0,0.5),(2.0,1.0),(3.0,2.0),(4.0,3.0)]
    print(f"{'c_inj/c_prod':>14s} {'基准NPV8(M$)':>13s} {'最佳算例':>26s} {'ΔNPV8(M$)':>11s} {'其注水量vs基准':>14s}")
    print("-"*88)
    rows=[]
    for ci, cp in grid:
        b = econ(base[0], ci, cp)
        best, bv = None, -9e18
        for s in sims:
            if Path(s).stem in ("baseline","ms_base","baseline_cal"): continue
            try: e = econ(s, ci, cp)
            except Exception: continue
            if e["npv8"] > bv: best, bv = (Path(s).stem, e), e["npv8"]
        if best is None: continue
        nm, e = best
        dw = 100*(e["winj"]/b["winj"]-1)
        print(f"{ci:5.1f}/{cp:<8.1f} {b['npv8']/1e6:13,.1f} {nm:>26s} {(bv-b['npv8'])/1e6:+11.1f} {dw:+13.2f}%")
        rows.append(dict(c_inj=ci, c_prod=cp, base_npv8=b["npv8"],
                         best=nm, delta_npv8=bv-b["npv8"], water_dev=dw/100))
    Path("_pipelines/fc_water_econ").mkdir(parents=True, exist_ok=True)
    json.dump({"grid":rows,"note":"含水成本重评价，复用已有模拟，未跑新算例"},
              open("_pipelines/fc_water_econ/sweep.json","w"), indent=1)
    print("\n→ _pipelines/fc_water_econ/sweep.json")

if __name__ == "__main__":
    main()
