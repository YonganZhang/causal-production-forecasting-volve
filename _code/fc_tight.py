#!/usr/bin/env python3
"""把主线定版方案的**实测**注水量收紧到 ±0.05%,并重报产油/NPV。

## 为什么

主线 `fc_ms.py` 用的容差是 1%,实际落点 short −0.62%、long +0.12%。
项目已实测过:1% 容差足以制造假赢家(注水多一点自然多产油)。
标题若要说 "same water",必须把**实测**总注水量校到 ±0.05%。

## 做法

沿用 `fc_decide.calib` 的割线法:θ 整体平移 log10 s(等价于所有目标注水率乘 s),
**不改井间份额**,所以"换分法"这个研究对象没被动过。
起点复用 multistart 已有的 `ms_{name}_{k}_c*.npz`(白送 2~3 个割线点)。

裁判是 OPM Flow;NPV 用 `fc_npv.BRENT`(EIA 真实历史年均 Brent)。

用法:
    python fc_tight.py --plan short --threads 12
    python fc_tight.py --collect
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

import forecast_gen as FG
import norne_bulk as NB
import fc_decide as FD
from fc_npv import BRENT, BBL_PER_M3, T_START

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_decide"
SIM = OUT / "sim"
NT, DAYS = FG.FC_N, FG.FC_GRID
N_STAGE, INJ = FG.N_STAGE, FG.INJ_W
SHORT_Y = 3.0
TOL = 5e-4
MAX_NEW = 5   # 可用 --max-new 覆盖


class _A:
    def __init__(self, threads):
        self.threads = threads


def metrics(key):
    """一次模拟 → 产油/前3年/实测注水/NPV。两套口径并列,便于对照。

    - `*_trap` = 主线口径:对 **40 个报告点的瞬时率**做梯形积分(fc_ms / fc_npv 用的)。
    - 其余 = **模拟器自报累计量** FOPT/FWIT(`field_cum`),没有求积误差。

    🔴 实测发现:报告网格 122.5 天/点,率曲线在步内会因井控切换而变,
    梯形法对注水量的残差达 **±0.5%**,是 ±0.05% 容差的 10 倍。
    所以注水量守恒必须用 **FWIT**,不能用率的梯形积分。
    """
    d = np.load(SIM / f"{key}.npz")
    ob = d["obs"]; n = len(NB.PRODUCERS)
    oil = np.stack([ob[(i * 3) * NT:(i * 3 + 1) * NT] for i in range(n)]).sum(0)
    fc = d["field_cum"].astype(np.float64)          # 0=FOPT 1=FWIT 2=FWPT 3=FPR
    fopt, fwit = fc[0], fc[1]
    w = np.gradient(DAYS).astype(float); w[0] *= .5; w[-1] *= .5
    ms = (DAYS - DAYS[0]) <= SHORT_Y * 365.25
    t_cut = DAYS[0] + SHORT_Y * 365.25

    r = {"water": float(fwit[-1] - fwit[0]),
         "oil": float(fopt[-1] - fopt[0]),
         "short": float(np.interp(t_cut, DAYS, fopt) - fopt[0]),
         "water_trap": float(d["field_cum"][1][-1] - d["field_cum"][1][0]),
         "oil_trap": float((oil * w).sum()),
         "short_trap": float(np.trapezoid(oil[ms], DAYS[ms]))}

    # NPV:用 FOPT 的**区间增量**(精确产量) × 该区间中点年份的 Brent × 中点折现
    dq = np.diff(fopt)
    tm = 0.5 * (DAYS[:-1] + DAYS[1:])
    yr = np.array([(T_START + dt.timedelta(days=float(x))).year for x in tm])
    pr = np.array([BRENT[y] for y in yr])
    t = (tm - DAYS[0]) / 365.25
    rev = dq * BBL_PER_M3 * pr
    # 主线口径 NPV(率 × 梯形权重),保留以便对照
    yr2 = np.array([(T_START + dt.timedelta(days=float(x))).year for x in DAYS])
    rev2 = oil * w * BBL_PER_M3 * np.array([BRENT[y] for y in yr2])
    t2 = (DAYS - DAYS[0]) / 365.25
    for dr in (0.0, 0.08, 0.10, 0.15):
        r[f"npv{dr:.0%}"] = float((rev / (1 + dr) ** t).sum())
        r[f"npv{dr:.0%}_trap"] = float((rev2 / (1 + dr) ** t2).sum())
    return r


def theta_of(key):
    return np.load(SIM / f"{key}.npz")["theta"].reshape(-1).astype(np.float64)


def run(a) -> int:
    ms = json.loads((OUT / "multistart.json").read_text())["results"]
    pick = ms[a.plan + "_pick"]
    th0 = np.asarray(pick["theta"], dtype=np.float64)          # 已含原 s
    W0 = metrics("ms_base")["water"]

    pts = []                                                    # (ds, dev, key)
    seeds = (sorted(SIM.glob(f"ms_{a.plan}_{pick['k']}_c*.npz"))
             + sorted(SIM.glob(f"tight_{a.plan}_*.npz")))
    for f in seeds:
        ds = float((theta_of(f.stem) - th0).mean())
        pts.append((ds, metrics(f.stem)["water"] / W0 - 1.0, f.stem))
        if ds < -0.02:                      # 远离根的旧点只会把割线拉歪
            pts.pop()
    pts.sort(key=lambda p: p[0])
    print(f"[{a.plan}] 基准实测注水 {W0:,.0f}   复用 {len(pts)} 个已有割线点")
    for p in pts:
        print(f"   种子 ds={p[0]:+.6f}  偏离 {p[1]:+.4%}   {p[2]}")

    for it in range(a.max_new):
        best = min(pts, key=lambda p: abs(p[1]))
        if abs(best[1]) <= TOL:
            break
        (x1, y1, _), (x2, y2, _) = sorted(pts, key=lambda p: abs(p[1]))[:2]
        ds = x2 - y2 * (x2 - x1) / (y2 - y1) if abs(y2 - y1) > 1e-12 else x2
        # 若已有正负号点则强制割线落在括号内(防外推跑飞)
        neg = [p[0] for p in pts if p[1] < 0]; pos = [p[0] for p in pts if p[1] > 0]
        if neg and pos:
            lo, hi = sorted((max(neg), min(pos)))
            if not (lo <= ds <= hi):
                ds = 0.5 * (lo + hi)
        ds = float(np.clip(ds, -0.3, 0.3))
        if any(abs(ds - p[0]) < 1e-9 for p in pts):
            print(f"   割线不再移动 (ds={ds:+.6f}),停"); break
        key = f"tight_{a.plan}_d{ds:+.6f}"   # 用 ds 命名,避免与已有算例撞名
        if not (SIM / f"{key}.npz").exists():
            FD._run_sim(key, (th0 + ds).astype(np.float32).reshape(N_STAGE, len(INJ)),
                        _A(a.threads))
        m = metrics(key); dev = m["water"] / W0 - 1.0
        pts.append((ds, dev, key))
        print(f"   iter{it} ds={ds:+.6f} (s=10^{ds:+.6f})  实测注水 {m['water']:,.0f}  "
              f"偏离 {dev:+.4%}  {'✔' if abs(dev) <= TOL else ''}", flush=True)

    best = min(pts, key=lambda p: abs(p[1]))
    zero = min(pts, key=lambda p: abs(p[0]))          # ds≈0 = 收紧前的主线定版算例
    assert abs(zero[0]) < 1e-6, zero
    rec = {"plan": a.plan, "k": pick["k"], "s_orig": pick["s"],
           "ds": best[0], "s_total": pick["s"] + best[0], "key": best[2],
           "W0": W0, "water_dev": best[1], "tol": TOL,
           "converged": bool(abs(best[1]) <= TOL),
           "n_new_sims": sum(1 for p in pts if p[2].startswith("tight_")),
           "before": {"key": zero[2], "water_dev": zero[1], **metrics(zero[2])},
           "after": metrics(best[2]),
           "trace": [{"ds": p[0], "dev": p[1], "key": p[2]} for p in sorted(pts)]}
    (OUT / f"_tight_{a.plan}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"[{a.plan}] 收敛={rec['converged']}  最终偏离 {best[1]:+.4%}  算例 {best[2]}")
    return 0


def collect(a) -> int:
    base = metrics("ms_base")
    R = {"tol": TOL,
         "judge": "OPM Flow (全物理)。注水量守恒用**模拟器自报累计** FWIT(field_cum[1]),"
                  "不用 40 点率梯形积分 —— 后者残差 ±0.5%,是本容差的 10 倍。",
         "water_metric": "FWIT (cumulative water injection total), 预测段 = FWIT[-1]-FWIT[0]",
         "brent_source": "EIA RBRTE annual (fc_npv.BRENT)",
         "baseline": {"key": "ms_base", **base}, "plans": {}}
    for nm in ("short", "long"):
        d = json.loads((OUT / f"_tight_{nm}.json").read_text())
        for side in ("before", "after"):
            d[side]["water_dev_trap"] = d[side]["water_trap"] / base["water_trap"] - 1.0
        d["before"]["water_dev"] = d["before"]["water"] / base["water"] - 1.0
        R["plans"][nm] = d
    # --- 头条对照 + 折现率翻转点 ---
    def npv_curve(key, r):
        d = np.load(SIM / f"{key}.npz"); fopt = d["field_cum"].astype(np.float64)[0]
        dq = np.diff(fopt); tm = 0.5 * (DAYS[:-1] + DAYS[1:])
        pr = np.array([BRENT[(T_START + dt.timedelta(days=float(x))).year] for x in tm])
        return float((dq * BBL_PER_M3 * pr / (1 + r) ** ((tm - DAYS[0]) / 365.25)).sum())

    def crossover(ks, kl):
        """短期方案 NPV 追平长期方案的折现率(二分)。"""
        f = lambda r: npv_curve(ks, r) - npv_curve(kl, r)
        lo, hi = 0.0, 0.30
        if f(lo) * f(hi) > 0:
            return None
        for _ in range(60):
            mid = .5 * (lo + hi)
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        return .5 * (lo + hi)

    hl = {}
    for side, ks, kl in (("before", R["plans"]["short"]["before"]["key"],
                          R["plans"]["long"]["before"]["key"]),
                         ("after", R["plans"]["short"]["key"], R["plans"]["long"]["key"])):
        ms_, ml_ = metrics(ks), metrics(kl)
        hl[side] = {
            "short_key": ks, "long_key": kl,
            "water_gap_short_vs_long": ms_["water"] / ml_["water"] - 1.0,
            "short3y_gain_short_vs_long": ms_["short"] / ml_["short"] - 1.0,
            "fullterm_cost_short_vs_long": ms_["oil"] / ml_["oil"] - 1.0,
            "short3y_gain_vs_base": ms_["short"] / base["short"] - 1.0,
            "crossover_rate": crossover(ks, kl),
            "trap_view": {"short3y_gain_vs_base": ms_["short_trap"] / base["short_trap"] - 1.0,
                          "fullterm_cost_short_vs_long": ms_["oil_trap"] / ml_["oil_trap"] - 1.0},
        }
    R["headline"] = hl
    R["caveats"] = [
        "🔴 注水量守恒必须用模拟器自报累计 FWIT。主线用的『40 点瞬时率梯形积分』"
        "对总注水量有 ±0.5% 残差(报告网格 122.5 天/点,井控在步内切换),"
        "该残差是 ±0.05% 容差的 10 倍,按它标定会得到非单调、无法收敛的目标函数。",
        "同一求积问题也影响产油与 NPV:基准全期产油 trap 7,465,043 vs FOPT 7,493,840,差 −0.38%。"
        "本文件的主口径用 FOPT 区间增量,并保留 *_trap 字段与主线数字对照。",
        "校正仍是 θ 整体平移(所有目标注水率乘同一标量 s),不改井间份额,研究对象未被动过。",
        "地质维度失效 / F-4H 是假旋钮 / 预测段真实工况是 WAG —— 三条主线限制照旧成立。",
    ]
    json.dump(R, open(OUT / "verify_tight.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if k != "plans" else
                          {kk: {"water_dev_before": vv["before"]["water_dev"],
                                "water_dev_after": vv["water_dev"]}
                           for kk, vv in v.items()}) for k, v in R.items()},
                     indent=1, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["short", "long"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    return collect(a) if a.collect else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
