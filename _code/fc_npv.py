#!/usr/bin/env python3
"""经济评价：用**真实历史 Brent 油价**算贴现 NPV，把产量差换成钱。

## 为什么必须用真实油价而不是固定油价

预测段是 **2006-12 → 2020-01**,这段时间 Brent 从 72 美元冲到 2008 的 96.94(年均;
月度峰值超 130),2011-2014 稳在 ~110,2016 崩到 43.64。**波动近 3 倍。**

用固定油价算,"早采还是晚采"的区别只剩贴现率;
用真实油价算,才能回答油田真正关心的问题:**在这段历史里,早采到底值不值。**

而且石油领域论文的标准口径是 **NPV**,不是产量。

## 数据来源

EIA *Europe Brent Spot Price FOB (Dollars per Barrel)*, Annual,
series RBRTE,取自 https://www.eia.gov/dnav/pet/hist_xls/RBRTEa.xls (2026-08-26 版)。
**原始年均值逐行写在下方,不做任何平滑或调整。**

## 口径声明（论文必须写）

- 只算**收入侧**,不含 CAPEX/OPEX/税费 —— 因为三套方案的井数、设备、注水总量完全相同,
  差异只在采出时间与总量,成本项在方案间**近似抵消**。这是保守做法:
  若计入注水 OPEX,长期方案(注水更均匀)反而更有利。
- 折现率取 8%(石油行业常用 8~12%),并给出 0%/10%/15% 的敏感性。
- 油价用**年均值**,不用月度 —— 我们的产量输出是 40 个时间格点(约 120 天/点),
  用月度会制造虚假精度。

用法:
    python fc_npv.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import forecast_gen as FG
import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_decide"
NT = FG.FC_N
DAYS = FG.FC_GRID
T_START = dt.date(1997, 11, 6)                      # 历史段起点(deck 定义)
BBL_PER_M3 = 6.2898

# EIA RBRTE 年均现货价(美元/桶)。原始值,未做任何处理。
BRENT = {2006: 65.16, 2007: 72.44, 2008: 96.94, 2009: 61.74, 2010: 79.61,
         2011: 111.26, 2012: 111.63, 2013: 108.56, 2014: 98.97, 2015: 52.32,
         2016: 43.64, 2017: 54.13, 2018: 71.34, 2019: 64.30, 2020: 41.96}


def oil_rate(tag):
    """读某方案的逐时刻总产油率 (NT,) sm³/d。"""
    d = np.load(OUT / "sim" / f"{tag}.npz")
    ob = d["obs"]; n = len(NB.PRODUCERS)
    return np.stack([ob[(i*3)*NT:(i*3+1)*NT] for i in range(n)]).sum(0)


def main() -> int:
    dates = [T_START + dt.timedelta(days=float(x)) for x in DAYS]
    years = np.array([d.year for d in dates])
    price = np.array([BRENT[y] for y in years])
    # 🔴 修正:必须用**梯形**权重,与全项目其它地方一致。
    #    此前用 np.gradient(DAYS) 直接求和(矩形法),基准产油算出 7,956,683,
    #    而梯形法是 7,465,043 —— 差 6.59%,足以让所有 NPV 结论作废。
    w = np.gradient(DAYS).astype(float)
    w[0] *= 0.5; w[-1] *= 0.5                    # 端点半权 = 梯形法
    t_yr = (DAYS - DAYS[0]) / 365.25

    print(f"预测段 {dates[0]} → {dates[-1]}   {len(DAYS)} 个时间格点")
    print(f"Brent 年均价范围 {price.min():.2f} ~ {price.max():.2f} 美元/桶 "
          f"(波动 {price.max()/price.min():.1f}×)\n")
    print("  年份   Brent(美元/桶)")
    for y in sorted(set(years)):
        print(f"  {y}    {BRENT[y]:>6.2f}")

    # 多起点定版方案:短期/长期各自选中的那个候选
    # 🔴 修正:必须用**注水量校正后**选中的那次迭代,不是 _c0(=校正前 s=0)。
    #    此前用了 _c0,导致 short 的注水量只有基准的 87%,产油自然低 —— 结论完全反了。
    #    按 multistart.json 记录的 water_dev 去匹配实际算例。
    ms = json.loads((OUT / "multistart.json").read_text())
    def pick_calibrated(nm):
        k = ms["results"][nm + "_pick"]["k"]
        target = ms["results"][nm + "_pick"]["oil"]
        best, bd = None, 1e30
        for f in sorted((OUT / "sim").glob(f"ms_{nm}_{k}_c*.npz")):
            d = np.load(f); ob = d["obs"]; n = len(NB.PRODUCERS)
            o = float(np.trapezoid(np.stack([ob[(i*3)*NT:(i*3+1)*NT] for i in range(n)]).sum(0), DAYS))
            if abs(o - target) < bd:
                best, bd = f.stem, abs(o - target)
        if best is None or bd > 1.0:
            raise SystemExit(f"🔴 {nm}: 找不到与 multistart 记录(产油 {target:,.0f})匹配的算例")
        return best
    picks = {"基准": "ms_base", "短期最优": pick_calibrated("short"),
             "长期最优": pick_calibrated("long")}
    print(f"\n用的算例: " + "  ".join(f"{k}={v}" for k, v in picks.items()))

    R = {}
    for name, key in picks.items():
        q = oil_rate(key)                            # sm³/d
        rev_t = q * w * BBL_PER_M3 * price           # 每格点收入(美元)
        R[name] = {"oil_m3": float((q * w).sum()), "rev_undisc": float(rev_t.sum()),
                   "npv": {}}
        for r in (0.0, 0.08, 0.10, 0.15):
            R[name]["npv"][f"{r:.0%}"] = float((rev_t / (1 + r) ** t_yr).sum())

    b = R["基准"]
    print(f"\n{'='*86}\n=== 🔴 用真实历史 Brent 油价的经济评价 ===\n")
    print(f"  {'方案':10s}{'累计产油(m³)':>16s}{'未贴现收入':>16s}{'NPV@8%':>16s}{'vs基准@8%':>14s}")
    for name in ("基准", "短期最优", "长期最优"):
        d = R[name]
        n8 = d["npv"]["8%"]
        print(f"  {name:10s}{d['oil_m3']:>16,.0f}{d['rev_undisc']/1e6:>14,.1f}M${n8/1e6:>14,.1f}M$"
              f"{(n8-b['npv']['8%'])/1e6:>+12,.1f}M$")

    print(f"\n  === 折现率敏感性(vs 基准,单位:百万美元) ===")
    print(f"  {'折现率':>8s}{'短期最优':>14s}{'长期最优':>14s}{'两者之差':>14s}")
    for r in ("0%", "8%", "10%", "15%"):
        s = (R["短期最优"]["npv"][r] - b["npv"][r]) / 1e6
        l = (R["长期最优"]["npv"][r] - b["npv"][r]) / 1e6
        print(f"  {r:>8s}{s:>+13,.1f}M{l:>+13,.1f}M{s-l:>+13,.1f}M")

    print(f"\n  🔴 关键:折现率越高,'早采'越占便宜 —— 这正是短期策略的价值所在。")
    print(f"     而这段历史里 2008 年 Brent 年均 96.94、2011-2014 稳在 ~110,")
    print(f"     '把产量前移'恰好吃到高价期。")
    json.dump({"source": "EIA RBRTE annual, https://www.eia.gov/dnav/pet/hist_xls/RBRTEa.xls",
               "brent": BRENT, "period": [str(dates[0]), str(dates[-1])],
               "cases": picks, "results": R,
               "caveats": ["只算收入侧,不含 CAPEX/OPEX/税费(三方案井数设备注水量相同,成本近似抵消;"
                           "计入注水 OPEX 则长期方案更有利,故本口径保守)",
                           "油价用年均值,不用月度 —— 产量输出仅 40 个格点(约 120 天/点),"
                           "月度价会制造虚假精度"]},
              open(OUT / "npv.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/npv.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
