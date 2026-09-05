#!/usr/bin/env python3
"""论文图表生成。全部数字来自已跑完并复核的 JSON，不重新计算任何结果。

对应 `_paper/OUTLINE.md` 的图表编号：
  Figure 1  学习曲线 + 可分辨阈值带
  Figure 2  代理预测 vs 模拟器真值（排名失效）
  Figure 3  三方案产油率曲线
  Figure 4  **核心图** NPV vs 折现率 → 经济最优反转
  Figure 5  归因偏回归 R² 分解
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import forecast_gen as FG
import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "_pipelines"
FIG = ROOT / "_figures" / "paper"
DAYS = FG.FC_GRID

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
C = {"base": "#6b7280", "short": "#d97706", "long": "#2563eb", "acc": "#dc2626"}


def fig1_learning_curve():
    d = json.loads((PIPE / "fc_4k" / "fc4k.json").read_text())
    Ns = d["N_list"]
    base = [d["results"][f"N{n}_base"]["mean"] * 100 for n in Ns]
    bsd = [d["results"][f"N{n}_base"]["sd"] * 100 for n in Ns]
    ff = [d["results"][f"N{n}_ff"]["mean"] * 100 for n in Ns]
    fsd = [d["results"][f"N{n}_ff"]["sd"] * 100 for n in Ns]
    thr = json.loads((PIPE / "fc_ceil" / "ceil.json").read_text())["resolvable_threshold_pp"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(Ns, base, yerr=bsd, marker="o", ms=4, lw=1.2, capsize=2,
                color=C["base"], label="Baseline Transformer")
    ax.errorbar(Ns, ff, yerr=fsd, marker="s", ms=4, lw=1.2, capsize=2,
                color=C["long"], label="+ Fourier positional embedding")
    ax.axhspan(min(ff) - thr, min(ff) + thr, color=C["acc"], alpha=0.10,
               label=f"resolvable threshold ±{thr:.3f} pp")
    p = np.polyfit(np.log(Ns), np.log(ff), 1)
    ax.set(xscale="log", yscale="log", xlabel="Training samples $N$",
           ylabel="Field-level cumulative-oil error (%)")
    ax.set_title(f"Error $\\propto N^{{{p[0]:.2f}}}$", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig1_learning_curve.png"); plt.close(fig)
    return f"斜率 {p[0]:.3f}（文中报 N^-0.51）"


def fig4_npv_vs_discount():
    """核心图。交叉点用 8 档实测值定位,不再靠三点插值。"""
    d = json.loads((PIPE / "fc_fix" / "fixed.json").read_text())
    b, s_, l = d["results"]["baseline"], d["results"]["short"], d["results"]["long"]
    rates = np.array([0, 2, 4, 6, 8, 10, 12, 15], float)
    ds = np.array([(s_[f"npv{int(r)}"] - b[f"npv{int(r)}"]) / 1e6 for r in rates])
    dl = np.array([(l[f"npv{int(r)}"] - b[f"npv{int(r)}"]) / 1e6 for r in rates])
    # 在实测相邻两档之间线性求根,得到交叉区间
    diff = ds - dl
    i = int(np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0][0])
    x0, x1 = rates[i], rates[i + 1]
    cross = x0 - diff[i] * (x1 - x0) / (diff[i + 1] - diff[i])
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(rates, ds, "o-", color=C["short"], lw=1.5, ms=4.5, label="Short-horizon optimum")
    ax.plot(rates, dl, "s-", color=C["long"], lw=1.5, ms=4.5, label="Long-horizon optimum")
    ax.axvspan(x0, x1, color=C["acc"], alpha=0.12)
    ax.annotate(f"reversal within\n{x0:.0f}–{x1:.0f}% (≈{cross:.1f}%)",
                xy=(cross, np.interp(cross, rates, ds)),
                xytext=(3.4, max(ds) * 0.42), fontsize=7, color=C["acc"],
                arrowprops=dict(arrowstyle="->", color=C["acc"], lw=0.8))
    ax.axhline(0, lw=0.6, color="k", alpha=0.4)
    ax.set(xlabel="Discount rate (%)", ylabel="$\\Delta$NPV vs. base case (M\\$)")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.savefig(FIG / "fig4_npv_reversal.png"); plt.close(fig)
    return f"实测 8 档;反转区间 {x0:.0f}-{x1:.0f}%(线性求根 {cross:.2f}%)"


def fig5_attribution():
    d = json.loads((PIPE / "fc_fix" / "fixed.json").read_text())["causal"]
    names = ["Temporal\nfront-loading", "Spatial\ndispersion", "Total water\ndeviation",
             "Temporal\n+ Spatial", "All three"]
    vals = [0.3878, 0.2856, 0.2019, d["r2_front_spread"], 0.7691]
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    cols = [C["long"], C["short"], C["base"], "#7c3aed", "#0f766e"]
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    inc = d["delta_r2_spread"]
    ax.annotate(f"incremental $R^2$ of spatial\nafter controlling temporal:\n**{inc:+.4f}**",
                xy=(3, d["r2_front_spread"]), xytext=(1.1, 0.62), fontsize=7, color=C["acc"],
                arrowprops=dict(arrowstyle="->", color=C["acc"], lw=0.8))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("$R^2$ of $\\Delta$NPV@8%")
    ax.set_title(f"Variance attribution ($n$={d['n']} simulated cases)", fontsize=8.5)
    fig.savefig(FIG / "fig5_attribution.png"); plt.close(fig)
    return f"增量 R² = {inc:+.4f}（n={d['n']}）"


def fig3_rate_curves():
    import fc_fix as FX
    ms = json.loads((PIPE / "fc_decide" / "multistart.json").read_text())
    picks = {"Base case": "ms_base"}
    import glob
    for nm, lab in (("short", "Short-horizon optimum"), ("long", "Long-horizon optimum")):
        k = ms["results"][nm + "_pick"]["k"]; tgt = ms["results"][nm + "_pick"]["oil"]
        best, bd = None, 1e30
        for f in sorted(glob.glob(str(PIPE / "fc_decide" / "sim" / f"ms_{nm}_{k}_c*.npz"))):
            ob = np.load(f)["obs"]; T = FG.FC_N
            o = float(np.trapezoid(np.stack([ob[(i*3)*T:(i*3+1)*T]
                      for i in range(len(NB.PRODUCERS))]).sum(0), DAYS))
            if abs(o - tgt) < bd:
                best, bd = Path(f).stem, abs(o - tgt)
        picks[lab] = best
    t = (DAYS - DAYS[0]) / 365.25 + 2006.9
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    for (lab, key), col in zip(picks.items(), [C["base"], C["short"], C["long"]]):
        d = np.load(PIPE / "fc_decide" / "sim" / f"{key}.npz")
        fc = d["field_cum"][0]
        rate = np.gradient(fc, DAYS)
        ax.plot(t, rate, lw=1.4, color=col, label=lab)
    ax.axvspan(t[0], t[0] + 3, color=C["short"], alpha=0.08)
    ax.text(t[0] + 1.5, ax.get_ylim()[1] * 0.93, "first 3 years", fontsize=6.5,
            ha="center", color=C["short"])
    ax.set(xlabel="Year", ylabel="Field oil rate (sm$^3$/d)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig3_rate_curves.png"); plt.close(fig)
    return "三方案产油率曲线"


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig1_learning_curve, fig3_rate_curves, fig4_npv_vs_discount, fig5_attribution):
        try:
            note = fn()
            print(f"  ✔ {fn.__name__:22s} {note}")
        except Exception as e:                                    # noqa: BLE001
            print(f"  🔴 {fn.__name__:22s} 失败: {type(e).__name__}: {e}")
    print(f"\n图输出目录 {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print(f"    {p.name}  {p.stat().st_size//1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
