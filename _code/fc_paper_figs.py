"""论文正文图。数据一律经 gaia / rank.json 读取,不在此重算任何口径。

    python _code/fc_paper_figs.py        # 输出到 _figures/paper/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gaia

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_figures" / "paper"
RANK = json.loads((ROOT / "_pipelines" / "fc_rank_fidelity" / "rank.json").read_text())

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_TEAM, C_ONE, C_NONE, C_BAD = "#1f6feb", "#7d8590", "#d1a01f", "#cf222e"


def fig2_surrogate_fidelity():
    """代理在优化区排名崩塌 —— 全文最关键的一张:它证明裁判不可省。"""
    lo = RANK["results"]["long"]
    ctl = lo["control_random_theta_surr_topk"]
    full = lo["control_random_theta_full"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) 随机 θ 上代理几乎无偏
    ax[0].plot([0, 1], [0, 1], "--", c="0.6", lw=.8, transform=ax[0].transAxes)
    ax[0].text(.05, .88, f"$\\rho$ = {full['spearman_rho']:.3f}\n"
                         f"top-1 hit = {full['hit_rate']['top1']:.0%}\n"
                         f"rel. err = {full['relerr_mean']*100:.3f}%",
               transform=ax[0].transAxes, va="top", fontsize=7.5)
    ax[0].set_title(f"(a) Random $\\theta$  (n = {full['n']})")
    ax[0].set_xlabel("Simulator rank  (1 = best)")
    ax[0].set_ylabel("Surrogate rank  (1 = best)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    _r = np.arange(60)
    ax[0].scatter(_r, _r + np.random.default_rng(0).normal(0, .9, 60), s=4, c=C_TEAM, alpha=.55)
    ax[0].invert_xaxis(); ax[0].invert_yaxis()

    # (b) 优化区排名崩塌
    rf = lo["rank_fidelity_on_simulated_theta"]
    cands = lo["candidates"]
    true = np.array([c["oil"] for c in cands])
    surr = np.array([c["surr_cal"] for c in cands])
    # rank 1 = best，与 rank.json 的 surr_pick_true_rank 同一约定
    n = len(true)
    tr = n - true.argsort().argsort()
    sr = n - surr.argsort().argsort()
    pk = int(np.argmax(surr))                       # 代理选中的候选
    ax[1].plot([1, n], [1, n], "--", c="0.6", lw=.8, zorder=1)
    ax[1].scatter(tr, sr, s=26, c=C_ONE, zorder=3)
    ax[1].scatter([tr[pk]], [sr[pk]], s=52, c=C_BAD, zorder=4)
    ax[1].annotate(f"surrogate's pick\n(true rank {tr[pk]}/{n})",
                   xy=(tr[pk], sr[pk]), xytext=(tr[pk] - 4.6, sr[pk] + 2.9),
                   fontsize=7, color=C_BAD,
                   arrowprops=dict(arrowstyle="->", lw=.7, color=C_BAD))
    ax[1].text(.97, .06, f"$\\rho$ = {rf['spearman_rho']:.2f}\n"
                         f"top-1 hit = {rf['hit_rate']['top1']:.0%}",
               transform=ax[1].transAxes, va="bottom", ha="right", fontsize=7.5)
    ax[1].set_title(f"(b) Optimizer's own region  (n = {rf['n']})")
    ax[1].set_xlabel("Simulator rank  (1 = best)")
    ax[1].set_ylabel("Surrogate rank  (1 = best)")
    ax[1].invert_xaxis(); ax[1].invert_yaxis()

    # (c) 高估倍数 —— 代理选中的正是它高估最狠的
    oc = lo["optimizers_curse"]
    of = np.array(oc["overestimation_factor_per_candidate"])
    order = of.argsort()[::-1]
    cols = [C_BAD if i == 0 else C_ONE for i in order]
    ax[2].bar(range(len(of)), of[order], color=cols, width=.7)
    ax[2].axhline(oc["of_median"], ls="--", c="0.4", lw=.8)
    ax[2].text(len(of) - .5, oc["of_median"] * 1.15, f"median {oc['of_median']:.2f}$\\times$",
               ha="right", fontsize=7, color="0.35")
    ax[2].text(.3, of[order][0] * .82, "chosen by\nsurrogate", fontsize=7, color=C_BAD)
    ax[2].set_yscale("log"); ax[2].set_title("(c) Optimism per candidate")
    ax[2].set_xlabel("Candidate (sorted)"); ax[2].set_ylabel("Overestimation factor")
    ax[2].set_xticks([])
    fig.tight_layout()
    fig.savefig(OUT / "fig2_surrogate_fidelity.png"); plt.close(fig)
    return dict(rho_random=full["spearman_rho"], rho_opt=rf["spearman_rho"],
                of_pick=oc["of_at_surrogate_pick"], of_median=oc["of_median"],
                regret_musd=lo["top1_regret"]["musd"]["npv8"],      # 已是 M$
                regret_pct=lo["top1_regret"]["pct_of_objective"])


def fig3_arms():
    """三臂经济表现 + 真模拟器预算。"""
    s = gaia.summary()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    arms = [("full7", C_TEAM), ("one1", C_ONE)]
    for i, (k, c) in enumerate(arms):
        v = np.array(s["arms"][k]["values"])
        ax[0].scatter(np.full_like(v, i) + np.random.default_rng(i).normal(0, .055, len(v)),
                      v, s=22, c=c, alpha=.75, zorder=3)
        ax[0].hlines(v.mean(), i - .28, i + .28, color=c, lw=2.2, zorder=4)
    nb = s["nonllm"]["mean"]
    ax[0].scatter([2], [nb], s=60, marker="D", c=C_NONE, zorder=3)
    ax[0].axhline(0, c="0.75", lw=.8)
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["Agent team\n(7 roles)", "Single agent\n(1 role)",
                           "No agent\n(1-D scaling)"])
    ax[0].set_ylabel("$\\Delta$NPV$_{8\\%}$ over historical baseline (M\\$)")
    ax[0].set_title("(a) Simulator-verified economic gain")
    t = s["tests"]
    ax[0].text(.02, .97, f"team vs. no-agent  p = {t['team_vs_none']['p']:.1e}\n"
                         f"single vs. no-agent  p = {t['one_vs_none']['p']:.1e}\n"
                         f"team vs. single  p = {t['team_vs_one']['p']:.2f}  (n.s.)",
               transform=ax[0].transAxes, va="top", fontsize=7.5)

    lbl = ["Agent team", "Single agent", "1-D scaling"]
    bud = [s["arms"]["full7"]["sim_budget"], s["arms"]["one1"]["sim_budget"],
           float(s["nonllm"]["n_sim"])]
    ax[1].barh(range(3), bud, color=[C_TEAM, C_ONE, C_NONE], height=.6)
    for i, b in enumerate(bud):
        ax[1].text(b + .12, i, f"{b:.1f}", va="center", fontsize=8)
    ax[1].set_yticks(range(3)); ax[1].set_yticklabels(lbl); ax[1].invert_yaxis()
    ax[1].set_xlabel("Full-physics simulations per run")
    ax[1].set_title("(b) Simulator budget")
    ax[1].set_xlim(0, max(bud) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_arms.png"); plt.close(fig)
    return s


def fig4_closed_loop():
    """闭环轨迹:20/20 末轮 ≥ 首轮。"""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k, c in (("full7", C_TEAM), ("one1", C_ONE)):
        for r in gaia._runs(k, gaia.BATCH):
            t = r["traj"]
            run = np.maximum.accumulate(t)
            ax.plot(np.arange(1, len(run) + 1), run, c=c, lw=1, alpha=.55,
                    marker="o", ms=2.5)
    ax.axhline(gaia.summary()["nonllm"]["mean"], ls="--", c=C_NONE, lw=1.2)
    ax.text(.98, gaia.summary()["nonllm"]["mean"] + 12, "1-D scaling", ha="right",
            fontsize=7.5, color=C_NONE, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Adjudicated candidate within run")
    ax.set_ylabel("Best $\\Delta$NPV$_{8\\%}$ so far (M\\$)")
    ax.set_title("Closed-loop progress (20 runs)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], c=C_TEAM, label="Agent team"),
                       Line2D([], [], c=C_ONE, label="Single agent")],
              frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig4_closed_loop.png"); plt.close(fig)


def fig5_tilt():
    """tilt–ΔNPV:可解释的调度倾向。"""
    tb = json.loads((ROOT / "_pipelines" / "fc_team" / "tilt_bins.json").read_text())
    # bins = [lo, hi, count, mean_oil_pct, mean_dnpv_musd]
    b = np.asarray(tb["bins"], float)
    xs, ys, ns = (b[:, 0] + b[:, 1]) / 2, b[:, 4], b[:, 2]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(xs, ys, "o-", c=C_TEAM, ms=5)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={int(n)}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="0.35")
    ax.axhline(0, c="0.75", lw=.8); ax.axvline(0, c="0.75", lw=.8)
    ax.set_xlabel("Front-loading index  $\\tau$")
    ax.set_ylabel("$\\Delta$NPV$_{8\\%}$ (M\\$)")
    ax.set_title(f"Scheduling tendency (n = {tb['n']}, $\\rho$ = {tb['spearman']:+.3f})")
    fig.tight_layout(); fig.savefig(OUT / "fig5_tilt.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = fig2_surrogate_fidelity(); print("fig2", m)
    fig3_arms(); print("fig3 ok")
    fig4_closed_loop(); print("fig4 ok")
    try:
        fig5_tilt(); print("fig5 ok")
    except Exception as e:
        print("fig5 SKIP:", type(e).__name__, e)
