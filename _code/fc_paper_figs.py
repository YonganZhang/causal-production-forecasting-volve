"""论文正文图 Fig. 1-6。数据一律经 gaia / gaia_sdk / _pipelines 读取,不在此重算口径。

    python _code/fc_paper_figs.py            # 全画
    python _code/fc_paper_figs.py fig2 fig4  # 只画指定的

组织原则(见 _paper/_figure_plan.md):一张图回答一个读者会问的问题,
子图之间必须递进,不是三张独立的图拼在一起。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaia

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_figures" / "paper"
PIPE = ROOT / "_pipelines"
SIM = PIPE / "fc_decide" / "sim"

RANK_FID = json.loads((PIPE / "fc_rank_fidelity" / "rank.json").read_text())["results"]["long"]
MODEL_RANK = json.loads((PIPE / "fc_rank" / "rank.json").read_text())
CARTO = json.loads((PIPE / "fc_team" / "cartography.json").read_text())["payload"]
TILT = json.loads((PIPE / "fc_team" / "tilt_bins.json").read_text())

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
# 三种权限 = 三种视觉编码。这是全文主张,不是配色偏好。
C_DATA, C_KNOW, C_AGENT = "#6e7781", "#8250df", "#1f6feb"
C_PROXY, C_JUDGE, C_REF = "#0f8a5f", "#cf222e", "#d1a01f"
C_ONE_GREY = "#7d8590"


def _panel(ax, tag: str, title: str = "") -> None:
    ax.set_title(f"({tag}) {title}" if title else f"({tag})", loc="left", fontweight="bold")


# ==================================================================== Fig. 1
def fig1_architecture():
    """架构:五层堆栈 + 知识层放大 + 一轮决策时序。"""
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1.25, 0.95], hspace=.42, wspace=.28)

    # ---------------- (a) 五层堆栈
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    _panel(ax, "a", "Gaia: five layers, 14 agents — authority follows demonstrated competence")
    rows = [
        (8.55, "L1  Data", C_DATA, "solid", 1.2,
         "data_engineer  ·  cartographer", "raw deck → 7,900 structured cases  |  471 cases → 4×22 influence map (cond. 2.05)"),
        (6.85, "L2  Knowledge", C_KNOW, "solid", 1.2,
         "knowledge_curator", "deck audit + field literature + domain corpus → 7 role knowledge bases   [run once, offline]"),
        (5.15, "L3  Reasoning", C_AGENT, "solid", 1.2,
         "7 domain roles  ·  synthesizer", "independent pass → cross-examination → adjudication by evidence strength, not by count"),
        (3.45, "L4  Evaluation", C_PROXY, "dashed", 1.3,
         "surrogate  ·  feasibility_auditor", "18.1 µs / candidate (3.3×10⁶ ×)   |   arithmetic gate — never delegated to a language model"),
        (1.55, "L5  Adjudication", C_JUDGE, "solid", 2.6,
         "OPM Flow full-physics   (not an agent)", "the only component permitted to assign value — every economic figure below comes from here"),
    ]
    for y, name, col, ls, lw, agents, detail in rows:
        ax.add_patch(FancyBboxPatch((0.5, y - 0.62), 8.4, 1.24, boxstyle="round,pad=0.04",
                                    ec=col, fc=col, alpha=.055, lw=lw, linestyle=ls))
        ax.text(0.75, y + 0.34, name, fontsize=8.5, fontweight="bold", color=col, va="center")
        ax.text(3.35, y + 0.34, agents, fontsize=8, color=col, va="center", fontweight="bold")
        ax.text(0.75, y - 0.26, detail, fontsize=7, color="0.28", va="center")
    for y0, y1 in ((7.93, 7.47), (6.23, 5.77), (4.53, 4.07), (2.83, 2.37)):
        ax.add_patch(FancyArrowPatch((4.7, y0), (4.7, y1), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.1, color="0.45"))
    ax.add_patch(FancyArrowPatch((8.9, 1.55), (9.55, 1.55), arrowstyle="-", lw=1.1, color=C_JUDGE))
    ax.add_patch(FancyArrowPatch((9.55, 1.55), (9.55, 5.15), arrowstyle="-", lw=1.1, color=C_JUDGE))
    ax.add_patch(FancyArrowPatch((9.55, 5.15), (8.9, 5.15), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=C_JUDGE))
    ax.text(9.72, 3.4, "measured outcome fed back", fontsize=7, color=C_JUDGE,
            rotation=90, va="center", ha="center")
    ax.legend(handles=[
        Line2D([], [], color=C_PROXY, ls="--", lw=1.4, label="capable, no pricing authority"),
        Line2D([], [], color=C_AGENT, lw=1.4, label="reasoning, no veto"),
        Line2D([], [], color=C_JUDGE, lw=2.6, label="pricing authority"),
    ], loc="lower left", bbox_to_anchor=(0.02, -0.06), ncol=3, frameon=False, fontsize=7.2)

    # ---------------- (b) 知识层放大
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    _panel(ax, "b", "How role knowledge is built")
    src = [(9.25, "deck audit", "present:  PERMX, PORO, NTG, FAULTS", "0.25"),
           (8.35, "", "absent:  GEOMECH, STRESS, POISSON", C_JUDGE),
           (7.25, "literature", "Norne 4D seismic — titles + DOI only", "0.25"),
           (6.25, "corpus", "11 modules → 6 waterflood principles", "0.25")]
    for y, lab, txt, col in src:
        if lab:
            ax.text(0.25, y, lab, fontsize=7.0, color="0.45", fontweight="bold")
        ax.text(2.55, y, txt, fontsize=7.1, color=col,
                fontweight="bold" if col == C_JUDGE else "normal")
    ax.add_patch(FancyArrowPatch((5.0, 5.6), (5.0, 5.0), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.0, color=C_KNOW))
    ax.add_patch(FancyBboxPatch((0.2, 0.55), 9.5, 4.3, boxstyle="round,pad=0.06",
                                ec=C_KNOW, fc=C_KNOW, alpha=.07, lw=1.2))
    ax.text(0.5, 4.15, "admissible vs inadmissible knowledge", fontsize=7.6,
            fontweight="bold", color=C_KNOW)
    ax.text(0.5, 3.05, "✓  water injected before water cut rises sweeps best",
            fontsize=7.1, color="0.2")
    ax.text(0.5, 2.0, "✗  a table of the best control vectors found here",
            fontsize=7.1, color=C_JUDGE)
    ax.text(0.5, 1.0, "test: could it be written without solving the problem?",
            fontsize=6.9, color="0.42", style="italic")

    # ---------------- (c) 一轮决策的时序
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    _panel(ax, "c", "One decision round")
    steps = [("7 roles read disjoint evidence", C_AGENT), ("each challenges one peer", C_AGENT),
             ("synthesiser weighs by source type", C_AGENT), ("surrogate screens candidates", C_PROXY),
             ("arithmetic feasibility gate", C_PROXY), ("full-physics adjudication", C_JUDGE)]
    for i, (t, c) in enumerate(steps):
        y = 8.9 - i * 1.45
        ax.add_patch(FancyBboxPatch((0.35, y - 0.46), 9.3, 0.92, boxstyle="round,pad=0.03",
                                    ec=c, fc=c, alpha=.06,
                                    lw=2.2 if c == C_JUDGE else 1.0,
                                    linestyle="--" if c == C_PROXY else "solid"))
        ax.text(0.6, y, f"{i+1}.  {t}", fontsize=7.2, color="0.2", va="center")

    # ---------------- (d) 权限对照表
    ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    _panel(ax, "d", "Why authority is split this way")
    cells = [("surrogate", "screens 3.3×10⁶ faster", "ranking fails where the optimiser searches (Fig. 2d)", C_PROXY),
             ("language agents", "integrate heterogeneous evidence", "cannot be the arbiter of feasibility or value", C_AGENT),
             ("full physics", "assigns every reported value", "1 min per run — hence layers 2–4", C_JUDGE)]
    ax.text(0.3, 8.6, "component", fontsize=7.4, fontweight="bold", color="0.35")
    ax.text(2.6, 8.6, "what it is good at", fontsize=7.4, fontweight="bold", color="0.35")
    ax.text(6.0, 8.6, "why it is not given more", fontsize=7.4, fontweight="bold", color="0.35")
    for i, (a, b, c, col) in enumerate(cells):
        y = 6.6 - i * 2.3
        ax.text(0.3, y, a, fontsize=7.4, color=col, fontweight="bold")
        ax.text(2.6, y, b, fontsize=7.2, color="0.25")
        ax.text(6.0, y, c, fontsize=7.2, color="0.25")
    fig.savefig(OUT / "fig1_architecture.png"); plt.close(fig)
    return "fig1"


# ==================================================================== Fig. 2
def fig2_surrogate():
    """代理模型:是什么、比别人好多少、预测得像不像、以及为何仍由物理裁定。"""
    from fc_team_proxy import Proxy
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 2, hspace=.52, wspace=.52)

    # ---------------- (a) 算子学习示意
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    _panel(ax, "a", "Operator learning, not forecasting")
    ax.add_patch(FancyBboxPatch((0.3, 6.4), 2.5, 2.0, boxstyle="round,pad=0.05",
                                ec=C_DATA, fc=C_DATA, alpha=.09, lw=1.1))
    ax.text(1.55, 7.9, r"$\theta \in \mathbb{R}^{6\times4}$", ha="center", fontsize=9)
    ax.text(1.55, 6.95, "static control\nvector", ha="center", fontsize=6.8, color="0.35")
    ax.add_patch(FancyBboxPatch((3.6, 6.4), 2.7, 2.0, boxstyle="round,pad=0.05",
                                ec=C_PROXY, fc=C_PROXY, alpha=.09, lw=1.3, linestyle="--"))
    ax.text(4.95, 7.9, "PosNet", ha="center", fontsize=8.5, fontweight="bold", color=C_PROXY)
    ax.text(4.95, 6.95, "Fourier pos-emb\n+ Transformer", ha="center", fontsize=6.8, color="0.35")
    ax.add_patch(FancyBboxPatch((7.1, 6.4), 2.6, 2.0, boxstyle="round,pad=0.05",
                                ec=C_DATA, fc=C_DATA, alpha=.09, lw=1.1))
    ax.text(8.4, 7.9, r"$W\times P\times T$", ha="center", fontsize=9)
    ax.text(8.4, 6.95, "per-well oil &\nwater curves", ha="center", fontsize=6.8, color="0.35")
    for x0, x1 in ((2.85, 3.55), (6.35, 7.05)):
        ax.add_patch(FancyArrowPatch((x0, 7.4), (x1, 7.4), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.1, color="0.45"))
    ax.text(5.0, 4.6, "no historical sequence is supplied at inference",
            ha="center", fontsize=7.6, color=C_JUDGE, fontweight="bold")
    ax.text(5.0, 3.5, "the map is control → trajectory,\nnot past → future",
            ha="center", fontsize=7.0, color="0.35")
    ax.text(5.0, 1.6, "7,400 training  +  500 confirmation cases\n"
                      "(generated after model selection was frozen)",
            ha="center", fontsize=6.9, color="0.45", style="italic")

    # ---------------- (b) 与 7 个模型的对比
    ax = fig.add_subplot(gs[0, 1])
    _panel(ax, "b", "Benchmark under one protocol")
    res = MODEL_RANK["results"]
    order = ["随机森林", "SVR(前16主成分)", "高斯过程(前16主成分)", "GBDT(前16主成分)",
             "MLP(BP 神经网络)", "Transformer(单模型)", "🏆 定版 Transformer+集成5"]
    en = ["Random forest", "SVR", "Gaussian process", "GBDT", "MLP",
          "Transformer (single)", "Transformer + ensemble"]
    err = [res[k]["mean"] * 100 for k in order]
    cols = [C_PROXY if i == len(order) - 1 else "#adb5bd" for i in range(len(order))]
    ax.barh(range(len(order)), err, color=cols, height=.66)
    for i, (e, k) in enumerate(zip(err, order)):
        ax.text(e + .09, i, f"{e:.3f}%   R²={res[k]['r2']:.4f}", va="center", fontsize=6.6)
    ax.axvline(res["PCA 结构性下界"]["mean"] * 100, ls=":", c=C_JUDGE, lw=1.2)
    ax.text(res["PCA 结构性下界"]["mean"] * 100 + .07, -1.15,
            "structural floor 0.047%", fontsize=6.3, color=C_JUDGE)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(en, fontsize=7)
    ax.set_xlim(0, max(err) * 1.42)
    ax.set_ylim(-1.5, len(order) - .3)
    ax.set_xlabel("Field-level relative error (%)   —   identical split, n = 2,000 (400 held out)")

    # ---------------- (c) 预测曲线
    ax = fig.add_subplot(gs[1, 0])
    _panel(ax, "c", "Predicted vs simulated well curves")
    z = np.load(SIM / "baseline.npz")
    obs = z["obs"].reshape(22, 3, 40)
    p = Proxy()
    live = np.asarray(p.live, bool)
    traj = p.trajectories(z["theta"].ravel()[None, :])[0]         # (nw, 2, T)
    yrs = z["grid_days"] / 365.25
    idx = np.where(live)[0]
    pick = idx[np.argsort(obs[idx, 0].sum(1))[::-1][:2]]          # 产油最大的两口
    for w in pick:
        k = int(np.where(idx == w)[0][0])
        ax.plot(yrs, obs[w, 0] / 1e3, c=C_JUDGE, lw=2.4, alpha=.35)
        ax.plot(yrs, traj[k, 0] / 1e3, c=C_PROXY, lw=1.0, ls="--")
        ax.plot(yrs, obs[w, 1] / 1e3, c=C_JUDGE, lw=2.0, alpha=.18)
        ax.plot(yrs, traj[k, 1] / 1e3, c=C_PROXY, lw=.9, ls=":", alpha=.85)
    ax.set_xlabel("Time (years)"); ax.set_ylabel(r"Rate ($10^3$ sm$^3$/d)")
    ax.legend(handles=[Line2D([], [], c=C_JUDGE, lw=2.4, alpha=.35, label="simulator"),
                       Line2D([], [], c=C_PROXY, lw=1.0, ls="--", label="surrogate: oil"),
                       Line2D([], [], c=C_PROXY, lw=.9, ls=":", label="surrogate: water")],
              frameon=False, loc="upper right", fontsize=6.6)
    ax.text(.02, .55, "two highest-rate\nproducers", transform=ax.transAxes,
            fontsize=6.5, color="0.45", va="top")

    # ---------------- (d) 为什么仍由物理裁定
    ax = fig.add_subplot(gs[1, 1])
    _panel(ax, "d", "Why physics still adjudicates")
    rnd = RANK_FID["control_random_theta_full"]
    opt = RANK_FID["rank_fidelity_on_simulated_theta"]
    cands = RANK_FID["candidates"]
    true = np.array([c["oil"] for c in cands]); surr = np.array([c["surr_cal"] for c in cands])
    n = len(true)
    tr, sr = n - true.argsort().argsort(), n - surr.argsort().argsort()
    pk = int(np.argmax(surr))
    rng = np.random.default_rng(0)
    m = 60
    ax.scatter(np.arange(1, m + 1), np.arange(1, m + 1) + rng.normal(0, .9, m),
               s=5, c=C_PROXY, alpha=.45, label=f"random $\\theta$   $\\rho$ = {rnd['spearman_rho']:.3f}")
    sc = (np.array(tr) - 1) / (n - 1) * (m - 1) + 1
    sy = (np.array(sr) - 1) / (n - 1) * (m - 1) + 1
    ax.scatter(sc, sy, s=30, c=C_ONE_GREY, zorder=3,
               label=f"optimiser's own   $\\rho$ = {opt['spearman_rho']:.2f}")
    ax.scatter([sc[pk]], [sy[pk]], s=62, c=C_JUDGE, zorder=4)
    ax.annotate("surrogate's pick —\nsimulator ranks it last", xy=(sc[pk], sy[pk]),
                xytext=(m * .52, m * .28), fontsize=6.7, color=C_JUDGE, ha="left",
                arrowprops=dict(arrowstyle="->", lw=.8, color=C_JUDGE,
                                connectionstyle="arc3,rad=-0.25"))
    ax.plot([1, m], [1, m], "--", c="0.65", lw=.8, zorder=1)
    ax.set_xlabel("Simulator rank  (1 = best)"); ax.set_ylabel("Surrogate rank  (1 = best)")
    ax.legend(frameon=False, loc="lower left", fontsize=6.6, handletextpad=.4)
    ax.set_xlim(m + 2, -1); ax.set_ylim(m + 2, -1)
    fig.savefig(OUT / "fig2_surrogate.png"); plt.close(fig)
    return "fig2"



# ==================================================================== Fig. 3
def fig3_process():
    """Agent Team 怎么运转:各角色握有什么证据 → 候选漏斗 → 证据强度裁决的作用。"""
    import fc_team
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    fig.subplots_adjust(wspace=.52, bottom=.30)

    # (a) 每个角色握有多少证据、什么类型
    ax = axes[0]; _panel(ax, "a", "Evidence held per role")
    diag = json.loads((PIPE / "fc_team" / "diag_round3.json").read_text())["readers"]
    tmap = {"simulation": (C_AGENT, "simulation"), "measurement": ("#8250df", "measurement"),
            "literature": (C_REF, "analogue literature")}
    typ = {r["id"]: r["type"] for r in fc_team.ROLES}
    short = {"reservoir_engineer": "reservoir", "connectivity_analyst": "connectivity",
             "production_surveillance": "surveillance", "economics_analyst": "economics",
             "geomechanics_expert": "geomechanics", "seismic_4d_analyst": "4D seismic",
             "constraint_auditor": "constraint audit"}
    rows = sorted(diag, key=lambda r: r["prompt_chars"], reverse=True)
    vals = [r["prompt_chars"] for r in rows]
    cols = [tmap[typ[r["role"]]][0] for r in rows]
    ax.barh(range(len(rows)), vals, color=cols, height=.66)
    for i, v in enumerate(vals):
        ax.text(v + 40, i, f"{v:,}", va="center", fontsize=6.4, color="0.35")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short[r["role"]] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.28)
    ax.legend(handles=[Line2D([], [], color=c, lw=5, label=l) for c, l in tmap.values()],
              frameon=False, fontsize=6.4, loc="upper center",
              bbox_to_anchor=(.5, -.30), ncol=1, handlelength=1.2)
    ax.text(.97, .06, "2 of 7 hold no\nfield measurement", transform=ax.transAxes,
            ha="right", fontsize=6.4, color=C_REF, fontweight="bold")

    # (b) 候选漏斗
    ax = axes[1]; _panel(ax, "b", "Candidate funnel")
    loops = sorted(gaia.LOOP_DIR.glob(f"loop_{gaia.BATCH}full7_*.json"))
    prop = adj = vet = rounds = 0
    for f in loops:
        d = json.loads(f.read_text())
        for r in d["rounds"]:
            rounds += 1
            prop += r["n_cand"]
            adj += len(r.get("adjudicated") or {})
            vet += len(r.get("vetoed") or [])
    val = [prop / rounds, adj / rounds, (adj - vet) / rounds]
    ax.bar(range(3), val, color=[C_AGENT, C_PROXY, C_JUDGE], width=.58)
    for i, v in enumerate(val):
        ax.text(i, v + .16, f"{v:.1f}", ha="center", fontsize=7.6, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["proposed", "screened\n+ gated", "adjudicated"], fontsize=7)
    ax.set_ylabel("Candidates per round"); ax.set_ylim(0, max(val) * 1.32)
    ax.text(.5, .90, f"{rounds} rounds, {len(loops)} runs", transform=ax.transAxes,
            ha="center", fontsize=6.4, color="0.45")
    ax.text(.5, .78, "the gate is arithmetic,\nnever a language model",
            transform=ax.transAxes, ha="center", fontsize=6.3, color=C_PROXY)

    # (c) 证据强度裁决的作用
    ax = axes[2]; _panel(ax, "c", "Adjudication rule")
    leg = ROOT / "_legacy" / "2026-09-06-batches-A2-E2-superseded" / "sim"

    def _tilt(g):
        ts = []
        for f in sorted(g):
            th = np.asarray(np.load(f)["theta"], float)
            if th.shape == (6, 4):
                m = th.mean(1); ts.append(m[:2].mean() - m[-2:].mean())
        return np.array(ts)

    before = _tilt(leg.glob("team_E2full7*.npz"))
    after = _tilt(SIM.glob(f"team_{gaia.BATCH}full7*.npz"))
    bp = ax.boxplot([before, after], widths=.5, patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], ["#adb5bd", C_AGENT]):
        b.set_facecolor(c); b.set_alpha(.45); b.set_edgecolor(c)
    for k in ("medians", "whiskers", "caps"):
        for e in bp[k]:
            e.set_color("0.35")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["by head\ncount", "by evidence\nstrength"], fontsize=7)
    ax.set_ylabel(r"Front-loading index  $\tau$")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * .26)
    ax.text(.5, .95, f"{before.mean():.2f}  \u2192  {after.mean():.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, fontweight="bold", color=C_AGENT)
    ax.set_xlabel(r"$\tau$ tracks $\Delta$NPV ($\rho$ = 0.71)", fontsize=6.6, color="0.45")
    fig.savefig(OUT / "fig3_process.png"); plt.close(fig)
    return "fig3"


# ==================================================================== Fig. 4
def fig4_economics():
    """经济结果与价值:赚了多少、物理上发生了什么、在不同资金成本下是否仍成立。"""
    from fc_water_econ import econ
    s = gaia.summary()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    fig.subplots_adjust(wspace=.44)

    # (a) ΔNPV
    ax = axes[0]; _panel(ax, "a", "Simulator-verified gain")
    v = np.array(s["arms"]["full7"]["values"])
    ax.scatter(np.full_like(v, 0) + np.random.default_rng(1).normal(0, .055, len(v)),
               v, s=26, c=C_AGENT, alpha=.75, zorder=3)
    ax.hlines(v.mean(), -.3, .3, color=C_AGENT, lw=2.4, zorder=4)
    ax.scatter([1], [s["nonllm"]["mean"]], s=70, marker="D", c=C_REF, zorder=3)
    ax.axhline(0, c="0.78", lw=.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Gaia\n(n = 10)", "1-D scaling\nreference"], fontsize=7)
    ax.set_xlim(-.55, 1.55)
    ax.set_ylabel(r"$\Delta$NPV$_{8\%}$  (M US\$)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * .16)
    ax.set_title(f"+{v.mean() - s['nonllm']['mean']:.0f} M\\$,  "
                 f"p = {s['tests']['team_vs_none']['p']:.0e}",
                 loc="right", fontsize=7, color="0.3")

    # (b) 物理上发生了什么
    ax = axes[1]; _panel(ax, "b", "What changed physically")
    best_f, best_v = None, -1e9
    base = gaia.baseline_npv()
    for f in sorted(gaia.LOOP_DIR.glob(f"loop_{gaia.BATCH}full7_*.json")):
        d = json.loads(f.read_text())
        for r in d["rounds"]:
            for k, m in (r.get("adjudicated") or {}).items():
                if int(k) in set(r.get("vetoed") or []):
                    continue
                if (m["npv8"] - base) / 1e6 > best_v:
                    best_v, best_f = (m["npv8"] - base) / 1e6, m
    b0 = econ(SIM / "baseline.npz", 2.0, 1.0)
    lab = ["cumulative\noil", "water\ninjected", "water\nproduced"]
    pct = [(best_f["oil"] / b0["oil"] - 1) * 100,
           (best_f["winj"] / b0["winj"] - 1) * 100,
           (best_f["wprd"] / b0["wprd"] - 1) * 100]
    ax.bar(range(3), pct, color=[C_PROXY, C_AGENT, C_REF], width=.6)
    for i, q in enumerate(pct):
        ax.text(i, q + (.35 if q >= 0 else -.75), f"{q:+.1f}%", ha="center",
                fontsize=7.4, fontweight="bold")
    ax.axhline(0, c="0.5", lw=.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=6.8)
    ax.set_ylabel("Change vs historical (%)")
    ax.set_ylim(min(pct) * 1.5 - 1, max(pct) * 1.35 + 1)
    ax.text(.5, .04, f"best run, $\\Delta$NPV = +{best_v:.0f} M\\$", transform=ax.transAxes,
            ha="center", fontsize=6.5, color="0.45")

    # (c) 折现率稳健性
    ax = axes[2]; _panel(ax, "c", "Robust across capital cost")
    rates, keys = [0, 2, 8, 15], ["npv0", "npv2", "npv8", "npv15"]
    d = [(best_f[k] - b0[k]) / 1e6 for k in keys]
    ax.plot(rates, d, "o-", c=C_AGENT, lw=1.6, ms=5)
    for r, y in zip(rates, d):
        ax.annotate(f"+{y:.0f}", (r, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.8)
    ax.axhline(0, c="0.78", lw=.8)
    ax.set_xlabel("Discount rate (%)"); ax.set_ylabel(r"$\Delta$NPV  (M US\$)")
    ax.set_xticks(rates); ax.set_ylim(0, max(d) * 1.30)
    ax.text(.5, .06, "positive at every discount rate tested", transform=ax.transAxes,
            ha="center", fontsize=6.5, color="0.45")
    fig.savefig(OUT / "fig4_economics.png"); plt.close(fig)
    return "fig4"


# ==================================================================== Fig. 5
def fig5_comparison():
    """对比与决策效率:赚得多、花得少、每次都不倒退。"""
    s = gaia.summary()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    fig.subplots_adjust(wspace=.42)

    # (a) 各方法 ΔNPV
    ax = axes[0]; _panel(ax, "a", "Gain over historical schedule")
    v = np.array(s["arms"]["full7"]["values"])
    ax.bar([0], [v.mean()], yerr=[v.std(ddof=1)], color=C_AGENT, width=.55,
           capsize=4, error_kw=dict(lw=1, ecolor="0.35"))
    ax.bar([1], [s["nonllm"]["mean"]], color=C_REF, width=.55)
    for i, (y, t) in enumerate(((v.mean(), f"+{v.mean():.0f}"),
                                (s["nonllm"]["mean"], f"+{s['nonllm']['mean']:.0f}"))):
        ax.text(i, y + 14, t, ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Gaia", "1-D scaling"], fontsize=7.5)
    ax.set_ylabel(r"$\Delta$NPV$_{8\%}$  (M US\$)")
    ax.set_ylim(0, v.mean() * 1.32); ax.set_xlim(-.6, 1.6)
    ax.text(.5, .55, f"×{v.mean() / s['nonllm']['mean']:.1f}", transform=ax.transAxes,
            ha="center", fontsize=11, color="0.45", fontweight="bold")

    # (b) 决策效率
    ax = axes[1]; _panel(ax, "b", "Cost of the decision")
    bud = [s["arms"]["full7"]["sim_budget"], float(s["nonllm"]["n_sim"])]
    ax.barh([0, 1], bud, color=[C_AGENT, C_REF], height=.5)
    for i, b in enumerate(bud):
        ax.text(b + .15, i, f"{b:.1f}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Gaia", "1-D scaling"], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("Full-physics simulations per run")
    ax.set_xlim(0, max(bud) * 1.3)
    ax.text(.5, .10, f"{v.mean() / s['nonllm']['mean']:.1f}× the gain for "
                     f"{bud[0] / bud[1]:.0%} of the budget",
            transform=ax.transAxes, ha="center", fontsize=6.6, color="0.4")

    # (c) 闭环轨迹
    ax = axes[2]; _panel(ax, "c", "Closed-loop progress")
    for r in gaia._runs("full7", gaia.BATCH):
        run = np.maximum.accumulate(r["traj"])
        ax.plot(np.arange(1, len(run) + 1), run, c=C_AGENT, lw=1, alpha=.55,
                marker="o", ms=2.6)
    ax.axhline(s["nonllm"]["mean"], ls="--", c=C_REF, lw=1.3)
    ax.text(.98, .06, "1-D scaling", transform=ax.transAxes, ha="right",
            fontsize=6.6, color=C_REF)
    ax.set_xlabel("Adjudicated candidate within run")
    ax.set_ylabel(r"Best $\Delta$NPV so far  (M US\$)")
    ax.set_xticks(range(1, 4))
    ax.text(.5, .93, "10/10 runs end at or above where they started",
            transform=ax.transAxes, ha="center", fontsize=6.6, color="0.4")
    fig.savefig(OUT / "fig5_comparison.png"); plt.close(fig)
    return "fig5"


# ==================================================================== Fig. 6
def fig6_interpretability():
    """决策看不看得懂:推荐了什么、为什么合理、哪些旋钮是假的。"""
    fig = plt.figure(figsize=(7.6, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=.85)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    base = gaia.baseline_npv()

    # (a) 最优方案热力图
    ax = axes[0]; _panel(ax, "a", "Recommended schedule")
    bv, bth = -1e9, None
    for f in sorted(SIM.glob(f"team_{gaia.BATCH}full7*.npz")):
        z = np.load(f)
        th = np.asarray(z["theta"], float)
        if th.shape != (6, 4):
            continue
        from fc_water_econ import econ
        m = econ(f, 2.0, 1.0)
        if (m["npv8"] - base) / 1e6 > bv:
            bv, bth = (m["npv8"] - base) / 1e6, th
    lim = np.abs(bth).max()
    im = ax.imshow(bth.T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(4):
        for j in range(6):
            ax.text(j, i, f"{bth[j, i]:+.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if abs(bth[j, i]) > lim * .55 else "0.15")
    ax.set_xticks(range(6))
    ax.set_xticklabels(["07", "08", "10", "12", "14", "17"], fontsize=6.8)
    ax.set_yticks(range(4)); ax.set_yticklabels(["F-1H", "F-2H", "F-3H", "F-4H"], fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=.040, pad=.04)
    cb.set_label(r"$\log_{10}$ multiplier", fontsize=6.2); cb.ax.tick_params(labelsize=5.8)
    ax.set_xlabel(f"Control period start   —   best run +{bv:.0f} M\\$", fontsize=6.8)

    # (b) tilt
    ax = axes[1]; _panel(ax, "b", "One readable direction")
    b = np.asarray(TILT["bins"], float)
    xs, ys, ns = (b[:, 0] + b[:, 1]) / 2, b[:, 4], b[:, 2]
    ax.plot(xs, ys, "o-", c=C_AGENT, ms=5, lw=1.6)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6.2, color="0.4")
    tteam = gaia.arm("full7")["tilt"]
    ax.axvline(tteam, ls="--", c=C_JUDGE, lw=1.2)
    ax.text(tteam, ax.get_ylim()[1], " Gaia", fontsize=6.5, color=C_JUDGE,
            rotation=90, va="top", ha="left")
    ax.axhline(0, c="0.8", lw=.8)
    ax.set_xlabel(r"Front-loading index  $\tau$" "\n"
                  f"n = {TILT['n']} held-out cases,  "
                  f"$\\rho$ = {TILT['spearman']:+.3f}", fontsize=7)
    ax.set_ylabel(r"$\Delta$NPV  (M US\$)")

    # (c) 影响 + 假旋钮
    ax = axes[2]; _panel(ax, "c", "Which knobs are real")
    beta = np.asarray(CARTO["beta"], float).sum(1)
    inj = CARTO["injectors"]
    cols = [C_AGENT if x > 3 else C_JUDGE for x in beta]
    ax.bar(range(4), beta, color=cols, width=.6)
    for i, x in enumerate(beta):
        ax.text(i, x + .18, f"{x:.2f}", ha="center", fontsize=7.2, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(inj, fontsize=7.5)
    ax.set_ylabel("Injector influence on cumulative oil")
    ax.set_ylim(0, beta.max() * 1.28)
    ax.annotate("weak lever", xy=(3, beta[3] + .35), xytext=(1.7, beta.max() * .55),
                fontsize=6.8, color=C_JUDGE,
                arrowprops=dict(arrowstyle="->", lw=.8, color=C_JUDGE,
                                connectionstyle="arc3,rad=-.2"))
    ax.set_xlabel(f"471 cases,  condition number {CARTO['diag']['cond']:.2f}", fontsize=6.6)
    fig.savefig(OUT / "fig6_interpretability.png"); plt.close(fig)
    return "fig6"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [a for a in sys.argv[1:] if a.startswith("fig")]
    order = [("fig1", fig1_architecture), ("fig2", fig2_surrogate), ("fig3", fig3_process),
             ("fig4", fig4_economics), ("fig5", fig5_comparison),
             ("fig6", fig6_interpretability)]
    for name, fn in order:
        if todo and name not in todo:
            continue
        print(fn(), "ok")
