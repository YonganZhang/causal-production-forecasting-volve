"""论文正文数字核对。把 04_results.md / 03_methods.md 里的每个数字回连到真源。

    python _code/fc_paper_check.py     # 全绿才允许说"数字已核"

新增一个正文数字，就在 CHECKS 里加一行；对不上直接红。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

import gaia

ROOT = Path(__file__).resolve().parent.parent
RANK = json.loads((ROOT / "_pipelines" / "fc_rank_fidelity" / "rank.json").read_text())["results"]["long"]
CART = json.loads((ROOT / "_pipelines" / "fc_team" / "cartography.json").read_text())["payload"]
TILT = json.loads((ROOT / "_pipelines" / "fc_team" / "tilt_bins.json").read_text())
S = gaia.summary()


def _beta_totals():
    b = np.asarray(CART["beta"], float).sum(1)
    return dict(zip(CART["injectors"], b))


def _ci(v, ref):
    d = np.asarray(v) - ref
    return st.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=st.sem(d))


A, B = np.array(S["arms"]["full7"]["values"]), np.array(S["arms"]["one1"]["values"])
NB = S["nonllm"]["mean"]
bt = _beta_totals()

CHECKS = [
    # (正文里的数字, 真源算出来的值, 容差, 说明)
    (1773.4, S["baseline_npv_musd"], 0.1, "基准 NPV@8% (M$)"),
    (361.3, S["arms"]["full7"]["mean"], 0.1, "Team 均值"),
    (356.9, S["arms"]["full7"]["median"], 0.1, "Team 中位"),
    (50.7, S["arms"]["full7"]["sd"], 0.1, "Team sd"),
    (454.7, S["arms"]["full7"]["best"], 0.1, "Team 最好"),
    (347.4, S["arms"]["one1"]["mean"], 0.1, "单 Agent 均值"),
    (343.3, S["arms"]["one1"]["median"], 0.1, "单 Agent 中位"),
    (72.9, S["arms"]["one1"]["sd"], 0.1, "单 Agent sd"),
    (508.0, S["arms"]["one1"]["best"], 0.1, "单 Agent 最好"),
    (62.6, NB, 0.1, "一维缩放"),
    (298.6, S["tests"]["team_vs_none"]["diff"], 0.1, "Team − 无 Agent"),
    (284.8, S["tests"]["one_vs_none"]["diff"], 0.1, "单 Agent − 无 Agent"),
    (13.8, S["tests"]["team_vs_one"]["diff"], 0.1, "Team − 单 Agent"),
    (262.4, _ci(A, NB)[0], 0.2, "Team vs 无 Agent 95%CI 下界"),
    (334.9, _ci(A, NB)[1], 0.2, "Team vs 无 Agent 95%CI 上界"),
    (232.7, _ci(B, NB)[0], 0.2, "单 Agent vs 无 Agent 95%CI 下界"),
    (337.0, _ci(B, NB)[1], 0.2, "单 Agent vs 无 Agent 95%CI 上界"),
    (2.7, S["arms"]["full7"]["sim_budget"], 0.05, "Team 真模拟/次"),
    (2.6, S["arms"]["one1"]["sim_budget"], 0.05, "单 Agent 真模拟/次"),
    (7, S["nonllm"]["n_sim"], 0.01, "一维缩放真模拟"),
    (0.63, S["tests"]["team_vs_one"]["p"], 0.005, "Team vs 单 Agent Welch p"),
    (0.52, S["tests"]["team_vs_one"]["p_mw"], 0.005, "Team vs 单 Agent MW p"),
    (0.45, st.levene(A, B).pvalue, 0.005, "Levene p"),
    (0.29, st.bartlett(A, B).pvalue, 0.005, "Bartlett p"),
    (0.66, S["arms"]["full7"]["tilt"], 0.005, "Team 平均 tilt"),
    (0.92, S["arms"]["one1"]["tilt"], 0.005, "单 Agent 平均 tilt"),
    (0.714, TILT["spearman"], 0.001, "tilt–ΔNPV spearman"),
    (275, TILT["n"], 0.5, "tilt 样本量"),
    (0.998, RANK["control_random_theta_full"]["spearman_rho"], 0.001, "随机 θ 排名 rho"),
    (-0.21, RANK["rank_fidelity_on_simulated_theta"]["spearman_rho"], 0.005, "优化区排名 rho"),
    (1.66, RANK["optimizers_curse"]["of_median"], 0.005, "高估倍数中位"),
    (10.6, RANK["optimizers_curse"]["of_at_surrogate_pick"], 0.05, "代理选中者的高估倍数"),
    (0.97, RANK["optimizers_curse"]["corr_surr_vs_err"], 0.005, "预测值与自身误差的相关"),
    (1.39, RANK["top1_regret"]["pct_of_objective"], 0.005, "top-1 regret (%)"),
    (46.0, RANK["top1_regret"]["musd"]["npv8"], 0.05, "top-1 regret (M$)"),
    (28.6, RANK["top1_regret"]["gain_capture_ratio"] * 100, 0.05, "代理捕获的增益比例 (%)"),
    (1.44, RANK["point_accuracy_at_candidates"]["relerr_mean"] * 100, 0.005, "优化区点误差 (%)"),
    (0.369, RANK["control_random_theta_full"]["relerr_mean"] * 100, 0.001, "确认集场级误差 (%)"),
    (132, RANK["control_random_theta_full"]["signal_to_error_ratio"], 0.5, "信噪比"),
    (0.95, RANK["control_random_theta_surr_topk"]["spearman_rho"], 0.005, "随机池 surr-top10 rho"),
    (0.94, RANK["control_random_theta_true_topk"]["spearman_rho"], 0.005, "随机池 true-top10 rho"),
    (5.0, RANK["control_random_theta_surr_topk"]["true_spread_pct"], 0.05, "随机池 top10 目标跨度 (%)"),
    (471, CART["n_case"], 0.5, "连通性算例数"),
    (2.05, CART["diag"]["cond"], 0.005, "设计矩阵条件数"),
    (8.57, bt["F-1H"], 0.005, "F-1H 影响"),
    (6.22, bt["F-2H"], 0.005, "F-2H 影响"),
    (6.06, bt["F-3H"], 0.005, "F-3H 影响"),
    (1.76, bt["F-4H"], 0.005, "F-4H 影响"),
]


def main() -> int:
    bad = []
    for paper, truth, tol, name in CHECKS:
        if abs(float(paper) - float(truth)) > tol:
            bad.append(f"  ✗ {name}: 正文 {paper}  真源 {truth:.4f}  (容差 {tol})")
    # 正文声称的定性关系
    order = sorted(bt, key=bt.get, reverse=True)
    if order != ["F-1H", "F-2H", "F-3H", "F-4H"]:
        bad.append(f"  ✗ 井影响排序: 实测 {order}")
    up = sum(S["arms"][k]["closed_loop_up"] for k in gaia.ARMS)
    tot = sum(S["arms"][k]["closed_loop_total"] for k in gaia.ARMS)
    if (up, tot) != (20, 20):
        bad.append(f"  ✗ 闭环正向率: 实测 {up}/{tot}, 正文写 20/20")
    if RANK["rank_fidelity_on_simulated_theta"]["hit_rate"]["top1"] != 0.0:
        bad.append("  ✗ 优化区 top-1 命中率不为 0")
    if RANK["optimizers_curse"]["err_rank_of_surrogate_pick"] != 1:
        bad.append("  ✗ 代理选中者并非它高估最狠的候选")
    if bad:
        print(f"❌ {len(bad)} 处对不上:"); print("\n".join(bad)); return 1
    print(f"✅ {len(CHECKS)} 个正文数字 + 4 条定性关系全部与真源一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
