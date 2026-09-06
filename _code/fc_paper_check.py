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
    (62.6, NB, 0.1, "一维缩放"),
    (298.6, S["tests"]["team_vs_none"]["diff"], 0.1, "Team − 无 Agent"),
    (262.4, _ci(A, NB)[0], 0.2, "Team vs 无 Agent 95%CI 下界"),
    (334.9, _ci(A, NB)[1], 0.2, "Team vs 无 Agent 95%CI 上界"),
    (2.7, S["arms"]["full7"]["sim_budget"], 0.05, "Team 真模拟/次"),
    (7, S["nonllm"]["n_sim"], 0.01, "一维缩放真模拟"),
    (0.66, S["arms"]["full7"]["tilt"], 0.005, "Team 平均 tilt"),
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


# --------------------------------------------------------------- 禁语闸门
#: 数据不支持的表述。value = 允许出现的上下文(否定式/物理描述),其余一律红。
BANNED = {
    "monotonic": ["not monotone", "monotonically increases oil"],
    # 允许的都是**否定式**用法:明确说"不主张/无法建立/不是"。
    "equivalence": ["cannot establish equivalence", "no claim of equivalence",
                    "about equivalence", "not an assertion of equivalence",
                    "non-equivalences"],
    "robustness": [],
    "causal law": [],
    "transferable": ["than establishing a transferable rule"],
    "expert-level": [],
    "state-of-the-art": ["rather than a state-of-the-art", "not a state-of-the-art",
                         "is not a state-of-the-art"],
}


def check_banned() -> list[str]:
    """按**段落**扫,不按行 —— 正文是硬换行的,否定词常落在上一行。

    行级扫描会把 "not an assertion of\nequivalence" 判成违规。
    """
    bad = []
    for f in sorted((ROOT / "_paper").glob("0*.md")):
        txt = f.read_text()
        off = 1
        for para in txt.split("\n\n"):
            flat = " ".join(para.split()).lower()
            for word, allowed in BANNED.items():
                if word in flat and not any(a.lower() in flat for a in allowed):
                    bad.append(f"  ✗ {f.name}:~{off} 禁语「{word}」: {flat[:100]}")
            off += para.count("\n") + 2
    return bad



def main() -> int:
    bad = []
    for paper, truth, tol, name in CHECKS:
        if abs(float(paper) - float(truth)) > tol:
            bad.append(f"  ✗ {name}: 正文 {paper}  真源 {truth:.4f}  (容差 {tol})")
    # 🔴 论文只报 Gaia 团队臂。单 Agent 臂的数据仍在 gaia.py 与 _pipelines 中,
    #    但不进正文(2026-09-06 用户裁定),故此处不核它的数字。

    # 正文声称的定性关系
    order = sorted(bt, key=bt.get, reverse=True)
    if order != ["F-1H", "F-2H", "F-3H", "F-4H"]:
        bad.append(f"  ✗ 井影响排序: 实测 {order}")
    a = S["arms"]["full7"]
    if (a["closed_loop_up"], a["closed_loop_total"]) != (10, 10):
        bad.append(f"  ✗ 闭环正向率: 实测 {a['closed_loop_up']}/{a['closed_loop_total']}, 正文写 10/10")
    if RANK["rank_fidelity_on_simulated_theta"]["hit_rate"]["top1"] != 0.0:
        bad.append("  ✗ 优化区 top-1 命中率不为 0")
    if RANK["optimizers_curse"]["err_rank_of_surrogate_pick"] != 1:
        bad.append("  ✗ 代理选中者并非它高估最狠的候选")
    bad += check_banned()
    if bad:
        print(f"❌ {len(bad)} 处对不上:"); print("\n".join(bad)); return 1
    print(f"✅ {len(CHECKS)} 个正文数字 + 4 条定性关系 + {len(BANNED)} 条禁语全部通过")
    return 0




if __name__ == "__main__":
    sys.exit(main())
