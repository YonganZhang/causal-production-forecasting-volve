"""Gaia 注水决策框架的**唯一读取接口**。

本模块不实现任何新算法,只把已固化的各环节接到一个稳定表面上:

    归属        实现在                      本模块暴露
    实测口径    fc_truth                    measured_winj / measured_oil / measured_wprod
    经济        fc_water_econ.econ          econ / baseline_npv
    智能体臂    _pipelines/fc_team/loop_*   arm() / ARMS
    无智能体    fc_baselines                nonllm()
    汇总        --                          summary() / report()

🔴 不变量
  1. ΔNPV 一律相对 ``baseline_npv()``,一律经 ``fc_water_econ.econ``。
  2. 实测累计水量一律经 ``fc_truth.measured_winj``(= 模拟器自报 FWIT)。
  3. **新增任何读数点必须调用本模块**,不得在临时脚本里重写一遍口径 ——
     那正是 ``fc_truth`` 当初要消灭的那一类缺陷(同一个量两套算法,
     14 个算例给出相反的可行性判定)。
  4. 本模块只读。它不跑模拟、不调 LLM、不写 ``_pipelines/``。

正式批次见 ``BATCH``;历史批次已归档到 ``_legacy/``,不参与汇总。

CLI::

    python _code/gaia.py            # 打印完整指标表
    python _code/gaia.py --json     # 机器可读
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fc_truth import measured_oil, measured_winj, measured_wprod  # noqa: F401  (再导出)
from fc_water_econ import econ

# ---------------------------------------------------------------- 常量

#: 正式批次。E2(证据强度加权前)、A2(对等化前)已归档,见 _legacy/。
BATCH = "F2"

#: 三个对照臂。``full7`` = Agent Team,``one1`` = 单 Agent,``B2`` = 无 Agent。
ARMS = ("full7", "one1")

#: 无 Agent 主基线 —— 一维缩放(全井全阶段同一乘子,唯一决策变量)。
NONLLM_MAIN = "B2"

C_INJ, C_PROD = 2.0, 1.0          # 注水 / 采出水处理成本 USD/bbl
RATE = "npv8"                     # 折现率 8%

ROOT = Path(__file__).resolve().parent.parent
LOOP_DIR = ROOT / "_pipelines" / "fc_team"
SIM_DIR = ROOT / "_pipelines" / "fc_decide" / "sim"
NONLLM_JSON = ROOT / "_pipelines" / "fc_baselines" / "baselines_v2.json"

_LABEL = {
    "full7": "Agent Team",
    "one1": "单 Agent",
    "B2": "无 Agent(一维缩放)",
    "B4": "无 Agent(等预算随机)",
    "BG": "无 Agent(贪心短期)",
}


def label(key: str) -> str:
    return _LABEL.get(key, key)


# ---------------------------------------------------------------- 基准

def baseline_npv(rate: str = RATE) -> float:
    """历史注水方案的 NPV(USD)。所有 ΔNPV 的公共减数。"""
    return float(econ(SIM_DIR / "baseline.npz", C_INJ, C_PROD)[rate])


# ---------------------------------------------------------------- 智能体臂

def _tilt(theta) -> float:
    """前期倾斜度 = 前两阶段均值 - 后两阶段均值(井向取平均)。

    与 ΔNPV 的 spearman = +0.714(275 个随机采样算例,已排除智能体自解)。
    """
    th = np.asarray(theta, float)
    if th.ndim != 2 or th.shape[1] != 4:
        return float("nan")
    s = th.mean(axis=1)
    return float(s[:2].mean() - s[-2:].mean())


def _runs(arm: str, batch: str):
    """逐次实验的轨迹。每次返回按轮次排列的 ΔNPV(百万美元)列表。"""
    base = baseline_npv()
    out = []
    for f in sorted(LOOP_DIR.glob(f"loop_{batch}{arm}_*.json")):
        d = json.loads(f.read_text())
        traj = []
        for r in d["rounds"]:
            vetoed = set(r.get("vetoed") or [])
            for k, m in (r.get("adjudicated") or {}).items():
                if int(k) in vetoed:          # 被模拟器实测否决的方案不计入
                    continue
                traj.append((m[RATE] - base) / 1e6)
        if traj:
            out.append({"file": f.name, "traj": traj})
    return out


def arm(name: str, batch: str = BATCH) -> dict:
    """一个智能体臂的完整指标。

    ``best`` 是该次实验中**经模拟器裁定**的最好方案(否决的已剔除);
    ``closed_loop_up`` 统计末轮 >= 首轮的次数,即闭环是否越跑越好。
    """
    runs = _runs(name, batch)
    best = np.array([max(r["traj"]) for r in runs])
    up = sum(1 for r in runs if r["traj"][-1] >= r["traj"][0])
    tilts = [_tilt(np.load(f)["theta"]) for f in sorted(SIM_DIR.glob(f"team_{batch}{name}*.npz"))]
    tilts = [t for t in tilts if np.isfinite(t)]
    return {
        "key": name,
        "label": label(name),
        "batch": batch,
        "n": len(best),
        "mean": float(best.mean()) if len(best) else float("nan"),
        "median": float(np.median(best)) if len(best) else float("nan"),
        "sd": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
        "best": float(best.max()) if len(best) else float("nan"),
        "values": best.tolist(),
        "closed_loop_up": up,
        "closed_loop_total": len(runs),
        "tilt": float(np.mean(tilts)) if tilts else float("nan"),
    }


# ---------------------------------------------------------------- 无 Agent

def nonllm(key: str = NONLLM_MAIN) -> dict:
    """非 LLM 基线。全部不含任何智能体,只有算法。"""
    d = json.loads(NONLLM_JSON.read_text())
    a = d["arms"][key]
    return {
        "key": key,
        "label": label(key),
        "n_sim": int(a["n_sim"]),
        "mean": (a["best"][RATE] - baseline_npv()) / 1e6,
        "method": a.get("method", ""),
    }


# ---------------------------------------------------------------- 汇总

def summary(batch: str = BATCH) -> dict:
    """正式结果的唯一装配点。论文表、报告、测试都从这里取数。"""
    from scipy import stats as st

    arms = {k: arm(k, batch) for k in ARMS}
    ref = nonllm(NONLLM_MAIN)
    tests = {}
    a, b = np.array(arms["full7"]["values"]), np.array(arms["one1"]["values"])
    if len(a) > 1 and len(b) > 1:
        tests["team_vs_one"] = {
            "diff": float(a.mean() - b.mean()),
            "p": float(st.ttest_ind(a, b, equal_var=False).pvalue),
            "p_mw": float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue),
        }
    for k, v in (("team_vs_none", a), ("one_vs_none", b)):
        if len(v) > 1:
            tests[k] = {
                "diff": float(v.mean() - ref["mean"]),
                "p": float(st.ttest_1samp(v, ref["mean"]).pvalue),
            }
    return {
        "batch": batch,
        "baseline_npv_musd": baseline_npv() / 1e6,
        "water_price": {"c_inj": C_INJ, "c_prod": C_PROD},
        "rate": RATE,
        "arms": arms,
        "nonllm_main": ref,
        "nonllm_all": {k: nonllm(k) for k in ("B2", "B4", "BG")},
        "tests": tests,
    }


def report(batch: str = BATCH) -> str:
    s = summary(batch)
    L = [
        f"基准 NPV@8% = {s['baseline_npv_musd']:.1f} M$"
        f"    水价 {C_INJ}/{C_PROD}    批次 {batch}    ΔNPV 越高越好",
        "=" * 74,
        f"{'臂':<22s}{'n':>4s}{'均值':>10s}{'中位':>10s}{'sd':>8s}{'最好':>10s}{'tilt':>8s}",
        "-" * 74,
    ]
    for k in ARMS:
        a = s["arms"][k]
        L.append(f"{a['label']:<22s}{a['n']:>4d}{a['mean']:>+9.1f}M{a['median']:>+9.1f}M"
                 f"{a['sd']:>8.1f}{a['best']:>+9.1f}M{a['tilt']:>+8.3f}")
    r = s["nonllm_main"]
    L.append(f"{r['label']:<22s}{r['n_sim']:>4d}{r['mean']:>+9.1f}M{'':>10s}{'':>8s}{'':>10s}{'不含 LLM':>8s}")
    L += ["-" * 74, "其余非 LLM 基线(未作主对照,列此以免藏基线):"]
    for k in ("B4", "BG"):
        n = s["nonllm_all"][k]
        L.append(f"    {n['label']:<26s}{n['n_sim']:>3d} 次模拟{n['mean']:>+9.1f}M")
    L += ["=" * 74, "", "检验:"]
    t = s["tests"]
    if "team_vs_one" in t:
        v = t["team_vs_one"]
        L.append(f"  Team vs 单 Agent   {v['diff']:+8.1f}M   Welch p={v['p']:.3f}   MW p={v['p_mw']:.3f}"
                 f"   {'✅' if v['p'] < .05 else '⚪ 未检出'}")
    for k, n in (("team_vs_none", "Team vs 无 Agent "), ("one_vs_none", "单 Ag vs 无 Agent")):
        if k in t:
            L.append(f"  {n}  {t[k]['diff']:+8.1f}M   单样本 p={t[k]['p']:.2e}   {'✅' if t[k]['p'] < .05 else '⚪'}")
    up = sum(s["arms"][k]["closed_loop_up"] for k in ARMS)
    tot = sum(s["arms"][k]["closed_loop_total"] for k in ARMS)
    L += ["", f"闭环正向率: {up}/{tot}   (末轮 ΔNPV >= 首轮)"]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        print(json.dumps(summary(), ensure_ascii=False, indent=2))
    else:
        print(report())
