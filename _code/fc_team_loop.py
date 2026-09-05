"""Gaia 团队主环路:Readers → adaptive Auditor → Synthesizer。

范式借自 deep_discover(知识发现项目),关键是**自适应终止**:
不是跑固定轮数,而是「连续两轮零 high/medium 风险」才收工。

分工(这是本方法的核心,不是工程细节):
  代理模型  = 模拟器的快速替身,每轮批量评估上千方案(实测 18 µs/方案,GPU)
  模拟器    = 唯一裁判,只跑 top-k,取值一律用 field_cum
  角色团队  = 提出方案并互相质疑;每条断言必须带 provenance
"""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_gen as FG                                       # noqa: E402
import norne_bulk as NB                                         # noqa: E402
from fc_team import (ROLES, Flag, Message, Provenance,          # noqa: E402
                     OUT, sha256_arr, sha256_file, validate)
from fc_team_proxy import Proxy                                 # noqa: E402
import fc_decide as FD                                          # noqa: E402
from fc_water_econ import econ                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "_pipelines" / "fc_decide" / "sim"
INJ, N_STAGE = FG.INJ_W, FG.N_STAGE
BASE_W = np.array([FG.BASE_WINJ[w] for w in INJ], float)
DAYS = np.asarray(FG.FC_GRID, float)


# 时间格点权重与时段索引(解析算注水量用，不花任何模拟)
TW = np.gradient(DAYS); TW[0] *= .5; TW[-1] *= .5
_yr = (DAYS - DAYS[0]) / 365.25 + FG.STAGE_AT[0]
SI = np.clip(np.searchsorted(np.asarray(FG.STAGE_AT, float), _yr, "right") - 1,
             0, N_STAGE - 1)
# 🔴 每口注水井的**实测注入能力上限**，从全部 289 个已有算例重标定(0 次新模拟)。
#    旧的 fc_loop/water_cap.npy 是在接近基准的算例上拟合的，外推到 +58% 注水时
#    低估达 18.4%。重标定后平均误差 3.7%。
#    重标定同时暴露一个关键事实:**四口井全部饱和**，不只 F-4H ——
#    目标率范围 227~98,510(430 倍)，实测最大约 20,000。
#    名义控制自由度远大于有效控制自由度。
_CAPF = ROOT / "_pipelines" / "fc_team" / "cap_refit.npy"
CAP = (np.load(_CAPF)[:, None] * np.ones((1, len(DAYS))) if _CAPF.exists()
       else np.repeat(BASE_W[:, None] * 2.5, len(DAYS), 1))
BBL = 6.2898
PRICE_AVG = 74.0        # 2006-2019 Brent 年均价的简单均值，仅用于代理侧排序
from fc_truth import baseline as _truth_baseline      # 🔴 唯一真源，禁止硬编码
BASE_WATER = _truth_baseline()["winj"]
# 🔴 水价:模块级单一来源，由 run() 从命令行参数写入;角色切片与提示词一律读这里
C_INJ, C_PROD = 2.0, 1.0
SHARED_BG = ""                         # 全体角色共享的背景，由 run() 构造


_WMODEL = ROOT / "_pipelines" / "fc_team" / "water_model.npy"
_WB = np.load(_WMODEL) if _WMODEL.exists() else None


def inj_water(th: np.ndarray) -> float:
    """由 θ 预测**实测**注水总量。

    🔴 2026-09-03 重大修正:此前用「基准率 × 10^θ，再按每口井一个固定上限截断」。
       该近似的 R² 只有 0.219，MAE 6.7pp，**最大误差 40.2pp**，而且是系统性高估 ——
       θ 越大高估越离谱，因为真实注入能力随油藏压力下降而下降，不是常数。

       后果极其严重:硬约束据此否决方案，于是**收益最高的 4 个方案(实测注水
       仅 -2.35% ~ +5.36%，完全合规)被判为 +38% ~ +44% 超标而否决**。
       这一条同时解释了此前所有反常:55% 候选被否决、Agent Team 命中率仅 3%、
       "越听指导越亏"—— 因为指导与守门都建立在同一个坏近似上。
       不是智能体有害，是守门员在杀好人。

       现改为对 791 个已模拟算例做 θ 的二次回归:R² 0.966，MAE 1.5pp。
       收益最高的 12 个方案在新模型下 0 个被误否决(原为 4 个)。
    """
    t = np.asarray(th, float).ravel()
    if _WB is not None:
        # 🔴 无截距(θ=0 必须给 0)，且 θ 先裁剪到训练域 —— 二次式在域外会发散:
        #    未裁剪时 θ=2.0 会预测 +605%，物理上注入能力早已饱和。
        tc = np.clip(t, -1.0, 1.0)
        dev = float(np.r_[tc, tc ** 2] @ _WB) / 100.0
        return BASE_WATER * (1.0 + dev)
    tgt = BASE_W[:, None] * 10.0 ** (t.reshape(N_STAGE, len(INJ))[SI, :].T)
    return float((np.minimum(tgt, CAP) * TW).sum())


def hard_check(th: np.ndarray, w_max_dev: float) -> list[Flag]:
    """🔴 程序化硬约束。不交给 LLM 判断 —— LLM 会被说服，数字不会。

    上一版红队只能报警不能否决，总工照样我行我素，注水一路 +35%→+40%→+58%。
    合法性必须由代码裁定，审计员的自然语言意见只是补充。
    """
    f, thm = [], th.reshape(N_STAGE, len(INJ))
    dev = inj_water(th) / BASE_WATER - 1.0
    # 🔴 2026-09-03:注水预算**不再作为 high 级否决**，降为 medium 警告。
    #    根因:任何由 θ 预测实测注水量的近似都不够准 —— 固定上限式 R²=0.219、
    #    最大误差 40.2pp;二次回归 R²=0.960 但在高 θ 稀疏区仍差 29pp。
    #    代价是实打实的:收益最高的 4 个方案(实测注水 -2.35%~+5.36%，完全合规)
    #    曾被判为 +38%~+44% 超标而否决 —— **守门员在杀好人**。
    #    真正准的守门员只有模拟器本身:它给出实测 FWIT。
    #    因此改为:预测超标只警告并送模拟器裁定，由**实测**注水量决定去留。
    if abs(dev) > w_max_dev * 2.5:
        f.append(Flag("high", "water_budget_extreme",
                      f"预测注水偏离 {dev*100:+.1f}%，远超 ±{w_max_dev*100:.0f}% "
                      f"(预测模型 MAE 1.5pp、高 θ 区最大 29pp，故只在极端偏离时否决)"))
    elif abs(dev) > w_max_dev:
        f.append(Flag("medium", "water_budget_predicted",
                      f"预测注水偏离 {dev*100:+.1f}% 超出 ±{w_max_dev*100:.0f}%，"
                      f"但预测在高 θ 区不可靠，交由模拟器实测裁定"))
    if np.abs(thm).max() > 1.0:
        f.append(Flag("high", "theta_out_of_box",
                      f"θ 绝对值最大 {np.abs(thm).max():.2f} > 1.0，超出代理训练分布"))
    # 目标率远超实测能力上限 = 名义自由度浪费，方案不可执行
    tgt = BASE_W[:, None] * 10.0 ** (thm[SI, :].T)
    over = (tgt > CAP * 1.05).mean()
    if over > 0.5:
        f.append(Flag("medium", "capacity_saturated",
                      f"{over*100:.0f}% 的时段目标率超过实测注入能力上限，该部分 θ 无效"))
    return f


class _A:                                   # _run_sim 需要的最小参数对象
    def __init__(self, threads): self.threads = threads


def _theta_key(th) -> str:
    """方案的**内容指纹** —— 全局唯一、跨轮稳定、与位置无关。"""
    return "sol-" + hashlib.sha256(np.ascontiguousarray(
        np.asarray(th, np.float32).ravel()).tobytes()).hexdigest()[:8]


def adjudicate(th: np.ndarray, key: str, threads: int,
               c_inj: float = 1.0, c_prod: float = 0.5,
               w_max: float = 0.25) -> dict | None:
    """🔴 唯一裁判:OPM Flow 全物理模拟。取值一律用模拟器自报的 field_cum，
    绝不用代理预测，也不用对井曲线做梯形积分（两者实测相差 0.58%，是容差的数十倍）。

    不再把注水总量校回基准 —— 水已进入目标函数计价（见 fc_water_econ），
    "注多少水"因此是**内生决策**而非外加约束。"""
    # 🔴 2026-09-05 审计抓到:缓存键此前只由 臂+运行号+轮次+候选序号 拼成，**不含 θ**。
    #    反复重跑同一 tag 时，后一次的 r2_c3 会静默复用前一次完全不同 θ 的模拟结果 ——
    #    不重跑、不报错、θ 与 ΔNPV 直接错配。已在生产数据中发生。
    #    现把 θ 的 SHA-256 前 10 位并入 key。
    _fp = hashlib.sha256(np.ascontiguousarray(
        np.asarray(th, np.float32).ravel()).tobytes()).hexdigest()[:10]
    key = f"{key}_{_fp}"
    f = SIM / f"{key}.npz"
    if not f.exists():
        try:
            FD._run_sim(key, th.astype(np.float32).reshape(N_STAGE, len(INJ)), _A(threads))
        # 🔴 _run_sim 抛的是 SystemExit(fc_decide.py:424)，它继承自 BaseException，
        #    **不是 Exception** —— 此前的 `except Exception` 接不住，
        #    模拟一失败整次运行直接崩掉且不落盘。
        except (Exception, SystemExit) as e:                     # noqa: BLE001
            print(f"    🔴 模拟失败 {key}: {str(e)[:90]}"); return None
    if not f.exists():
        return None
    m = econ(f, c_inj=c_inj, c_prod=c_prod)
    d = np.load(f)
    m["water_inj"] = float(d["field_cum"][1][-1] - d["field_cum"][1][0])
    # 🔴 真正的预算合规以**实测** FWIT 判定，不用任何近似
    m["water_dev"] = m["water_inj"] / BASE_WATER - 1.0
    m["budget_ok"] = abs(m["water_dev"]) <= w_max
    return m


# ============================================================ 角色视角
# 🔴 修订史:
#   v1 无立场 → 七个角色建议全部撞车(还叠加了我在固定提示词里的结论剧透)。
#   v2 派方向性立场制造分歧 → 分歧确实出现，但我把牌堆歪了:
#      1 个增产派对 6 个减注/设限派，团队被系统性拖向保守;
#      而单 Agent 臂恰好只留那个唯一的增产派 —— 对照组带 buff，比较无效。
#      证据:full7 三次差结果产油 -0.07%/+0.04%/-2.70%(只省水不产油)，
#      总工日志原文出现"reservoir_engineer 少数派压力保全"。
#   v3(当前):**取消方向性立场**。分歧应当来自**不同的数据**，不是我派的偏见。
#      每个角色只声明自己盯什么、对什么负责;方向由它从自己的数据里读出来。
PERSPECTIVE = {
 "reservoir_engineer":
   "你盯的是地层能量与波及:压力剖面、含水演化、注采是否平衡。"
   "压力和波及支持增注你就说增，支持收敛你就说收敛 —— 方向从数据读，不预设。",
 "connectivity_analyst":
   "你盯的是井间通道:哪口注水井的水真正到达了哪口生产井，哪些是无效或负贡献。"
   "你对'水打给谁'负责，不对'总量多少'预设立场。",
 "production_surveillance":
   "你盯的是单井动态:谁在产油、谁只在产水、谁已水淹、谁还有余量。"
   "你按每口井的实际油水表现给增减建议，不搞一刀切。",
 "economics_analyst":
   "你盯的是扣除注水与采出水成本后的 NPV。它支持增注你就支持增注，"
   "支持收缩你就支持收缩 —— 你对钱负责，不对某个方向负责。",
 "geomechanics_expert":
   "你盯的是注入强度的物理边界:压力、压实、诱导裂缝风险。"
   "你的产出是**可行区间的上限**，不是'该增还是该减'的主张。"
   "🔴 你没有本油藏的任何实测应力数据，只有同类砂岩的类比经验。"
   "因此:给上限时必须说明它是类比估计、不确定性有多大;"
   "**不要因为不确定就把上限压低** —— 那是把无知当成保守，"
   "会让团队错过实测数据支持的方向。若实测数据显示某方向有效而你无证据反对，"
   "请明确说'本角色无证据反对'。",
 "seismic_4d_analyst":
   "你盯的是水驱前缘的不确定性:哪些方案对前缘假设最敏感、判断错了代价最大。"
   "你的产出是**风险敞口排序**，不是方向建议。"
   "🔴 你只有文献标题与 DOI，没有本油藏的前缘实测。"
   "因此只排序风险、不否决方向;无证据时明确说'本角色无证据反对'。",
 "constraint_auditor":
   "你盯的是可执行性:目标注水率能否兑现、哪口井的 theta 是空转、方案是否越界。"
   "你的产出是**可行/不可行的判定**，与增减方向无关。",
}

# ============================================================ 数据切片






def _econ_limit_table() -> str:
    """经济极限含水率随时间的变化 —— 由油价与成本直接算出，非最优解。

    经济极限含水率 fw* 满足:多采一方液体时，油收入 = 采出水处理成本，即
        (1 - fw*) * p_oil = fw* * c_prod   =>   fw* = p_oil / (p_oil + c_prod)
    含水率越过 fw* 之后，继续采液在经济上就是净亏。
    这是任何油藏经济评价教科书都会算的量，属于领域先验而非本案例的解。
    """
    b = np.load(SIM / "baseline.npz")
    ob = b["obs"].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    o, w = ob[:, 0, :].sum(0), ob[:, 1, :].sum(0)
    wc = w / np.maximum(o + w, 1e-9)
    import datetime as _dt
    from fc_water_econ import BRENT as _BR, T_START as _TS
    yrs = np.array([(_TS + _dt.timedelta(days=float(x))).year for x in DAYS])
    px = np.array([_BR[min(max(y, min(_BR)), max(_BR))] for y in yrs])
    lim = px / (px + 1.0)                       # c_prod = 1.0 USD/bbl
    t = (DAYS - DAYS[0]) / 365.25
    rows = "\n".join(
        f"    第{t[i]:4.1f}年  实际含水 {100*wc[i]:5.1f}%   经济极限 {100*lim[i]:5.1f}%   "
        f"余量 {100*(lim[i]-wc[i]):+5.1f} pp"
        for i in range(0, len(DAYS), 5))
    return ("\n  基准情形的含水率 vs 经济极限含水率(fw* = p_oil/(p_oil+c_prod)):\n"
            + rows +
            "\n  → 余量越小，该时段每多采一方液体的净收益越低。")


_TILT_BINS = ROOT / "_pipelines" / "fc_team" / "tilt_bins.json"


def tilt_prior() -> str:
    """注水时程倾斜度(tilt)与收益的统计关系 —— 由 1061 个已模拟算例导出。

    🔴 为什么必须显式给(2026-09-05 子智能体元思维分析的核心发现):
       本问题的 ΔNPV 几乎是**一个标量**的函数:
         tilt = 前两段均值 - 后两段均值,  spearman = +0.894,  R² = 0.700
       24 维决策空间里一个标量承载 70% 的收益信号;控制住 tilt 后，
       全部"哪口井打多少水"的空间自由度合计只补 12 个百分点。
       而七个角色按**油藏工程学科**分工(压力/连通性/生产/经济/岩石力学/地震/约束)，
       其中五个的关注轴上 NPV 梯度≈0，于是团队系统性把 tilt 压平:
         full7 tilt +0.551  vs  单角色 +1.054   (Welch p = 3.8e-05)
       实测:诊断模式下(无历史反馈)七角色合议的 tilt 只有 +0.231。

       🔴 这张表是**统计规律**不是答案:它给方向与量级，不给任何具体 θ。
          给它不构成泄露 —— 判据是"这条信息是否只有解完题之后才写得出来"，
          而 tilt-收益关系从任意一批扰动算例都能回归出来。
    """
    if not _TILT_BINS.exists():
        return ""
    d = json.loads(_TILT_BINS.read_text())
    rows = "\n".join(
        f"    tilt [{lo:+.2f}, {hi:+.2f})   n={n:4d}   平均 ΔNPV {mu:+7.1f} M$   p90 {p90:+7.1f} M$"
        for lo, hi, n, mu, p90 in d["bins"])
    return (f"\n\n【注水时程倾斜度(tilt)与收益的统计关系 —— {d['n']} 个算例】\n"
            f"  样本范围:{d.get('source','')}\n"
            "  定义:tilt = (第1、2段的平均 θ) − (第5、6段的平均 θ)，即"
            "「早期相对后期多注多少」。\n"
            f"  实测 spearman(tilt, ΔNPV) = {d['spearman']:+.3f}。\n"
            "  分箱统计:\n" + rows +
            "\n  → 这是本问题**最主要的收益轴**。井间分配(哪口井打多少水)在控制 tilt 后\n"
            "     只额外解释约 12 个百分点。\n"
            "  🔴 本表给的是方向与量级，不含任何具体方案。你仍需自己决定\n"
            "     各段各井的具体数值、并满足注水总量硬约束。")


def domain_rules() -> str:
    """水驱井控的通用领域规则。

    🔴 出处必须是**文献与油藏工程常识**，不得是本项目跑出来的最优解。
       (2026-09-04)此前给的是"方案库里最好的 8 个 θ 逐格打印"+"顶级配方统计"，
       那是**答案**不是知识 —— 等于考前发答案，三个臂都在抄，结果全部作废。
       规则与答案的界线:
         ❌ "最好的方案是 +0.62 +0.62 +0.62 +0.62 +0.03 -0.63"
         ✅ "含水率上升前注入的水驱油效率最高"
       前者是解题结果，后者是任何油藏工程师上班第一天就知道的东西。

    条目出处:petro-knowledge/05-生产优化.md 及其引用的
    Jansen 2008 (J. Process Control 18:846)、Van Essen 2011 (SPE J 16(1))。
    """
    return """

【水驱井控的通用领域规则(来自油藏工程文献，非本案例的解)】
  1. 波及效率随含水率上升而下降:在综合含水率仍低时注入的水，驱油效率最高;
     含水率越过经济极限后，注入的水基本原样采出，只增加处理成本而不增油。
  2. 存在长期采收率与短期现金流的冲突。折现率越高，越倾向前置产量;
     折现率接近零时，最大化最终采收率才是最优。(Van Essen 2011)
  3. Reactive control 是行业参照策略:生产井含水超过阈值即关停或减注上游注水。
     在高含水油田它经常出人意料地强。(Jansen 2008)
  4. 注水井受最低井底流压约束时，提高目标注水率**不会**提高实际注入量。
     对这类井做速率调整是无效控制,应先判断哪些井的控制是真正有效的。
  5. 井间连通性决定注水去向。注入到与高含水生产井强连通通道的水，
     边际增油低;应优先支持仍有产油潜力的连通方向。
  6. 在总注入量受约束时，价值来自**重新分配**而非增加总量:
     把水从低效时段/井挪到高效时段/井。
"""


def _dyn(last: Path | None, role_id: str) -> str:
    """把**上一轮实测算例**在该角色自己的数据域里的表现拼出来。

    🔴 P4.13 根因(2026-09-02):此前 slice_for 的 hist 形参从未被函数体使用，
       每个专家每一轮都重读同一个 baseline.npz —— 数据恒定，
       于是三轮里 economics 逐轮重复"最优 97.7%"、connectivity 重复"F-4H 影响力 2.81"。
       专家提供的是**静态先验而非自适应分析**，真正在学习的只有总工一人。
       这同时解释了:多专家不优于单专家(先验重复不叠加)、
       质疑轮 0 个改主意(无新信息)、裸 LLM 打平(总工独自拿反馈也能学)。
    """
    if last is None or not last.exists():
        return "\n\n【上一轮实测】首轮，尚无实测反馈。"
    b, c = np.load(SIM / "baseline.npz"), np.load(last)
    fb, fcc = b["field_cum"], c["field_cum"]
    ob = b["obs"].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    oc = c["obs"].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    d_oil = (fcc[0][-1] - fcc[0][0]) / (fb[0][-1] - fb[0][0]) - 1
    d_wat = (fcc[1][-1] - fcc[1][0]) / (fb[1][-1] - fb[1][0]) - 1
    head = (f"\n\n【上一轮实测算例 {last.stem}(模拟器裁定，非代理预测)】\n"
            f"  全期产油 {100*d_oil:+.2f}% vs 基准，实测注水 {100*d_wat:+.2f}% vs 基准")

    if role_id == "reservoir_engineer":
        return head + (
            f"\n  你的数据域:油藏压力 FPR 末值 {fcc[3][-1]:.1f} bar(基准 {fb[3][-1]:.1f})，"
            f"中期 {fcc[3][len(DAYS)//2]:.1f}(基准 {fb[3][len(DAYS)//2]:.1f})。"
            "\n  → 这套注水实际把压力推到了哪里?与你上一轮的预期一致吗?")
    if role_id == "production_surveillance":
        wo, ww = oc[:, 0, :].sum(1), oc[:, 1, :].sum(1)
        bo, bw = ob[:, 0, :].sum(1), ob[:, 1, :].sum(1)
        m = np.argsort(-bo)[:5]
        rows = "\n".join(
            f"    {NB.PRODUCERS[i]:>7s} 油 {100*(wo[i]/max(bo[i],1e-9)-1):+6.1f}%  "
            f"水 {100*(ww[i]/max(bw[i],1e-9)-1):+6.1f}%" for i in m)
        return head + "\n  你的数据域:主力生产井相对基准的变化\n" + rows + \
            "\n  → 哪些井真的多产了油?哪些只是多产了水?"
    if role_id == "constraint_auditor":
        ia, ib = c["inj_actual"], b["inj_actual"]
        rows = "\n".join(
            f"    {w:>6s} 实测峰值 {ia[k].max():8,.0f}(基准 {ib[k].max():8,.0f})"
            for k, w in enumerate(INJ))
        return head + "\n  你的数据域:各注水井实测峰值\n" + rows + \
            "\n  → 目标值兑现了吗?哪口井的 theta 又是空转?"
        return head + "\n  你的数据域:各注水井实测峰值\n" + rows + \
            "\n  → 目标值兑现了吗?哪口井的 theta 又是空转?"
    if role_id == "connectivity_analyst":
        wo = oc[:, 0, :].sum(1); bo = ob[:, 0, :].sum(1)
        m = np.argsort(-bo)[:5]
        rows = ", ".join(f"{NB.PRODUCERS[i]} {100*(wo[i]/max(bo[i],1e-9)-1):+.1f}%" for i in m)
        return head + f"\n  你的数据域:主力生产井产油变化 {rows}" + \
            "\n  → 实际响应的井，和你按 beta 预测的井是同一批吗?"
    return head + "\n  → 这个结果与你的判断是否一致?"


def slice_for(role_id: str, carto: dict, last: Path | None = None) -> tuple[str, Provenance]:
    """给角色装配它**真正能读到**的数据:静态背景 + 上一轮实测在其数据域的表现。"""
    b = np.load(SIM / "baseline.npz")
    fc = b["field_cum"]; ob = b["obs"].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    yrs = (DAYS - DAYS[0]) / 365.25

    if role_id == "reservoir_engineer":
        fpr = fc[3]; wc = ob[:, 1, :].sum(0) / np.maximum(ob[:, 0, :].sum(0) + ob[:, 1, :].sum(0), 1e-9)
        txt = ("基准情形油藏压力 FPR(bar) 与综合含水率随时间:\n" +
               "\n".join(f"  第{y:4.1f}年  FPR={p:7.1f}  含水={w*100:5.1f}%"
                         for y, p, w in zip(yrs[::5], fpr[::5], wc[::5])))
        return txt + _dyn(last, role_id), Provenance("sim/baseline.npz", "field_cum[FPR] + obs[WWPR,WOPR]",
                               "simulation", sha256_file(SIM / "baseline.npz"))

    if role_id == "connectivity_analyst":
        B = np.array(carto["payload"]["beta"]); prods = carto["payload"]["producers"]
        rows = [(p, B[:, j]) for j, p in enumerate(prods) if np.abs(B[:, j]).max() >= .05]
        txt = ("注水井→生产井 标准化偏回归系数(由 283 个已模拟算例反推，条件数 3.09 可归因):\n"
               f"  {'生产井':>8s} " + " ".join(f"{w:>8s}" for w in INJ) + "\n" +
               "\n".join(f"  {p:>8s} " + " ".join(f"{v:+8.3f}" for v in c) for p, c in rows) +
               f"\n各注水井总影响力(|β|行和): " +
               ", ".join(f"{w}={np.abs(B[k]).sum():.2f}" for k, w in enumerate(INJ)))
        return txt + _dyn(last, role_id), Provenance("_pipelines/fc_team/cartography.json", "payload.beta",
                               "simulation", carto["provenance"]["source_sha256"])

    if role_id == "production_surveillance":
        oil = ob[:, 0, :].sum(1); wat = ob[:, 1, :].sum(1)
        wc = wat / np.maximum(oil + wat, 1e-9)
        order = np.argsort(-oil)
        txt = ("基准情形各生产井(按累计产油排序，含水率高者优先考虑减注上游水):\n" +
               "\n".join(f"  {NB.PRODUCERS[i]:>7s}  累计油={oil[i]:10,.0f}  含水={wc[i]*100:5.1f}%"
                         for i in order[:12] if oil[i] > 0))
        return txt + _dyn(last, role_id), Provenance("sim/baseline.npz", "obs[:, WOPR/WWPR, :]",
                               "simulation", sha256_file(SIM / "baseline.npz"))

    if role_id == "economics_analyst":
        # 🔴 2026-09-05 第二处泄露(用户怀疑单 Agent 作弊，查实):
        #    此前这里读 _pipelines/fc_water_econ/sweep.json，而该文件每一行都是
        #    「水价 X → 最佳方案 = <算例名>，ΔNPV = +514.3M，注水量 = -2.35%」——
        #    **那是已知最优解的答案卡片**，不是市场事实。
        #    更糟的是单 Agent 臂保留的恰好就是本角色，于是:
        #      单 Agent   = 唯一能看到"最优注水 -2.35%、可赚 +514M"的角色 → +490M
        #      Agent Team = 同一角色被稀释在七人中并需与他人争论        → +197M
        #    排序反常正是这处泄露造成的。
        #    现改为只给**市场与成本事实**:油价序列、水成本、折现率 —— 不含任何最优解。
        import datetime as _dt
        from fc_water_econ import BRENT as _BR, T_START as _TS, BBL as _BBL
        _CI, _CP = C_INJ, C_PROD
        _yrs = sorted({(_TS + _dt.timedelta(days=float(x))).year for x in DAYS})
        _px = "\n".join(f"    {y}  {_BR[y]:6.2f} USD/bbl" for y in _yrs if y in _BR)
        txt = ("经济口径(市场与成本事实，不含任何方案评价结果):\n"
               "  预测期各年 EIA Brent 年均现货价:\n" + _px +
               f"\n  注水成本 {_CI:.1f} USD/bbl = {_CI*_BBL:.2f} USD/Sm3\n"
               f"  采出水处理 {_CP:.1f} USD/bbl = {_CP*_BBL:.2f} USD/Sm3\n"
               f"  折现率 8%/年;体积换算 1 Sm3 = {_BBL:.4f} bbl\n"
               "  目标 = 折现后的(油收入 - 注水成本 - 采出水处理成本)\n"
               + _econ_limit_table())
        return txt + _dyn(last, role_id), Provenance(
            "EIA-RBRTE + cost assumptions", "annual Brent price series + unit costs",
            "measurement", sha256_arr(np.array(sorted(_BR), dtype="U")))


    if role_id == "constraint_auditor":
        ia = b["inj_actual"]
        txt = ("目标注水率 vs 实测(基准情形峰值，Sm3/day):\n" +
               "\n".join(f"  {w:>6s}  基准率={BASE_W[k]:8,.0f}  实测峰值={ia[k].max():8,.0f}  "
                         f"达成率={100*ia[k].max()/BASE_W[k]:5.1f}%" for k, w in enumerate(INJ)) +
               "\n🔴 F-4H 受最低井底流压上限截断，目标守恒 ≠ 实测守恒。任何方案都要复核实测注水量。")
        return txt + _dyn(last, role_id), Provenance("sim/baseline.npz", "inj_actual",
                               "simulation", sha256_file(SIM / "baseline.npz"))

    if role_id == "geomechanics_expert":
        txt = ("🔴 本油藏 deck 不含 GEOMECH/STRESS/YOUNGMOD/POISSON 关键字，"
               "无 Norne 实测岩石力学数据。你只能基于北海同类砂岩油藏的**类比**经验，"
               "给出注入压力上限与压实/破裂风险的**定性**边界。"
               "严禁声称掌握 Norne 的实测应力数据。")
        return txt + _dyn(last, role_id), Provenance("north-sea-sandstone-analogue", "qualitative-prior",
                               "literature", sha256_arr(np.array([role_id], dtype="U")))

    if role_id == "seismic_4d_analyst":
        txt = ("Norne 有丰富的 4D 时移地震文献(检索所得，示例):\n"
               "  doi:10.4043/19049-ms  Estimating 4D Velocity Changes and Contact Movement, Norne\n"
               "  doi:10.1190/segam2019-3216321.1  Time-lapse seismic inversion, Norne Field\n"
               "  doi:10.2523/iptc-10894-ms  Integrated Reservoir Management: Time-Lapse Acquisition\n"
               "据此给出水驱前缘推进与流体接触面移动的**定性**先验。"
               "你只有文献标题与 DOI，没有其中的数值，不得编造具体数字。")
        return txt + _dyn(last, role_id), Provenance("crossref:norne-4d-seismic", "titles+DOIs",
                               "literature", sha256_arr(np.array(["4d"], dtype="U")))

    return "(无专属数据，仅综合他人消息)" + _dyn(last, role_id), Provenance(
        "internal", "messages", "reasoning", sha256_arr(np.array([role_id], dtype="U")))


# ============================================================ LLM
LLM_CALLS = {"n": 0, "chars_in": 0}


def ask(prompt: str, model: str, tmo: int, retry: int = 1) -> tuple[str, dict]:
    """无状态调用独立 LLM 实例。cwd=/tmp 且不给工具,避免它看到本项目文件。"""
    # 🔴 P4.10:加质疑轮后每个 worker 的 LLM 调用翻倍，6 worker × 8 并发 ≈ 48 个同时调用，
    #    实测 6 个臂在同一分钟内集体 rc=1(stderr 为空)—— 过载而非逻辑错。退避重试。
    LLM_CALLS["n"] += 1; LLM_CALLS["chars_in"] += len(prompt)
    r = None
    for k in range(4):
        r = subprocess.run(["claude", "-p", "--model", model, "--allowed-tools", ""],
                           input=prompt, cwd="/tmp", capture_output=True,
                           text=True, timeout=tmo)
        if r.returncode == 0:
            break
        time.sleep(5 * 2 ** k + random.uniform(0, 3))
    if r.returncode != 0:
        raise RuntimeError(f"LLM rc={r.returncode} (重试 4 次仍失败): {r.stderr[:400]}")
    s = r.stdout.strip()
    if "```" in s:
        s = s.split("```")[1]
        s = s[s.find("{"):] if s.lstrip().startswith("json") else s
    try:
        return r.stdout, json.loads(s[s.find("{"): s.rfind("}") + 1])
    except json.JSONDecodeError:
        # 🔴 实测:审计员偶尔吐出带未转义引号的 JSON，整轮就崩了。重试一次并加硬提示。
        if retry <= 0:
            raise
        return ask(prompt + "\n\n🔴 上次输出不是合法 JSON。只输出一个 JSON 对象，"
                            "字符串内不得出现未转义的引号或换行。", model, tmo, retry - 1)


FIXED = """你是一个油藏注水决策团队中的角色:**{role}**。
职责:{duty}

【固定背景 —— Norne 油田】
北海砂岩油藏，46x112x22 网格，44431 活网格。
决策变量 theta 是 {ns} 个时段 x {ni} 口注水井 = 24 个数，
每个数是该井该时段注水率的 log10 乘子(0 表示保持基准，+0.3 约为 2 倍，-0.3 约为一半)。
注水井 {inj}，基准率(Sm3/day) {basew}。
时段起始年 {stages}。预测期约 13 年。
【本次优化的经济口径】
注水成本 {cinj} USD/bbl，采出水处理 {cprod} USD/bbl，油价用 2006-2019 真实 Brent 年均价。
水不是免费的:多注水通常多产油，但目标是**扣除水成本后的 NPV**，不是产油量。
最优注水总量该往哪个方向走，由你从下面的数据自己判断 —— 不要假设答案。

【共享背景:全部角色一致(不区分角色)】
{shared}

【你的专业视角】
{stance}
🔴 你和其他角色读的是**不同的数据**，所以出现分歧是正常的、也是团队价值所在。
   请如实报告你的数据支持什么，即使它与别人相反；
   不要为了显得稳妥而向中庸靠拢，也不要为了制造分歧而故意唱反调。

🔴 你只能基于下面给你的数据发言。数据支持到什么程度，confidence 就给到什么程度 ——
   有硬数据支撑就大胆给高分，纯属类比或猜测就如实给低分。不要一律往低报。

【你能读到的数据】
{data}
"""

DYN = """
【本轮动态信息 —— 第 {r} 轮】
{hist}
"""

# 🔴 ASK_READER 直接拼接、不走 .format()，所以这里必须写单大括号；
#    此前写成 {{ }} 是照抄了 ASK_CRITIQUE(它走 format)的转义，双括号原样漏进了提示词。
ASK_READER = """
请给出你的专业判断。只输出 JSON:
{"assessment": "<=120字的判断", "recommendation": "<=100字，具体说哪些井哪些时段该增/减注水",
 "confidence": 0.0~1.0, "concerns": ["<=3条风险"],
 "evidence": "<=400字。从你上面读到的数据里**逐字摘录**你认为总工必须亲眼看到的部分(表格照抄即可)。总工看不到你的原始数据，只能通过这个字段拿到证据。不要在这里写你的观点。"}
"""

ASK_CRITIQUE = """
【第二轮 —— 你现在能看到其他角色的意见了】
{peers}

请以你的立场审视上面的分歧。只输出 JSON:
{{"challenge": "<=120字，指名道姓反驳你最不同意的那条意见，说清理由>",
  "revision": "<=100字，你自己的建议要不要改；不改就写'维持'并说明为什么其他人没说服你>",
  "confidence": 0.0~1.0}}
🔴 如果其他人的数据说服了你，就明说改变判断 —— 被证据说服不是让步。
   如果没被说服，说清楚你的数据为什么更有说服力。
"""

ASK_CHIEF = """
你是总工。

{incumbent}
【第零部分:模拟器实测(全物理裁定，非任何模型的预测)】
{truth}

【第一部分:各角色**原文转呈**的关键证据(未经他们概括，逐字转发)】
{facts}

【第二部分:各角色的判断与互相质疑】
{msgs}

🔴 三部分的优先级:第零部分是**唯一裁判给出的事实**，优先级最高;
   第一部分是角色转呈的原始证据;第二部分是他们对证据的**解读**。
   三者冲突时，一律以第零部分为准。
   两者冲突时以证据为准;各角色都没提到的证据要点，你仍应自行采纳。
   🔴 若第一部分为空，说明本臂没有配备领域角色，你只能依据任务描述与
      模拟器反馈自行判断 —— 这是有意的对照设置，不是遗漏。

🔴 各角色读的是**不同的数据**，出现分歧是正常的。你的职责不是取平均，
   而是判断在当前证据下谁更有道理。

🔴 **按证据强度裁决，不按人头。** 每条意见都标了 `依据类型`:
     simulation / measurement = 本油藏的实测或模拟数据 —— 权重最高;
     literature               = 同类油藏的类比文献，**不含本油藏任何实测** —— 仅作定性边界;
     reasoning                = 无专属数据，只是综合他人意见 —— 不独立构成证据。
   一条 literature 类的顾虑**不足以否决**一条 simulation 类的正向证据;
   若二者冲突，应采纳实测方向并在方案里为该风险留出余量，而不是直接放弃该方向。
   同理:多个角色都表达同一顾虑，若它们的依据类型都是 literature，
   那仍然只是**一份**类比证据，不因人多而变强。
   🔴 注意:角色按油藏工程学科分工(压力/连通性/生产/经济/岩石力学/地震/约束)，
      这个分工**不等于**目标函数的敏感方向。若多数角色的关注点都落在
      对 NPV 影响很小的维度上，你不应因为"人多"就跟随;
      请以第零部分的实测数字判断哪个方向真正改变了 NPV。

请提出 {n} 个互不相同的候选方案。只输出 JSON:
{{"candidates": [{{"theta": [[{ni}个数] x {ns}段], "rationale": "<=80字，注明采纳了哪些角色的意见"}}],
  "confidence": 0.0~1.0}}
每个 theta 是 **{ns} 行 × {ni} 列**的二维数组:**第一维是时段(共 {ns} 段)，第二维是注水井(共 {ni} 口)**。
即 theta[段][井]。不要写成 {ni} 行 × {ns} 列 —— 两者元素个数相同，写反了方案含义完全不同。
元素取值建议在 -0.8 ~ +0.8 之间。

🔴 **硬约束(违反者会被程序直接否决，不进模拟器)**:
   1. 全期实测注水总量必须落在基准的 ±{wdev:.0f}% 以内。
   2. 每个 theta 分量的绝对值不得超过 1.0。

🔴 **不要因为怕超预算就把幅度压小。** 预算是否超标由**模拟器实测**判定，
   不由任何近似公式判定。预测超标只给一条警告，方案照样送模拟器;
   只有预测极端偏离(超过容差 2.5 倍)才会被直接否决。
   幅度不足是常见的失败原因。
"""

ASK_AUDIT = """
你是红队审计员。下面是总工的候选方案和代理模型的快速评估:
{cands}

各角色的意见与置信度:
{msgs}

请挑毛病。只输出 JSON:
{{"flags": [{{"severity": "high|medium|low", "code": "<短标识>", "detail": "<=80字",
             "candidate": <候选编号，若是全局问题则填 -1>}}],
  "verdict": "pass|revise", "confidence": 0.0~1.0}}
🔴 被你标为 high 的候选会被**直接否决**，不进入模拟器，也不会被采纳。请慎用 high。
🔴 下面已列出程序化硬约束检查的结果；那些是数字判定，你不必重复，请找它们之外的问题。
"""


def run(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    carto = json.loads((OUT / "cartography.json").read_text())
    proxy = Proxy(a.device)
    active = [r for r in ROLES if r["id"] not in a.drop and r["id"] != "chief_engineer"]
    print(f"=== Gaia 团队环路 ===\n角色 {len(active)} 个"
          + (f"(消融剔除 {a.drop})" if a.drop else "") + f"  模型={a.model}\n")

    hist, all_msgs, clean_streak, rounds = "首轮，尚无历史反馈。", [], 0, []
    last_case: Path | None = None          # 上一轮被模拟器裁定的最佳算例
    # 🔴 精英保留(2026-09-05):此前每轮都重新提六个候选，好方案跑丢就找不回来。
    #    实测 Team 七次里六次"越跑越差":首轮均值 +431.4M → 终态 +291.8M，
    #    闭环净效果 -139.6M(-32%);而单 Agent 只掉 12.9M。
    #    原因是七个角色每轮都提新顾虑，总工每轮响应最新的那个，
    #    不断推翻上一轮**已被模拟器验证过**的好方案。
    #    这不是"角色有害"，是迭代优化缺了 elitism。
    global C_INJ, C_PROD, SHARED_BG
    C_INJ, C_PROD = a.c_inj, a.c_prod          # 角色切片与提示词从模块级变量读，不硬编码
    # 🔴 2026-09-05 对等性修复(用户指出单 Agent 仍占优 + 审计 major):
    #    tilt 先验与领域规则此前只发给 7 个角色中的 4 个，而没拿到的 3 个
    #    (constraint_auditor / geomechanics_expert / seismic_4d_analyst)
    #    恰好是 PERSPECTIVE 限定"只能给上限/风险/否决"的角色 ——
    #    既不给数据又限定只能唱反调，它们在团队里纯稀释。
    #    更糟的是单 Agent 臂保留的 economics_analyst 恰好在"拿到"组。
    #    现把二者提升为**全体共享背景**:分歧只来自各自的专属数据，
    #    而不是"谁拿到了共享背景"。
    SHARED_BG = (tilt_prior() + domain_rules()).strip()
    base_m = econ(SIM / "baseline.npz", c_inj=a.c_inj, c_prod=a.c_prod)
    prev_truth = ""                        # 累积的模拟器实测记录，直连总工
    incumbent: dict | None = None          # 迄今被模拟器验证的最优
    incumbent_th: np.ndarray | None = None
    for rnd in range(1, a.max_rounds + 1):
        print(f"--- 第 {rnd} 轮 ---")
        # 1) Readers 并行
        def one(role):
            data, prov = slice_for(role["id"], carto, last_case)
            p = (FIXED.format(role=role["id"], duty=role["duty"], ns=N_STAGE, ni=len(INJ),
                              inj=INJ, basew=BASE_W.round(0).tolist(),
                              stages=FG.STAGE_AT, data=data,
                              cinj=a.c_inj, cprod=a.c_prod,
                              stance=PERSPECTIVE.get(role["id"], "按你能读到的数据说话。"),
                              shared=SHARED_BG)
                 + DYN.format(r=rnd, hist=hist) + ASK_READER)
            try:
                _, ans = ask(p, a.model, a.timeout)
            except Exception as e:                                    # noqa: BLE001
                return role["id"], None, f"{type(e).__name__}: {e}"
            m = Message(role["id"], "reading", ans,
                        float(np.clip(ans.get("confidence", .5), 0, 1)), prov)
            return role["id"], m, validate(m)

        # 🔴 粗粒度消融需要「零分析师」臂(裸 LLM 只剩总工)。
        #    旧版 active 为空时会走到"全部角色失败"并中止 —— 那不是失败，是实验设计。
        res = []
        if active:
            with ThreadPoolExecutor(max_workers=min(4, len(active))) as ex:
                res = list(ex.map(one, active))
        msgs = []
        for rid, m, err in res:
            if m is None:
                print(f"  🔴 {rid:<24s} 调用失败: {err}"); continue
            ok = "✅" if not err else f"🔴 {err}"
            print(f"  {rid:<24s} conf={m.confidence:.2f} {ok}  {m.payload.get('assessment','')[:52]}")
            if not err:
                msgs.append(m); all_msgs.append(m)
        if not msgs and active:
            print("  🔴 本轮全部角色失败，中止"); return 1

        # 🔴 P4.9:角色互看 + 质疑。此前 7 个角色完全并行、互不可见，
        #    总工只做一次拼接 —— 那是问卷调查，不是团队。
        crit = {}
        if a.critique and len(msgs) > 1:
            peers = "\n".join(
                f"[{m.agent_id}] {m.payload.get('assessment','')} "
                f"建议: {m.payload.get('recommendation','')}" for m in msgs)

            def one_crit(m):
                role = next(r for r in ROLES if r["id"] == m.agent_id)
                data, _ = slice_for(m.agent_id, carto, last_case)
                others = "\n".join(l for l in peers.split("\n")
                                    if not l.startswith(f"[{m.agent_id}]"))
                p = (FIXED.format(role=role["id"], duty=role["duty"], ns=N_STAGE,
                                  ni=len(INJ), inj=INJ, basew=BASE_W.round(0).tolist(),
                                  stages=FG.STAGE_AT, data=data, cinj=a.c_inj,
                                  cprod=a.c_prod,
                                  stance=PERSPECTIVE.get(role["id"], ""),
                                  shared=SHARED_BG)
                     + ASK_CRITIQUE.format(peers=others))
                try:
                    _, ans = ask(p, a.model, a.timeout)
                    return m.agent_id, ans
                except Exception:                                  # noqa: BLE001
                    return m.agent_id, None

            with ThreadPoolExecutor(max_workers=min(4, len(msgs))) as ex:
                for rid, ans in ex.map(one_crit, msgs):
                    if ans:
                        crit[rid] = ans
            n_rev = sum(1 for v in crit.values()
                        if "维持" not in str(v.get("revision", "")))
            print(f"  质疑轮: {len(crit)}/{len(msgs)} 回应，{n_rev} 个改了主意")
            for rid, v in list(crit.items())[:3]:
                print(f"    {rid:<24s} 反驳: {str(v.get('challenge',''))[:56]}")

        # 🔴 领域证据只能**经由角色**到达总工。无角色臂 → 空 —— 这是有意的对照设置。
        evidence_txt = "\n\n".join(
            f"[{m.agent_id} 转呈]\n{m.payload.get('evidence','')}"
            for m in msgs if str(m.payload.get("evidence", "")).strip()) or \
            "(本臂未配备领域角色，无证据转呈)"

        # 🔴 2026-09-05:按证据强度排序后再呈给总工。
        #    实测(E2 批次)各角色的"反 tilt 用词"净倾向:
        #      geomechanics_expert +0.42(最强反向，而它只有 156 字纯文献、零本项目数据)
        #      economics_analyst  -0.75(最强正向，有本油藏经济数据)
        #    而 tilt 与收益相关 +0.714 —— 让零数据的类比角色与有实测的角色等权、
        #    且排在同一列表里不分先后，是团队 tilt 被压平(+0.570 vs 单 Agent +0.865)的机制。
        _RANK = {"simulation": 0, "measurement": 0, "deck": 0, "literature": 1, "reasoning": 2}
        msgs = sorted(msgs, key=lambda m: (_RANK.get(m.provenance.source_type, 2),
                                           -m.confidence))
        msg_txt = ("(本臂无分析师角色，总工独立决策)" if not active else "\n".join(
            f"[{m.agent_id} | conf={m.confidence:.2f} | 依据类型={m.provenance.source_type}] "
            f"{m.payload.get('assessment','')} 建议: {m.payload.get('recommendation','')}"
            + (f"\n    ↳ 质疑他人: {crit[m.agent_id].get('challenge','')}"
               f"\n    ↳ 二轮结论: {crit[m.agent_id].get('revision','')}"
               if m.agent_id in crit else "")
            for m in msgs))

        # 2) Synthesizer
        try:
            inc_txt = ""
            if incumbent is not None and incumbent_th is not None:
                _t = incumbent_th.reshape(N_STAGE, len(INJ))
                inc_txt = (
                    "【迄今已被模拟器验证的最优方案(现任冠军)】\n"
                    f"  ΔNPV@8% = {(incumbent['npv8']-base_m['npv8'])/1e6:+.1f} M$   "
                    f"实测注水 {incumbent['water_dev']*100:+.2f}%\n"
                    "  它的 theta[段][井]:\n" + "\n".join(
                        "    " + " ".join(f"{_t[_s, _j]:+6.2f}" for _j in range(len(INJ)))
                        for _s in range(N_STAGE)) +
                    "\n  🔴 这是**本轮之前已经真跑出来的成绩**，不是猜测。\n"
                    "     你的新候选里必须至少有一个是在它基础上的小幅改进"
                    "(而不是推倒重来);\n"
                    "     若某个角色的意见会让你明显偏离它，请先说明为什么值得冒这个险。\n"
                    "     角色每轮都会提出新顾虑，但**已被验证的成绩不应被未验证的顾虑推翻**。\n\n")
            # 🔴 2026-09-05 接口修复(子智能体元思维分析的首要发现):
            #    此前 hist(:737)只进 readers、sim_txt(:931)只进审计员，
            #    **总工从未直接读到过 OPM Flow 的任何数字** ——
            #    全系统唯一真正写 θ 的智能体，只能通过 7 个 LLM 的转述获得真值。
            #    这是一条转述信道，保真度随中继节点数下降，故角色越多损害越大:
            #    实测 full7 的 tilt 被压到 +0.551 而 one1 保持 +1.054(p=3.8e-5)，
            #    而 tilt 与 ΔNPV 的 spearman 高达 +0.894(R²=0.70, n=1058)。
            truth_txt = (prev_truth + "\n" + hist) if prev_truth else hist
            _, syn = ask(ASK_CHIEF.format(msgs=msg_txt, facts=evidence_txt,
                                          incumbent=inc_txt, truth=truth_txt,
                                          n=a.n_cand, ni=len(INJ), ns=N_STAGE,
                                          wdev=a.w_max_dev*100),
                         a.model, a.timeout)
        except Exception as e:                                        # noqa: BLE001
            print(f"  🔴 总工失败: {e}"); return 1
        cands = []
        n_transposed = 0
        for c in syn.get("candidates", []):
            try:
                th = np.asarray(c["theta"], float)
            except Exception:                                          # noqa: BLE001
                continue
            # 🔴 定时炸弹(2026-09-03 抓到):LLM 偶尔返回 (井, 时段) 而不是 (时段, 井)。
            #    两者都是 24 个元素，直接 reshape(6,4) **会静默成功并把井与时段彻底错位**，
            #    不抛任何异常。必须先看形状再决定是否转置。
            if th.shape == (len(INJ), N_STAGE):
                th = th.T; n_transposed += 1
            elif th.shape != (N_STAGE, len(INJ)):
                if th.size != N_STAGE * len(INJ):
                    continue
                th = th.reshape(N_STAGE, len(INJ))
            cands.append((th, c.get("rationale", "")))
        if n_transposed:
            print(f"  ⚠ 自动转置 {n_transposed} 个候选(LLM 返回了 井×时段 而非 时段×井)")
        if not cands:
            print("  🔴 总工未给出合法候选"); return 1

        # 3) 代理快速评估(这就是代理的用途:替代慢模拟器做批量测试)
        # 🔴 冠军作为候选参与排序，但**不重复模拟**:它的成绩已知，
        #    若再跑一次会白白吃掉本轮唯一的一次全物理模拟预算(--n-sim 1)。
        inc_idx = None
        if incumbent_th is not None:
            inc_idx = len(cands)
            cands.append((incumbent_th.reshape(N_STAGE, len(INJ)), "[现任冠军·精英保留]"))
        TH = np.stack([t.ravel() for t, _ in cands]).astype(np.float32)
        t0 = time.time(); ev = proxy.evaluate(TH); dt = time.time() - t0
        # 🔴 排序必须用**经济**口径,不能用纯产油。
        #    上一版按 ev["oil"] 排序,团队于是一路加水(+35%→+40%→+58%),
        #    因为水在排序里不要钱 —— 这是目标函数缺项,不是智能体的错。
        hard = {i: hard_check(TH[i], a.w_max_dev) for i in range(len(TH))}
        vetoed = {i for i, fs in hard.items() if any(x.severity == "high" for x in fs)}
        wi = np.array([inj_water(t) for t in TH])
        ev["score"] = (ev["oil"] * BBL * PRICE_AVG
                       - wi * BBL * a.c_inj - ev["water_prod"] * BBL * a.c_prod)
        ev["inj_water"] = wi
        order = [int(i) for i in np.argsort(-ev["score"]) if i not in vetoed]
        if vetoed:
            print(f"  🔴 硬约束否决 {len(vetoed)}/{len(TH)} 个候选:")
            for i in sorted(vetoed):
                print(f"     #{i}: " + "; ".join(f"{x.code}({x.detail[:40]})"
                                                 for x in hard[i] if x.severity == "high"))
        if not order:
            print("  🔴 全部候选被硬约束否决，要求总工重来")
            hist = ("上一轮你给的**全部**候选都因违反硬约束被否决:"
                    + "; ".join(sorted({x.code for fs in hard.values() for x in fs
                                        if x.severity == "high"}))
                    + f"。注水量必须落在基准的 ±{a.w_max_dev*100:.0f}% 内，"
                      "且 θ 每个分量绝对值不得超过 1.0。请重新给方案。")
            rounds.append({"round": rnd, "all_vetoed": True,
                           "n_cand": len(TH), "n_veto": len(vetoed)})
            clean_streak = 0
            continue
        print(f"  总工给出 {len(cands)} 个候选 → 代理评估耗时 {1e3*dt:.1f} ms "
              f"({1e6*dt/len(cands):.0f} µs/方案)")
        for i in order[:3]:
            print(f"    #{i} 代理:产油 {ev['oil'][i]:,.0f}  注水 "
                  f"{100*(ev['inj_water'][i]/BASE_WATER-1):+6.1f}%  "
                  f"经济分 {ev['score'][i]/1e6:,.0f}M$  {cands[i][1][:38]}")

        # 3b) 🔴 模拟器裁定 top-k —— 代理只负责筛，真值只认 OPM Flow
        adj, errs_pp, sim_keys = {}, [], {}
        for i in order[:a.n_sim]:
            if i == inc_idx:                     # 冠军无需重跑，成绩已知
                adj[int(i)] = incumbent
                print(f"    候选 #{i} 是现任冠军，复用已有实测(省一次模拟)")
                continue
            key = f"team{a.tag}_r{rnd}_c{i}"
            print(f"    模拟裁定 #{i} …", flush=True)
            m = adjudicate(TH[i], key, a.threads, a.c_inj, a.c_prod, a.w_max_dev)
            sim_keys[int(i)] = _theta_key(TH[i])
            if m is None:
                continue
            if not m.get("budget_ok", True):
                print(f"      🔴 实测注水 {m['water_dev']*100:+.1f}% 超出预算，该算例不计入")
                continue
            adj[int(i)] = m
            e = 100.0 * (ev["oil"][i] / m["oil"] - 1.0)
            errs_pp.append(e)
            print(f"      真值产油 {m['oil']:,.0f} (基准 {base_m['oil']:,.0f}, "
                  f"{100*(m['oil']/base_m['oil']-1):+.2f}%)  注水 "
                  f"{100*(m['water_inj']/base_m['winj']-1):+.2f}%  "
                  f"ΔNPV8={(m['npv8']-base_m['npv8'])/1e6:+,.1f}M$  代理误差 {e:+.2f}%")
        sim_txt = ("\n【模拟器(唯一裁判)对上述候选的实测结果】\n" + "\n".join(
            f"  候选#{i}: 实测累计产油 {m['oil']:,.0f} ({100*(m['oil']/base_m['oil']-1):+.2f}% vs 基准)，"
            f"实测注水 {100*(m['water_inj']/base_m['winj']-1):+.2f}%，"
            f"ΔNPV@8%={(m['npv8']-base_m['npv8'])/1e6:+,.1f}M$，"
            f"代理对它高估了 {100*(ev['oil'][i]/m['oil']-1):+.2f}%"
            for i, m in adj.items())) if adj else "\n(本轮无模拟结果)"

        cand_txt = "\n".join(
            f"  候选#{i}: 代理预测累计产油 {ev['oil'][i]:,.0f}(基准 {base_m['oil']:,.0f})，"
            f"注水量为基准的 {100*ev['inj_water'][i]/BASE_WATER:.1f}%，"
            f"扣水成本后经济分 {ev['score'][i]/1e6:,.0f}M$，理由: {cands[i][1]}"
            for i in order[:a.n_cand])

        # 4) Auditor
        try:
            hard_txt = ("\n【程序化硬约束检查(数字判定，已生效)】\n" + ("\n".join(
                f"  候选#{i}: " + "; ".join(f"[{x.severity}]{x.code} {x.detail}" for x in fs)
                for i, fs in hard.items() if fs) or "  全部候选通过硬约束"))
            _, aud = ask(ASK_AUDIT.format(cands=cand_txt + sim_txt + hard_txt, msgs=msg_txt),
                         a.model, a.timeout)
        except Exception as e:                                        # noqa: BLE001
            print(f"  🔴 审计失败: {e}"); return 1
        aud_flags = aud.get("flags", [])
        flags = [Flag(f.get("severity", "low"), f.get("code", "?"), f.get("detail", ""))
                 for f in aud_flags]
        flags += [x for fs in hard.values() for x in fs]
        # 🔴 审计员点名 high 的候选也被否决 —— 这就是"报警"变成"否决权"
        aud_veto = {int(f["candidate"]) for f in aud_flags
                    if f.get("severity") == "high"
                    and isinstance(f.get("candidate"), (int, float))
                    and 0 <= int(f["candidate"]) < len(TH)}
        if aud_veto:
            print(f"  🔴 审计员否决候选 {sorted(aud_veto)}")
        vetoed |= aud_veto
        hi = [f for f in flags if f.severity in ("high", "medium")]
        print(f"  审计: verdict={aud.get('verdict')} flags={len(flags)} "
              f"(high/medium {len(hi)})")
        for f in flags[:4]:
            print(f"    [{f.severity}] {f.code}: {f.detail[:64]}")

        rounds.append({"round": rnd, "n_msg": len(msgs), "n_cand": len(cands),
                       "proxy_best_oil": float(ev["oil"][order[0]]),
                       "flags": [asdict(f) for f in flags],
                       "verdict": aud.get("verdict"),
                       "adjudicated": {str(k): v for k, v in adj.items()},
                       "vetoed": sorted(vetoed),
                       "hard_flags": {str(i): [asdict(x) for x in fs]
                                      for i, fs in hard.items() if fs},
                       "proxy_err_pp": errs_pp,
                       "best_theta": TH[order[0]].tolist()})

        # 🔴 2026-09-05 Workflow 审计抓到:此处曾把 incumbent 更新与 prev_truth 累积
        #    写在 break **之后**，而我上一轮"修复"只改了注释、没验证顺序 ——
        #    注释断言"已在上文更新"，代码里它在下文。自适应终止那一轮的成绩会被静默丢弃。
        #    现改为:先更新状态，再判终止。
        for _i, _m in adj.items():
            if incumbent is None or _m["npv8"] > incumbent["npv8"]:
                incumbent, incumbent_th = _m, TH[_i].copy()
        if adj:
            prev_truth += ("" if not prev_truth else "\n") + "\n".join(
                f"  [第{rnd}轮·{sim_keys.get(int(i),'?')}] 实测 ΔNPV@8% {(m['npv8']-base_m['npv8'])/1e6:+.1f} M$   "
                f"产油 {100*(m['oil']/base_m['oil']-1):+.2f}%   实测注水 {m['water_dev']*100:+.2f}%   "
                f"θ各段均值 " + " ".join(f"{x:+.2f}" for x in TH[i].reshape(N_STAGE, len(INJ)).mean(1))
                for i, m in adj.items())
        # 5) 自适应终止:连续两轮零 high/medium —— 状态已在上方更新完毕
        clean_streak = clean_streak + 1 if not hi else 0
        print(f"  连续无高/中风险轮数 = {clean_streak}/2")
        if clean_streak >= 2:
            print("\n✅ 达到终止条件(连续两轮零 high/medium 风险)")
            break

        # 🔴 类层修复 #2「标识符用位置而非内容」(2026-09-05 审计):
        #    此前 hist 写「上一轮模拟器实测最佳:候选#4」，而候选编号每轮从 #0 重编 ——
        #    下一轮的「候选#4」是另一个方案。实证:run_Rnogeo_4.log 里
        #    production_surveillance 引用「候选#4验证有效」时，指的已是别的方案。
        #    同一份代码里两套编号约定:prev_truth 带轮次，hist 不带。
        #    现统一为**内容决定的全局 key**(含 θ 指纹)，两处一致。
        best_sim = max(adj.items(), key=lambda kv: kv[1]["npv8"], default=None)
        if best_sim is not None:
            last_case = SIM / f"team{a.tag}_r{rnd}_c{best_sim[0]}.npz"   # 喂给下一轮各专家
        hist = (
            (f"上一轮(第{rnd}轮)模拟器实测最佳:{sim_keys.get(best_sim[0], '?')}"
             f"  产油 {best_sim[1]['oil']:,.0f}"
             f"({100*(best_sim[1]['oil']/base_m['oil']-1):+.2f}% vs 基准)，"
             f"实测注水 {100*(best_sim[1]['water_inj']/base_m['winj']-1):+.2f}%，"
             f"ΔNPV@8%={(best_sim[1]['npv8']-base_m['npv8'])/1e6:+,.1f}M$。"
             if best_sim else "上一轮无有效模拟结果。")
            + (f" 🔴 代理在这些候选上平均高估 {np.mean(errs_pp):+.2f}%，"
               f"因此**不要相信代理的绝对数值**，只用它排序。" if errs_pp else "")
            + f" 审计给出 {len(hi)} 条高/中风险:"
            + "; ".join(f"{f.code}—{f.detail}" for f in hi[:3]))

    res = {"incumbent_npv8": None if incumbent is None else incumbent["npv8"],
           "incumbent_theta": None if incumbent_th is None else incumbent_th.tolist(),
           "rounds": rounds, "roles": [r["id"] for r in active], "dropped": a.drop,
           "model": a.model, "llm_calls": LLM_CALLS["n"],
           "llm_prompt_chars": LLM_CALLS["chars_in"],
           "note": "代理仅用于筛选；最终取值须由 OPM Flow 的 field_cum 裁定。"}
    (OUT / f"loop{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n→ {OUT/f'loop{a.tag}.json'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--n-cand", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--drop", nargs="*", default=[], help="消融:剔除的角色 id")
    ap.add_argument("--critique", action="store_true", default=True,
                    help="启用角色互看质疑轮(P4.9)")
    ap.add_argument("--no-critique", dest="critique", action="store_false")
    ap.add_argument("--w-max-dev", type=float, default=0.25,
                    help="允许的注水量偏离基准的上限(硬约束)")
    ap.add_argument("--c-inj", type=float, default=1.0, help="注水成本 USD/bbl")
    ap.add_argument("--c-prod", type=float, default=0.5, help="采出水处理 USD/bbl")
    ap.add_argument("--n-sim", type=int, default=2, help="每轮交模拟器裁定的候选数")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tag", default="")
    sys.exit(run(ap.parse_args()))
