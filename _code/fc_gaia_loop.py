#!/usr/bin/env python3
"""闭环盖亚（Gaia-in-the-loop）：Agent 管理模拟预算，OPM Flow 做裁判。

## 与单发臂（fc_agent.py）的唯一区别

单发：Agent 猜一次 θ → 校正 → 跑 OPM → 完。**没有反馈。**
闭环：Agent 给 θ → 确定性校正 + OPM 真跑 → **把真实结果喂回去** → Agent 修正 → …

诊断（2026-08-28 四臂实验）：盖亚 G2 单发 NPV −229.8 M$，是唯一低于基准的臂；
无知识库 G0 单发 +177.6 M$。结论是"病根不是猜得准不准，是没有反馈"。本脚本验证它。

## 🔴 职责边界（share-agent-development §1.6：调度器不吃 LLM）

| 谁 | 干什么 |
|---|---|
| **确定性代码** | 注水量校正（解标量 s）、调用 OPM Flow、算指标、判终止、管预算 |
| **LLM** | 只做一件事：看真实反馈，给下一个 θ 与推理 |

LLM 从不决定"再跑一次吗"、"s 取多少"、"哪一轮最好"。

## 注水量校正：为什么能省 OPM 调用

约束是**实测**总注水量 = 基准 ±0.05%。老做法（fc_agent.evaluate）从 s=0 起步盲试，
每套 θ 要 2~3 次 OPM。5 次预算下连两轮都跑不完。

改进：先用一个**解析约束模型**把 s 算出来再跑。模型（见 `predict_water`）：
    实测注水率_k(t) = min( 基准率_k × 10^(θ_k(t)+s),  cap_k(t) )
cap_k(t) 是 BHP 500 bar 上限对应的注入能力，**从已有的 67 个历史模拟里拟合**
（`fitcap` 子命令，不花任何新的 OPM 调用）。
F-4H 的 cap 约 540~590 sm³/d，而基准目标率 2071 —— 这正是"假旋钮"的量化形式。

在 |θ|≤0.45 的范围里这个模型的中位误差 0.28%。它不够直接满足 ±0.05%，
但足以（a）给一个好的初值，（b）提供准确的局部斜率 d(实测水)/ds，
于是通常 1~2 次 OPM 就收敛，而不是 3 次。每次真跑后模型再做一次
乘性偏差更新（`bias`），下一轮起步更准。**全过程无 LLM 参与。**

用法:
    python fc_loop.py fitcap                     # 从已有模拟拟合 cap（0 次 OPM）
    python fc_loop.py check                      # 交叉验证约束模型精度（0 次 OPM）
    python fc_loop.py run --budget 5 --threads 12
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import forecast_gen as FG
import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_loop"
SIMDIR = ROOT / "_pipelines" / "fc_decide" / "sim"
CAPF = OUT / "water_cap.npy"

INJ = FG.INJ_W
NW = len(INJ)
BASE = np.array([FG.BASE_WINJ[w] for w in INJ], float)
DAYS = FG.FC_GRID
TW = np.gradient(DAYS).astype(float); TW[0] *= .5; TW[-1] *= .5   # 梯形权重
T_START = dt.date(1997, 11, 6)

# ---- 时段 → 时间格点映射（与 forecast_gen.write_pred_sch 的 DATES 循环逐行对应）
_MON = {"JAN": 1, "JUL": 7}
_BD = np.array([(dt.date(y, _MON[m], d) - T_START).days for (d, m, y) in FG.PRED_DATES], float)
_SL, _s = [], 0
for (_d, _m, _y) in FG.PRED_DATES:
    if _s < FG.N_STAGE - 1 and _y >= FG.STAGE_AT[_s + 1]:
        _s += 1
    _SL.append(_s)
_SL = np.array(_SL)
SI = np.array([_SL[min(int(np.searchsorted(_BD, x, side="left")), len(_BD) - 1)] for x in DAYS])


# ================================================================ 确定性：约束模型
def fit_cap() -> np.ndarray:
    """从已有模拟拟合每口注水井每个时间格点的注入能力上限 cap_k(t)。0 次 OPM。"""
    S = []
    for p in sorted(glob.glob(str(SIMDIR / "*.npz"))):
        d = np.load(p)
        if d["theta"].shape != (FG.N_STAGE, NW):
            continue
        S.append((d["theta"], d["inj_actual"].astype(float)))
    if len(S) < 10:
        raise SystemExit("🔴 已有模拟太少，无法拟合 cap")
    cap = np.zeros((NW, FG.FC_N))
    for k in range(NW):
        for t in range(FG.FC_N):
            tg = np.array([BASE[k] * 10 ** th[SI[t], k] for th, _ in S])
            me = np.array([ia[k, t] for _, ia in S])
            hi = tg > np.percentile(tg, 70)       # 目标率高 → 一定被 cap 卡住
            cap[k, t] = np.median(me[hi]) if hi.sum() > 3 else me.max()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(CAPF, cap)
    print(f"cap 拟合完成（{len(S)} 个已有模拟，0 次新 OPM）→ {CAPF}")
    for k, w in enumerate(INJ):
        print(f"  {w}: 基准目标率 {BASE[k]:7.1f}  cap 中位 {np.median(cap[k]):7.1f} sm3/d"
              f"  → 能力/目标 {np.median(cap[k])/BASE[k]:.3f}")
    return cap


def _cap() -> np.ndarray:
    if not CAPF.exists():
        return fit_cap()
    return np.load(CAPF)


def predict_water(th: np.ndarray, s: float, cap: np.ndarray) -> float:
    """解析预测实测总注水量。纯函数，无模拟。"""
    tgt = BASE[:, None] * 10.0 ** (th[SI, :].T + s)
    return float((np.minimum(tgt, cap) * TW).sum())


def solve_s(th: np.ndarray, target_W: float, cap: np.ndarray, bias: float = 1.0) -> float:
    """二分求标量 s，使 bias * predict_water(th, s) == target_W。确定性。"""
    lo, hi = -0.8, 0.8
    for _ in range(60):
        mid = .5 * (lo + hi)
        if bias * predict_water(th, mid, cap) < target_W:
            lo = mid
        else:
            hi = mid
    return .5 * (lo + hi)


# ================================================================ 确定性：跑模拟 + 指标
def metrics(key: str) -> dict:
    from fc_npv import BRENT, BBL_PER_M3
    d = np.load(SIMDIR / f"{key}.npz")
    ob = d["obs"]; n = len(NB.PRODUCERS); NT = FG.FC_N
    oil_w = np.stack([ob[(i * 3) * NT:(i * 3 + 1) * NT] for i in range(n)])   # (22, 40)
    oil = oil_w.sum(0)
    yrs = np.array([(T_START + dt.timedelta(days=float(x))).year for x in DAYS])
    price = np.array([BRENT[y] for y in yrs])
    t = (DAYS - DAYS[0]) / 365.25
    rev = oil * TW * BBL_PER_M3 * price
    ms = (DAYS - DAYS[0]) <= 3.0 * 365.25
    inj = d["inj_actual"].astype(float)
    return {"oil": float((oil * TW).sum()),
            "short": float((oil * TW * ms).sum()),
            "water": float((inj * TW).sum()),
            "npv0": float(rev.sum()),
            "npv8": float((rev / 1.08 ** t).sum()),
            "npv15": float((rev / 1.15 ** t).sum()),
            "oil_by_well": {w: float((oil_w[i] * TW).sum())
                            for i, w in enumerate(NB.PRODUCERS) if (oil_w[i] * TW).sum() > 1.0},
            "inj_by_well": {w: float((inj[k] * TW).sum()) for k, w in enumerate(INJ)}}


def run_sim(key: str, th_shifted: np.ndarray, threads: int) -> None:
    import fc_decide as FD

    class _A:
        pass
    _A.threads = threads
    FD._run_sim(key, th_shifted.astype(np.float32), _A)


# ================================================================ 确定性：校正 + 裁判
def evaluate(tag, th, W0, cap, budget_left, threads, tol, state, log, max_cal=2):
    """一套 θ → 解析定 s → OPM 真跑 → 若超差则确定性修正再跑。返回 (best, calls, trials)."""
    calls, pts = 0, []
    budget_left = min(budget_left, max_cal)      # 每轮校正上限：保证预算能覆盖多轮
    bias = state.get("bias", 1.0)
    s = solve_s(th, W0, cap, bias)
    while True:
        if calls >= budget_left:
            break
        import hashlib
        h = hashlib.md5(np.round(th + s, 6).tobytes()).hexdigest()[:8]   # θ+s 指纹，杜绝跨轮误复用
        key = f"loop_{tag}_c{calls}_{h}"
        if not (SIMDIR / f"{key}.npz").exists():
            t0 = time.time()
            run_sim(key, th + s, threads)
            log(f"    OPM 真跑 {key}  s=10^{s:+.5f}  ({time.time()-t0:.0f}s)")
        else:
            log(f"    复用已有 {key}  s=10^{s:+.5f}")
        calls += 1
        m = metrics(key)
        dev = m["water"] / W0 - 1
        pts.append({"key": key, "s": float(s), "water_dev": float(dev), **m})
        log(f"    实测注水偏离 {dev:+.4%}  NPV@8% {m['npv8']/1e6:,.1f}M$  (预算已用 {calls})")
        # ---- 确定性偏差更新：真跑实测 / 模型预测（乘性偏差，供下一步与下一轮起步）
        state["bias"] = bias = m["water"] / predict_water(th, s, cap)
        if abs(dev) <= tol:
            break
        if calls >= budget_left:
            break
        if len(pts) >= 2:            # 割线法（真跑数据）
            (s0, d0), (s1, d1) = (pts[-2]["s"], pts[-2]["water_dev"]), (pts[-1]["s"], pts[-1]["water_dev"])
            s = float(np.clip(s1 - d1 * (s1 - s0) / (d1 - d0 + 1e-12), -0.8, 0.8))
        else:                        # 用解析模型的斜率做牛顿步
            s = solve_s(th, W0, cap, state["bias"])
    best = min(pts, key=lambda p: abs(p["water_dev"]))
    return best, calls, pts


# ================================================================ LLM：只提方案
LOOP_NOTE = """
============================================================

【本次是闭环实验：你有多次机会，每次都会拿到真实模拟结果】

1. **反馈是真的**。你每给一套 θ，我就用 OPM Flow 全物理模拟器真跑一次（不是代理模型、
   不是估算），然后把累计产油、前 3 年产油、NPV、逐井产油量、逐井**实测**注水量原样贴给你。
   你据此修正，再给下一套。

2. **注水量硬约束由代码处理，不用你操心整体水平**。
   我会对你给的 24 个 θ **统一加一个标量 s**（等价于所有井所有时段的注水率同乘 10^s），
   把**实测**总注水量校到基准的 ±0.05%。
   → 因此：你只需要决定 θ 的**形状**——井之间怎么分、时段之间怎么分。
   → θ 里的公共常数项没有意义（会被 s 抵消掉）。加 0.1 或减 0.1 到全部 24 个数是同一个方案。

3. **预算有限**。总共只有 {budget} 次 OPM 调用（校正用掉的也算）。每轮会告诉你还剩几次。
   预算用完就停，所以早期的探索要有信息量，不要浪费。

4. **评分指标是 NPV@8%**（EIA 真实历史 Brent 年均价，2006-12 → 2020-01，只算收入侧，
   井数与总注水量在各方案间相同故成本项抵消）。这是最终评判。
   `objective` 字段仍然要填（"short" 或 "long"），但胜负按 NPV@8% 判。

5. 基准方案（θ 全零，维持当前注水率）的真实成绩，作为你的参照系：
{baseline}

【输出格式】严格的 JSON，不要有别的内容，不要 markdown 代码块围栏：
{{"objective": "short" 或 "long",
  "theta": [[6 行 × 4 列的数字]],
  "rationale": "你这一轮为什么这么改（中文，300 字以内；第 2 轮起必须明确写出你从上一轮
                真实结果里读到了什么、因此改了哪几个数）",
  "expected_tradeoff": "你预期的代价（中文，一句话）"}}
"""

HIST_HDR = """
============================================================

【历史：你之前提交的方案与 OPM Flow 的真实裁定】
"""


def fmt_round(i, rec, base):
    m = rec["result"]
    lines = [f"--- 第 {i} 轮 ---",
             f"你给的 θ（6 段 × 4 井 {INJ}）:",
             json.dumps([[round(float(x), 4) for x in r] for r in rec["theta"]]),
             f"代码施加的注水量校正标量 s = {m['s']:+.5f}"
             f"（实际执行的 θ = 你给的 θ + {m['s']:+.5f}，实测注水偏离基准 {m['water_dev']:+.4%}）",
             "OPM Flow 真实结果:",
             f"  13 年累计产油 {m['oil']:,.0f} sm3   (基准 {base['oil']:,.0f}，"
             f"{(m['oil']/base['oil']-1)*100:+.2f}%)",
             f"  前 3 年累计产油 {m['short']:,.0f} sm3  (基准 {base['short']:,.0f}，"
             f"{(m['short']/base['short']-1)*100:+.2f}%)",
             f"  NPV@8%  {m['npv8']/1e6:,.1f} M$   (基准 {base['npv8']/1e6:,.1f}，"
             f"{(m['npv8']-base['npv8'])/1e6:+.1f} M$)  ← 评分指标",
             f"  NPV@0%  {m['npv0']/1e6:,.1f} M$ (基准 {base['npv0']/1e6:,.1f})   "
             f"NPV@15% {m['npv15']/1e6:,.1f} M$ (基准 {base['npv15']/1e6:,.1f})",
             "  逐口注水井 13 年**实测**注水量 (sm3，括号内为基准):",
             "    " + "  ".join(f"{w} {m['inj_by_well'][w]:,.0f}({base['inj_by_well'][w]:,.0f})"
                                for w in INJ),
             "  逐口生产井 13 年累计产油 (sm3，括号内为基准；未列出的井全程为 0):"]
    for w in sorted(m["oil_by_well"], key=lambda x: -m["oil_by_well"][x]):
        b = base["oil_by_well"].get(w, 0.0)
        lines.append(f"    {w:8s} {m['oil_by_well'][w]:>11,.0f}  ({b:>11,.0f}, "
                     f"{(m['oil_by_well'][w]/b-1)*100:+6.2f}%)" if b > 0 else
                     f"    {w:8s} {m['oil_by_well'][w]:>11,.0f}")
    lines.append(f"  你当时的理由: {rec.get('rationale','')}")
    return "\n".join(lines)


def ask_llm(prompt: str, model: str, tmo: int) -> tuple[str, dict]:
    """无状态调用一个**独立的** LLM 实例。cwd=/tmp，不给工具，避免它看到本项目任何文件。"""
    pf = OUT / "_prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    cmd = ["claude", "-p", "--model", model, "--allowed-tools", ""]
    r = subprocess.run(cmd, stdin=open(pf, "rb"), cwd="/tmp", capture_output=True,
                       text=True, timeout=tmo)
    txt = r.stdout.strip()
    if r.returncode != 0:
        raise SystemExit(f"🔴 LLM 调用失败 rc={r.returncode}\n{r.stderr[:2000]}")
    s = txt
    if "```" in s:
        s = s.split("```")[1]
        s = s[s.find("{"):] if s.lstrip().startswith("json") else s
    s = s[s.find("{"): s.rfind("}") + 1]
    return txt, json.loads(s)


# ================================================================ 主循环
def run(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "loop.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    cap = _cap()
    if not (SIMDIR / "ms_base.npz").exists():
        raise SystemExit("🔴 缺基准模拟 ms_base.npz")
    base = metrics("ms_base")
    W0 = base["water"]
    log(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] 闭环盖亚 arm={a.arm} "
        f"预算={a.budget} 次 OPM  容差=±{a.tol:.2%}  模型={a.model}")
    log(f"  基准: 累计产油 {base['oil']:,.0f}  前3年 {base['short']:,.0f}  "
        f"NPV@8% {base['npv8']/1e6:,.1f}M$  实测注水 {W0:,.0f}")

    brief = (ROOT / "_pipelines" / "fc_agent" / f"brief_{a.arm}.txt").read_text(encoding="utf-8")
    base_txt = (f"   13 年累计产油 {base['oil']:,.0f} sm3\n"
                f"   前 3 年累计产油 {base['short']:,.0f} sm3\n"
                f"   NPV@8% {base['npv8']/1e6:,.1f} M$   NPV@0% {base['npv0']/1e6:,.1f} M$   "
                f"NPV@15% {base['npv15']/1e6:,.1f} M$\n"
                f"   13 年实测总注水量 {W0:,.0f} sm3")
    note = LOOP_NOTE.format(budget=a.budget, baseline=base_txt)

    hist, used, rnd = [], 0, 0
    state = {"bias": 1.0}
    while used < a.budget and rnd < a.max_rounds:
        rnd += 1
        p = brief + note
        if hist:
            p += HIST_HDR + "\n\n".join(fmt_round(i + 1, h, base) for i, h in enumerate(hist))
            p += (f"\n\n【现在是第 {rnd} 轮】剩余 OPM 预算 {a.budget-used} 次。"
                  f"看清上面的真实数字，给出你修正后的 θ。只输出 JSON。")
        else:
            p += f"\n\n【现在是第 1 轮】剩余 OPM 预算 {a.budget} 次。给出你的第一套 θ。只输出 JSON。"
        (OUT / f"prompt_{a.seed_tag}_r{rnd}.txt").write_text(p, encoding="utf-8")
        log(f"\n  [第 {rnd} 轮] 提示词 {len(p):,} 字符 → 调 LLM …")
        raw, ans = ask_llm(p, a.model, a.llm_timeout)
        (OUT / f"ans_{a.seed_tag}_r{rnd}.json").write_text(json.dumps(ans, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        th = np.asarray(ans["theta"], float).reshape(FG.N_STAGE, NW)
        th = th - th.mean()          # 去掉无意义的公共常数（s 会重新决定水平）
        log(f"    objective={ans.get('objective')}  θ 范围 [{th.min():+.3f},{th.max():+.3f}]")
        log(f"    理由: {str(ans.get('rationale',''))[:300]}")
        best, calls, trials = evaluate(f"{a.arm}{a.seed_tag}r{rnd}", th, W0, cap, a.budget - used,
                                       a.threads, a.tol, state, log, a.max_cal)
        used += calls
        hist.append({"round": rnd, "theta": th.tolist(), "objective": ans.get("objective"),
                     "rationale": ans.get("rationale"),
                     "expected_tradeoff": ans.get("expected_tradeoff"),
                     "opm_calls_this_round": calls, "opm_calls_cumulative": used,
                     "result": best, "all_trials": trials})
        log(f"  [第 {rnd} 轮 完] NPV@8% {best['npv8']/1e6:,.1f}M$ "
            f"({(best['npv8']-base['npv8'])/1e6:+.1f} vs 基准)  本轮 OPM {calls} 次  累计 {used} 次")
        _dump(OUT / f"loop_G2_{a.seed_tag}.json", a, base, hist, used)
        if abs(best["water_dev"]) > a.tol:
            log("  🔴 本轮未能把实测注水校进 ±0.05%，预算耗尽，结果不计入合格解")

    _dump(OUT / f"loop_G2_{a.seed_tag}.json", a, base, hist, used)
    valid = [h for h in hist if abs(h["result"]["water_dev"]) <= a.tol]
    log(f"\n{'='*70}\n  轮次   OPM(本轮/累计)   注水偏离     NPV@8%        vs基准")
    for h in hist:
        r = h["result"]
        log(f"   {h['round']:>2d}      {h['opm_calls_this_round']}/{h['opm_calls_cumulative']}"
            f"        {r['water_dev']:>+9.4%}  {r['npv8']/1e6:>9,.1f}M$  "
            f"{(r['npv8']-base['npv8'])/1e6:>+8.1f}M$"
            + ("" if abs(r["water_dev"]) <= a.tol else "   ← 不合格"))
    if valid:
        bh = max(valid, key=lambda h: h["result"]["npv8"])
        log(f"\n  最优合格解: 第 {bh['round']} 轮，NPV@8% {bh['result']['npv8']/1e6:,.1f}M$ "
            f"({(bh['result']['npv8']-base['npv8'])/1e6:+.1f}M$ vs 基准)，"
            f"达成所用 OPM 累计 {bh['opm_calls_cumulative']} 次")
        log(f"  对照 G0 单发: 1 轮 / 5 次 OPM → +177.6M$ ；数值优化多起点: 33 次 → +91.3M$")
    logf.close()
    return 0


def _dump(path, a, base, hist, used):
    valid = [h for h in hist if abs(h["result"]["water_dev"]) <= a.tol]
    bh = max(valid, key=lambda h: h["result"]["npv8"]) if valid else None
    j = {"seed_tag": a.seed_tag, "experiment": "闭环盖亚 (Gaia-in-the-loop)",
         "arm": a.arm, "model": a.model, "budget_opm": a.budget,
         "water_tol": a.tol, "threads": a.threads,
         "judge": "OPM Flow 全物理模拟（非代理模型、非 LLM 打分）",
         "roles": {"deterministic_code": ["注水量标量校正 solve_s", "OPM 调用", "指标计算",
                                          "预算与终止判断", "θ 去公共常数"],
                   "llm": ["提出 θ", "写推理"]},
         "baseline": {k: v for k, v in base.items() if k not in ("oil_by_well", "inj_by_well")},
         "references": {"G0_single_shot": {"opm_calls": 5, "npv8_delta_MUSD": 177.6},
                        "numerical_multistart": {"opm_calls": 33,
                                                 "npv8_delta_MUSD_short": 91.3,
                                                 "npv8_delta_MUSD_long": 42.9}},
         "rounds": hist, "opm_calls_total": used,
         "best": None if bh is None else {
             "round": bh["round"], "npv8": bh["result"]["npv8"],
             "npv8_delta_MUSD": (bh["result"]["npv8"] - base["npv8"]) / 1e6,
             "opm_calls_to_reach": bh["opm_calls_cumulative"],
             "sim_npz": str(SIMDIR / (bh["result"]["key"] + ".npz")),
             "water_dev": bh["result"]["water_dev"]},
         "beats_G0_single_shot": None if bh is None else
             bool((bh["result"]["npv8"] - base["npv8"]) / 1e6 > 177.6)}
    path.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")


def check(a) -> int:
    """交叉验证约束模型精度。0 次 OPM。"""
    cap = _cap()
    errs = []
    for p in sorted(glob.glob(str(SIMDIR / "*.npz"))):
        d = np.load(p)
        th = d["theta"]
        if th.shape != (FG.N_STAGE, NW):
            continue
        act = float((d["inj_actual"].astype(float) * TW).sum())
        errs.append((abs(predict_water(th, 0.0, cap) / act - 1), Path(p).name,
                     float(np.abs(th).max())))
    mod = [e for e in errs if e[2] <= 0.45]
    print(f"全部 {len(errs)} 个模拟: 中位误差 {100*np.median([e[0] for e in errs]):.3f}%")
    print(f"|θ|≤0.45 子集 {len(mod)} 个: 中位 {100*np.median([e[0] for e in mod]):.3f}%  "
          f"最大 {100*max(e[0] for e in mod):.3f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fitcap", "check", "run"])
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--seed-tag", default="s1", help="重复实验编号，只影响文件名")
    ap.add_argument("--budget", type=int, default=5, help="OPM Flow 调用总上限（含校正）")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tol", type=float, default=5e-4, help="实测注水量容差")
    ap.add_argument("--max-cal", type=int, default=2, help="每轮注水校正最多用几次 OPM")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--llm-timeout", type=int, default=900)
    a = ap.parse_args()
    if a.cmd == "fitcap":
        fit_cap(); return 0
    if a.cmd == "check":
        return check(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
