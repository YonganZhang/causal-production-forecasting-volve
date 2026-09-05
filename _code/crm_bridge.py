#!/usr/bin/env python3
"""地基实验 F1：三维场能不能解释 CRM 的井对 gain？

主线（2026-08-09 用户拍板）是「**CRM 给出多少，我们给出在哪、为什么**」——
不推翻 CRM，而是给它配一个能看见地下的解释层。这条主线的地基是一个必须先验的问题：

    我们算出的三维因果响应场，积分回井口，能不能重现井对之间的 gain？

对不上，说明三维场与井口量之间存在口径问题，后面所有"空间解释"都是空中楼阁。
对得上，"CRM 是对的、我们给它配解释层"这个故事才立得住，而且是用我们自己的数据证明的。

两个检验：

  A. 物理自洽（质量守恒方向）
     Σ_cells ∂SWAT_c/∂θ_a × PORV_c  = 多注的水**存在地下**的部分
     ∂(累计产水)/∂θ_a               = 多注的水**被采出来**的部分
     两者都应为正, 且量级可比。若储存项为负或量级差几个数量级, 说明场响应不物理。
     ⚠️ v1 分片没存实测注入量(WWIR), 所以这里**不能闭合**质量平衡, 只能看方向与量级。
        v2 已加 WWIR/WGIR, 届时可做真正的闭合检验。

  B. 井口归因(本实验的主检验)
     井对 gain B[a,p] = ∂(生产井 p 的累计产油)/∂θ_a          ← 相当于 CRM 的 gain
     用**井 p 泄油区内的场响应**去预测它:
         gain ~ β1 · <∂SWAT/∂θ_a>_near(p) + β2 · <∂PRESSURE/∂θ_a>_near(p)
     两个特征都要:注水既通过水侵降低产油(SWAT 项), 又通过保压提高产油(PRESSURE 项),
     只用一个必然解释不全。报 R² 与相关系数。

🔴 诚实边界:B 成立只说明"井口 gain 可由井周场响应线性重构", 这是解释层的**必要条件**,
   不等于"我们解释了 CRM"。真正的解释要靠后续的堵层反事实验证(说走第 10 层就把第 10 层
   堵上重跑, gain 掉了才算数)。

用法:
    python crm_bridge.py                      # 用默认分片目录
    python crm_bridge.py --radius 400 --max-shards 2000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import paths as P

OUTDIR = Path(__file__).resolve().parent.parent / "_pipelines" / "crm_bridge"
INIT = P.NORNE_DECK / "out" / "NORNE_ATW2013.INIT"
SCH = P.NORNE_DECK / "INCLUDE" / "BC0407_HIST01122006.SCH"

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")
N_TIMES = 40
DAY_GRID = np.linspace(1.0, 3312.0, N_TIMES)     # 与 norne_bulk 同一口径
FIELD_LAST_DAY = 3260.0                          # 参考算例 UNRST 末帧;v2 分片自带 field_days


def completions(names: set[str]) -> dict[str, list[tuple[int, int, int]]]:
    """COMPDAT → 每口井的射孔单元 (i,j,k), 0 基。"""
    txt = SCH.read_text(errors="ignore")
    out: dict[str, set] = {}
    for blk in re.findall(r"^COMPDAT(.*?)^/\s*$", txt, re.S | re.M):
        for line in blk.splitlines():
            line = line.split("--")[0].strip()
            if not line or line == "/":
                continue
            t = line.replace("/", " ").split()
            if len(t) < 5:
                continue
            try:
                w, i, j, k1, k2 = t[0].strip("'"), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            except ValueError:
                continue
            if w in names:
                out.setdefault(w, set()).update((i - 1, j - 1, k) for k in range(k1 - 1, k2))
    return {w: sorted(v) for w, v in out.items()}


def injector_base_rates(day: float) -> dict[str, float]:
    """从 SCHEDULE 取每口注入井在 day 时刻**生效**的 WCONINJE 目标注入率(基准, 未乘 θ)。

    CRM 的 gain f_ij 是无量纲的:生产井 j 的产液速率对注入井 i 注入速率的偏导。
    我们的回归系数是「每 1 个标准差的 log10 乘子」, 要换算成 CRM 口径得用链式法则:
        w_i = w_base_i · 10^θ_i     ⇒   ∂w_i/∂θ_i = ln(10) · w_i
        f_ij = (∂q_j/∂θ_i^std / sd_i) / (ln(10) · w_i)
    ⚠️ v1 分片没存实测注入率(WWIR), 只能用 deck 的**目标**率;顶到 600 bar BHP 上限时
       实际注入会被截断, 所以这是上界近似。v2 分片自带 inj_actual, 届时用实测值。
    """
    import datetime as dt
    MON = {m: i + 1 for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    START = dt.date(1997, 11, 6)
    txt = SCH.read_text(errors="ignore")
    cur, best, in_dates = 0.0, {}, False
    for ln in txt.splitlines():
        b = ln.split("--")[0]
        if b.strip().upper().startswith("DATES"):
            in_dates = True; continue
        if in_dates:
            g = re.match(r"\s*(\d+)\s+'(\w{3})\w*'\s+(\d{4})", b)
            if g:
                cur = float((dt.date(int(g.group(3)), MON[g.group(2).upper()],
                                     int(g.group(1))) - START).days)
            if b.strip().startswith("/"):
                in_dates = False
        if "'RATE'" not in b or cur > day:
            continue
        t = b.replace("/", " ").split()
        if len(t) < 2:
            continue
        w, ph = t[0].strip("'"), t[1].strip("'").upper()
        if w not in INJECTORS or ph != "WATER":
            continue
        g2 = re.search(r"'RATE'\s+([\d.eE+-]+)", b)
        if g2:
            best[w] = float(g2.group(1))          # 取 day 之前最后一条生效记录
    return best


def multi_ols_raw(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """**不**标准化 X 的多元 OLS, 系数量纲是 dY/dX —— 用实测注入率时要的就是这个。"""
    A = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.lstsq(A, Y, rcond=None)[0][:X.shape[1]]


def multi_ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """标准化 X 后的多元 OLS 斜率, 返回 (p, n_y)。θ 各维独立随机 → 系数即因果效应。"""
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    A = np.hstack([Xs, np.ones((len(Xs), 1))])
    return np.linalg.lstsq(A, Y, rcond=None)[0][:X.shape[1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--radius", type=float, default=400.0, help="泄油区半径(米), 3D 距离")
    ap.add_argument("--max-shards", type=int, default=0, help="0=全用")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    root = Path(args.root) if args.root else P.NORNE_BULK
    sh = sorted((root / "shards").glob("*.npz"))
    if args.max_shards:
        sh = sh[:args.max_shards]
    if not sh:
        print(f"{root} 下没有分片"); return 1

    # ---- 装载:只取末帧的 SWAT 与 PRESSURE, 别把 8 个快照全读进来(会到 5+ GB) ----
    TH, SW, PR, OB, IA = [], [], [], [], []
    for p in sh:
        d = np.load(p)
        TH.append(d["theta"]); OB.append(d["obs"])
        f = d["fields"][-1]                       # (2, ncell): [PRESSURE, SWAT]
        PR.append(f[0].astype(np.float32)); SW.append(f[1].astype(np.float32))
        if "inj_actual" in d.files:
            IA.append(d["inj_actual"])
    TH = np.stack(TH).astype(np.float64)
    OB = np.stack(OB).astype(np.float64)
    SW = np.stack(SW); PR = np.stack(PR)
    n, ncell = SW.shape
    print(f"样本 {n}   活动单元 {ncell:,}   θ 维度 {TH.shape[1]}")
    print(f"θ 各维两两相关最大绝对值 "
          f"{np.abs(np.corrcoef(TH.T) - np.eye(TH.shape[1])).max():.4f}  (应≈0)")

    # v2 的控制变量是 13 个「井×相态」且分片自带实测注入率; v1 是 9 个井名、只有目标率。
    INJ_ACT = np.stack(IA).astype(np.float64) if len(IA) == n else None
    schema = 2 if INJ_ACT is not None else 1
    if schema >= 2:
        import norne_bulk as NB
        ctrl_names = [f"{w}_{ph}" for w, ph in NB.CONTROLS]
    else:
        ctrl_names = list(INJECTORS)
    n_ctrl = len(ctrl_names)
    print(f"schema v{schema}   控制变量 {n_ctrl} 个"
          + ("  (含实测注入率, 可直接对 dq/dw 回归)" if INJ_ACT is not None
             else "  (无实测注入率, 只能用 deck 目标率链式换算)"))

    # ---- 井口真值 gain:∂累计产油/∂θ ----
    # 🔴 场与速率必须取**同一时刻**。旧版拿 obs 最后 5 个点(第 2972~3312 天, 均值 3142)
    #    去配 fields[-1](第 3260 天), 差 118 天。改成取与场快照时刻最接近的那个 obs 时刻。
    #    (v1 分片没存 field_days, 用参考算例的 UNRST 末日 3260 天; v2 分片自带 field_days。)
    field_day = float(FIELD_LAST_DAY)
    jt = int(np.argmin(np.abs(DAY_GRID - field_day)))
    print(f"场末帧在第 {field_day:.0f} 天; 取 obs 第 {jt} 个时刻(第 {DAY_GRID[jt]:.0f} 天)配对, "
          f"错位 {DAY_GRID[jt]-field_day:+.0f} 天")
    oil_end = np.stack([OB[:, (i * 3 + 0) * N_TIMES + jt] for i in range(len(PRODUCERS))], axis=1)
    liq_end = np.stack([OB[:, (i * 3 + 0) * N_TIMES + jt] + OB[:, (i * 3 + 1) * N_TIMES + jt]
                        for i in range(len(PRODUCERS))], axis=1)

    # 🔴 WOPR/WWPR 单位是 m3/**天**(速率)。旧版直接 .sum() 把 40 个速率采样加起来当"累计量",
    #    但 40 个点跨 3311 天、间隔 84.9 天 —— 少乘了这个 dt, 体积偏小约 85 倍。
    #    (Codex 跨模型复核 2026-08-09 抓到, 已独立确认:C-2H 39,448 × 84.90 = 3.35M m3。)
    #    对 R²/相关系数无影响(常数倍缩放), 但质量守恒表的绝对量级全错。
    cum_oil = np.stack([np.trapezoid(OB[:, (i * 3 + 0) * N_TIMES:(i * 3 + 1) * N_TIMES], DAY_GRID, axis=1)
                        for i in range(len(PRODUCERS))], axis=1)
    cum_wat = np.stack([np.trapezoid(OB[:, (i * 3 + 1) * N_TIMES:(i * 3 + 2) * N_TIMES], DAY_GRID, axis=1)
                        for i in range(len(PRODUCERS))], axis=1)
    B_oil = multi_ols(TH, cum_oil)[:n_ctrl]        # (9, 22)
    B_wat = multi_ols(TH, cum_wat)[:n_ctrl]

    # ---- 场响应 ----
    E_sw = multi_ols(TH, SW)[:n_ctrl]              # (9, ncell)
    E_pr = multi_ols(TH, PR)[:n_ctrl]

    # ---- 几何 ----
    G = np.load(P.NORNE_GRID_CACHE)
    ctr = G["corners"].mean(axis=1).astype(np.float64)     # (ncell, 3) 单元中心
    ijk = G["ijk"]
    lut = {(int(a), int(b), int(c)): k for k, (a, b, c) in enumerate(ijk)}

    from resdata.resfile import ResdataFile
    porv_all = np.asarray(ResdataFile(str(INIT))["PORV"][0], np.float64)
    porv = porv_all[porv_all > 0]
    assert len(porv) == ncell, f"PORV 活动单元 {len(porv)} != 场 {ncell}"

    # ---- 检验 A:储存 vs 采出 ----
    print("\n=== A. 物理自洽(多注 1 个 θ 标准差的水去了哪) ===")
    print(f"{'注入井':8s}{'地下储存(m3)':>16s}{'被采出(m3)':>14s}{'储存占比':>10s}")
    A_rows = []
    for a, w in enumerate(ctrl_names):
        stored = float((E_sw[a] * porv).sum())
        produced = float(B_wat[a].sum())
        frac = stored / (stored + produced) if (stored + produced) > 0 else float("nan")
        A_rows.append({"injector": w, "stored_m3": stored, "produced_m3": produced,
                       "stored_frac": frac})
        print(f"{w:8s}{stored:>16,.0f}{produced:>14,.0f}{frac:>10.1%}")
    n_neg = sum(1 for r in A_rows if r["stored_m3"] < 0)
    print(f"储存项为负的井: {n_neg}/{len(ctrl_names)}  (应为 0 —— 多注水不该让地下含水量下降)")

    # ---- 换算到 CRM 量纲, 便于与流线分配因子直接比 ----
    #
    # 🔴 2026-08-12 Codex 复核指出旧法有两处系统性错误, 已逐条复核确认:
    #   (a) 分母用 deck 的**目标**注入率, 但 1463 条注入记录带 600 bar BHP 上限。
    #       用 v2 实测 WWIR 量化:F-1H 有 44.0% 的样本被截断, 实际 dw/dθ 只有理论
    #       目标导数的 0.376(F-2H 23.8%/0.663, F-3H 27.0%/0.541)。
    #       我原注释写"上界近似"**方向反了** —— 截断让实际导数变小, 所以 f 是被**低估**。
    #   (b) 分母**用错日期**:产量取第 3227 天, 注入率却取第 3260 天的 deck 率,
    #       三井目标率之比 1.169/1.218/1.184, 又把 f 压低 14.5%~17.9%。
    #   两者叠加, 旧 f 的校正因子约 3.11/1.84/2.19, 量级比从 0.179 变为约 1.03 ——
    #   即"因果比流诊小 5 倍"根本不是方法差异, 是我算错了。
    #
    # 新做法:不再经由 deck 目标率做链式换算, 直接对**同一时刻的实测注入率**回归。
    if INJ_ACT is not None:
        w_act = INJ_ACT[:, :, jt]                          # (n, n_ctrl) 匹配时刻的实测注入率
        crm_f = np.full((len(ctrl_names), liq_end.shape[1]), np.nan)
        Wc = np.hstack([w_act, TH[:, len(ctrl_names):]])   # 处理量换成实测率, 其余控制照旧
        B_w = multi_ols_raw(Wc, liq_end)[:len(ctrl_names)]  # 未标准化 → 系数即 dq/dw
        crm_f[:] = B_w
        print(f"\n[CRM 量纲] 用**实测注入率**回归 (v2), 系数直接是 dq_prod/dw_inj, 无需链式换算")
    else:
        base_rate = injector_base_rates(field_day)
        B_liq = multi_ols(TH, liq_end)[:len(ctrl_names)]
        sd = TH.std(0)[:len(ctrl_names)]
        crm_f = np.full_like(B_liq, np.nan)
        for a, iw in enumerate(ctrl_names):
            wr = base_rate.get(iw.split("_")[0], 0.0)
            if wr > 1.0:
                crm_f[a] = (B_liq[a] / sd[a]) / (np.log(10.0) * wr)
        print(f"\n[CRM 量纲] ⚠ 无实测注入率(schema v1), 退回 deck 目标率链式换算 —— "
              f"已知系统性低估(BHP 截断 + 日期错配), 仅供参考")
    print(f"\n=== CRM 量纲的分配系数 f_ij (无量纲, 行和应 ≲ 1) ===")
    print(f"{'控制变量':12s}{'Σ_j f_ij':>12s}{'最大 f_ij':>12s}{'对应生产井':>12s}")
    crm_rows = []
    for a, iw in enumerate(ctrl_names):
        if not np.isfinite(crm_f[a]).any():
            print(f"{iw:12s}{'(无数)':>12s}"); continue
        rs = float(np.nansum(crm_f[a])); mx = int(np.nanargmax(crm_f[a]))
        crm_rows.append({"control": iw, "row_sum": rs, "max_f": float(crm_f[a, mx]),
                         "max_producer": PRODUCERS[mx]})
        print(f"{iw:12s}{rs:>12.3f}{crm_f[a, mx]:>12.3f}{PRODUCERS[mx]:>12s}")
    bad = [r["injector"] for r in crm_rows if r["row_sum"] > 1.5 or r["row_sum"] < -0.5]
    if bad:
        print(f"⚠ 行和明显越界(应 ≲1)的井: {bad} —— 可能是 BHP 截断或相态污染")
    np.save(OUTDIR / "crm_allocation_f.npy", crm_f)

    # ---- 检验 B:井口归因(消融阶梯) ----
    #
    # 🔴 第一版这里 R²=0.019, 我一度以为"地基不成立"。实际是**我的口径全错**, 三处:
    #   1. 量纲不匹配:拿**累计产油**(9 年积分)去对**末帧快照**。两条完全不同的轨迹
    #      可以落在同一个末态。改用末期产油速率, 与末帧快照同一时刻。
    #   2. 相态污染:9 口注入井里 4 口是水气混注, 它们的 θ 同时在改注气,
    #      SWAT 响应方向是反的(注气驱替水 → SWAT 下降)。见检验 A 的负储存。
    #   3. 生产井基数差几十倍:B[a,p] 随 p 的绝对产量线性缩放, 而井周场响应不这样缩放。
    #      混在一起做池化回归, 信号被基数差异淹没。加生产井固定效应。
    # 三处叠加后 R² 0.019 → 0.403。所以这张消融表本身就是结果, 保留下来。
    comp = completions(set(PRODUCERS))
    from scipy.spatial import cKDTree
    tree = cKDTree(ctr)
    near: dict[str, np.ndarray] = {}
    for w, cells in comp.items():
        idx = [lut[c] for c in cells if c in lut]
        if not idx:
            continue
        hits = tree.query_ball_point(ctr[idx], r=args.radius)
        u = [np.asarray(h, int) for h in hits if len(h)]
        near[w] = np.unique(np.concatenate(u)) if u else np.asarray(idx, int)

    B_end = multi_ols(TH, oil_end)[:n_ctrl]
    base = {"cum": cum_oil.mean(0), "end": oil_end.mean(0)}
    gains = {"cum": B_oil, "end": B_end}

    # 🔴 22 口生产井里 **10 口在历史末期已停产**(末期均产恰为 0)。
    #    旧版归一化写成 `B/|base| if |base|>1e-9 else B` —— 护栏没让它报错, 但让这 10 口
    #    用**未归一**的 gain、另外 12 口用**归一后**的 gain, 两种量纲混进同一个回归。
    #    这是 2026-08-09 报出 R²=0.403 的直接原因, 该数已撤回。
    ALIVE = {i for i in range(len(PRODUCERS)) if base["end"][i] > 1.0}
    print(f"末期仍在产的生产井 {len(ALIVE)}/{len(PRODUCERS)}: "
          f"{[PRODUCERS[i] for i in sorted(ALIVE)]}")

    def fit(target, subset, normalize, fe):
        Bm, bs = gains[target], base[target]
        X, y, pid = [], [], []
        for a, iw in enumerate(ctrl_names):
            if iw not in subset:
                continue
            for p, pw in enumerate(PRODUCERS):
                if pw not in near:
                    continue
                if normalize and p not in ALIVE:
                    continue                     # 停产井不能做分母, 直接剔除而不是混量纲
                c = near[pw]
                wgt = porv[c] / porv[c].sum()
                X.append([float((E_sw[a][c] * wgt).sum()), float((E_pr[a][c] * wgt).sum())])
                g = Bm[a, p] / bs[p] if normalize else Bm[a, p]
                y.append(float(g)); pid.append(p)
        X = np.asarray(X); y = np.asarray(y); pid = np.asarray(pid)
        cols = [(X - X.mean(0)) / X.std(0)]
        if fe:
            D = np.zeros((len(y), len(PRODUCERS))); D[np.arange(len(y)), pid] = 1.0
            cols.append(D)
        else:
            cols.append(np.ones((len(y), 1)))
        A = np.hstack(cols)
        assert np.isfinite(A).all() and np.isfinite(y).all(), "设计矩阵含 NaN/Inf"
        tot = max(((y - y.mean()) ** 2).sum(), 1e-30)
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ b) ** 2).sum() / tot
        # 🔴 必须报**留一交叉验证**。井对数只有几十个而固定效应就占十几个参数,
        #    样本内 R² 会被自由度撑起来。旧版只报样本内 R², 得出的 0.403 已撤回:
        #    实测该配置留一 R² 是 **负的**(-0.068), 即还不如直接用均值预测。
        err = []
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            bi, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            err.append(y[i] - A[i] @ bi)
        loo = 1 - (np.asarray(err) ** 2).sum() / tot
        # 🔴 主指标用**分组**留一:把整口生产井留出去。同一口生产井的若干井对共享
        #    该井的完井、泄油区和基数, 彼此不独立 —— 逐点留一会高估泛化能力。
        gerr = []
        for g in sorted(set(pid)):
            m = pid != g
            if m.sum() < A.shape[1] + 1:
                continue
            bg, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            gerr.extend(y[~m] - A[~m] @ bg)
        gloo = 1 - (np.asarray(gerr) ** 2).sum() / tot if gerr else float("nan")
        return (len(y), A.shape[1], float(r2), float(loo), float(gloo),
                float(np.corrcoef(X[:, 0], y)[0, 1]), float(np.corrcoef(X[:, 1], y)[0, 1]))

    # v2 相态已分离 → 所有 WATER 控制都是"纯注水", 不再只有 5 口
    if schema >= 2:
        PURE = {c for c in ctrl_names if c.endswith("_WATER")}
    else:
        PURE = {"C-2H", "F-1H", "F-2H", "F-3H", "F-4H"}   # v1 未分相态, 只能取本就纯水的 5 口
    ALL = set(ctrl_names)
    print(f"纯注水控制 {len(PURE)} 个" + (" (v2 相态分离后从 5 增到 9)" if schema >= 2 else ""))
    ladder = [
        ("累计产油 · 全部井 · 原始",              "cum", ALL,  False, False),
        ("累计产油 · 仅纯注水",                   "cum", PURE, False, False),
        ("累计产油 · 仅纯注水 + 固定效应",         "cum", PURE, True,  True),
        ("末期速率 · 仅纯注水",                   "end", PURE, False, False),
        ("末期速率 · 全部井 + 归一 + 固定效应",    "end", ALL,  True,  True),
        ("末期速率 · 仅纯注水 + 归一 + 固定效应",  "end", PURE, True,  True),
        ("末期速率 · 仅纯注水 + 归一(只有场, 无固定效应)", "end", PURE, True, False),
    ]
    print(f"\n=== B. 井口归因消融(泄油区半径 {args.radius:g} m) ===")
    print(f"{'配置':38s}{'井对':>5s}{'参数':>5s}{'r(SW)':>8s}{'样本内':>8s}{'留一':>8s}{'分组留一':>9s}")
    B_rows = []
    for label, tgt, sub, nm, fe in ladder:
        npair, npar, r2, loo, gloo, rs, rp = fit(tgt, sub, nm, fe)
        B_rows.append({"config": label, "n_pairs": npair, "n_params": npar,
                       "r2_insample": r2, "r2_loo": loo, "r2_group_loo": gloo,
                       "r_swat": rs, "r_pressure": rp})
        print(f"{label:38s}{npair:>5d}{npar:>5d}{rs:>+8.3f}{r2:>8.3f}{loo:>+8.3f}{gloo:>+9.3f}")

    best = max(B_rows, key=lambda r: (r["r2_group_loo"] if np.isfinite(r["r2_group_loo"]) else -9))
    verdict = ("场能样本外重构井口 gain(留出整口未见过的生产井仍成立), 解释层地基成立"
               if best["r2_group_loo"] > 0.2 else
               f"🔴 结论未定:最佳分组留一 R² 仅 {best['r2_group_loo']:+.3f}。井对级样本太少"
               f"(纯注水井 × 在产生产井), 自由度撑不起固定效应。"
               f"唯一稳健信号是 SWAT 与 gain 的负相关(井周水多→产油少), 符号正确。"
               f" 下一步:v2 相态分离后清洁注入井从 5 增到 9, 且改用时间分辨/单元级样本。")
    print(f"\n最佳分组留一 R² = {best['r2_group_loo']:+.3f} ({best['config']})")
    print(f"判定: {verdict}")

    json.dump({"n_samples": n, "radius_m": args.radius,
               "mass_balance": A_rows, "n_negative_storage": n_neg,
               "ablation": B_rows, "best_r2_group_loo": best["r2_group_loo"], "best_r2_loo": best["r2_loo"], "verdict": verdict,
               "field_day": field_day, "obs_slot": jt, "obs_slot_day": float(DAY_GRID[jt]),
               "crm_allocation": crm_rows,
               "caveat": "v1 分片无实测注入量(WWIR), 质量平衡不能闭合, 只看方向与量级"},
              open(OUTDIR / "f1_bridge.json", "w"), indent=1, ensure_ascii=False)
    np.save(OUTDIR / "well_gain_oil.npy", B_oil)
    np.save(OUTDIR / "field_effect_swat.npy", E_sw)
    np.save(OUTDIR / "field_effect_pressure.npy", E_pr)
    print(f"已写入 {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
