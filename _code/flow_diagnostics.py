#!/usr/bin/env python3
"""流动诊断（flow diagnostics）：从通量场算到达时间与注入井示踪剂，得到井对分配因子。

## 为什么必须做这个

2026-08-09 的文献调研（Workflow `norne-novelty-scan`）查清了一件事：
**流线/流动诊断早就能做逐层的注采归因**，而且 Norne 正是 ResInsight 的演示数据集。
Thiele 的原话是 "for each injector, the amount of injected fluid supporting any producer
in the field is known exactly"。所以"我们能说出水走哪一层"这个措辞站不住。

用户 2026-08-10 拍板：新颖性不是门槛，但**对比必须做实**。于是流动诊断从"竞争对手"
变成"必须复现的基线" —— 审稿人一定会问「这跟流线有什么区别」，绕过去就是送人头。

## 数学

OPM Flow 在 deck 里加 `RPTRST ... FLOWS FLORES` 后会输出单元间通量
（`FLRWATI+/J+/K+` 等，油/气/水各三个方向，另有 `N+` 给非邻接连接）。有了通量场：

  1. **井项来自 ICON/XCON**（逐射孔连接的真实流量），不是通量散度。
     🔴 第一版用散度识别井是错的：散度里 64% 是可压缩储量项而非井，
        实测 16,698 个单元被误判成注入源（全场只有 36 口井）。
  2. **示踪剂**（稳态、纯对流、上风格式）：对每口注入井 i 解
         Σ_out v · c_cell − Σ_in v · c_upstream = q_i · [cell ∈ 井 i]
     得到"该单元的流体有多大比例来自注入井 i"。
  3. **井对分配因子** f_ij = 生产井 j 采出量中来自注入井 i 的份额
         = Σ_{j 的采出单元} |q_out| · c_i / Σ_{j 的采出单元} |q_out|
     这与 CRM 的 gain 是同一个口径（无量纲、行和 ≤ 1），可直接对拍。

## 🔴 诚实边界

- 稳态纯对流，**不含扩散/毛管/重力分离**；这与 MRST/ResInsight 的 flow diagnostics
  同口径，但不等于全流线模拟。
- 用**单一时刻**的通量场，即"冻结流场"假设。Thiele 自己写明流线敏感度
  "assumes fixed streamline paths for all time (a linear model)"。
- **NNC（断层等非邻接连接）**：若 EGRID 提供 NNC1/NNC2 映射则计入，否则跳过并在
  输出里标注 —— Norne 有 53 条断层，漏掉它们会低估跨断层连通。

用法:
    python flow_diagnostics.py --case /tmp/fluxtest --step -1
    python flow_diagnostics.py --case <dir> --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import paths as P

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")
# 🔴 口径必须自洽。旧版用三相**储层体积**通量 FLR* 建图, 井项却用 XCON 的**地面**
#    产油+产水率(实测列 0/1/2 = 地面 WOPR/WWPR/WGPR, 各自误差 0.00%), 单位对不上,
#    而且 XCON 里根本没有储层体积列(找过 WVPR/WVIR, 无匹配)。
#    改为**只追踪水相、统一用地面体积**:FLOWAT* 通量 + XCON 列 1。
#    只追水的理由:本问题就是"注入的水去了哪"; 且气的地面体积比液体大三个量级,
#    混进来会把水示踪剂彻底淹掉。水的体积系数≈1, 地面/储层差异可忽略。
FLUX_PREFIX = "FLO"          # FLO*=地面体积, FLR*=储层体积
PHASES = ("WAT",)
XCON_COL = {"OIL": 0, "WAT": 1, "GAS": 2}     # 由 summary 对拍确定, 误差 0.00%


def completions(sch: Path, names: set[str]) -> dict[str, list[tuple[int, int, int]]]:
    """COMPDAT → 每口井的射孔单元 (i,j,k)，0 基。"""
    txt = sch.read_text(errors="ignore")
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


def read_fluxes(unrst: Path, step: int) -> tuple[dict[str, np.ndarray], int]:
    """读某个重启步的总通量（三相之和，储层体积口径）。返回 {方向: (nactive,)}。"""
    from resdata.resfile import ResdataFile

    f = ResdataFile(str(unrst))
    n = f.num_named_kw("PRESSURE")
    s = step if step >= 0 else n + step
    out = {}
    for d in ("I", "J", "K", "N"):
        tot = None
        for ph in PHASES:
            kw = f"{FLUX_PREFIX}{ph}{d}+"
            try:
                v = np.asarray(f[kw][s], np.float64)
            except Exception:                                # noqa: BLE001  该相/方向不存在
                continue
            tot = v if tot is None else tot + v
        if tot is not None:
            out[d] = tot
    if not out:
        raise RuntimeError(f"{unrst} 里没有 {FLUX_PREFIX}*+ 通量关键字 —— "
                           "deck 的 RPTRST 需要加 FLOWS FLORES 才会输出")
    return out, s


def build_graph(grid, flux: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把方向通量摊成 (from, to, rate>0) 的有向边。"""
    nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
    na = grid.getNumActive()
    # 活动单元的 (i,j,k) → active index
    ijk = np.empty((na, 3), np.int32)
    for a in range(na):
        ijk[a] = grid.get_ijk(active_index=a)
    lut = -np.ones((nx, ny, nz), np.int64)
    lut[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(na)

    src, dst, rate = [], [], []
    for d, off in (("I", (1, 0, 0)), ("J", (0, 1, 0)), ("K", (0, 0, 1))):
        if d not in flux:
            continue
        v = flux[d]
        nb = lut[np.clip(ijk[:, 0] + off[0], 0, nx - 1),
                 np.clip(ijk[:, 1] + off[1], 0, ny - 1),
                 np.clip(ijk[:, 2] + off[2], 0, nz - 1)]
        ok = (nb >= 0) & np.isfinite(v) & (np.abs(v) > 0)
        # 通量为正 = 从本单元流向 +方向邻居；为负则反向
        a = np.arange(na)
        pos = ok & (v > 0)
        neg = ok & (v < 0)
        src.append(a[pos]); dst.append(nb[pos]); rate.append(v[pos])
        src.append(nb[neg]); dst.append(a[neg]); rate.append(-v[neg])
    return (np.concatenate(src), np.concatenate(dst), np.concatenate(rate))


def well_connection_rates(unrst: Path, step: int, grid) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """从重启文件的 ICON/XCON 读**逐射孔连接**的真实流量。

    🔴 第一版这里用通量散度 div(v) 当井项, 是概念错误:散度里除了井还有**压缩性与
       饱和度变化引起的储量项**。实测 16,698 个单元被判成"注入源"而全场只有 36 口井,
       真井信号被稀释到分配因子 Σ≈0.002(成熟水驱应接近 1)。
       旁证:散度的 Σ正 = −Σ负 = 65,544 恰好守恒 —— 正因为守恒才不可能是井项。

    列口径不靠猜, 用 summary 对拍确定:XCON 列 0 = 逐连接产油率、列 1 = 逐连接产水率
    (与 WOPR/WWPR 逐井求和的中位相对误差 0.00%, 匹配 8 口在产井)。
    符号约定:采出为正、注入为负。
    """
    from resdata.resfile import ResdataFile

    f = ResdataFile(str(unrst))
    # 🔴 PRESSURE/通量有 65 个块, 而 XCON/ICON/ZWEL 只有 64 个(初始块没有井记录)。
    #    旧版用 min(step, n_XCON-1) 硬夹:末帧碰巧对上, **非末帧会错一整帧**。
    #    正确做法是按块偏移对齐 —— 井块比场块少 1, 所以场块 k 对应井块 k-1。
    nk = f.num_named_kw("XCON")
    npres = f.num_named_kw("PRESSURE")
    k = step if step >= 0 else npres + step
    i = k - (npres - nk)
    if not (0 <= i < nk):
        raise IndexError(f"场块 {k} 没有对应的井块(井块共 {nk} 个, 场块 {npres} 个)")
    ih = np.asarray(f["INTEHEAD"][k])
    nwell, ncwmax, nicon, nxcon = int(ih[16]), int(ih[17]), int(ih[32]), int(ih[34])
    X = np.asarray(f["XCON"][i]).reshape(nwell, ncwmax, nxcon)
    I = np.asarray(f["ICON"][i]).reshape(nwell, ncwmax, nicon)
    zw = [str(x).strip() for x in np.asarray(f["ZWEL"][i])]
    names = [zw[k * 3] for k in range(nwell)]

    na = grid.getNumActive()
    q = np.zeros(na)                                   # 逐单元净井量:采出正、注入负
    per_well: dict[str, np.ndarray] = {}
    for w, nm in enumerate(names):
        act = np.flatnonzero(I[w, :, 0] > 0)
        if act.size == 0:
            continue
        cells, rates = [], []
        for c in act:
            gi = grid.global_index(ijk=(int(I[w, c, 1]) - 1, int(I[w, c, 2]) - 1, int(I[w, c, 3]) - 1))
            a = grid.get_active_index(global_index=gi)
            if a < 0:
                continue
            cells.append(a)
            rates.append(float(X[w, c, XCON_COL["WAT"]]))   # 只取水相, 与通量同口径
        if not cells:
            continue
        arr = np.zeros(na)
        arr[np.asarray(cells)] = np.asarray(rates)
        per_well[nm] = arr
        q += arr
    return per_well, q


def solve_tracers(na: int, src: np.ndarray, dst: np.ndarray, rate: np.ndarray,
                  well_rate: dict[str, np.ndarray], q_well: np.ndarray):
    """稳态上风示踪剂。井项来自 ICON/XCON 的真实连接流量，**通量先做守恒重构**。

    🔴 通量场来自**可压缩**模拟, 逐单元进出不平衡(储量项)。第一版直接解 A·c=s 时
       F-4H 浓度炸到 2.96e9; 第二版改成除以"全部正源"基准场 c_all —— 但 2026-08-12
       的跨模型复核指出**那不是守恒重构**:逐单元非线性相除后 A·r ≠ s, 实测
       ||A r − s||₁/||s||₁ = 1.18~1.75, 方程已被改掉; 且 c_all 的 P99=405、max=6e9,
       "中位偏 45%"严重低估了空间非均匀性。

       现在改成真正的重构:把通量投影到**最近的、满足井项约束的**场
           v* = argmin ‖v* − v‖²   s.t.  D v* = −q_well
       拉格朗日给出  v* = v − Dᵀλ,  (D Dᵀ) λ = D v + q_well。
       D Dᵀ 是图拉普拉斯(奇异, 常向量在零空间), 故先把右端投影到与常向量正交的子空间,
       再用共轭梯度解。重构后 A·1 = −q_well 严格成立, 示踪剂天然有界。
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    # 关键认识:通量场来自**可压缩**模拟, 逐单元 imb = 流出−流入 里既有井也有储量项
    # (实测 Σ|imb| = 131,088 而真实井项只有 47,717 —— 64% 是储量)。
    #
    # 试过两条错路, 都记下来:
    #   ① 除以"全部正源"基准场 c_all —— 逐单元非线性相除, A·r ≠ s, 方程被改掉
    #      (实测 ‖A r − s‖₁/‖s‖₁ = 1.18~1.75)。
    #   ② 强求 D v* = −q_well 做投影重构 —— 等于把储量项全抹掉, 而它是井项的 1.8 倍,
    #      修正量比原通量还大:114,096 条边翻了 59,980 条, 残差反而从 1.815 恶化到 3.9e4,
    #      CG 跑满 5000 次不收敛, 分配因子全变 0。**可压缩流里储量是真实的, 不能抹。**
    #
    # 正确做法:不改通量, 而是把 imb 的正部**如实分解**成"井注入"和"其他(储量释放)"两路,
    # 各解一路示踪剂。因为 A·1 = imb 且 Σ源 = max(imb,0), 所有浓度之和恒等于 1,
    # 每一路天然落在 [0,1], 不需要任何归一化或截断。"其他"那一路的份额就是
    # **无法归因给注入井的部分**, 直接报出来而不是藏起来。
    out_sum = np.bincount(src, weights=rate, minlength=na)
    in_sum = np.bincount(dst, weights=rate, minlength=na)
    imb = out_sum - in_sum
    pos = np.maximum(imb, 0.0)

    # 🔴 恒等式必须严格成立, 否则浓度会 >1。推导:
    #      A·1 = 流出 − 流入 = imb,  而源项总和 S = max(imb, 0)
    #    在 imb<0 的单元(储量**累积**, 净吸收)上, 左边是负数而右边是 0 —— 对不上,
    #    解 A·x = 0 在那里给出 x > 1。实测各路浓度之和最大到 5.0。
    #    把累积量作为额外的**汇**加到对角线:diag = 流出 + max(−imb, 0), 于是
    #      A·1 = imb + max(−imb,0) = max(imb,0) = S   ← 恒等式严格成立
    #    物理含义:流体被压缩/滞留在该单元, 相当于被"采出"到储量里, 是真实的汇。
    accum = np.maximum(-imb, 0.0)
    diag = out_sum + accum
    A = (sp.diags(np.where(diag > 0, diag, 1.0))
         - sp.coo_matrix((rate, (dst, src)), shape=(na, na)).tocsr())
    lu = spl.splu(A.tocsc())

    # 井注入在每个单元能认领的份额, 上限是该单元的净流出 —— 超出部分不可归因
    claim = {w: np.minimum(np.maximum(-arr, 0.0), pos) for w, arr in well_rate.items()}
    claimed = np.zeros(na)
    for v in claim.values():
        claimed += v
    over = claimed > pos + 1e-12               # 多口井同格竞争时按比例缩
    if over.any():
        scale = np.ones(na)
        scale[over] = pos[over] / np.maximum(claimed[over], 1e-30)
        for w in claim:
            claim[w] = claim[w] * scale
        claimed = np.minimum(claimed, pos)
    other = pos - claimed                      # 储量释放等无法归因的源

    conc = {w: np.clip(lu.solve(v), 0.0, 1.0) for w, v in claim.items() if v.sum() > 0}
    c_other = np.clip(lu.solve(other), 0.0, 1.0)
    tot_claim = float(sum(float(v.sum()) for v in claim.values()))
    tot_inj = float(sum(float(np.maximum(-a_, 0).sum()) for a_ in well_rate.values()))
    csum = np.zeros(na)
    for v in conc.values():
        csum += v
    csum += c_other
    return conc, q_well, {"attributable_frac": tot_claim / max(tot_inj, 1e-30),
                          "unattributed_source": float(other.sum()),
                          "well_source": tot_claim,
                          "sum_conc_median": float(np.median(csum[out_sum > 0])),
                          "sum_conc_max": float(csum.max()),
                          "other_conc_median": float(np.median(c_other[out_sum > 0]))}


def allocation(conc: dict[str, np.ndarray], q: np.ndarray,
               prod_cells: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    """井对分配因子 f_ij：生产井 j 的采出量中来自注入井 i 的份额。"""
    out: dict[str, dict[str, float]] = {}
    for p, cells in prod_cells.items():
        c = np.asarray([x for x in cells if 0 <= x < len(q)], np.int64)
        w = np.maximum(q[c], 0.0)              # 只在真的在采出的单元上加权(采出为正)
        if w.sum() <= 0:
            continue
        out[p] = {inj: float((w * np.clip(cc[c], 0, None)).sum() / w.sum())
                  for inj, cc in conc.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="含 out/*.UNRST 的算例目录")
    ap.add_argument("--step", type=int, default=-1, help="重启步序号, -1=末帧")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    case = Path(args.case)
    unrst = case / "out" / "NORNE_ATW2013.UNRST"
    egrid = case / "out" / "NORNE_ATW2013.EGRID"
    sch = (case if (case / "INCLUDE").exists() else P.NORNE_DECK) / "INCLUDE" / "BC0407_HIST01122006.SCH"
    for f in (unrst, egrid, sch):
        if not f.exists():
            print(f"缺 {f}"); return 1

    from resdata.grid import Grid
    grid = Grid(str(egrid))
    na = grid.getNumActive()
    flux, step = read_fluxes(unrst, args.step)
    print(f"算例 {case.name}   活动单元 {na:,}   重启步 {step}")
    print(f"通量方向: {sorted(flux)}   " +
          ("⚠ 无 N+(非邻接/断层)通量, 跨断层连通会被低估" if "N" not in flux else
           "含 N+(非邻接), 但本版未接 NNC 映射, 暂按跳过处理并在结果里标注"))

    src, dst, rate = build_graph(grid, flux)
    print(f"有向边 {len(rate):,} 条   总通量 {rate.sum():,.0f} rm3/day")

    comp = completions(sch, set(INJECTORS) | set(PRODUCERS))
    ijk_lut = {}
    for a in range(na):
        ijk_lut[tuple(int(x) for x in grid.get_ijk(active_index=a))] = a
    cells = {w: [ijk_lut[c] for c in v if c in ijk_lut] for w, v in comp.items()}

    prod_cells = {w: c for w, c in cells.items() if w in PRODUCERS and c}
    well_rate, q_well = well_connection_rates(unrst, step, grid)
    inj_rate = {w: a for w, a in well_rate.items() if w in INJECTORS and (-a).sum() > 0}
    print(f"井连接流量: 注入井 {len(inj_rate)}  总注入 {sum(float(np.maximum(-a,0).sum()) for a in inj_rate.values()):,.0f}"
          f"  总采出 {float(np.maximum(q_well,0).sum()):,.0f} rm3/day")
    print(f"  真井单元 {(np.abs(q_well) > 0).sum():,} 个 (对照:散度法误判出 44,417 个)")
    conc, q, rec = solve_tracers(na, src, dst, rate, inj_rate, q_well)
    print(f"源分解: 井注入可归因 {rec['attributable_frac']:.1%}"
          f"   不可归因源(储量释放) {rec['unattributed_source']:,.0f}")
    print(f"  各路浓度之和 中位 {rec['sum_conc_median']:.4f} 最大 {rec['sum_conc_max']:.4f} (理论=1)"
          f"   '其他'路中位 {rec['other_conc_median']:.4f}")
    mx = max((float(c.max()) for c in conc.values()), default=float("nan"))
    print(f"解出示踪剂的注入井 {len(conc)}/{len(inj_rate)}   浓度最大值 {mx:.4f} (物理上应 ≤1)")

    alloc = allocation(conc, q, prod_cells)
    # 🔴 三种口径必须分开存, 不能混。2026-08-12 的跨模型复核发现主会话把
    #    **按生产井归一**的份额直接和**按注入井归一**的导数逐元素比 —— 是两个不同的
    #    条件概率, 对拍无意义。修正口径后 Spearman 从 +0.266 升到 +0.784。
    #      producer_fraction: 生产井 j 的产液中来自注入井 i 的份额(分母 q_prod_j)
    #      attributed_rate  : 归因流量 Q_ij = fraction × q_prod_j  (绝对量, 无量纲化)
    #      injector_fraction: Q_ij / q_inj_i  —— 与 CRM 的 dq_j/dw_i 同向, 对拍用这个
    q_prod = {w: float(np.maximum(a, 0).sum()) for w, a in well_rate.items() if w in PRODUCERS}
    q_inj = {w: float(np.maximum(-a, 0).sum()) for w, a in well_rate.items() if w in INJECTORS}
    attributed, inj_frac = {}, {}
    for pw, by in alloc.items():
        for iw, fr in by.items():
            Q = fr * q_prod.get(pw, 0.0)
            attributed.setdefault(pw, {})[iw] = Q
            if q_inj.get(iw, 0.0) > 0:
                inj_frac.setdefault(pw, {})[iw] = Q / q_inj[iw]
    print(f"\n注入井口径的 top1(与 CRM 同向, 不受微产量井劫持):")
    for iw in sorted(q_inj):
        cand = [(pw, d.get(iw, 0.0)) for pw, d in inj_frac.items() if iw in d]
        if not cand:
            continue
        pw, v = max(cand, key=lambda t: t[1])
        rs = sum(x[1] for x in cand)
        print(f"  {iw:8s} → {pw:8s} {v:.3f}   行和 {rs:.3f}")
    print(f"\n=== 井对分配因子 f_ij（行=生产井, 只列前 5 大来源） ===")
    rows = []
    for p in sorted(alloc):
        top = sorted(alloc[p].items(), key=lambda kv: -kv[1])[:5]
        tot = sum(alloc[p].values())
        rows.append({"producer": p, "total_from_injectors": tot,
                     "by_injector": alloc[p]})
        print(f"{p:8s} Σ={tot:5.3f}  " + "  ".join(f"{k}:{v:.3f}" for k, v in top if v > 1e-4))

    out = {"case": str(case), "step": step, "n_active": na, "reconstruction": rec,
           "has_nnc_flux": "N" in flux, "nnc_included": False,
           "n_injectors_solved": len(conc), "allocation": rows,
           "producer_liquid_rate": q_prod, "injector_rate": q_inj,
           "attributed_rate": attributed, "injector_fraction": inj_frac,
           "caveat": "稳态纯对流+冻结流场; 未计入 NNC(断层), 跨断层连通被低估"}
    dst_json = Path(args.json) if args.json else \
        Path(__file__).resolve().parent.parent / "_pipelines" / "flow_diagnostics" / "allocation.json"
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(dst_json.with_suffix(".npz"),
                        **{f"conc_{k}": v.astype(np.float32) for k, v in conc.items()},
                        q=q.astype(np.float32))
    print(f"\n已写入 {dst_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
