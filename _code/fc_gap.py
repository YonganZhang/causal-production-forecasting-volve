#!/usr/bin/env python3
"""Gap 分析：先查清 Transformer 到底错在哪，再决定改哪一层。

## 先把任务类型定性 —— 这决定了该用哪个领域的模型

| | 本任务 |
|---|---|
| 输入 | **24 个静态数** (6 决策时点 × 4 注水井的注水乘子) |
| 输出 | 11 井 × 2 相 × 40 时刻 = **880 个数**的轨迹 |
| 推理时是否看历史曲线 | **否** |

**所以这不是时序预测任务。** 时序预测模型(Autoformer / Informer / Chronos / PatchTST)
的架构前提是"给一段历史,预测后一段",它们的核心组件(自相关、序列分解、patch 嵌入)
都作用在**历史序列**上。本任务推理时根本没有历史可看,把它们直接搬过来,
编码器端是空的 —— 这是架构错配,不是调参问题。

本任务的正确名字是**参数化算子学习 / 参数→场 的代理**(parametric operator learning):
学一个映射 G: θ ∈ R²⁴ → u(t) ∈ R^(22×40)。这在油藏代理模型文献里的主流是
**FNO(Fourier Neural Operator)** 与 **DeepONet**,不是时序预测模型。

⚠️ 但 Autoformer 仍要作为 baseline 报 —— 因为它是"合理但错配"的对照,
   审稿人会问,而且它能证明"错配"这个论断不是空谈。

## 本脚本查什么

拿定版单模型 Transformer 的留出集残差,逐项定位误差来源:

1. **误差在时间上怎么分布** —— 早期/晚期? 对应什么物理阶段?
2. **误差的频率成分** —— 低频(整体水平错)还是高频(细节抖动)?
   这决定了该上谱方法(FNO)还是该上平滑/分解。
3. **逐井误差** —— 集中在少数井还是均摊?
4. **异方差** —— 误差随产量大小怎么变?
5. **物理违背** —— 负产率、累计非单调、井间总量关系。
6. **水突破时刻** —— 含水率拐点预测得准不准。这是油藏工程最关心的特征,
   也是最容易被 MSE 抹平的**高频/相位**信息。
7. **可学习上界** —— 同一 θ 附近的样本之间响应差多少(数据本身的噪声底)。

用法:
    python fc_gap.py --gpu 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG
import norne_bulk as NB
from fc_improve2 import TF

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_gap"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    WELLS = [NB.PRODUCERS[i] for i in np.where(live)[0]]
    Y = Y[:, live]; nw = len(WELLS); n = len(TH)
    rng = np.random.default_rng(args.split_seed); idx = rng.permutation(n)
    te, trva = idx[:400], idx[600:]
    trva = np.concatenate([idx[600:], idx[400:600]])
    Yf = Y.reshape(n, -1)
    ym, ys = Yf[trva].mean(0), Yf[trva].std(0) + 1e-8
    xm, xs = TH[trva].mean(0), TH[trva].std(0) + 1e-8
    X = ((TH - xm) / xs).astype(np.float32)

    print(f"=== 0. 任务定性 ===")
    print(f"  输入 {TH.shape[1]} 维静态参数 → 输出 {nw}井 × {NPH}相 × {NT}时刻 = {nw*NPH*NT} 维轨迹")
    print(f"  推理时不看任何历史曲线 → **不是时序预测任务, 是参数化算子学习**")

    # ---- 训练定版单模型 ----
    torch.manual_seed(0)
    net = TF(X.shape[1], nw * NPH, NT, 128, 4, 3, 256).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    A = torch.tensor(X[trva], device=dev)
    B = torch.tensor(((Yf[trva] - ym) / ys).reshape(-1, nw * NPH, NT).astype(np.float32), device=dev)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        pm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 64):
            b = pm[i:i + 64]; opt.zero_grad()
            torch.nn.functional.mse_loss(net(A[b]), B[b]).backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(X[te], device=dev)).cpu().numpy().reshape(len(te), -1) * ys + ym
    P = P.reshape(-1, nw, NPH, NT); T = Yf[te].reshape(-1, nw, NPH, NT)
    E = P - T
    oilP, oilT = P[:, :, 0, :], T[:, :, 0, :]
    watP, watT = P[:, :, 1, :], T[:, :, 1, :]
    G = {}

    def rel(a, b):
        return float(np.abs(a - b).sum() / np.maximum(np.abs(b).sum(), 1e-9))

    cum = lambda A_: np.trapezoid(A_, DAYS, axis=-1).sum(1)
    ct, cp = cum(oilT), cum(oilP)
    base_rel = float(np.mean(np.abs(cp - ct) / np.abs(ct)))
    print(f"\n  复现定版单模型: 累计产油平均相对误差 {base_rel:.3%}")

    # ---- 1. 误差的时间分布 ----
    print(f"\n=== 1. 误差在时间上怎么分布 ===")
    e_t = np.abs(E[:, :, 0, :]).mean((0, 1)); s_t = np.abs(T[:, :, 0, :]).mean((0, 1))
    nrm = e_t / np.maximum(s_t, 1e-9)
    yrs = 2006 + (DAYS - DAYS[0]) / 365.25
    for k in range(0, NT, 6):
        print(f"  {yrs[k]:.1f}年 (第{DAYS[k]:.0f}天)  归一化误差 {nrm[k]:.3%}   平均油率 {s_t[k]:8.1f}")
    G["err_by_time"] = {"years": yrs.tolist(), "norm_err": nrm.tolist(), "mean_oil": s_t.tolist()}
    print(f"  🔴 误差最大的时段: {yrs[int(np.argmax(nrm))]:.1f} 年 ({nrm.max():.2%})"
          f"   最小: {yrs[int(np.argmin(nrm))]:.1f} 年 ({nrm.min():.2%})")

    # ---- 2. 频率成分:低频还是高频 ----
    print(f"\n=== 2. 误差的频率成分(决定该上谱方法还是平滑) ===")
    FT = np.fft.rfft(T[:, :, 0, :], axis=-1); FE = np.fft.rfft(E[:, :, 0, :], axis=-1)
    ps_t = (np.abs(FT) ** 2).mean((0, 1)); ps_e = (np.abs(FE) ** 2).mean((0, 1))
    half = len(ps_t) // 2
    lo_s, hi_s = ps_t[:half].sum(), ps_t[half:].sum()
    lo_e, hi_e = ps_e[:half].sum(), ps_e[half:].sum()
    print(f"  信号能量  低频 {lo_s/(lo_s+hi_s):.2%}  高频 {hi_s/(lo_s+hi_s):.2%}")
    print(f"  误差能量  低频 {lo_e/(lo_e+hi_e):.2%}  高频 {hi_e/(lo_e+hi_e):.2%}")
    snr = ps_t / np.maximum(ps_e, 1e-30)
    print(f"  逐频段 信号/误差 功率比(越低=该频段学得越差):")
    for k in range(0, len(ps_t), max(1, len(ps_t) // 8)):
        print(f"    模态 {k:2d}  信噪比 {snr[k]:10.1f}")
    G["spectrum"] = {"signal_lo_frac": float(lo_s/(lo_s+hi_s)), "err_lo_frac": float(lo_e/(lo_e+hi_e)),
                     "snr_per_mode": snr.tolist()}
    k_worst = int(np.argmin(snr))
    print(f"  🔴 学得最差的模态 k={k_worst} (信噪比 {snr[k_worst]:.1f});"
          f" 模态 0(总水平)信噪比 {snr[0]:.1f}")

    # ---- 3. 逐井 ----
    print(f"\n=== 3. 逐井误差(集中还是均摊) ===")
    pw = np.array([rel(oilP[:, i], oilT[:, i]) for i in range(nw)])
    shr = np.array([np.abs(E[:, i, 0, :]).sum() for i in range(nw)]); shr = shr / shr.sum()
    for i in np.argsort(-pw):
        print(f"  {WELLS[i]:8s} 相对误差 {pw[i]:>7.2%}   占总误差 {shr[i]:>6.1%}"
              f"   平均油率 {oilT[:,i].mean():8.1f}")
    G["per_well"] = {WELLS[i]: {"rel": float(pw[i]), "share": float(shr[i])} for i in range(nw)}
    print(f"  🔴 前 3 口井占总误差 {np.sort(shr)[::-1][:3].sum():.1%}")

    # ---- 4. 异方差 ----
    print(f"\n=== 4. 误差随产量大小怎么变(异方差?) ===")
    r = np.abs(cp - ct) / np.abs(ct)
    q = np.quantile(ct, [0, .25, .5, .75, 1.])
    for i in range(4):
        m = (ct >= q[i]) & (ct <= q[i+1])
        print(f"  累计产油 {q[i]/1e6:.2f}~{q[i+1]/1e6:.2f}e6 m³  相对误差 {r[m].mean():.3%}  n={m.sum()}")
    G["heteroscedastic"] = {"corr_err_vs_size": float(np.corrcoef(ct, r)[0, 1])}
    print(f"  误差与产量大小的相关 {np.corrcoef(ct, r)[0,1]:+.3f}")

    # ---- 5. 物理违背 ----
    print(f"\n=== 5. 物理违背 ===")
    negm = float(np.trapezoid(np.maximum(-oilP, 0), DAYS, axis=-1).sum(1).mean())
    cumP = np.cumsum(oilP, -1)
    nonmono = float((np.diff(cumP, axis=-1) < 0).mean())
    wcP = watP / np.maximum(oilP + watP, 1e-9); wcT = watT / np.maximum(oilT + watT, 1e-9)
    wc_bad = float(((wcP < -0.01) | (wcP > 1.01)).mean())
    print(f"  负产油'质量' {negm:.1f} m³ / 样本 (占累计 {negm/ct.mean():.2e})")
    print(f"  累计产油非单调格点 {nonmono:.2%}")
    print(f"  含水率越界(<0 或 >1) 格点 {wc_bad:.2%}")
    G["physics"] = {"neg_mass": negm, "nonmono": nonmono, "wc_out_of_range": wc_bad}

    # ---- 6. 水突破时刻(相位) ----
    print(f"\n=== 6. 水突破时刻:最容易被 MSE 抹平的相位信息 ===")
    def bt(wc, thr=0.5):
        o = np.argmax(wc > thr, -1).astype(float)
        o[~(wc > thr).any(-1)] = np.nan
        return o
    bT, bP = bt(wcT), bt(wcP)
    m = np.isfinite(bT) & np.isfinite(bP)
    dt = (bP - bT)[m]
    print(f"  有突破的井×样本 {int(m.sum())}/{m.size}  ({m.mean():.1%})")
    print(f"  突破时刻误差 中位 {np.median(np.abs(dt)):.2f} 格点 = {np.median(np.abs(dt))*(DAYS[1]-DAYS[0]):.0f} 天")
    print(f"                 p90 {np.quantile(np.abs(dt),0.9):.2f} 格点 = {np.quantile(np.abs(dt),0.9)*(DAYS[1]-DAYS[0]):.0f} 天")
    print(f"  漏报(真有突破但预测无) {int((np.isfinite(bT)&~np.isfinite(bP)).sum())}"
          f"   虚报 {int((~np.isfinite(bT)&np.isfinite(bP)).sum())}")
    G["breakthrough"] = {"n": int(m.sum()), "median_grid_err": float(np.median(np.abs(dt))),
                         "p90_grid_err": float(np.quantile(np.abs(dt), 0.9)),
                         "days_per_grid": float(DAYS[1] - DAYS[0])}

    # ---- 7. 水平 vs 形状:误差是"整体高低"还是"曲线形状" ----
    print(f"\n=== 7. 误差拆解:整体水平错 vs 曲线形状错 ===")
    # 🔴 2026-08-27 修正:此前用 L1 和直接比 —— e_lvl 只有 1 个标量项,
    #    e_shp 有 40 个时刻项,项数差 40 倍,归一化无效。
    #    错误结果 level 1.2%/shape 98.8% 曾被 fc_loop.py 的 deriv_l2 当作立项理由。
    #    正确做法:按**能量**(平方和)分解,两个分量在同一空间正交,可直接相加。
    mT, mP = oilT.mean(-1, keepdims=True), oilP.mean(-1, keepdims=True)
    e_lvl = float((((mP - mT) ** 2) * oilT.shape[-1]).sum())      # 常数分量在 40 个时刻上各占一份
    e_shp = float((((oilP - mP) - (oilT - mT)) ** 2).sum())
    print(f"  水平分量误差 {e_lvl/(e_lvl+e_shp):.1%}   形状分量误差 {e_shp/(e_lvl+e_shp):.1%}")
    G["level_vs_shape"] = {"level": float(e_lvl/(e_lvl+e_shp)), "shape": float(e_shp/(e_lvl+e_shp))}

    json.dump({"base_relerr": base_rel, "n_test": len(te), "wells": WELLS, **G},
              open(OUT / "gap.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/gap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
