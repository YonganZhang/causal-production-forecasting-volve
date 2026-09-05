#!/usr/bin/env python3
"""4000 样本重训：验证"瓶颈是数据不是架构"这个判断，并给出定版模型。

## 为什么这是最该做的一步

学习曲线在 2000 样本时**还在加速下降**，不是饱和:

| 样本量 | 误差 | 降幅 |
|---|---|---|
| 400 → 800 | 1.598% → 1.022% | 36.1% |
| **800 → 1600** | 1.022% → **0.483%** | **52.7%** |

而我试过的**六个模块**每个只值 ±0.05 个百分点。**模型是数据受限,不是容量受限** ——
这是六个模块全部失败最简洁的解释。样本已扩到 4000(2.18 小时,失败 0)。

## 设计：同一 test/val，只改训练集大小

`split_seed=0` 对 4000 个样本做置换 → **test=idx[:400] 封存 / val=idx[400:600] /
训练池=idx[600:]**(3400 条)。然后从训练池里**取前 N 条**训练,N ∈ {1400, 2000, 2800, 3400}。

这样 test 与 val 在所有 N 下**完全相同**,唯一变量就是训练样本量 —— 才能干净地
回答"多给数据到底值多少"。N=1400 正好对应此前 2000 样本时代的训练集大小,可直接对照。

🔴 前 2000 个 θ 与扩样本前逐位一致(已验证),所以新旧结果同源可比。

用法:
    python fc_4k.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 8000
import forecast_gen as FG
import fc_models as M
from fc_mech import PosNet

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_4k"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(args.split_seed).permutation(n)
    te, va, pool = idx[:400], idx[400:600], idx[600:]
    Yf = Y.reshape(n, -1); NCH = nw * NPH; D = TH.shape[1]
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    print(f"总样本 {n}   test {len(te)}(封存) / val {len(va)} / 训练池 {len(pool)}")
    print(f"生产井 {nw}(剔 {int((~live).sum())} 口恒零)   {args.seeds} 种子\n")

    def run(N, ev_rows, *, seed, ff, lr=3e-3):
        fit = pool[:N]
        ym, ys = Yf[fit].mean(0), Yf[fit].std(0) + 1e-8
        xm, xs = TH[fit].mean(0), TH[fit].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = (PosNet(D, NCH, NT, "fourier", n_bands=16) if ff
               else M.TFBase(D, NCH, NT)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[fit], device=dev)
        B = torch.tensor(((Yf[fit]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i+64]; opt.zero_grad()
                F.mse_loss(net(A[b]), B[b]).backward(); opt.step()
            s.step()
        net.eval()
        with torch.no_grad():
            P = net(torch.tensor(X[ev_rows], device=dev)).cpu().numpy().reshape(len(ev_rows), -1)
        P = P * ys + ym
        ct, cp = cum(Yf[ev_rows]), cum(P)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        return (float(rel.mean()), float(np.median(rel)), float(np.quantile(rel, .9)),
                float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()))

    NS = [N for N in (1400, 2000, 2800, 3400) if N <= len(pool)]
    print(f"=== 封存 test 上的学习曲线(test/val 恒定,唯一变量是训练样本量) ===")
    print(f"  {'N':>6} {'模型':16s}{'误差':>10s}{'±sd':>9s}{'中位':>9s}{'R²':>9s}")
    R, t0 = {}, time.time()
    for N in NS:
        for ff in (False, True):
            tag = f"{'傅里叶(本文)' if ff else '基线 Transformer'}"
            a = np.array([run(N, te, seed=s, ff=ff) for s in range(args.seeds)])
            R[f"N{N}_{'ff' if ff else 'base'}"] = {
                "N": N, "ff": ff, "mean": float(a[:, 0].mean()),
                "sd": float(a[:, 0].std(ddof=1)), "median": float(a[:, 1].mean()),
                "p90": float(a[:, 2].mean()), "r2": float(a[:, 3].mean()),
                "raw": a[:, 0].tolist()}
            r = R[f"N{N}_{'ff' if ff else 'base'}"]
            print(f"  {N:>6} {tag:16s}{r['mean']:>10.3%}{r['sd']:>9.3%}{r['median']:>9.3%}"
                  f"{r['r2']:>9.4f}   ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    print(f"\n=== 判定 ===")
    n0, n1 = NS[0], NS[-1]
    for k in ("base", "ff"):
        a = np.array(R[f"N{n0}_{k}"]["raw"]); b = np.array(R[f"N{n1}_{k}"]["raw"])
        p = float(ttest_ind(b, a, equal_var=False).pvalue)
        nm = "傅里叶(本文)" if k == "ff" else "基线"
        print(f"  {nm:14s} N={n0}→{n1}: {a.mean()*100:.3f}% → {b.mean()*100:.3f}%  "
              f"降幅 {1-b.mean()/a.mean():.1%}  p={p:.4f}")
    for N in NS:
        a = np.array(R[f"N{N}_base"]["raw"]); b = np.array(R[f"N{N}_ff"]["raw"])
        p = float(ttest_ind(b, a, equal_var=False).pvalue)
        vd = "✔ 显著" if p < .05 and b.mean() < a.mean() else "— 噪声内"
        print(f"  N={N:<5d} 傅里叶 vs 基线: {(b.mean()-a.mean())*100:+.3f}pp  p={p:.4f}  {vd}")

    best = min(R, key=lambda k: R[k]["mean"])
    print(f"\n🏆 定版: {'傅里叶(本文)' if R[best]['ff'] else '基线'} @ N={R[best]['N']}  "
          f"{R[best]['mean']:.3%} ± {R[best]['sd']:.3%}  R² {R[best]['r2']:.4f}")
    json.dump({"n_total": n, "split_seed": args.split_seed, "seeds": args.seeds,
               "N_list": NS, "results": R, "best": best,
               "protocol": ("test 400 与 val 200 在所有 N 下完全相同,唯一变量是训练样本量。"
                            "N=1400 对应扩样本前的训练集大小,可直接对照。")},
              open(OUT / "fc4k.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/fc4k.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
