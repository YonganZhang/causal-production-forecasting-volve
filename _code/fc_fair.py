#!/usr/bin/env python3
"""参数预算对齐的公平重测：修掉"有的模型 25K 参数、有的 148 万"这个不公平。

## 为什么不能直接删掉表现差的模型

用户提出把第 8 名(N-BEATS)删掉。**不能删** —— 删掉表现差的基线是选择性报告,
是审稿人最容易识破、也最伤论文可信度的做法。**但用户指出的问题是真的**:

| 模型 | 参数量 | 名次 |
|---|---|---|
| DLinear | **25,280** | 10 |
| TimesNet | 159,446 | 7 |
| Autoformer | 278,444 | 3 |
| 本文 | 407,702 | **1** |
| DeepONet | 1,027,568 | 9 |
| **N-BEATS** | **1,479,776** | **8** |

参数量跨 **58 倍**,这个比较本身就不公平 —— 两个方向都不公平:
N-BEATS/DeepONet 可能**容量过剩而过拟合**(1400 个训练样本喂 148 万参数),
DLinear/TimesNet 可能**容量不足**。

**正确做法是把预算对齐,不是删掉输家。** 修好之后如果 N-BEATS 还是差,
那才是可以写进论文的结论;现在的差是我们没给它公平机会。

## 本脚本

给每个模型一组容量档位,**在 val 上选**(容量 + 学习率一起选),目标是让各模型
都能在 ~400K 参数附近发挥,然后在封存 test 上评一次。

🔴 DLinear 例外:它的全部主张就是"极简线性够用",把它撑到 400K 就不是 DLinear 了。
   所以保留原版,另加一个 `DLinear-wide` 作为补充数据点,两个都报,并注明。

用法:
    python fc_fair.py --gpu 4 --split-seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG
import fc_models as M1
import fc_models2 as M2
from fc_mech import PosNet

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_fair"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID
BUDGET = 410_000                                   # 目标参数预算


class DLinearWide(M2.DLinear):
    """DLinear 加一个隐藏层,仅用于参数预算对齐的补充对照。
    🔴 这**不是** Zeng et al. 的 DLinear —— 原版的主张就是"不要隐藏层"。两个都报。"""

    def __init__(self, d_in, n_ch, n_t, hidden=1024, **kw):
        super().__init__(d_in, n_ch, n_t, **kw)
        self.stem = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_ch * n_t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tune-seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(args.split_seed).permutation(n)
    te, va, tr = idx[:400], idx[400:600], idx[600:]
    trva = np.concatenate([tr, va])
    Yf = Y.reshape(n, -1); NCH = nw * NPH
    D = TH.shape[1]
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)

    # 每个模型给若干容量档位,让它们都有机会落在预算附近
    CAND = {
        "本文(傅里叶位置嵌入)": [lambda d=d: PosNet(D, NCH, NT, "fourier", n_bands=16, d=d,
                                                  layers=l) for d in (128, 160) for l in (3, 4)],
        "Transformer":      [lambda d=d, l=l: M1.TFBase(D, NCH, NT, d=d, layers=l)
                             for d in (128, 160) for l in (3, 4)],
        "Autoformer":       [lambda d=d, l=l: M1.Autoformer(D, NCH, NT, d=d, layers=l)
                             for d in (128, 160, 192) for l in (2, 3)],
        "iTransformer":     [lambda d=d, l=l: M2.ITransformer(D, NCH, NT, d=d, layers=l)
                             for d in (128, 160) for l in (3, 4)],
        "PatchTST":         [lambda d=d, p=p: M2.PatchTST(D, NCH, NT, d=d, patch=p)
                             for d in (128, 160) for p in (5, 8)],
        "TimesNet":         [lambda d=d, l=l: M2.TimesNet(D, NCH, NT, d=d, layers=l)
                             for d in (128, 192, 256) for l in (2, 3)],
        "FNO":              [lambda w=w, m=m: M1.FNO1d(D, NCH, NT, width=w, modes=m, layers=4)
                             for w in (64, 96, 128) for m in (8, 16)],
        "N-BEATS":          [lambda w=w, b=b, ba=ba: M2.NBeats(D, NCH, NT, width=w, blocks=b, basis=ba)
                             for w in (128, 192, 256) for b in (2, 3) for ba in (16, 32)],
        "DeepONet":         [lambda p=p, w=w: M1.DeepONet(D, NCH, NT, p=p, width=w)
                             for p in (48, 64, 96) for w in (128, 192)],
        "DLinear(原版)":     [lambda: M2.DLinear(D, NCH, NT)],
        "DLinear-wide(补充)": [lambda h=h: DLinearWide(D, NCH, NT, hidden=h) for h in (256, 512)],
    }

    def npar(f):
        return sum(p.numel() for p in f().parameters())

    def run(f, fit_rows, ev_rows, *, seed, lr):
        ym, ys = Yf[fit_rows].mean(0), Yf[fit_rows].std(0) + 1e-8
        xm, xs = TH[fit_rows].mean(0), TH[fit_rows].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = f().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[fit_rows], device=dev)
        B = torch.tensor(((Yf[fit_rows]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
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

    t0 = time.time()
    print(f"split_seed={args.split_seed}  目标参数预算 ≈ {BUDGET:,}\n")
    print("=== 1. 各模型在 val 上选容量 + 学习率(不在 test 上做任何选择) ===")
    BEST = {}
    for name, cands in CAND.items():
        # 只保留参数量最接近预算的 3 个候选,避免网格爆炸
        cands = sorted(cands, key=lambda f: abs(npar(f) - BUDGET))[:3]
        rows = []
        for ci, f in enumerate(cands):
            for lr in (1e-3, 3e-3):
                e = np.mean([run(f, tr, va, seed=s, lr=lr)[0] for s in range(args.tune_seeds)])
                rows.append({"ci": ci, "lr": lr, "val": float(e), "params": npar(f)})
        b = min(rows, key=lambda r: r["val"])
        BEST[name] = {"f": cands[b["ci"]], "lr": b["lr"], "params": b["params"], "val": b["val"]}
        print(f"  {name:22s} 参数 {b['params']:>9,d}  lr={b['lr']:<6g}  val {b['val']:.3%}"
              f"  ({(time.time()-t0)/60:.0f}min)", flush=True)

    print(f"\n=== 2. 封存 test({args.seeds} 种子, 单模型不集成) ===")
    R = {}
    for name, cfg in BEST.items():
        a = np.array([run(cfg["f"], trva, te, seed=s, lr=cfg["lr"]) for s in range(args.seeds)])
        R[name] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                   "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                   "r2": float(a[:, 3].mean()), "params": cfg["params"], "lr": cfg["lr"],
                   "raw": a[:, 0].tolist()}
        r = R[name]
        print(f"  {name:22s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"R² {r['r2']:.4f}  参数 {r['params']:>9,d}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    ours = np.array(R["本文(傅里叶位置嵌入)"]["raw"])
    rk = sorted(R, key=lambda k: R[k]["mean"])
    print(f"\n=== 3. 参数预算对齐后的排名(封存 test) ===")
    print(f"  {'#':>2} {'模型':22s}{'误差':>10s}{'R²':>9s}{'参数':>11s}{'p(vs本文)':>11s}")
    for i, k in enumerate(rk, 1):
        p = float(ttest_ind(np.array(R[k]["raw"]), ours, equal_var=False).pvalue) if k != "本文(傅里叶位置嵌入)" else 1.0
        print(f"  {i:>2} {k:22s}{R[k]['mean']:>10.3%}{R[k]['r2']:>9.4f}{R[k]['params']:>11,d}{p:>11.4f}")
    json.dump({"split_seed": args.split_seed, "budget": BUDGET, "seeds": args.seeds,
               "results": R, "ranking": rk,
               "note": ("参数预算对齐重测。此前 DLinear 25K vs N-BEATS 148万,跨 58 倍,比较不公平。"
                        "本轮每个模型在 val 上选容量+学习率,test 只评一次。"
                        "🔴 未删除任何模型 —— 删掉表现差的基线是选择性报告。")},
              open(OUT / f"fair_split{args.split_seed}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/fair_split{args.split_seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
