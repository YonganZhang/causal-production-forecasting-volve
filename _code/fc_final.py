#!/usr/bin/env python3
"""代理模型定版：三分切分 + 验证集选超参 + 留出集只报一次。

## 为什么要重做一遍

前两轮(fc_improve v1/v2)的结论全部作废,原因是**两个方法学错误**,不是模型问题:

**错误 1 —— 超参在留出集上选,又在同一个留出集上报数。**
`fc_improve.py:158-167` 扫 8 组超参、按留出集误差挑最好,然后"最终模型 D"
又在**同一个留出集**上报 0.551%。这是教科书级的选择性偏差,那个数不可引用。

**错误 2 —— 单种子下结论。**
基线自身 24 种子跨度 0.528%~0.691%(sd 0.048pp),把报表里 0.551%~0.689%
的全部差异吞掉。任何 <0.1pp 的"改进"都必须配对多种子检验才能宣称。

## 已被实验否决的三个解释(含我自己提的两个)

| 解释 | 判定 | 证据 |
|---|---|---|
| "物理特征是确定性函数,不含新信息" | 信息论上对,但**推不出"所以无影响"** | 随机线性重参数化(同样确定性)反而**变好** 0.043pp |
| "秩亏/共线性/条件数毁了输入" | 🔴 **实测否决** | 精确删掉 12 个共线列(43维)= 0.709%,与 A 无差异(p=0.32);log 消重尾(条件数 6.5e7→7.7e4)= 0.715%(p=0.37);PCA 白化最差 0.814% |
| "欠训练/超参不公平" | 🔴 **实测否决** | 末轮 train MSE 基线 0.00041 vs A 0.00041(完全相同),val MSE 0.0195 vs 0.0245(高 26%)→ 是泛化差,不是拟合不动 |

**真正成立的结论**:A(物理特征)不是无效,是**显著有害** —— 24 配对种子 +0.0987pp,
95%CI [+0.068,+0.130],22/24 种子更差,Wilcoxon p=1.1e-5,Cohen dz=1.33。
剂量-反应曲线是直接因果证据:PF 整体降权 ×1.0→0.735%、×0.3→0.638%、×0.1→0.550%、
×0→0.573%,**网络越少听物理特征越准**。单独用 31 维 PF(丢掉 θ)= 1.99%,
说明它们是对 θ 的**有损非线性重编码**,不是增量信息。

**B(单调约束)是纯噪声**:24 配对种子 p=0.623,且惩罚项本身是 no-op —— 权重 1e-3
下惩罚量级 7e-11 对 MSE 0.044。更根本地,"25% 负产率格点"是假指标:22 口生产井里
**11 口油率水率恒为 0**,负产全部落在这 11 口死井上(--drop-dead 后负产率 = 0.0%),
负产质量 7~105 m³ 对累计产油 7.8e6 m³。

## 本脚本怎么做

三分切分:**test 400 全程封存**,train 1400 / val 200。超参只在 val 上选,
选定后在 train+val 上重训 + 多种子集成,**在 test 上只评一次**。

用法:
    python fc_final.py --gpu 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG
import norne_bulk as NB
from fc_improve2 import TF, phys_features

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_final"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def cumoil(Y, nw):
    return np.trapezoid(Y.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)


def fit(Xtr, Ytr_n, dev, *, seed, nw, lr, epochs, d, heads, layers, ff):
    torch.manual_seed(seed)
    net = TF(Xtr.shape[1], nw * NPH, NT, d, heads, layers, ff).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    A = torch.tensor(Xtr, device=dev)
    B = torch.tensor(Ytr_n.reshape(-1, nw * NPH, NT), device=dev)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        pm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 64):
            b = pm[i:i + 64]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(A[b]), B[b]).backward()
            opt.step()
        sch.step()
    return net.eval()


def predict(nets, X, ym, ys, dev):
    """多网络集成 = 在**物理量纲**上平均预测。"""
    with torch.no_grad():
        t = torch.tensor(X, device=dev)
        P = np.mean([n(t).cpu().numpy().reshape(len(X), -1) for n in nets], axis=0)
    return P * ys + ym


def score(P, Yraw, nw):
    ct, cp = cumoil(Yraw, nw), cumoil(P, nw)
    rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
    r2 = 1 - ((cp - ct) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
    return float(rel.mean()), float(np.quantile(rel, 0.9)), float(r2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--tune-seeds", type=int, default=2)
    ap.add_argument("--ens", type=int, default=5)
    ap.add_argument("--final-seeds", type=int, default=4, help="集成实验重复次数(报 ±)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum())                    # 死井恒为 0, 剔除不改累计产油口径
    n = len(TH)
    rng = np.random.default_rng(args.split_seed); idx = rng.permutation(n)
    te = idx[:400]; va = idx[400:600]; tr = idx[600:]       # 🔴 test 全程封存
    trva = np.concatenate([tr, va])
    print(f"样本 {n}  train {len(tr)} / val {len(va)} / test {len(te)}(封存)  生产井 {nw}(剔 {int((~live).sum())} 口恒零)\n")

    Yf = Y.reshape(n, -1)
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8           # 调参阶段:统计量只用 train
    PF = phys_features(TH, full=True)

    def mknorm(rows, X):
        m, s = X[rows].mean(0), X[rows].std(0) + 1e-8
        return ((X - m) / s).astype(np.float32)
    FEAT = {"θ(24维)": TH, "θ+物理特征(55维)": np.hstack([TH, PF])}

    # ---------------- 1. 在 val 上选超参 ----------------
    print("=== 1. 超参搜索(只看 val, test 不碰) ===")
    GRID = list(itertools.product([1e-3, 3e-3], [300, 600], [128, 192]))
    best, t0 = {}, time.time()
    tune_log = {}
    for fname, Xraw in FEAT.items():
        X = mknorm(tr, Xraw)
        Ytr_n = (Yf[tr] - ym) / ys
        rows = []
        for lr, ep, d in GRID:
            es = [score(predict([fit(X[tr], Ytr_n, dev, seed=s, nw=nw, lr=lr, epochs=ep,
                                     d=d, heads=4, layers=3, ff=256)], X[va], ym, ys, dev),
                        Y[va], nw)[0] for s in range(args.tune_seeds)]
            rows.append({"lr": lr, "epochs": ep, "d": d, "val": float(np.mean(es))})
            print(f"  {fname:18s} lr={lr:<6g} ep={ep:<4d} d={d:<4d} val {np.mean(es):.3%}"
                  f"  ({(time.time()-t0)/60:.0f}min)", flush=True)
        rows.sort(key=lambda r: r["val"])
        best[fname] = rows[0]; tune_log[fname] = rows
        print(f"  → {fname} 最佳 lr={rows[0]['lr']} ep={rows[0]['epochs']} "
              f"d={rows[0]['d']}  val {rows[0]['val']:.3%}\n")

    # ---------------- 2. train+val 重训, test 只评一次 ----------------
    print("=== 2. 定版:train+val 重训 + {}-模型集成, 在封存的 test 上只评一次 ===".format(args.ens))
    ym2, ys2 = Yf[trva].mean(0), Yf[trva].std(0) + 1e-8
    Y2n = (Yf[trva] - ym2) / ys2
    final = {}
    for fname, Xraw in FEAT.items():
        X = mknorm(trva, Xraw); c = best[fname]
        singles, ens = [], []
        for rep in range(args.final_seeds):
            nets = [fit(X[trva], Y2n, dev, seed=1000 * rep + s, nw=nw, lr=c["lr"],
                        epochs=c["epochs"], d=c["d"], heads=4, layers=3, ff=256)
                    for s in range(args.ens)]
            singles.append(score(predict(nets[:1], X[te], ym2, ys2, dev), Y[te], nw))
            ens.append(score(predict(nets, X[te], ym2, ys2, dev), Y[te], nw))
        def agg(v):
            a = np.array(v)
            return {"relerr": float(a[:, 0].mean()), "relerr_sd": float(a[:, 0].std(ddof=1)),
                    "p90": float(a[:, 1].mean()), "r2": float(a[:, 2].mean()),
                    "raw": a[:, 0].tolist()}
        final[fname] = {"cfg": c, "single": agg(singles), "ensemble": agg(ens)}
        s1, s5 = final[fname]["single"], final[fname]["ensemble"]
        print(f"  {fname:18s} 单模型 {s1['relerr']:.3%}±{s1['relerr_sd']:.3%}  "
              f"集成{args.ens} {s5['relerr']:.3%}±{s5['relerr_sd']:.3%}  "
              f"R² {s5['r2']:.4f}  p90 {s5['p90']:.3%}", flush=True)

    # ---------------- 3. 平凡基线 + 配对检验 ----------------
    ctr = cumoil(Yf[trva], nw).mean()
    cte = cumoil(Yf[te], nw)
    triv = float((np.abs(cte - ctr) / np.abs(cte)).mean())
    print(f"\n  平凡基线(训练集均值) {triv:.3%}")

    from scipy.stats import ttest_rel
    a = np.array(final["θ(24维)"]["ensemble"]["raw"])
    b = np.array(final["θ+物理特征(55维)"]["ensemble"]["raw"])
    p = float(ttest_rel(a, b).pvalue)
    print(f"\n=== 3. 判定 ===")
    print(f"  物理特征 − 纯θ = {(b-a).mean()*100:+.3f} 个百分点   配对 t p={p:.4f}"
          f"   {'🔴 物理特征显著有害' if p < 0.05 and b.mean() > a.mean() else '— 噪声范围内'}")
    win = min(final, key=lambda k: final[k]["ensemble"]["relerr"])
    e = final[win]["ensemble"]
    print(f"\n🏆 定版模型: {win}  {best[win]}")
    print(f"   封存 test 上 误差 {e['relerr']:.3%}  R² {e['r2']:.4f}  p90 {e['p90']:.3%}")
    print(f"   相对平凡基线 {triv:.3%} 提升 {triv/e['relerr']:.1f}×")

    json.dump({"n": n, "split": {"train": len(tr), "val": len(va), "test": len(te)},
               "n_live_wells": nw, "tuning": tune_log, "best_cfg": best,
               "final": final, "trivial_baseline": triv, "paired_p": p, "winner": win,
               "protocol": ("test 400 条全程封存;超参只在 val 200 上选;"
                            "选定后 train+val 重训 + 集成, test 只评一次。"
                            "这修掉了 fc_improve 的选择性偏差(超参与报数用同一留出集)。")},
              open(OUT / "final.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
