#!/usr/bin/env python3
"""全模型总排名 v2：把五个新骨干拉进同一协议，产出论文用的完整基线表。

## 协议（与 fc_final2 / fc_nb 完全一致，所以可直接并表）

- 切分 `split_seed=0`：**test 400 全程封存** / val 200 / train 1400；
- 每个骨干**各自在 val 上选学习率**（避免拿基线超参跑新架构这种隐性不公平）；
- 选定后 train+val 重训，**test 只评一次**；
- **全部单模型，不做集成**；多种子报均值±标准差 + Welch 检验。

## 🔴 适配声明

五个新骨干原本都是 history→future 预测器，本任务推理时**没有历史序列**。
适配方式统一为"把历史编码器换成 θ 条件注入，保留各自的特征机制"。
这个适配对所有模型一视同仁（含已有的 Transformer/Autoformer），横向比较公平，
但它们**没有跑在原生设定上** —— 论文必须原样声明。

用法:
    python fc_zoo.py --gpu 1
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
N_PIN = 4000
import forecast_gen as FG
import fc_models as M1
from fc_models2 import ZOO2
from fc_mech import PosNet

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_zoo"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    te, va, tr = idx[:400], idx[400:600], idx[600:]
    trva = np.concatenate([tr, va])
    Yf = Y.reshape(n, -1); NCH = nw * NPH
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}(封存)  输出 {nw}井×2相×{NT}时刻\n")

    BUILD = {
        "Transformer(基线)": lambda: M1.TFBase(TH.shape[1], NCH, NT),
        "🏆 傅里叶位置嵌入(本文)": lambda: PosNet(TH.shape[1], NCH, NT, "fourier", n_bands=16),
        "Autoformer": lambda: M1.Autoformer(TH.shape[1], NCH, NT),
        "FNO": lambda: M1.FNO1d(TH.shape[1], NCH, NT, width=64, modes=8, layers=4),
        "DeepONet": lambda: M1.DeepONet(TH.shape[1], NCH, NT),
        **{k: (lambda c=v: c(TH.shape[1], NCH, NT)) for k, v in ZOO2.items()},
    }

    def run(name, fit_rows, ev_rows, *, seed, lr):
        ym, ys = Yf[fit_rows].mean(0), Yf[fit_rows].std(0) + 1e-8
        xm, xs = TH[fit_rows].mean(0), TH[fit_rows].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = BUILD[name]().to(dev)
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
        net.eval(); t0 = time.time()
        with torch.no_grad():
            P = net(torch.tensor(X[ev_rows], device=dev)).cpu().numpy().reshape(len(ev_rows), -1)
        inf_ms = (time.time() - t0) * 1000 / len(ev_rows)
        P = P * ys + ym
        ct, cp = cum(Yf[ev_rows]), cum(P)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        return (float(rel.mean()), float(np.median(rel)), float(np.quantile(rel, .9)),
                float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()), inf_ms,
                sum(p.numel() for p in net.parameters()))

    t0 = time.time()
    print("=== 1. 各骨干在 val 上选学习率 ===")
    LR = {}
    for k in BUILD:
        best = None
        for lr in (1e-3, 3e-3):
            e = run(k, tr, va, seed=0, lr=lr)[0]
            if best is None or e < best[1]:
                best = (lr, e)
        LR[k] = best[0]
        print(f"  {k:24s} 选 lr={best[0]:<6g} (val {best[1]:.3%})  ({(time.time()-t0)/60:.0f}min)",
              flush=True)

    print(f"\n=== 2. 封存 test({args.seeds} 种子, **单模型不集成**) ===")
    R = {}
    for k in BUILD:
        a = np.array([run(k, trva, te, seed=s, lr=LR[k]) for s in range(args.seeds)])
        R[k] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                "r2": float(a[:, 3].mean()), "infer_ms": float(a[:, 4].mean()),
                "params": int(a[0, 5]), "lr": LR[k], "raw": a[:, 0].tolist()}
        r = R[k]
        print(f"  {k:24s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"R² {r['r2']:.4f}  参数 {r['params']:>9,d}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    ours = np.array(R["🏆 傅里叶位置嵌入(本文)"]["raw"])
    base = np.array(R["Transformer(基线)"]["raw"])
    rk = sorted(R, key=lambda k: R[k]["mean"])
    print(f"\n=== 3. 排名(封存 test, 平均相对误差) ===")
    print(f"  {'#':>2} {'模型':24s}{'平均误差':>10s}{'R²':>9s}{'参数':>11s}{'vs 冠军':>9s}{'p(vs本文)':>11s}")
    top = R[rk[0]]["mean"]
    for i, k in enumerate(rk, 1):
        p = ttest_ind(np.array(R[k]["raw"]), ours, equal_var=False).pvalue if k != rk[0] or True else 1
        print(f"  {i:>2} {k:24s}{R[k]['mean']:>10.3%}{R[k]['r2']:>9.4f}"
              f"{R[k]['params']:>11,d}{R[k]['mean']/top:>8.2f}×{p:>11.4f}")
    second = [k for k in rk if k != "🏆 傅里叶位置嵌入(本文)"][0]
    s2 = R[second]
    w = R["🏆 傅里叶位置嵌入(本文)"]
    pw = float(ttest_ind(np.array(s2["raw"]), ours, equal_var=False).pvalue)
    print(f"\n本文 {w['mean']:.3%} ± {w['sd']:.3%}")
    print(f"  最强对手(非本文) = {second} {s2['mean']:.3%}")
    print(f"  → 误差是它的 1/{s2['mean']/w['mean']:.2f}, 降低 {1-w['mean']/s2['mean']:.1%}, p={pw:.4f}")
    json.dump({"results": R, "ranking": rk, "lr_selected": LR, "seeds": args.seeds,
               "runner_up": second, "p_vs_runner_up": pw,
               "protocol": "test 400 封存;各骨干各自在 val 选 lr;单模型不集成",
               "adaptation": ("五个新骨干原为 history→future;本任务无历史,统一改为"
                              "θ 条件注入 + 保留各自特征机制。对所有模型一视同仁,"
                              "但**非原生设定**,论文须声明。")},
              open(OUT / "zoo.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/zoo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
