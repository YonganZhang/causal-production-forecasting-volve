#!/usr/bin/env python3
"""噪声地板与可达上限：判定"架构差异到底测不测得出来"。

## 为什么必须做这个

3400 样本下四个结构差异巨大的架构挤在 val 0.345%~0.362%(跨度 0.017pp),
而单模型的种子标准差就有 0.008~0.034pp。若**指标本身的噪声地板**就有这个量级,
那"七个模块全失败"根本不是异常 —— 是**这个规模下压根分辨不出来**。

这条探针在上一轮 Workflow 里因 API 断连挂了,本脚本补上。

## 四个量

1. **test 采样噪声**:400 条 test 的均值相对误差,自举重采样给标准误。
   若它 ≥0.02pp,则所有 <0.05pp 的模块差异都不可靠。
2. **种子噪声**:同一配置多种子的标准差。
3. **偏差-方差分解**:误差里多少是所有种子**共有的系统偏差**(可以靠更好的模型/更多数据消掉),
   多少是种子方差(只能靠集成消掉)。系统偏差占比高 = 还有余量可挖。
4. **过参数化探底**:训练一个刻意过大的模型到训练集近乎零误差,看留出误差停在哪 ——
   这给出"当前数据量下的泛化下限"的经验估计。

用法:
    python fc_ceil.py --gpu 7
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
from fc_mech import PosNet

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_ceil"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--ntrain", type=int, default=3400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    te, va, pool = idx[:400], idx[400:600], idx[600:]
    tr = pool[:min(args.ntrain, len(pool))]
    Yf = Y.reshape(n, -1); NCH = nw * NPH; D = TH.shape[1]
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    ct = cum(Yf[te])
    print(f"总样本 {n}  训练 {len(tr)} / test {len(te)}(封存)   {args.seeds} 种子\n")

    def fit(seed, d=128, layers=3, epochs=None):
        ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
        xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        ep = epochs or args.epochs
        torch.manual_seed(seed)
        net = PosNet(D, NCH, NT, "fourier", n_bands=16, d=d, layers=layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        A = torch.tensor(X[tr], device=dev)
        B = torch.tensor(((Yf[tr]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        for _ in range(ep):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i+64]; opt.zero_grad()
                F.mse_loss(net(A[b]), B[b]).backward(); opt.step()
            s.step()
        net.eval()
        with torch.no_grad():
            trn = float(F.mse_loss(net(A), B))
            P = net(torch.tensor(X[te], device=dev)).cpu().numpy().reshape(len(te), -1)
        return P * ys + ym, trn, sum(p.numel() for p in net.parameters())

    t0 = time.time()
    print(f"=== 1. 定版模型 {args.seeds} 种子(逐样本预测全部留存) ===")
    Ps, trns = [], []
    for s in range(args.seeds):
        P, trn, npar = fit(s)
        Ps.append(P); trns.append(trn)
        e = float(np.mean(np.abs(cum(P) - ct) / np.abs(ct)))
        print(f"  seed {s}: test {e:.4%}  train_mse {trn:.2e}   ({(time.time()-t0)/60:.0f}min)",
              flush=True)
    CP = np.stack([cum(P) for P in Ps])                    # (seeds, 400)
    per_seed = np.abs(CP - ct[None]) / np.abs(ct)[None]
    seed_means = per_seed.mean(1)

    # ---- 2. test 采样噪声(自举) ----
    rng = np.random.default_rng(0)
    boot = np.array([per_seed.mean(0)[rng.integers(0, len(te), len(te))].mean()
                     for _ in range(4000)])
    se_test = float(boot.std(ddof=1))
    print(f"\n=== 2. 噪声地板 ===")
    print(f"  test 采样噪声(400 条自举, 标准误)      {se_test*100:.4f} pp")
    print(f"  种子噪声(同配置 {args.seeds} 种子 sd)      {seed_means.std(ddof=1)*100:.4f} pp")
    thresh = 2 * np.sqrt(se_test ** 2 + (seed_means.std(ddof=1) / np.sqrt(6)) ** 2) * 100
    print(f"  🔴 可分辨阈值(2σ, 6 种子)              {thresh:.4f} pp")
    print(f"     → 小于这个值的架构差异,在当前规模下**测不出来**")

    # ---- 3. 偏差-方差分解 ----
    signed = (CP - ct[None]) / np.abs(ct)[None]
    bias = signed.mean(0)                                   # 所有种子共有的系统偏差
    var = signed.var(0)
    b2, v = float((bias ** 2).mean()), float(var.mean())
    print(f"\n=== 3. 偏差-方差分解(相对误差平方) ===")
    print(f"  系统偏差²  {b2:.3e}  占 {b2/(b2+v):.1%}   ← 可靠更好的模型/更多数据消掉")
    print(f"  种子方差   {v:.3e}  占 {v/(b2+v):.1%}   ← 只能靠集成消掉")
    print(f"  → {'系统性余量仍占主导,不是纯噪声限制' if b2/(b2+v) > 0.5 else '已被种子方差主导,单模型接近极限'}")

    # ---- 4. 过参数化探底 ----
    print(f"\n=== 4. 过参数化探底(训练集拟到近零,看留出停在哪) ===")
    ov = {}
    for d, ly, ep in ((128, 3, 600), (256, 4, 1200)):
        P, trn, npar = fit(100, d=d, layers=ly, epochs=ep)
        e = float(np.mean(np.abs(cum(P) - ct) / np.abs(ct)))
        ov[f"d{d}_L{ly}_ep{ep}"] = {"test": e, "train_mse": trn, "params": npar}
        print(f"  d={d} L={ly} ep={ep}  参数 {npar:>9,d}  train_mse {trn:.2e}  test {e:.4%}"
              f"   ({(time.time()-t0)/60:.0f}min)", flush=True)

    ens = float(np.mean(np.abs(CP.mean(0) - ct) / np.abs(ct)))
    print(f"\n=== 5. 汇总 ===")
    print(f"  单模型均值        {seed_means.mean():.4%} ± {seed_means.std(ddof=1):.4%}")
    print(f"  {args.seeds}-模型集成(方差地板) {ens:.4%}")
    print(f"  可分辨阈值        {thresh:.4f} pp")
    print(f"  → 已知架构差异: 轴向 −定版 = +0.014pp, Autoformer −定版 = +0.074pp")
    print(f"    {'轴向与定版的差异低于可分辨阈值,不能说谁更好' if 0.014 < thresh else '轴向差异可分辨'}")
    json.dump({"n_train": len(tr), "seeds": args.seeds,
               "seed_means": seed_means.tolist(),
               "se_test_pp": se_test * 100, "seed_sd_pp": float(seed_means.std(ddof=1) * 100),
               "resolvable_threshold_pp": float(thresh),
               "bias2_frac": float(b2 / (b2 + v)), "var_frac": float(v / (b2 + v)),
               "overparam": ov, "ensemble": ens,
               "note": ("回答'架构差异测不测得出来'。可分辨阈值 = 2σ,"
                        "综合 test 采样噪声与 6 种子下的种子噪声。")},
              open(OUT / "ceil.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/ceil.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
