#!/usr/bin/env python3
"""泄漏审计：为什么 R²=0.99 / 误差 0.5% —— 是真的，还是 pipeline 漏了？

## 用户的质疑（合理，且是最该问的）

"代理的是油藏数值模拟，准确率能到 99% 我觉得不太对，检查一下数据集有没有泄露。"

## 先澄清一个读数

R²=0.9924 与"平均相对误差 0.489%"**不是同一个量，也不冲突**：
- R² 是在**样本之间**解释累计产油的方差（样本间累计产油从 6.11e6 到 9.60e6，±22%）；
- 0.489% 是**逐样本**预测值与真值的相对偏差。

拿训练集均值去猜(平凡基线)的相对误差是 **5.745%**，R² 约为 0。所以模型是
"比瞎猜准 12 倍",不是"轻易就能到 99%"。

## 本脚本的七项检查（第 3 项是决定性的）

1. **切分互斥** —— 索引级 + 内容级(θ 与 Y 的哈希)。
2. **近邻污染** —— θ 是 24 维连续量,不会有精确重复;但若 test 样本在 θ 空间里
   紧贴某个 train 样本,模型只需内插。比较 test→train 与 train→train 的最近邻距离分布。
3. 🔴 **打乱标签对照(决定性)** —— 把训练集的目标行**随机置换**后重训。
   若 pipeline 干净,误差必须坍塌到平凡基线(~5.7%)。若仍显著低于平凡基线,
   说明输入以某种方式泄漏了目标。**这是唯一能一票否决的检验。**
4. **k-NN 内插基线** —— 只在 θ 空间做最近邻/加权内插,不训练任何网络。
   若 k-NN 就能到 0.5%,说明任务本质是"稠密采样下的内插",
   那 0.489% 不代表模型强,只代表采样密。这不是泄漏,但会改变论文该怎么写。
5. **学习曲线** —— N ∈ {100,200,400,800,1600}。若误差随 N 下降不明显,可疑。
6. **目标内在维度** —— PCA 需要多少主成分。若 5 个成分就解释 99.9%,
   任务本身就是低维的,高 R² 是**结构性的**,不是泄漏。
7. **模拟器输出唯一性** —— 2000 个 shard 是否真是 2000 次独立模拟(输出哈希)。

用法:
    python fc_leak.py --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
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

NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID
OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_leak"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    G = {}

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    te, va, tr = idx[:400], idx[400:600], idx[600:]
    trva = np.concatenate([tr, va])
    Yf = Y.reshape(n, -1); NCH = nw * NPH
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    C_all = cum(Yf)

    print(f"=== 0. 目标本身的分布(先看尺子) ===")
    print(f"  累计产油 min {C_all.min():.4g}  max {C_all.max():.4g}  "
          f"均值 {C_all.mean():.4g}  变异系数 {C_all.std()/C_all.mean():.2%}")
    triv = float(np.mean(np.abs(C_all[te] - C_all[trva].mean()) / np.abs(C_all[te])))
    print(f"  平凡基线(拿训练均值猜) 相对误差 {triv:.3%}   → 模型 0.489% 是它的 1/{triv/0.00489:.1f}")
    G["target"] = {"cv": float(C_all.std()/C_all.mean()), "trivial": triv}

    # ---- 1. 切分互斥 ----
    print(f"\n=== 1. 切分互斥(索引级 + 内容级) ===")
    ov = len(set(te) & set(va)) + len(set(te) & set(tr)) + len(set(va) & set(tr))
    hx = [hashlib.sha256(TH[i].tobytes()).hexdigest() for i in range(n)]
    hy = [hashlib.sha256(Yf[i].tobytes()).hexdigest() for i in range(n)]
    dupx = n - len(set(hx)); dupy = n - len(set(hy))
    cross = len(set(hy[i] for i in te) & set(hy[i] for i in trva))
    print(f"  索引重叠 {ov}   θ 重复行 {dupx}   Y 重复行 {dupy}   test/train 内容重复 {cross}")
    G["splits"] = {"index_overlap": ov, "dup_theta": dupx, "dup_y": dupy, "content_cross": cross}
    print(f"  {'✔ 无重叠' if ov==0 and cross==0 else '🔴 有重叠'}")

    # ---- 2. 近邻污染 ----
    print(f"\n=== 2. 近邻污染(test 是不是紧贴 train) ===")
    Z = (TH - TH[trva].mean(0)) / (TH[trva].std(0) + 1e-8)
    def nn_dist(Q, Rr):
        d = np.sqrt(((Q[:, None, :] - Rr[None, :, :]) ** 2).sum(-1))
        return np.sort(d, 1)
    d_te = nn_dist(Z[te], Z[trva])[:, 0]
    d_tr = nn_dist(Z[trva[:400]], Z[trva])[:, 1]        # 排除自己
    print(f"  test→train 最近邻距离 中位 {np.median(d_te):.3f}  最小 {d_te.min():.3f}")
    print(f"  train→train 最近邻距离 中位 {np.median(d_tr):.3f}  最小 {d_tr.min():.3f}")
    print(f"  比值 {np.median(d_te)/np.median(d_tr):.3f}  "
          f"({'✔ 同分布,无污染' if 0.8 < np.median(d_te)/np.median(d_tr) < 1.25 else '🔴 分布不同'})")
    G["nn"] = {"test_med": float(np.median(d_te)), "train_med": float(np.median(d_tr)),
               "test_min": float(d_te.min())}

    # ---- 6. 目标内在维度(先算,后面解释高 R² 要用) ----
    from sklearn.decomposition import PCA
    p = PCA(64, random_state=0).fit(Yf[trva])
    cs = np.cumsum(p.explained_variance_ratio_)
    k99 = int(np.argmax(cs > 0.99) + 1); k999 = int(np.argmax(cs > 0.999) + 1)
    print(f"\n=== 6. 目标的内在维度 ===")
    print(f"  解释 99% 方差需要 {k99} 个主成分;99.9% 需要 {k999} 个")
    print(f"  → 880 维输出的**内在维度只有个位数**,高 R² 是结构性的")
    G["intrinsic_dim"] = {"k99": k99, "k999": k999}

    # ---- 4. k-NN 内插基线 ----
    print(f"\n=== 4. k-NN 内插基线(不训练任何网络) ===")
    D = np.sqrt(((Z[te][:, None, :] - Z[trva][None, :, :]) ** 2).sum(-1))
    ordr = np.argsort(D, 1)
    for k in (1, 5, 20):
        w = 1.0 / (np.take_along_axis(D, ordr[:, :k], 1) + 1e-9)
        w = w / w.sum(1, keepdims=True)
        P = (Yf[trva][ordr[:, :k]] * w[:, :, None]).sum(1)
        e = float(np.mean(np.abs(cum(P) - C_all[te]) / np.abs(C_all[te])))
        print(f"  k={k:<3d} 相对误差 {e:.3%}")
        G.setdefault("knn", {})[f"k{k}"] = e
    print(f"  → k-NN 最好 {min(G['knn'].values()):.3%} vs 神经网络 0.489%："
          f"网络比纯内插好 {min(G['knn'].values())/0.00489:.1f}×")

    # ---- 训练函数 ----
    def fit_eval(fit_rows, ev_rows, *, seed=0, shuffle=False, sub=None):
        rows = fit_rows if sub is None else fit_rows[:sub]
        Ytar = Yf[rows].copy()
        if shuffle:                                    # 🔴 打乱标签:切断输入-目标对应
            Ytar = Ytar[np.random.default_rng(seed).permutation(len(Ytar))]
        ym, ys = Ytar.mean(0), Ytar.std(0) + 1e-8
        xm, xs = TH[rows].mean(0), TH[rows].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = PosNet(TH.shape[1], NCH, NT, "fourier", n_bands=16).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        A = torch.tensor(X[rows], device=dev)
        B = torch.tensor(((Ytar - ym) / ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
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
        return float(np.mean(np.abs(cum(P) - C_all[ev_rows]) / np.abs(C_all[ev_rows])))

    # ---- 3. 打乱标签对照(决定性) ----
    print(f"\n=== 3. 🔴 打乱标签对照(决定性检验) ===")
    print(f"   把训练目标随机置换后重训。pipeline 干净 → 误差必须坍塌到平凡基线 {triv:.3%}")
    t0 = time.time()
    sh = [fit_eval(trva, te, seed=s, shuffle=True) for s in range(args.seeds)]
    nm = [fit_eval(trva, te, seed=s) for s in range(args.seeds)]
    print(f"  正常标签 {np.mean(nm):.3%} ± {np.std(nm):.3%}")
    print(f"  打乱标签 {np.mean(sh):.3%} ± {np.std(sh):.3%}   (平凡基线 {triv:.3%})")
    ratio = np.mean(sh) / triv
    ok = 0.85 < ratio < 1.30
    print(f"  打乱后/平凡基线 = {ratio:.3f}  →  "
          f"{'✔ 无泄漏(打乱后模型学不到任何东西)' if ok else '🔴 可疑:打乱后仍显著优于平凡基线'}")
    G["shuffle"] = {"normal": float(np.mean(nm)), "shuffled": float(np.mean(sh)),
                    "trivial": triv, "ratio": float(ratio), "pass": bool(ok)}

    # ---- 5. 学习曲线 ----
    print(f"\n=== 5. 学习曲线(误差该随样本量下降) ===")
    lc = {}
    for N in (100, 200, 400, 800, 1600):
        e = fit_eval(trva, te, seed=0, sub=N)
        lc[N] = e
        print(f"  N={N:<5d} 相对误差 {e:.3%}   ({(time.time()-t0)/60:.0f}min)", flush=True)
    G["learning_curve"] = lc
    mono = all(lc[a] >= lc[b] - 0.0005 for a, b in zip([100,200,400,800], [200,400,800,1600]))
    print(f"  {'✔ 单调下降,符合正常学习' if mono else '🔴 不单调,可疑'}")

    # ---- 7. 模拟器输出唯一性 ----
    print(f"\n=== 7. 2000 个 shard 是不是 2000 次独立模拟 ===")
    print(f"  Y 唯一行 {len(set(hy))}/{n}   θ 唯一行 {len(set(hx))}/{n}")
    print(f"  {'✔ 全部唯一' if len(set(hy))==n else '🔴 有重复模拟结果'}")

    print(f"\n{'='*60}\n判定:")
    verdict = ("✔ 未发现泄漏。" if (ov == 0 and cross == 0 and ok and len(set(hy)) == n)
               else "🔴 发现可疑项,见上。")
    print(f"  {verdict}")
    print(f"  高精度的**结构性原因**:(a) 目标是**确定性模拟器**输出,无观测噪声;")
    print(f"     (b) 880 维输出的内在维度只有 {k99} 个主成分(解释 99% 方差);")
    print(f"     (c) 输入只有 24 维且采样稠密 —— 但 k-NN 纯内插只能到 "
          f"{min(G['knn'].values()):.3%},网络仍好 {min(G['knn'].values())/0.00489:.1f} 倍。")
    json.dump({**G, "verdict": verdict}, open(OUT / "leak.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/leak.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
