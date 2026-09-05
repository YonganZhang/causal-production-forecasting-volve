#!/usr/bin/env python3
"""代理模型总排名：所有模型放进**同一个协议、同一个指标**里重跑一次。

## 为什么必须重跑,不能直接引用 fc_benchmark 的表

`fc_benchmark.json` 与 `fc_final.json` **不可直接比较**,并排放就是范畴错误:

| | fc_benchmark | fc_final |
|---|---|---|
| 指标 | `relerr_median` | `relerr_mean` |
| 切分 | 两分(1600/400),超参未在独立集上选 | 三分,test 封存,超参只在 val 上选 |
| 重复 | 单种子 | 多种子 + 集成 |

本项目已因"拿不同口径的数并排"撤稿过一次(撤稿 #7:范畴错误)。所以本脚本把
**每个模型**都放进 fc_final 的协议里:同一个 split_seed、同一批封存的 400 条 test、
同一套指标(mean / median / p90 / R²),随机性模型报多次重复的均值±标准差。

## 口径声明

- **死井剔除**:22 口生产井里 11 口油率水率恒为 0,全部模型统一剔除。
  累计产油是求和,死井贡献 0,所以指标口径不变。
- **PCA 下界**:把真值投影到前 k 个主成分再还原的误差 —— 任何靠 PCA 降维的模型
  都不可能优于它。这是**结构性下界**,不是一个模型。
- 🔴 **Chronos-2 信息不对等**:它额外看了预测段前 16 个真实时刻的曲线,
  其余模型只看 24 维注水方案。它的数**不参与排名**,只作脚注。

用法:
    python fc_rank.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG
import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_rank"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=4, help="随机性模型的重复次数")
    ap.add_argument("--n-pca", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum())
    n = len(TH)
    # 🔴 与 fc_final 完全同一个切分,否则不可比
    rng = np.random.default_rng(args.split_seed); idx = rng.permutation(n)
    te, va, tr = idx[:400], idx[400:600], idx[600:]
    trva = np.concatenate([tr, va])
    Yf = Y.reshape(n, -1)

    def cumoil(P):
        return np.trapezoid(P.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)

    ct = cumoil(Yf[te])

    def metrics(P):
        cp = cumoil(P)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        return {"mean": float(rel.mean()), "median": float(np.median(rel)),
                "p90": float(np.quantile(rel, 0.9)),
                "r2": float(1 - ((cp - ct) ** 2).sum() / ((ct - ct.mean()) ** 2).sum())}

    # 训练集统计量只用 train+val(test 全程不参与任何拟合/选择)
    ym, ys = Yf[trva].mean(0), Yf[trva].std(0) + 1e-8
    xm, xs = TH[trva].mean(0), TH[trva].std(0) + 1e-8
    Xall = ((TH - xm) / xs).astype(np.float32)
    Xtr, Xte = Xall[trva], Xall[te]

    from sklearn.decomposition import PCA
    pca = PCA(args.n_pca, random_state=0).fit((Yf[trva] - ym) / ys)
    Ztr = pca.transform((Yf[trva] - ym) / ys)
    back = lambda Z: pca.inverse_transform(Z) * ys + ym
    print(f"样本 {n}  拟合 {len(trva)}  封存 test {len(te)}  生产井 {nw}(剔 {int((~live).sum())} 口恒零)")
    print(f"PCA {args.n_pca} 主成分解释方差 {pca.explained_variance_ratio_.sum():.4%}\n")

    R = {}

    def run(name, fn, reps=1, note=None):
        ms, t0 = [], time.time()
        for r in range(reps):
            P, infer = fn(r)
            ms.append(metrics(P))
        a = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        a["mean_sd"] = float(np.std([m["mean"] for m in ms], ddof=1)) if reps > 1 else 0.0
        a["reps"] = reps; a["sec"] = round((time.time() - t0) / reps, 1)
        a["infer_ms"] = round(infer * 1000 / len(te), 3)
        if note:
            a["note"] = note
        R[name] = a
        print(f"  {name:26s} 平均 {a['mean']:.3%}  中位 {a['median']:.3%}  "
              f"p90 {a['p90']:.3%}  R² {a['r2']:>8.4f}  {a['sec']:>6.1f}s", flush=True)

    # ---- 参照物(不是模型) ----
    run("平凡基线(训练均值)", lambda r: (np.tile(Yf[trva].mean(0), (len(te), 1)), 0.0))
    run("PCA 结构性下界", lambda r: (back(pca.transform((Yf[te] - ym) / ys)), 0.0),
        note="真值投影再还原,任何 PCA 降维模型都不可能优于它")

    # ---- 传统机器学习 ----
    def m_rf(r):
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(300, n_jobs=8, random_state=r).fit(Xtr, Ztr)
        t = time.time(); P = back(m.predict(Xte)); return P, time.time() - t
    run("随机森林", m_rf, reps=2)

    def m_gbdt(r):
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        Zp = np.zeros((len(Xte), args.n_pca))
        t0 = time.time()
        for k in range(16):
            Zp[:, k] = H(random_state=r).fit(Xtr, Ztr[:, k]).predict(Xte)
        return back(Zp), time.time() - t0
    run("GBDT(前16主成分)", m_gbdt)

    def m_svr(r):
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR
        m = MultiOutputRegressor(SVR(C=10.0, epsilon=0.01), n_jobs=8).fit(Xtr, Ztr[:, :16])
        t = time.time(); Zp = np.zeros((len(Xte), args.n_pca)); Zp[:, :16] = m.predict(Xte)
        return back(Zp), time.time() - t
    run("SVR(前16主成分)", m_svr)

    def m_gp(r):
        from sklearn.gaussian_process import GaussianProcessRegressor as G
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
        k = C(1.0) * RBF(np.ones(Xtr.shape[1])) + WhiteKernel(1e-3)
        sub = slice(None, min(800, len(Xtr)))
        m = G(kernel=k, normalize_y=True, random_state=0).fit(Xtr[sub], Ztr[sub, :16])
        t = time.time(); Zp = np.zeros((len(Xte), args.n_pca)); Zp[:, :16] = m.predict(Xte)
        return back(Zp), time.time() - t
    run("高斯过程(前16主成分)", m_gp, note="O(n³),只用 800 条")

    # ---- 神经网络 ----
    import torch
    from fc_improve2 import TF
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    Yn = ((Yf[trva] - ym) / ys).astype(np.float32)

    def m_mlp(r):
        torch.manual_seed(r)
        net = FD.build_net(TH.shape[1], Yf.shape[1], 768, 5).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        A = torch.tensor(Xtr, device=dev); B = torch.tensor(Yn, device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
        for _ in range(600):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i + 64]; opt.zero_grad()
                torch.nn.functional.mse_loss(net(A[b]), B[b]).backward(); opt.step()
            s.step()
        net.eval(); t = time.time()
        with torch.no_grad():
            P = net(torch.tensor(Xte, device=dev)).cpu().numpy() * ys + ym
        return P, time.time() - t
    run("MLP(BP 神经网络)", m_mlp, reps=args.reps)

    def tf_fit(seed, d=128, ep=600, lr=3e-3):
        torch.manual_seed(seed)
        net = TF(Xtr.shape[1], nw * NPH, NT, d, 4, 3, 256).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(Xtr, device=dev)
        B = torch.tensor(Yn.reshape(-1, nw * NPH, NT), device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        for _ in range(ep):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i + 64]; opt.zero_grad()
                torch.nn.functional.mse_loss(net(A[b]), B[b]).backward(); opt.step()
            s.step()
        return net.eval()

    def m_tf1(r):
        net = tf_fit(r); t = time.time()
        with torch.no_grad():
            P = net(torch.tensor(Xte, device=dev)).cpu().numpy().reshape(len(te), -1) * ys + ym
        return P, time.time() - t
    run("Transformer(单模型)", m_tf1, reps=args.reps)

    def m_tf5(r):
        nets = [tf_fit(1000 * r + s) for s in range(5)]
        t = time.time()
        with torch.no_grad():
            x = torch.tensor(Xte, device=dev)
            P = np.mean([nn(x).cpu().numpy().reshape(len(te), -1) for nn in nets], 0) * ys + ym
        return P, time.time() - t
    run("🏆 定版 Transformer+集成5", m_tf5, reps=args.reps,
        note="lr=3e-3/ep=600/d=128, 超参在 val 上选定, test 只评一次")

    # ---- 排名 ----
    REF = {"平凡基线(训练均值)", "PCA 结构性下界"}
    rank = sorted([k for k in R if k not in REF], key=lambda k: R[k]["mean"])
    print(f"\n=== 排名(封存 test, 平均相对误差) ===")
    print(f"  {'#':>2} {'模型':26s}{'平均误差':>10s}{'R²':>9s}{'vs 冠军':>9s}")
    top = R[rank[0]]["mean"]
    for i, k in enumerate(rank, 1):
        print(f"  {i:>2} {k:26s}{R[k]['mean']:>10.3%}{R[k]['r2']:>9.4f}"
              f"{R[k]['mean']/top:>8.2f}×")
    a, b = R[rank[0]], R[rank[1]]
    triv = R["平凡基线(训练均值)"]["mean"]
    print(f"\n冠军 {rank[0]}  {a['mean']:.3%} ± {a['mean_sd']:.3%}")
    print(f"亚军 {rank[1]}  {b['mean']:.3%}")
    print(f"  → 误差是亚军的 1/{b['mean']/a['mean']:.2f}, 即降低 {1-a['mean']/b['mean']:.1%}")
    print(f"  → 相对平凡基线({triv:.3%}) 误差降低 {triv/a['mean']:.1f}×")
    print(f"  → 距 PCA 结构性下界({R['PCA 结构性下界']['mean']:.3%}) 还差 "
          f"{a['mean']/R['PCA 结构性下界']['mean']:.1f}×")

    json.dump({"n": n, "n_fit": len(trva), "n_test": len(te), "n_live_wells": nw,
               "n_pca": args.n_pca, "pca_explained": float(pca.explained_variance_ratio_.sum()),
               "results": R, "ranking": rank,
               "protocol": ("与 fc_final 完全同一切分(split_seed=0, test 400 封存)、"
                            "同一指标(mean/median/p90/R²)。所有模型只在 train+val 上拟合。"
                            "Chronos-2 因信息不对等未纳入。")},
              open(OUT / "rank.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/rank.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
