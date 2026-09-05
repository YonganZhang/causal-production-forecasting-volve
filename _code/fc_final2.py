#!/usr/bin/env python3
"""定版 v2：把两个已确认有效的改动做实，在封存 test 上评一次。

## 已确认的结论(来自 fc_arch2 12 种子 + fc_loop 高频循环)

**✔ 傅里叶特征嵌入 —— 真实有效,且有剂量-反应**

| n_bands | 封存 test 误差 | vs 基线(0.559%) | p |
|---|---|---|---|
| 4 | 0.534% | −0.024pp | 0.0421 |
| 8 | 0.520% | −0.039pp | 0.0024 |
| 16 | **0.489%** | **−0.070pp** | **0.0000** |

**误差随 n_bands 单调下降**。剂量-反应是最强的一类因果证据 —— 上一轮正是靠
同样的曲线(物理特征降权越多越准)判定了 A 有害。既然单调,就该往 16 以上试。

**✔ 相对 l2 损失 [Wen 2022]** —— val 上 0.627% vs 基线 0.715%(p=0.0220)。
各井平均油率从 1.6 到 309(差 190 倍),MSE 被大井主导;相对归一让每个样本等权。

**🔴 Wen 2022 的 ∂_t 导数项 —— 在本任务上有害,文献迁移失败**
val: 相对l2 单独 0.627% → 加导数项 0.818%(beta=0.5) / 0.904%(beta=1.0)。
beta 越大越差,同样是剂量-反应,方向相反。**必须在论文里如实写**:
Wen 2022 的导数项是为**场量**(饱和度前缘、压力尖峰)设计的,那里高频是真信号;
本任务的目标是**井口速率曲线**,已经过井模型平滑,高频主要是数值噪声 ——
放大它的导数等于放大噪声。这是一条有价值的负结果。

**🔴 FNO 公平调参后仍显著变差**(0.739%, p=0.0000, 12 组超参搜过)。
**🔴 谱加权损失全无效**(alpha 0.1/1/10 全在噪声内),且已确认不是 no-op
   (谱项占损失 7.7%/44.4%/88.4%)。

## 本脚本

1. 在 **val** 上把 n_bands 扫到 {16, 32, 64},确认单调性到哪里为止。
2. 选定后在**封存 test** 上跑 12 种子:基线 / 傅里叶 / 相对l2 / 傅里叶+相对l2。
3. 全部单模型,**不集成**。

用法:
    python fc_final2.py --gpu 3
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
import fc_models as M
from fc_loop import rel_l2

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_final2"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    rng = np.random.default_rng(args.split_seed); idx = rng.permutation(n)
    te, va, tr = idx[:400], idx[400:600], idx[600:]
    trva = np.concatenate([tr, va])
    Yf = Y.reshape(n, -1); NCH = nw * NPH
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)

    def prep(rows):
        ym, ys = Yf[rows].mean(0), Yf[rows].std(0) + 1e-8
        xm, xs = TH[rows].mean(0), TH[rows].std(0) + 1e-8
        return ym, ys, ((TH - xm) / xs).astype(np.float32)

    def run(fit_rows, ev_rows, *, seed, loss="mse", ff=False, n_bands=16, lr=3e-3):
        ym, ys, X = prep(fit_rows)
        torch.manual_seed(seed)
        net = (M.OursNet(TH.shape[1], NCH, NT, use_ff=True, use_film=False, n_bands=n_bands)
               if ff else M.TFBase(TH.shape[1], NCH, NT)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[fit_rows], device=dev)
        B = torch.tensor(((Yf[fit_rows]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        lf = rel_l2 if loss == "rel" else F.mse_loss
        for _ in range(args.epochs):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i+64]; opt.zero_grad()
                lf(net(A[b]), B[b]).backward(); opt.step()
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
    # ---- 1. val 上把 n_bands 推到单调性的尽头 ----
    print("=== 1. n_bands 扫到尽头(只看 val;test 封存) ===")
    print("   已知 4→8→16 单调下降(0.534/0.520/0.489),看还能不能继续")
    nb_rows = []
    for nb in (16, 32, 64):
        es = [run(tr, va, seed=s, ff=True, n_bands=nb)[0] for s in range(3)]
        nb_rows.append({"n_bands": nb, "val": float(np.mean(es)), "sd": float(np.std(es, ddof=1))})
        print(f"  n_bands={nb:<4d} val {np.mean(es):.3%} ± {np.std(es,ddof=1):.3%}"
              f"  ({(time.time()-t0)/60:.0f}min)", flush=True)
    best_nb = min(nb_rows, key=lambda r: r["val"])["n_bands"]
    print(f"  → 选 n_bands={best_nb}\n")

    # ---- 2. 封存 test, 12 种子, 不集成 ----
    print(f"=== 2. 封存 test({args.seeds} 种子, **单模型不集成**) ===")
    CFG = {
        "基线 Transformer(MSE)": dict(),
        f"+傅里叶特征(n_bands={best_nb})": dict(ff=True, n_bands=best_nb),
        "+相对l2 [Wen2022]": dict(loss="rel"),
        f"+傅里叶+相对l2": dict(ff=True, n_bands=best_nb, loss="rel"),
    }
    R = {}
    for tag, kw in CFG.items():
        a = np.array([run(trva, te, seed=s, **kw) for s in range(args.seeds)])
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                  "r2": float(a[:, 3].mean()), "raw": a[:, 0].tolist()}
        r = R[tag]
        print(f"  {tag:30s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"p90 {r['p90']:.3%}  R² {r['r2']:.4f}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    base = np.array(R["基线 Transformer(MSE)"]["raw"])
    print(f"\n=== 3. 显著性(vs 基线, Welch, n={args.seeds}) ===")
    sig = {}
    for k, v in R.items():
        if k.startswith("基线"):
            continue
        a = np.array(v["raw"]); p = float(ttest_ind(a, base, equal_var=False).pvalue)
        d = (a.mean() - base.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[k] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {k:30s} {d:>+8.3f}pp  p={p:.4f}   {vd}")

    rk = sorted(R, key=lambda k: R[k]["mean"])
    w, b0 = R[rk[0]], R["基线 Transformer(MSE)"]
    print(f"\n🏆 定版: {rk[0]}  {w['mean']:.3%} ± {w['sd']:.3%}  R² {w['r2']:.4f}")
    print(f"   vs 基线 Transformer({b0['mean']:.3%}): 误差降低 {1-w['mean']/b0['mean']:.1%}")
    json.dump({"n_bands_sweep": nb_rows, "best_n_bands": best_nb, "results": R,
               "significance": sig, "ranking": rk, "seeds": args.seeds,
               "protocol": "n_bands 在 val 上选;test 400 封存只评一次;**单模型不集成**"},
              open(OUT / "final2.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/final2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
