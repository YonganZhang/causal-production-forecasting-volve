#!/usr/bin/env python3
"""第二轮：把第一轮的三个悬案查清。不是加新模块,是把已有结论做实。

## 第一轮的诚实结论(fc_arch.json)

| 改动 | 结果 | 判定 |
|---|---|---|
| 傅里叶特征嵌入 | 0.505% vs 基线 0.547% | −0.043pp, **p=0.0801 不显著** |
| FiLM 条件调制 | 1.030% | 🔴 **显著变差** +0.482pp |
| 谱加权损失 | 0.561% | — 噪声内 p=0.4748 |
| Ours(三个全开) | 0.821% | 🔴 **显著变差** —— 我设计的模型输给了基线 |
| FNO(算子学习) | 0.755% | 🔴 **显著变差** p=0.0001 |
| DeepONet | 1.240% | 🔴 显著变差 |
| Autoformer | 0.578% | — 噪声内 p=0.5066 |

**两个我自己的判断被否掉了**:
1. "算子学习模型(FNO/DeepONet)该赢" —— 实测**都输**。
2. "Autoformer 架构错配所以该差" —— 实测**与基线无差异**,错配论断**不被数字支持**。

## 本轮查三件事

**Q1 傅里叶特征到底有没有效?** p=0.0801 是 4 种子的欠功效结果,不能下结论。
用 12 种子重测,并扫 n_bands ∈ {4,8,16}。这是唯一还有希望的模块。

**Q2 FNO 是不是被不公平地判了死刑?** 第一轮只扫了 lr,width/modes/layers 全用默认。
FNO 是本领域代理模型的主流骨干,用默认超参判它输不公平。本轮在 val 上做真实搜索。
⚠️ 另有一个理论疑点要一并检验:FFT 隐含**周期边界**,而本任务 t=0 被 RESTART
   状态硬钉住、末端自由 —— 周期性假设被违反。若 FNO 调参后仍输,这是候选解释。

**Q3 谱损失是不是 no-op?** 上一次单调约束就栽在"惩罚项量级 7e-11"。
本轮扫 alpha ∈ {0.1, 1, 10} 并记录谱项占总损失的实际比例。

用法:
    python fc_arch2.py --gpu 4
"""
from __future__ import annotations

import argparse
import itertools
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

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_arch2"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
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

    def sc(P, rows):
        ct, cp = cum(Yf[rows]), cum(P)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        return float(rel.mean()), float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum())

    def run(kind, fit_rows, ev_rows, *, seed, lr, spec=0.0, **kw):
        ym, ys, X = prep(fit_rows)
        torch.manual_seed(seed)
        net = ({"TFBase": M.TFBase, "FNO1d": M.FNO1d, "Ours": M.OursNet}[kind]
               )(TH.shape[1], NCH, NT, **kw).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[fit_rows], device=dev)
        B = torch.tensor(((Yf[fit_rows]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        frac = 0.0
        for ep in range(args.epochs):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i+64]; opt.zero_grad()
                out = net(A[b]); mse = F.mse_loss(out, B[b])
                loss = M.spectral_loss(out, B[b], alpha=spec, beta=0.5) if spec > 0 else mse
                loss.backward(); opt.step()
                if ep == args.epochs - 1:
                    frac = float((loss - mse) / loss) if spec > 0 else 0.0
            s.step()
        net.eval()
        with torch.no_grad():
            P = net(torch.tensor(X[ev_rows], device=dev)).cpu().numpy().reshape(len(ev_rows), -1)
        return (*sc(P*ys+ym, ev_rows), frac)

    R, t0 = {}, time.time()

    def bench(tag, kind, *, seeds, lr=3e-3, spec=0.0, **kw):
        rs = [run(kind, trva, te, seed=s, lr=lr, spec=spec, **kw) for s in range(seeds)]
        a = np.array(rs)
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "r2": float(a[:, 1].mean()), "spec_frac": float(a[:, 2].mean()),
                  "seeds": seeds, "raw": a[:, 0].tolist(), "lr": lr}
        r = R[tag]
        print(f"  {tag:30s} {r['mean']:.3%} ± {r['sd']:.3%}  R² {r['r2']:.4f}"
              + (f"  谱项占损失 {r['spec_frac']:.1%}" if spec > 0 else "")
              + f"  ({(time.time()-t0)/60:.0f}min)", flush=True)

    # ---------- Q1 傅里叶特征:12 种子 ----------
    print(f"=== Q1 傅里叶特征({args.seeds} 种子,第一轮 4 种子给出 p=0.0801 欠功效) ===")
    bench("基线 Transformer", "TFBase", seeds=args.seeds)
    for nb in (4, 8, 16):
        bench(f"傅里叶特征 n_bands={nb}", "Ours", seeds=args.seeds,
              use_ff=True, use_film=False, n_bands=nb)

    # ---------- Q2 FNO 公平调参 ----------
    print(f"\n=== Q2 FNO 公平调参(第一轮只扫了 lr, width/modes/layers 全默认) ===")
    best, rows = None, []
    for w, md, ly, lr in itertools.product([64, 128], [8, 16, 20], [4], [1e-3, 3e-3]):
        e, r2, _ = run("FNO1d", tr, va, seed=0, lr=lr, width=w, modes=md, layers=ly)
        rows.append({"width": w, "modes": md, "layers": ly, "lr": lr, "val": e})
        print(f"  FNO width={w:<4d} modes={md:<3d} L={ly} lr={lr:<6g} val {e:.3%}"
              f"  ({(time.time()-t0)/60:.0f}min)", flush=True)
        if best is None or e < best["val"]:
            best = rows[-1]
    print(f"  → FNO 最佳 {best}")
    bench("FNO(调参后)", "FNO1d", seeds=4, lr=best["lr"],
          width=best["width"], modes=best["modes"], layers=best["layers"])

    # ---------- Q3 谱损失量级 ----------
    print(f"\n=== Q3 谱损失是不是 no-op(上次单调约束就栽在惩罚项 7e-11) ===")
    for al in (0.1, 1.0, 10.0):
        bench(f"谱损失 alpha={al}", "Ours", seeds=4, use_ff=False, use_film=False, spec=al)

    # ---------- 判定 ----------
    from scipy.stats import ttest_ind
    base = np.array(R["基线 Transformer"]["raw"])
    print(f"\n=== 判定(vs 基线 Transformer, Welch) ===")
    sig = {}
    for k, v in R.items():
        if k == "基线 Transformer":
            continue
        a = np.array(v["raw"]); p = float(ttest_ind(a, base, equal_var=False).pvalue)
        d = (a.mean() - base.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[k] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {k:30s} {d:>+8.3f}pp  p={p:.4f}   {vd}")
    rank = sorted(R, key=lambda k: R[k]["mean"])
    print(f"\n排名: " + " > ".join(f"{k}({R[k]['mean']:.3%})" for k in rank))
    json.dump({"results": R, "significance": sig, "fno_grid": rows, "fno_best": best,
               "ranking": rank, "seeds": args.seeds},
              open(OUT / "arch2.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/arch2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
