#!/usr/bin/env python3
"""架构对照 + 模块消融：在 fc_final 的协议下比骨干，再逐项拆我们加的模块。

## 纪律

- **不做集成**。集成不是模型贡献,写不进论文。全部单模型,多种子报均值±标准差。
- 每个骨干**各自在 val 上选学习率**,避免"拿基线的超参去跑新架构"这种隐性不公平。
- test 400 条全程封存,只在最后评一次。
- 与 `fc_rank` 同一个 `split_seed`,所以传统模型(随机森林/GBDT/GP/SVR/MLP)的数
  可以直接并列,不必重跑。

## 改动依据(全部来自 fc_gap.py 实测)

误差 98.8% 是**形状**错;误差能量高频占比 18.08% 而信号只占 4.63%;
逐模态信噪比从 15559 衰减到 896(**17×**)。→ 病灶是**谱偏差**。
物理违背为 0、水突破误差 0 天、无异方差 → 这三条路线**被实测排除**,不做。

用法:
    python fc_arch.py --gpu 3
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
import norne_bulk as NB
import fc_models as M

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_arch"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=4)
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
    Yf = Y.reshape(n, -1)
    NCH = nw * NPH
    print(f"样本 {n}  train {len(tr)} / val {len(va)} / test {len(te)}(封存)  "
          f"输出 {nw}井×{NPH}相×{NT}时刻\n")

    def prep(rows):
        ym, ys = Yf[rows].mean(0), Yf[rows].std(0) + 1e-8
        xm, xs = TH[rows].mean(0), TH[rows].std(0) + 1e-8
        return ym, ys, ((TH - xm) / xs).astype(np.float32)

    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)

    def sc(P, rows):
        ct, cp = cum(Yf[rows]), cum(P)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        return (float(rel.mean()), float(np.median(rel)), float(np.quantile(rel, .9)),
                float(1 - ((cp - ct) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()))

    def build(kind, **kw):
        A = {"TFBase": M.TFBase, "Autoformer": M.Autoformer,
             "DeepONet": M.DeepONet, "FNO1d": M.FNO1d, "Ours": M.OursNet}[kind]
        return A(TH.shape[1], NCH, NT, **kw)

    def train_eval(kind, fit_rows, ev_rows, *, seed, lr, spec=0.0, **kw):
        ym, ys, X = prep(fit_rows)
        torch.manual_seed(seed)
        net = build(kind, **kw).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[fit_rows], device=dev)
        B = torch.tensor(((Yf[fit_rows] - ym) / ys).reshape(-1, NCH, NT).astype(np.float32),
                         device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i + 64]; opt.zero_grad()
                out = net(A[b])
                loss = (M.spectral_loss(out, B[b], alpha=spec, beta=0.5) if spec > 0
                        else F.mse_loss(out, B[b]))
                loss.backward(); opt.step()
            s.step()
        net.eval()
        with torch.no_grad():
            P = net(torch.tensor(X[ev_rows], device=dev)).cpu().numpy().reshape(len(ev_rows), -1)
        return sc(P * ys + ym, ev_rows), sum(p.numel() for p in net.parameters())

    # ============ 1. 每个骨干各自在 val 上选学习率 ============
    print("=== 1. 各骨干在 val 上选学习率(避免拿基线超参跑新架构) ===")
    ARCH = ["TFBase", "Autoformer", "DeepONet", "FNO1d", "Ours"]
    LR = {}
    t0 = time.time()
    for k in ARCH:
        best = None
        for lr in (1e-3, 3e-3):
            (e, *_), _ = train_eval(k, tr, va, seed=0, lr=lr)
            print(f"  {k:12s} lr={lr:<6g} val {e:.3%}  ({(time.time()-t0)/60:.0f}min)", flush=True)
            if best is None or e < best[1]:
                best = (lr, e)
        LR[k] = best[0]
        print(f"  → {k} 选 lr={best[0]}\n", flush=True)

    # ============ 2. 骨干对照(封存 test, 多种子, 不集成) ============
    print("=== 2. 骨干对照(封存 test, {} 种子, **不做集成**) ===".format(args.seeds))
    R = {}

    def bench(tag, kind, *, spec=0.0, **kw):
        rs = [train_eval(kind, trva, te, seed=s, lr=LR[kind], spec=spec, **kw)
              for s in range(args.seeds)]
        a = np.array([r[0] for r in rs]); npar = rs[0][1]
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                  "r2": float(a[:, 3].mean()), "params": int(npar),
                  "lr": LR[kind], "raw": a[:, 0].tolist()}
        r = R[tag]
        print(f"  {tag:34s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"R² {r['r2']:.4f}  参数 {npar:,}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    bench("Transformer(基线)", "TFBase")
    bench("Autoformer(时序模型,架构错配)", "Autoformer")
    bench("DeepONet(算子学习)", "DeepONet")
    bench("FNO(算子学习,谱卷积)", "FNO1d")

    # ============ 3. 模块消融 ============
    print(f"\n=== 3. 模块消融(在 Transformer 骨干上逐项加) ===")
    bench("+傅里叶特征嵌入", "Ours", use_ff=True, use_film=False)
    bench("+FiLM 条件调制", "Ours", use_ff=False, use_film=True)
    bench("+谱加权损失", "Ours", use_ff=False, use_film=False, spec=1.0)
    bench("+傅里叶+FiLM", "Ours", use_ff=True, use_film=True)
    bench("🏆 Ours(傅里叶+FiLM+谱损失)", "Ours", use_ff=True, use_film=True, spec=1.0)

    # ============ 4. 判定 ============
    from scipy.stats import ttest_ind
    base = np.array(R["Transformer(基线)"]["raw"])
    print(f"\n=== 4. 显著性(vs Transformer 基线, Welch) ===")
    sig = {}
    for k, v in R.items():
        if k == "Transformer(基线)":
            continue
        a = np.array(v["raw"]); p = float(ttest_ind(a, base, equal_var=False).pvalue)
        d = (a.mean() - base.mean()) * 100
        verdict = ("✔ 显著变好" if p < .05 and d < 0 else
                   "🔴 显著变差" if p < .05 else "— 噪声内")
        sig[k] = {"delta_pp": d, "p": p, "verdict": verdict}
        print(f"  {k:34s} {d:>+8.3f}pp  p={p:.4f}   {verdict}")

    rank = sorted(R, key=lambda k: R[k]["mean"])
    print(f"\n=== 5. 排名(单模型, 无集成) ===")
    for i, k in enumerate(rank, 1):
        print(f"  {i:>2} {k:34s} {R[k]['mean']:.3%} ± {R[k]['sd']:.3%}   R² {R[k]['r2']:.4f}")
    w = R[rank[0]]; s2 = R[rank[1]]
    print(f"\n🏆 {rank[0]}  {w['mean']:.3%}")
    print(f"   vs 亚军 {rank[1]}({s2['mean']:.3%}): 误差降低 {1-w['mean']/s2['mean']:.1%}")
    print(f"   vs Transformer 基线({R['Transformer(基线)']['mean']:.3%}): "
          f"降低 {1-w['mean']/R['Transformer(基线)']['mean']:.1%}")

    json.dump({"n": n, "n_test": len(te), "seeds": args.seeds, "epochs": args.epochs,
               "lr_selected": LR, "results": R, "significance": sig, "ranking": rank,
               "protocol": "test 400 封存;各骨干各自在 val 选 lr;**全部单模型,不集成**",
               "gap_basis": ("改动依据 fc_gap:误差 98.8% 是形状错;误差高频占 18.08% vs "
                             "信号 4.63%;模态信噪比 15559→896(17×)。物理违背/水突破/"
                             "异方差三条路线已被实测排除。")},
              open(OUT / "arch.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/arch.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
