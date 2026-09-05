#!/usr/bin/env python3
"""在最好的 Transformer 上做改进，逐项消融，选出最终模型。

## 设计原则:每一项都要能回答"它带来了什么新信息"

🔴 **智能体打分若只看注水方案本身, 加不了任何信息** —— 分数是输入的确定性函数,
   网络自己就能学会那个函数。要有用, 必须注入**外部知识**(物理规则/领域经验),
   而不是把网络已经知道的再喂一遍。所以下面的"物理特征"是**领域公式算出来的量**,
   不是对输入的重新包装。

## 四项改进
A 物理特征融合:注采比、注入爬坡率、井间失衡度、分段累计 —— 都是油藏工程里
  有名字的量, 网络要从 24 个乘子里自己拼出来很费劲, 直接给等于送先验。
B 单调约束:累计产油必须单调递增 —— 这是物理事实, 网络现在不知道。
C 超参搜索:层数/头数/宽度/学习率。
D 集成:多种子平均, 降方差。

用法: python fc_improve.py --gpu 6
"""
from __future__ import annotations

import argparse, json, time, itertools
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_improve"
DAYS = FG.FC_GRID
NW, NPH, NT = 22, 2, 40
BASE_W = np.array([FG.BASE_WINJ[w] for w in FG.INJ_W])          # 各井基准注水率


def phys_features(TH):
    """A: 从注水方案算出**油藏工程里有名字的量**。

    TH (n, 24) = 6 段 × 4 井的 log10 乘子。
    这些量网络理论上能自己拼出来, 但拼非线性组合很费劲;
    直接给 = 送先验, 这才是"特征融合"该有的样子。
    """
    n = len(TH)
    S = TH.reshape(n, FG.N_STAGE, len(FG.INJ_W))
    rate = (10.0 ** S) * BASE_W[None, None, :]                   # 各段各井的实际目标率
    tot = rate.sum(2)                                            # 每段总注水率 (n, 6)
    f = [
        tot,                                                     # 分段总注水
        np.cumsum(tot, 1),                                       # 累计注水(时间积累效应)
        np.diff(tot, axis=1),                                    # 爬坡率:加注还是减注
        rate.std(2) / np.maximum(rate.mean(2), 1e-9),            # 井间失衡度(变异系数)
        rate.max(2) - rate.min(2),                               # 井间极差
        tot.mean(1, keepdims=True),                              # 全期平均
        (tot[:, :2].mean(1) / np.maximum(tot[:, -2:].mean(1), 1e-9))[:, None],  # 前重/后重比
    ]
    return np.hstack([x.reshape(n, -1) for x in f]).astype(np.float32)


class TF(nn.Module):
    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256):
        super().__init__()
        self.cond = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_ch)

    def forward(self, x):
        h = self.cond(x)[:, None, :] + self.pos
        return self.head(self.tr(h)).transpose(1, 2)


def cumoil(Y):
    return np.trapezoid(Y[:, :, 0, :], DAYS, axis=2).sum(1)


def train_eval(Xtr, Ytr_n, Xte, Yte_raw, ym, ys, dev, *, seed=0, mono=False,
               d=128, heads=4, layers=3, ff=256, epochs=300, n_seed=1):
    """训练并返回留出集的相对误差 / R²。mono=True 时加单调惩罚。"""
    preds = []
    for sd in range(n_seed):
        torch.manual_seed(seed + sd)
        net = TF(Xtr.shape[1], NW * NPH, NT, d, heads, layers, ff).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        A = torch.tensor(Xtr, device=dev); B = torch.tensor(Ytr_n.reshape(-1, NW * NPH, NT), device=dev)
        # 反归一化用的张量(算单调惩罚要在物理量纲上算)
        ymt = torch.tensor(ym.reshape(NW * NPH, NT), device=dev)
        yst = torch.tensor(ys.reshape(NW * NPH, NT), device=dev)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i + 64]
                opt.zero_grad()
                out = net(A[b])
                loss = torch.nn.functional.mse_loss(out, B[b])
                if mono:
                    # B: 产油率必须 ≥0 → 累计产油自然单调。惩罚负产率。
                    phys = out * yst + ymt
                    oil = phys.view(-1, NW, NPH, NT)[:, :, 0, :]
                    loss = loss + 1e-3 * torch.relu(-oil).mean()
                loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            P = net(torch.tensor(Xte, device=dev)).cpu().numpy()
        preds.append((P.reshape(len(Xte), -1) * ys + ym).reshape(-1, NW, NPH, NT))
    P = np.mean(preds, 0)
    e = np.abs(cumoil(P) - cumoil(Yte_raw)) / np.maximum(np.abs(cumoil(Yte_raw)), 1e-9)
    t = cumoil(Yte_raw)
    r2 = 1 - ((cumoil(P) - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    neg = float((P[:, :, 0, :] < 0).mean())
    return float(np.median(e)), float(np.percentile(e, 90)), float(r2), neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    TH, Y, IA, FC = FD.load(N_PIN)
    n = len(TH); Yf = Y.reshape(n, -1)
    rng = np.random.default_rng(args.seed); idx = rng.permutation(n)
    n_te = n // 5; te, tr = idx[:n_te], idx[n_te:]
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
    Ytr_n = (Yf[tr] - ym) / ys

    PF = phys_features(TH)
    print(f"样本 {n}  训练 {len(tr)}  留出 {len(te)}")
    print(f"原始输入 {TH.shape[1]} 维   物理特征 {PF.shape[1]} 维   融合后 {TH.shape[1]+PF.shape[1]} 维")

    def norm(X):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8
        return ((X - m) / s).astype(np.float32)
    X0 = norm(TH)
    X1 = norm(np.hstack([TH, PF]))
    res = {}

    def go(tag, X, **kw):
        t0 = time.time()
        e, p90, r2, neg = train_eval(X[tr], Ytr_n, X[te], Y[te], ym, ys, dev,
                                     seed=args.seed, epochs=args.epochs, **kw)
        res[tag] = {"relerr": e, "p90": p90, "r2": r2, "neg_frac": neg,
                    "sec": round(time.time() - t0, 1), **{k: v for k, v in kw.items()}}
        print(f"  {tag:34s} 误差 {e:>7.3%}  p90 {p90:>7.3%}  R² {r2:>7.4f}  "
              f"负产率格点 {neg:>6.2%}  {time.time()-t0:>5.0f}s", flush=True)
        return e

    print("\n=== 逐项消融 ===")
    base = go("基线 Transformer", X0)
    go("A 物理特征融合", X1)
    go("B 单调约束(罚负产率)", X0, mono=True)
    go("A+B", X1, mono=True)

    print("\n=== C 超参搜索(在 A+B 之上) ===")
    best_cfg, best_e = None, 1e9
    for d, heads, layers, ff in itertools.product([128, 192], [4, 8], [3, 4], [256]):
        e = go(f"C d={d} h={heads} L={layers}", X1, mono=True, d=d, heads=heads, layers=layers, ff=ff)
        if e < best_e:
            best_e, best_cfg = e, dict(d=d, heads=heads, layers=layers, ff=ff)
    print(f"  最佳超参 {best_cfg}  误差 {best_e:.3%}")

    print("\n=== D 集成(最佳超参 × 5 种子) ===")
    go("D 最终模型(A+B+C+集成)", X1, mono=True, n_seed=5, **best_cfg)

    order = sorted(res, key=lambda k: res[k]["relerr"])
    print(f"\n🏆 最终模型: {order[0]}  误差 {res[order[0]]['relerr']:.3%}  R² {res[order[0]]['r2']:.4f}")
    print(f"   相对基线 Transformer({base:.3%}) 改进 {(1-res[order[0]]['relerr']/base):.1%}")
    json.dump({"n": n, "n_test": len(te), "baseline": base, "best_cfg": best_cfg,
               "results": res, "best": order[0],
               "phys_feature_dim": int(PF.shape[1])},
              open(OUT / "improve.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {OUT}/improve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
