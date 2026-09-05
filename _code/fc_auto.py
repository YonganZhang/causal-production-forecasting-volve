#!/usr/bin/env python3
"""在 Autoformer 骨干上做两个模块的改进 + 完整消融。

## 为什么换骨干

用户指出:改进应该建在**最好的骨干**上,而不是基线 Transformer 上。
val 上 Autoformer(0.599%) 优于 Transformer(0.657%)。
⚠️ 但更早一轮在**封存 test** 上两者是 Autoformer 0.578% vs Transformer 0.547%,p=0.5066(无差异)。
两个口径不一致 —— 所以本脚本**两个骨干都做**,让 test 数字说话,不预设谁是"最好的"。

## 两个模块（各自有独立证据，不是凑数）

**模块 1 — 傅里叶位置嵌入**（已确认有效）
证据:Gap 分析测到谱偏差(误差高频占 18.08% vs 信号 4.63%;逐模态信噪比 15559→896);
封存 test 12 种子 p=0.0000;20 种子排除了"初始化幅度"这个混杂(A2−B=+0.060pp, p=0.0000)。

**模块 2 — 井间交互（跨通道注意力）**（新，是待检验假设）
证据链:
  (a) **物理**:22 个通道 = 11 井 × 2 相。注一口井影响多口产油井 ——
      这正是本项目流动诊断/CRM 连通性工作的核心对象,是真实物理不是假设;
  (b) **Gap 实测**:误差在井之间**不均匀**,前 3 口井占总误差 **41.6%**;
  (c) **架构缺口**:现有所有骨干的输出头都是 `Linear(d → n_ch)` ——
      **各通道在最后一层才分叉,彼此之间从不交互**。井间干扰无处建模;
  (d) **文献**:iTransformer(ICLR 2024)的核心机制就是"注意力作用在变量之间"。
      它作为**整个骨干**在本任务只排第 4,但这不等于该机制无用 ——
      本脚本把它拆成一个**模块**装进强骨干,单独检验。

实现:在输出 (B, n_ch, n_t) 上做残差式通道混合 ——
把每个通道的整条曲线投影成一个 token,在 n_ch 个 token 间自注意力,再投影回去。

## 协议

- 消融配置**全部预先写死**,不在 test 上做任何选择;
- 🔴 **两个切分**:`split_seed=0`(既有,已被多轮 test 评测"看过")与
  `split_seed=1`(**全新,从未被任何实验碰过**)。两个都报。
  这是对"test 复用导致置信度虚高"的直接补救 —— split 1 才是真正干净的确认集。

用法:
    python fc_auto.py --gpu 2 --split-seed 0
    python fc_auto.py --gpu 3 --split-seed 1
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
import fc_models as M

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_auto"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def fourier_pos(n_t, n_bands, device):
    t = torch.linspace(0, 1, n_t, device=device)
    f = 2.0 ** torch.arange(n_bands, device=device, dtype=t.dtype)
    a = t[:, None] * f[None, :] * 2 * np.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class ChannelMix(nn.Module):
    """模块 2：井间交互。

    现有骨干的输出头是 `Linear(d → n_ch)`，各通道在最后一层才分叉、彼此从不交互。
    本模块把每条通道曲线投影成一个 token，在通道之间做自注意力，再投影回去（残差）。
    对应真实物理：注水井 → 多口产油井的干扰。
    """

    def __init__(self, n_ch, n_t, d=128, heads=4):
        super().__init__()
        self.inp = nn.Linear(n_t, d)
        self.ch_emb = nn.Parameter(torch.randn(1, n_ch, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_t)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 残差初始为恒等

    def forward(self, y):                                    # (B, n_ch, n_t)
        h = self.norm(self.inp(y) + self.ch_emb)
        a, _ = self.attn(h, h, h)
        return y + self.out(a)


class Net(nn.Module):
    """骨干 × 两个模块的所有组合，供逐项消融。"""

    def __init__(self, d_in, n_ch, n_t, backbone="Autoformer", ff=False, cmix=False,
                 d=128, heads=4, layers=3, ffn=256, n_bands=16):
        super().__init__()
        self.ff, self.n_bands, self.n_t = ff, n_bands, n_t
        if backbone == "Autoformer":
            self.bb = M.Autoformer(d_in, n_ch, n_t, d, heads, max(2, layers - 1), ffn)
        else:
            self.bb = M.TFBase(d_in, n_ch, n_t, d, heads, layers, ffn)
        if ff:                                               # 模块 1:换掉自由学习的 pos
            # 用 forward 的 pos_override 注入,不去改骨干的 Parameter(那会破坏 nn.Module 语义)
            self.pos_proj = nn.Linear(2 * n_bands, d)
        self.cmix = ChannelMix(n_ch, n_t, d, heads) if cmix else None

    def forward(self, x):
        po = (self.pos_proj(fourier_pos(self.n_t, self.n_bands, x.device))[None]
              if self.ff else None)
        y = self.bb(x, pos_override=po)
        return self.cmix(y) if self.cmix is not None else y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--cfgset", choices=["auto", "tf"], default="auto",
                    help="auto=Autoformer 侧消融; tf=Transformer 侧消融(井间交互在更强骨干上的补测)")
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
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    tag_split = ("既有(已被多轮 test 评测看过)" if args.split_seed == 0
                 else "🔴 全新,从未被任何实验碰过")
    print(f"split_seed={args.split_seed} {tag_split}")
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}   {args.seeds} 种子\n")

    def run(ev_rows, *, seed, lr=3e-3, **kw):
        fit = trva
        ym, ys = Yf[fit].mean(0), Yf[fit].std(0) + 1e-8
        xm, xs = TH[fit].mean(0), TH[fit].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = Net(TH.shape[1], NCH, NT, **kw).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[fit], device=dev)
        B = torch.tensor(((Yf[fit]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
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
                float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()),
                sum(p.numel() for p in net.parameters()))

    CFG_TF = {
        "Transformer 骨干":                 dict(backbone="TF"),
        "Transformer + 傅里叶":              dict(backbone="TF", ff=True),
        "Transformer + 井间交互":             dict(backbone="TF", cmix=True),
        "🏆 Transformer + 傅里叶 + 井间交互":    dict(backbone="TF", ff=True, cmix=True),
    }
    CFG_AUTO = {
        "Transformer 骨干":                 dict(backbone="TF"),
        "Transformer + 傅里叶":              dict(backbone="TF", ff=True),
        "Autoformer 骨干":                  dict(backbone="Autoformer"),
        "Autoformer + 傅里叶":               dict(backbone="Autoformer", ff=True),
        "Autoformer + 井间交互":              dict(backbone="Autoformer", cmix=True),
        "🏆 Autoformer + 傅里叶 + 井间交互":     dict(backbone="Autoformer", ff=True, cmix=True),
    }
    CFG = CFG_TF if args.cfgset == "tf" else CFG_AUTO
    print(f"=== 封存 test(配置全部预先写死,test 上不做任何选择) ===")
    R, t0 = {}, time.time()
    for tag, kw in CFG.items():
        a = np.array([run(te, seed=s, **kw) for s in range(args.seeds)])
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                  "r2": float(a[:, 3].mean()), "params": int(a[0, 4]), "raw": a[:, 0].tolist()}
        r = R[tag]
        print(f"  {tag:32s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"R² {r['r2']:.4f}  参数 {r['params']:>9,d}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    print(f"\n=== 消融判定(Welch) ===")
    sig = {}
    PAIRS_TF = [("Transformer + 傅里叶", "Transformer 骨干"),
                ("Transformer + 井间交互", "Transformer 骨干"),
                ("🏆 Transformer + 傅里叶 + 井间交互", "Transformer 骨干"),
                ("🏆 Transformer + 傅里叶 + 井间交互", "Transformer + 傅里叶")]
    PAIRS = PAIRS_TF if args.cfgset == "tf" else [("Autoformer + 傅里叶", "Autoformer 骨干"),
             ("Autoformer + 井间交互", "Autoformer 骨干"),
             ("🏆 Autoformer + 傅里叶 + 井间交互", "Autoformer 骨干"),
             ("🏆 Autoformer + 傅里叶 + 井间交互", "Autoformer + 傅里叶"),
             ("🏆 Autoformer + 傅里叶 + 井间交互", "Transformer + 傅里叶"),
             ("Autoformer 骨干", "Transformer 骨干")]
    for a_, b_ in PAIRS:
        x, y = np.array(R[a_]["raw"]), np.array(R[b_]["raw"])
        p = float(ttest_ind(x, y, equal_var=False).pvalue); d = (x.mean() - y.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[f"{a_} vs {b_}"] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {a_:32s} vs {b_:26s} {d:>+8.3f}pp p={p:.4f}  {vd}")

    rk = sorted(R, key=lambda k: R[k]["mean"])
    print(f"\n=== 排名(封存 test) ===")
    for i, k in enumerate(rk, 1):
        print(f"  {i} {k:32s} {R[k]['mean']:.3%} ± {R[k]['sd']:.3%}   R² {R[k]['r2']:.4f}")
    json.dump({"split_seed": args.split_seed, "split_note": tag_split, "seeds": args.seeds,
               "results": R, "significance": sig, "ranking": rk},
              open(OUT / f"{args.cfgset}_split{args.split_seed}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/{args.cfgset}_split{args.split_seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
