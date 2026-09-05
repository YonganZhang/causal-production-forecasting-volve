#!/usr/bin/env python3
"""重写井间交互模块：修掉旧版的四个设计错误 + 引入物理连通性先验。

## 旧版为什么失败（我自己复查代码找出的，不是猜）

旧实现 `fc_auto.py:ChannelMix`：

    def forward(self, y):                      # y = (B, 22通道, 40时刻)
        h = self.norm(self.inp(y) + self.ch_emb)   # ← 错误 ②
        a, _ = self.attn(h, h, h)                  # ← 错误 ③
        return y + self.out(a)                     # ← 错误 ①

| # | 错误 | 后果 |
|---|---|---|
| ① | **挂在骨干输出之后**，拿到的是已预测好的曲线，看不见骨干内部表示 | 井间干扰应在**表示层**发生，不是事后修补 |
| ② | **LayerNorm 抹掉幅度**。各井平均油率从 1.6 到 309（**差 190 倍**），幅度恰是井间关系最强的信号 | 亲手把最有用的信息归一化掉 |
| ③ | **注意力内部漏了残差**（标准是 `h = h + attn(h)`），直接把 `attn(h)` 送进输出层 | 信息通路被破坏 |
| ④ | 新配置固定 `lr=3e-3`，**从未单独调过** | 与我自己批评过的"拿基线超参跑新架构"是同一个错 |

实测代价：Transformer 0.551% → **0.715%**（把基线打坏 30%），两个切分一致。

## 新版的三条改动

**1. 放进潜空间（修①）**
把输出头从 `Linear(d → n_ch)` 换成**通道查询**：n_ch 个可学习 token 去**跨注意力**
读取骨干的时间表示，得到每通道的潜向量，通道之间再自注意力，最后每通道各自解码出曲线。
通道在**表示层**交互，而不是拿预测结果互相修补。

**2. 保留幅度（修②③）**
去掉会抹平尺度的 LayerNorm，改用**预归一化残差**（`h = h + attn(norm(h))`）——
这是标准 Transformer 的写法，残差通路上的原始尺度完整保留。

**3. 🔴 物理连通性先验（本项目独有）**
`_pipelines/flow_diagnostics/allocation.json` 里有流动诊断算出的
**注入井 → 生产井分配矩阵**，是真实物理，别人没有。把它变成生产井之间的
**共享注入源相似度**，作为注意力的加性偏置：

    bias[i,j] = λ · cos( a_i , a_j )      a_i = 生产井 i 的注入来源向量

物理含义：**两口井若吃同一批注入水，它们的响应就该联动。**
这比让网络从 1,400 个样本里自由学 22×22 的通道关系合理得多 —— 后者正是它过拟合的原因。
λ 可学习，初始为 0（起点等于无先验），所以先验**帮不帮得上由数据说了算**。

## 纪律

- 每个配置**各自在 val 上调学习率**（修④）；
- 两个切分（`split_seed` 0 与 1）都跑；
- **配置预先写死**，test 上不做任何选择。

用法:
    python fc_block.py --gpu 5 --split-seed 0
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
import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_block"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def fourier_pos(n_t, n_bands, device):
    t = torch.linspace(0, 1, n_t, device=device)
    f = 2.0 ** torch.arange(n_bands, device=device, dtype=t.dtype)
    a = t[:, None] * f[None, :] * 2 * np.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def physics_prior(wells) -> np.ndarray | None:
    """从流动诊断的分配矩阵构造生产井之间的「共享注入源」相似度。

    返回 (n_ch, n_ch)，n_ch = len(wells) * 2（油相/水相各一份，同井共享同一行）。
    拿不到数据就返回 None —— **不静默退化成全零先验**，让调用方显式处理。
    """
    f = ROOT / "_pipelines" / "flow_diagnostics" / "allocation.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()).get("injector_fraction")
    if not d:
        return None
    injs = sorted({i for v in d.values() for i in v})
    A = np.array([[d.get(w, {}).get(i, 0.0) for i in injs] for w in wells], float)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    if (nrm == 0).all():
        return None
    A = A / np.maximum(nrm, 1e-9)
    S = A @ A.T                                        # 余弦相似度:共享注入源的程度
    return np.kron(S, np.ones((NPH, NPH)))             # 油/水两相共享同一井间结构


class ChannelHead(nn.Module):
    """通道查询解码头：在**潜空间**做井间交互（新版模块 1+2+3）。"""

    def __init__(self, n_ch, n_t, d, heads=4, prior=None):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_ch, d) * 0.02)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d) for _ in range(3))
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.self_ = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.dec = nn.Linear(d, n_t)
        self.heads = heads
        if prior is not None:
            self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))
            self.lam = nn.Parameter(torch.zeros(1))     # 初始为 0 → 起点等于无先验
        else:
            self.prior = None

    def forward(self, h):                               # h = 骨干的时间表示 (B, T, d)
        B = h.shape[0]
        q = self.q.expand(B, -1, -1)
        # 跨注意力:每个通道去时间表示里取自己需要的东西
        a, _ = self.cross(self.n1(q), h, h)
        q = q + a
        # 自注意力:通道之间交互(井间干扰),可挂物理先验偏置
        mask = None
        if self.prior is not None:
            m = (self.lam * self.prior)[None].expand(B * self.heads, -1, -1)
            mask = m.reshape(B * self.heads, *self.prior.shape)
        s, _ = self.self_(self.n2(q), self.n2(q), self.n2(q), attn_mask=mask)
        q = q + s
        q = q + self.ff(self.n3(q))                     # 预归一化残差,幅度不被抹掉
        return self.dec(q)                              # (B, n_ch, n_t)


class Net(nn.Module):
    """Transformer 骨干 + 傅里叶位置嵌入(已确认有效) + 可选的新版通道头。"""

    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256,
                 n_bands=16, use_ff=True, chan=False, prior=None):
        super().__init__()
        self.use_ff, self.n_bands, self.n_t = use_ff, n_bands, n_t
        self.cond = nn.Linear(d_in, d)
        if use_ff:
            self.pos_proj = nn.Linear(2 * n_bands, d)
        else:
            self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0), layers)
        self.head = ChannelHead(n_ch, n_t, d, heads, prior) if chan else nn.Linear(d, n_ch)
        self.chan = chan

    def forward(self, x):
        pos = (self.pos_proj(fourier_pos(self.n_t, self.n_bands, x.device))[None]
               if self.use_ff else self.pos)
        h = self.tr(self.cond(x)[:, None, :] + pos)
        return self.head(h) if self.chan else self.head(h).transpose(1, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    WELLS = [NB.PRODUCERS[i] for i in np.where(live)[0]]
    Y = Y[:, live]; nw = len(WELLS); n = len(TH)
    idx = np.random.default_rng(args.split_seed).permutation(n)
    te, va, tr = idx[:400], idx[400:600], idx[600:]
    trva = np.concatenate([tr, va])
    Yf = Y.reshape(n, -1); NCH = nw * NPH; D = TH.shape[1]
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)

    P = physics_prior(WELLS)
    if P is None:
        print("🔴 拿不到流动诊断分配矩阵,物理先验一档将跳过(不静默退化成全零)")
    else:
        off = P[~np.eye(len(P), dtype=bool)]
        print(f"物理先验 {P.shape}  非对角相似度 中位 {np.median(off):.3f} "
              f"最大 {off.max():.3f}  非零占比 {(off > 1e-6).mean():.1%}")
    print(f"split_seed={args.split_seed}  train {len(tr)} / val {len(va)} / test {len(te)}\n")

    def run(ev_rows, *, seed, lr, fit=None, **kw):
        fit = trva if fit is None else fit
        ym, ys = Yf[fit].mean(0), Yf[fit].std(0) + 1e-8
        xm, xs = TH[fit].mean(0), TH[fit].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = Net(D, NCH, NT, **kw).to(dev)
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
            Pr = net(torch.tensor(X[ev_rows], device=dev)).cpu().numpy().reshape(len(ev_rows), -1)
        Pr = Pr * ys + ym
        ct, cp = cum(Yf[ev_rows]), cum(Pr)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        lam = float(net.head.lam.item()) if (kw.get("chan") and net.head.prior is not None) else float("nan")
        return (float(rel.mean()), float(np.median(rel)), float(np.quantile(rel, .9)),
                float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()),
                sum(p.numel() for p in net.parameters()), lam)

    CFG = {
        "基线 Transformer":                  dict(use_ff=False),
        "+傅里叶(现最优)":                     dict(use_ff=True),
        "+傅里叶 +新版通道头":                  dict(use_ff=True, chan=True),
        "+傅里叶 +新版通道头 +物理先验":          dict(use_ff=True, chan=True, prior=P),
    }
    if P is None:
        CFG.pop("+傅里叶 +新版通道头 +物理先验")

    print("=== 1. 各配置**各自**在 val 上调学习率(修掉旧版固定 lr 的不公平) ===")
    LR = {}
    t0 = time.time()
    for tag, kw in CFG.items():
        best = None
        for lr in (1e-3, 3e-3):
            e = run(va, seed=0, lr=lr, fit=tr, **kw)[0]
            if best is None or e < best[1]:
                best = (lr, e)
        LR[tag] = best[0]
        print(f"  {tag:26s} 选 lr={best[0]:<6g} (val {best[1]:.3%})  ({(time.time()-t0)/60:.0f}min)",
              flush=True)

    print(f"\n=== 2. 封存 test({args.seeds} 种子) ===")
    R = {}
    for tag, kw in CFG.items():
        a = np.array([run(te, seed=s, lr=LR[tag], **kw) for s in range(args.seeds)])
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                  "r2": float(a[:, 3].mean()), "params": int(a[0, 4]),
                  "lambda": float(np.nanmean(a[:, 5])), "lr": LR[tag], "raw": a[:, 0].tolist()}
        r = R[tag]
        lam = f"  λ={r['lambda']:+.4f}" if not np.isnan(r["lambda"]) else ""
        print(f"  {tag:26s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"R² {r['r2']:.4f}  参数 {r['params']:>9,d}{lam}  "
              f"({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    ref = np.array(R["+傅里叶(现最优)"]["raw"])
    print(f"\n=== 3. 判定(vs 现最优「+傅里叶」, Welch) ===")
    sig = {}
    for tag in R:
        if tag == "+傅里叶(现最优)":
            continue
        a = np.array(R[tag]["raw"]); p = float(ttest_ind(a, ref, equal_var=False).pvalue)
        d = (a.mean() - ref.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[tag] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {tag:26s} {d:>+8.3f}pp  p={p:.4f}   {vd}")
    if "+傅里叶 +新版通道头 +物理先验" in R:
        lam = R["+傅里叶 +新版通道头 +物理先验"]["lambda"]
        print(f"\n  物理先验权重 λ 训练后 = {lam:+.4f}(初始 0)"
              f"  → {'网络确实用上了先验' if abs(lam) > 0.01 else '🔴 网络基本没用先验,λ 没离开 0'}")
    json.dump({"split_seed": args.split_seed, "results": R, "significance": sig,
               "lr_selected": LR, "seeds": args.seeds,
               "fixes": ["潜空间通道交互(修①)", "去掉抹平幅度的 LayerNorm + 预归一化残差(修②③)",
                         "各配置各自调 lr(修④)", "流动诊断物理连通性作注意力先验(新增)"]},
              open(OUT / f"block_split{args.split_seed}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/block_split{args.split_seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
