#!/usr/bin/env python3
"""把通道维度贯穿整个骨干：这是此前**不存在**的模块位置。

## 为什么四次独立设计全部失败 —— 结构性原因

复查 `fc_models.py:TFBase`:

    self.tr   = TransformerEncoder(...)    # 序列 = 40 个**时刻** token,维度 d
    self.head = nn.Linear(d, n_ch)         # ← 通道在**最后一层**才被创造出来

**骨干内部根本没有"通道"这个维度。** 于是任何井间交互模块都**只能**挂在末端:

| 尝试 | 位置 | 结果 |
|---|---|---|
| 我的 v1 | 输出之后做残差 | 0.551% → 0.715% |
| 我的 v2 | 替换输出头(潜空间通道查询) | 0.493% → 0.650% |
| v2 + 物理先验 | 同上 + 流动诊断偏置 | 0.657%,λ 训到 −0.72 |
| Codex 独立实现 | 输出上的井 token 残差(+1.3万参数) | 5/6 种子输,p=0.0334 |
| 调参后(32 组搜索, 3400 样本) | 替换输出头 | 0.394% vs 0.366%,p=0.0014 |

五次尝试都在**同一个位置**。不是"实现不好"五次,是**这个架构只提供了这一个位置**。

## 本脚本：真正把模块位置换掉

**轴向(factorized)骨干**:全程携带 `(B, n_ch, T, d)`,交替做
**时间轴注意力**(每个通道内部)与**通道轴注意力**(每个时刻跨井)。
井间交互发生在**每一层**,而不是末端一次。

这是 iTransformer(ICLR 2024)"把变量当 token"思想的完整版 ——
iTransformer 只在通道轴做注意力(丢掉时间轴),轴向骨干两轴都做。

计算量:22 通道 × 40 时刻 = 880 token。分解后两轴各自只在
40 或 22 个 token 上做注意力,所以比 880 全连接注意力便宜得多。

## 同时回答用户的另一个问题：换 Autoformer 骨干

此前 Autoformer 的排名是在 **1400 训练样本**下定的。样本涨到 3400 后
排名可能改变(数据规模会改变容量需求)。本脚本在 N=3400 下重测。

## 纪律

N=3400、封存 test 只评一次、6 种子、**单模型不集成**、配置预先写死。

用法:
    python fc_axial.py --gpu 4
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
from fc_mech import PosNet

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_axial"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def fourier_pos(n_t, n_bands, device):
    t = torch.linspace(0, 1, n_t, device=device)
    f = 2.0 ** torch.arange(n_bands, device=device, dtype=t.dtype)
    a = t[:, None] * f[None, :] * 2 * np.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class AxialBlock(nn.Module):
    """一层轴向块：先在时间轴做注意力，再在通道轴做注意力。

    张量始终是 (B, C, T, d) —— **通道维度贯穿全程**，
    所以井间交互发生在每一层，而不是末端补一刀。
    """

    def __init__(self, d, heads=4, ff=None, drop=0.0, prior=None):
        super().__init__()
        ff = ff or 2 * d
        self.nt1, self.nc1, self.nf = (nn.LayerNorm(d) for _ in range(3))
        self.att_t = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        self.att_c = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.heads = heads
        if prior is not None:
            self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))
            self.lam = nn.Parameter(torch.zeros(1))
        else:
            self.prior = None

    def forward(self, x):                                   # (B, C, T, d)
        B, C, T, d = x.shape
        # --- 时间轴:每口井内部的时序演化 ---
        h = self.nt1(x).reshape(B * C, T, d)
        a, _ = self.att_t(h, h, h)
        x = x + a.reshape(B, C, T, d)
        # --- 通道轴:每个时刻的井间干扰 ---
        h = self.nc1(x).permute(0, 2, 1, 3).reshape(B * T, C, d)
        mask = None
        if self.prior is not None:
            mask = (self.lam * self.prior)[None].expand(B * T * self.heads, C, C)
        a, _ = self.att_c(h, h, h, attn_mask=mask)
        x = x + a.reshape(B, T, C, d).permute(0, 2, 1, 3)
        return x + self.ff(self.nf(x))


class AxialNet(nn.Module):
    """通道维度贯穿全程的轴向骨干 + 傅里叶位置嵌入(已确认有效,保留)。"""

    def __init__(self, d_in, n_ch, n_t, d=64, heads=4, layers=3, n_bands=16,
                 drop=0.0, prior=None):
        super().__init__()
        self.n_bands, self.n_t, self.n_ch, self.d = n_bands, n_t, n_ch, d
        self.cond = nn.Linear(d_in, d)
        self.pos_proj = nn.Linear(2 * n_bands, d)
        self.ch_emb = nn.Parameter(torch.randn(1, n_ch, 1, d) * 0.02)
        self.blocks = nn.ModuleList([AxialBlock(d, heads, drop=drop, prior=prior)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.dec = nn.Linear(d, 1)

    def forward(self, x):
        B = x.shape[0]
        pos = self.pos_proj(fourier_pos(self.n_t, self.n_bands, x.device))   # (T, d)
        h = (self.cond(x)[:, None, None, :] + pos[None, None] + self.ch_emb)
        h = h.expand(B, self.n_ch, self.n_t, self.d).contiguous()
        for b in self.blocks:
            h = b(h)
        return self.dec(self.norm(h)).squeeze(-1)           # (B, n_ch, n_t)


def physics_prior(wells):
    f = Path(__file__).resolve().parent.parent / "_pipelines" / "flow_diagnostics" / "allocation.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()).get("injector_fraction")
    if not d:
        return None
    injs = sorted({i for v in d.values() for i in v})
    A = np.array([[d.get(w, {}).get(i, 0.0) for i in injs] for w in wells], float)
    nz = np.linalg.norm(A, axis=1, keepdims=True)
    if (nz == 0).all():
        return None
    A = A / np.maximum(nz, 1e-9)
    return np.kron(A @ A.T, np.ones((NPH, NPH)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--ntrain", type=int, default=3400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    import norne_bulk as NB
    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    WELLS = [NB.PRODUCERS[i] for i in np.where(live)[0]]
    Y = Y[:, live]; nw = len(WELLS); n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    te, va, pool = idx[:400], idx[400:600], idx[600:]
    tr = pool[:min(args.ntrain, len(pool))]
    Yf = Y.reshape(n, -1); NCH = nw * NPH; D = TH.shape[1]
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    P = physics_prior(WELLS)
    print(f"总样本 {n}  训练 {len(tr)} / val {len(va)} / test {len(te)}(封存)")
    print(f"物理先验 {None if P is None else P.shape}   {args.seeds} 种子\n")

    def run(build, ev_rows, *, seed, lr):
        ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
        xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = build().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[tr], device=dev)
        B = torch.tensor(((Yf[tr]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
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
        return (float(rel.mean()), float(np.median(rel)), float(np.quantile(rel, .9)),
                float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()),
                sum(p.numel() for p in net.parameters()))

    CFG = {
        "Transformer+傅里叶(现定版)":    lambda: PosNet(D, NCH, NT, "fourier", n_bands=16),
        "Autoformer 骨干":            lambda: M.Autoformer(D, NCH, NT),
        "🔴 轴向骨干(通道贯穿全程)":       lambda: AxialNet(D, NCH, NT, d=64, layers=3),
        "🔴 轴向 + 物理先验":            lambda: AxialNet(D, NCH, NT, d=64, layers=3, prior=P),
    }
    if P is None:
        CFG.pop("🔴 轴向 + 物理先验")

    print("=== 1. 各配置在 val 上选学习率 ===")
    LR, t0 = {}, time.time()
    for tag, f in CFG.items():
        best = None
        for lr in (1e-3, 3e-3):
            e = run(f, va, seed=0, lr=lr)[0]
            if best is None or e < best[1]:
                best = (lr, e)
        LR[tag] = best[0]
        print(f"  {tag:26s} lr={best[0]:<6g} val {best[1]:.3%}  ({(time.time()-t0)/60:.0f}min)",
              flush=True)

    print(f"\n=== 2. 封存 test({args.seeds} 种子, 单模型不集成) ===")
    R = {}
    for tag, f in CFG.items():
        a = np.array([run(f, te, seed=s, lr=LR[tag]) for s in range(args.seeds)])
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "median": float(a[:, 1].mean()), "p90": float(a[:, 2].mean()),
                  "r2": float(a[:, 3].mean()), "params": int(a[0, 4]),
                  "lr": LR[tag], "raw": a[:, 0].tolist()}
        r = R[tag]
        print(f"  {tag:26s} {r['mean']:.3%} ± {r['sd']:.3%}  中位 {r['median']:.3%}  "
              f"R² {r['r2']:.4f}  参数 {r['params']:>9,d}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    ref = np.array(R["Transformer+傅里叶(现定版)"]["raw"])
    print(f"\n=== 3. 判定(vs 现定版) ===")
    sig = {}
    for tag in R:
        if tag.startswith("Transformer"):
            continue
        a = np.array(R[tag]["raw"]); p = float(ttest_ind(a, ref, equal_var=False).pvalue)
        d = (a.mean() - ref.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[tag] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {tag:26s} {d:>+8.3f}pp  p={p:.4f}   {vd}")
    rk = sorted(R, key=lambda k: R[k]["mean"])
    print(f"\n排名: " + " > ".join(f"{k}({R[k]['mean']:.3%})" for k in rk))
    json.dump({"n_train": len(tr), "results": R, "significance": sig, "ranking": rk,
               "seeds": args.seeds, "lr": LR,
               "why": ("此前五次井间交互尝试全部挂在**末端**,因为 TFBase 骨干内部"
                       "根本没有通道维度(通道在 head=Linear(d,n_ch) 才被创造)。"
                       "轴向骨干全程携带 (B,C,T,d),交替做时间轴与通道轴注意力,"
                       "井间交互发生在每一层 —— 这是此前不存在的模块位置。")},
              open(OUT / "axial.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/axial.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
