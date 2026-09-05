#!/usr/bin/env python3
"""即插即用模块消融：抄 NLP/时序论文里公认"稳定涨点"的那批组件。

## 为什么这批和之前七个不一样

之前七个模块都是我**自己设计**的（FiLM、谱损失、井间交互…），全部失败。
本轮换思路：只用**文献里被反复验证、宣称即插即用、且被主流模型广泛采用**的组件，
且优先选**不增加或减少参数**的（前七次失败的共同点之一就是往上堆参数）。

## 🔴 先查出一个真问题

我们的骨干用 `nn.TransformerEncoderLayer(...)`，其 `norm_first` **默认 False = post-LN**。
而 Xiong et al. (ICML 2020) *On Layer Normalization in the Transformer Architecture*
证明 **pre-LN** 梯度尺度良性、无需 warmup、更稳更好训，现代实现（GPT/LLaMA/ViT）几乎全用 pre-LN。
**我从头到尾用的都是 post-LN，而且从没试过换。** 这是最有先验的一项。

## 本轮八项（全部来自公开论文，出处见下）

| 项 | 出处 | 参数变化 | 机理 |
|---|---|---|---|
| **preln** | Xiong et al. ICML 2020 | **0** | 归一化放到残差分支内部 |
| **rmsnorm** | Zhang & Sennrich NeurIPS 2019 | **−** | 去掉均值中心化与 bias（LLaMA 用） |
| **swiglu** | Shazeer 2020 (GLU Variants) | ~0（按 2/3 缩宽保持等参） | 门控 FFN（PaLM/LLaMA 用） |
| **layerscale** | Touvron et al. ICCV 2021 (CaiT) | +2d/层（极小） | 每通道可学习残差缩放，初值 1e-4 |
| **rezero** | Bachlechner et al. UAI 2021 | +1/层 | 残差门初值 0，起点是恒等映射 |
| **rope** | Su et al. Neurocomputing 2024 | **0** | 旋转位置编码（NLP 标配），与我们的傅里叶嵌入正面对比 |
| **droppath** | Huang et al. ECCV 2016 | **0** | 随机深度正则 |
| **levelshape** | RevIN 思想 (Kim et al. ICLR 2022) 的适配 | +小 | 分头预测「整体水平」与「归一化形状」再相乘 |

🔴 `levelshape` 是适配版不是原版：RevIN 归一化的是**输入序列**，而本任务推理时没有输入序列，
   所以只能用它的**输出侧**思想。论文里必须写明是适配。

## 纪律

- 全部叠在**定版**（Transformer + 傅里叶位置嵌入）上，一次只开一项；
- 先在 **val** 上筛（4 种子），只有筛出来的才上封存 test（8 种子）；
- 上一轮已确认**当前规模下 <0.05pp 的差异可能测不出来**，所以任何入选项
  必须同时满足「val 与 test 方向一致」且「test 上 p<0.05」。

用法:
    python fc_plug.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 8000
import forecast_gen as FG

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_plug"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def fourier_pos(n_t, n_bands, device):
    t = torch.linspace(0, 1, n_t, device=device)
    f = 2.0 ** torch.arange(n_bands, device=device, dtype=t.dtype)
    a = t[:, None] * f[None, :] * 2 * math.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class RMSNorm(nn.Module):
    """Zhang & Sennrich NeurIPS 2019。去掉均值中心化与 bias，只按 RMS 缩放。"""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """Shazeer 2020。门控 FFN。按 2/3 缩宽以保持与普通 FFN 大致等参。"""

    def __init__(self, d, ff):
        super().__init__()
        h = int(ff * 2 / 3 / 8) * 8 or 8
        self.w1, self.w2, self.w3 = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def rope(x, heads):
    """Su et al. 旋转位置编码。对 Q/K 的每两维做与位置相关的旋转。"""
    B, T, D = x.shape
    dh = D // heads
    x = x.view(B, T, heads, dh)
    half = dh // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / half))
    ang = torch.arange(T, device=x.device, dtype=x.dtype)[:, None] * inv[None]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    a, b = x[..., :half], x[..., half:2 * half]
    out = torch.cat([a * cos - b * sin, a * sin + b * cos], -1)
    if dh % 2:
        out = torch.cat([out, x[..., 2 * half:]], -1)
    return out.reshape(B, T, D)


class Layer(nn.Module):
    """可逐项开关的编码器层。默认等价于 nn.TransformerEncoderLayer(post-LN)。"""

    def __init__(self, d, heads, ff, *, preln=False, rms=False, swiglu=False,
                 layerscale=False, rezero=False, use_rope=False, droppath=0.0,
                 ls_init=1e-1):
        super().__init__()
        N = RMSNorm if rms else nn.LayerNorm
        self.n1, self.n2 = N(d), N(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = SwiGLU(d, ff) if swiglu else nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.preln, self.heads, self.use_rope, self.dp = preln, heads, use_rope, droppath
        # 🔴 修正:CaiT 的 1e-4 是给 24+ 层深网的;3 层用 1e-4 等于把残差分支关掉。
        #    原文对浅层用 1e-1。此处按深度可配,默认 1e-1。
        self.ls1 = nn.Parameter(torch.full((d,), ls_init)) if layerscale else None
        self.ls2 = nn.Parameter(torch.full((d,), ls_init)) if layerscale else None
        self.rz = nn.Parameter(torch.zeros(1)) if rezero else None

    def _drop(self, y):
        if self.training and self.dp > 0:
            keep = 1.0 - self.dp
            m = torch.rand(y.shape[0], 1, 1, device=y.device) < keep
            return y * m / keep
        return y

    def _scale(self, y, ls):
        if ls is not None:
            y = y * ls
        if self.rz is not None:
            y = y * self.rz
        return self._drop(y)

    def _sa(self, h):
        q = k = rope(h, self.heads) if self.use_rope else h
        return self.attn(q, k, h)[0]

    def forward(self, x):
        if self.preln:                                   # pre-LN:归一化在残差分支内
            x = x + self._scale(self._sa(self.n1(x)), self.ls1)
            return x + self._scale(self.ff(self.n2(x)), self.ls2)
        x = self.n1(x + self._scale(self._sa(x), self.ls1))      # post-LN(现状)
        return self.n2(x + self._scale(self.ff(x), self.ls2))


class Net(nn.Module):
    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256, n_bands=16,
                 levelshape=False, **kw):
        super().__init__()
        self.n_bands, self.n_t, self.n_ch = n_bands, n_t, n_ch
        self.cond = nn.Linear(d_in, d)
        self.pos_proj = nn.Linear(2 * n_bands, d)
        self.blocks = nn.ModuleList([Layer(d, heads, ff, **kw) for _ in range(layers)])
        # 🔴 修正(2026-08-28):pre-LN 的残差流从不被归一化,必须在最后补一层 norm。
        #    GPT / ViT / LLaMA 的标准实现全都有这一层;我此前漏了,导致 head 拿到未归一化的表示。
        #    这是 pre-LN 在封存 test 上反而变差(0.383% vs 0.372%)、且方差是基线 2.3 倍的直接原因候选。
        N = RMSNorm if kw.get("rms") else nn.LayerNorm
        self.final_norm = N(d) if kw.get("preln") else None
        self.head = nn.Linear(d, n_ch)
        self.levelshape = levelshape
        if levelshape:
            # RevIN 思想的输出侧适配:分头预测「每通道整体水平」与「归一化形状」
            self.lvl = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, n_ch))

    def forward(self, x):
        pos = self.pos_proj(fourier_pos(self.n_t, self.n_bands, x.device))[None]
        h = self.cond(x)[:, None, :] + pos
        for b in self.blocks:
            h = b(h)
        if self.final_norm is not None:
            h = self.final_norm(h)
        y = self.head(h).transpose(1, 2)
        if self.levelshape:
            y = y - y.mean(-1, keepdim=True)              # 只留形状
            y = y + self.lvl(x)[:, :, None]               # 水平由专门的头给
        return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--screen-seeds", type=int, default=4)
    ap.add_argument("--test-seeds", type=int, default=8)
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
    print(f"总样本 {n}  训练 {len(tr)} / val {len(va)} / test {len(te)}(封存)\n")

    def run(ev_rows, *, seed, lr=3e-3, **kw):
        ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
        xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        net = Net(D, NCH, NT, **kw).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        A = torch.tensor(X[tr], device=dev)
        B = torch.tensor(((Yf[tr]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            net.train()
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
        return (float(rel.mean()),
                float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()),
                sum(p.numel() for p in net.parameters()))

    PLUG = {
        "定版(post-LN, 现状)":            dict(),
        "🔴 preln [ICML2020]":            dict(preln=True),
        "rmsnorm [NeurIPS2019]":          dict(rms=True),
        "swiglu [Shazeer2020]":           dict(swiglu=True),
        "layerscale [ICCV2021]":          dict(layerscale=True),
        "rezero [UAI2021]":               dict(rezero=True),
        "rope [Su2024]":                  dict(use_rope=True),
        "droppath 0.1 [ECCV2016]":        dict(droppath=0.1),
        "levelshape [RevIN 适配]":         dict(levelshape=True),
    }
    t0 = time.time()
    # 🔴 修正:此前所有配置共用 lr=3e-3(post-LN 调出来的值)。
    #    pre-LN 的核心卖点恰恰是**允许更大的学习率**(无需 warmup),
    #    拿 post-LN 的 lr 去跑它是我自己批评过的那种不公平。本轮逐配置扫 lr。
    LRS = (1e-3, 3e-3, 1e-2)
    print(f"=== 1. val 筛选({args.screen_seeds} 种子;**逐配置扫 lr {LRS}**;test 封存不碰) ===")
    scr = {}
    for tag, kw in PLUG.items():
        best = None
        for lr in LRS:
            a = np.array([run(va, seed=s, lr=lr, **kw) for s in range(args.screen_seeds)])
            m = float(a[:, 0].mean())
            if best is None or m < best[1]:
                best = (lr, m, float(a[:, 0].std(ddof=1)), a[:, 0].tolist(), int(a[0, 2]))
        scr[tag] = {"lr": best[0], "mean": best[1], "sd": best[2],
                    "raw": best[3], "params": best[4], "kw": {k: str(v) for k, v in kw.items()}}
        r = scr[tag]
        print(f"  {tag:28s} val {r['mean']:.3%} ± {r['sd']:.3%}  lr={r['lr']:<6g} "
              f"参数 {r['params']:>9,d}   ({(time.time()-t0)/60:.0f}min)", flush=True)

    base_v = scr["定版(post-LN, 现状)"]["mean"]
    keep = [t for t in PLUG if t != "定版(post-LN, 现状)" and scr[t]["mean"] < base_v]
    print(f"\n  → val 上优于定版的: {keep if keep else '无'}")

    print(f"\n=== 2. 封存 test({args.test_seeds} 种子;只评入选项 + 定版) ===")
    R = {}
    for tag in ["定版(post-LN, 现状)"] + keep:
        a = np.array([run(te, seed=s, lr=scr[tag]["lr"], **PLUG[tag]) for s in range(args.test_seeds)])
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "r2": float(a[:, 1].mean()), "params": int(a[0, 2]), "raw": a[:, 0].tolist()}
        r = R[tag]
        print(f"  {tag:28s} {r['mean']:.3%} ± {r['sd']:.3%}  R² {r['r2']:.4f}  "
              f"参数 {r['params']:>9,d}   ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    ref = np.array(R["定版(post-LN, 现状)"]["raw"])
    print(f"\n=== 3. 判定(vs 定版) ===")
    sig = {}
    for tag in R:
        if tag.startswith("定版"):
            continue
        a = np.array(R[tag]["raw"]); p = float(ttest_ind(a, ref, equal_var=False).pvalue)
        d = (a.mean() - ref.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[tag] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {tag:28s} {d:>+8.3f}pp  p={p:.4f}   {vd}")
    win = [t for t in sig if sig[t]["verdict"].startswith("✔")]
    print(f"\n{'🏆 有效的即插即用项: ' + ', '.join(win) if win else '🔴 无一项在 test 上显著变好'}")
    json.dump({"n_train": len(tr), "screen": scr, "test": R, "significance": sig,
               "winners": win, "screen_seeds": args.screen_seeds, "test_seeds": args.test_seeds,
               "note": "全部叠在定版(Transformer+傅里叶)上,一次只开一项;val 先筛,test 只评入选项。"},
              open(OUT / "plug.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/plug.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
