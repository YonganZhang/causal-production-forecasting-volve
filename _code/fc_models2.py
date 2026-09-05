#!/usr/bin/env python3
"""第二批骨干：把近年公认最强的时序模型拉进来做对照。

## 🔴 适配声明（必须写在最前面，审稿人一定会问）

这些模型原本都是 **history → future** 的预测器。本任务是 **θ(24维静态) → 轨迹(880维)**，
**推理时没有历史序列可编码**。所以每个模型都做了同一种适配：

> **把它的历史编码器换成 θ 条件注入，保留它各自的特征机制不动。**

这个适配对所有模型**一视同仁**（包括已有的 Transformer / Autoformer 基线），
所以横向比较是公平的；但它确实意味着这些模型**没有跑在它们的原生设定上**。
论文里必须原样声明这一句，不能假装是原版对比。

## 五个新骨干（选型依据：LTSF 领域公认的强基线 + 机制彼此正交）

| 模型 | 出处 | 特征机制 | 为什么值得试 |
|---|---|---|---|
| **DLinear** | Zeng et al. AAAI 2023 | 序列分解 + **沿时间轴的线性层** | 该文的著名结论是"线性层打败 Transformer"。**必须作为反面控制** —— 若它接近最优，说明所有深模型都是过度设计 |
| **PatchTST** | Nie et al. ICLR 2023 | **分块(patching)** + 通道独立 | 分块把时间轴降到 5 个 token，是与自注意力正交的归纳偏置 |
| **iTransformer** | Liu et al. ICLR 2024 | **倒置维度**：注意力作用在**变量间**而非时间上 | 本任务有 22 个通道(11井×2相)，井间干扰是真实物理，跨通道注意力天然对口 |
| **TimesNet** | Wu et al. ICLR 2023 | FFT 找主周期 → **1D 重排成 2D** → Inception 卷积 | 与我们已确认有效的傅里叶发现同源，但走的是"周期→二维"路线 |
| **N-BEATS** | Oreshkin et al. ICLR 2020 | **基展开** + 残差堆叠 | 🔴 最相关：我们唯一成立的改进就是基函数(傅里叶位置嵌入)。N-BEATS 是把基展开做成整个架构的模型 |

选 DLinear 作反面控制是刻意的：本项目已经反复出现"复杂的输给简单的"
(FNO 输给普通 Transformer、我的三模块输给基线)。若 DLinear 也很强，
那才是这篇论文真正该讲的故事。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Decomp(nn.Module):
    def __init__(self, k=13):
        super().__init__()
        self.k = k; self.avg = nn.AvgPool1d(k, stride=1)

    def forward(self, x):                                    # (B, C, T)
        p = self.k // 2
        xp = torch.cat([x[..., :1].repeat(1, 1, p), x, x[..., -1:].repeat(1, 1, p)], -1)
        tr = self.avg(xp)
        return x - tr, tr


class DLinear(nn.Module):
    """Zeng et al. AAAI 2023。机制:序列分解 + **沿时间轴**的线性层(逐通道共享)。

    适配:先用一层线性从 θ 得到初始轨迹估计,再按 DLinear 原样做分解 + 双分支时间线性。
    这样"时间轴线性层"这个核心机制被完整保留。
    """

    def __init__(self, d_in, n_ch, n_t, k=13, **_):
        super().__init__()
        self.n_ch, self.n_t = n_ch, n_t
        self.stem = nn.Linear(d_in, n_ch * n_t)
        self.dec = _Decomp(k)
        self.lin_s = nn.Linear(n_t, n_t)
        self.lin_t = nn.Linear(n_t, n_t)

    def forward(self, x):
        z = self.stem(x).view(-1, self.n_ch, self.n_t)
        s, t = self.dec(z)
        return self.lin_s(s) + self.lin_t(t)


class PatchTST(nn.Module):
    """Nie et al. ICLR 2023。机制:把时间轴切成 patch,注意力作用在 patch 之间。

    适配:没有历史可切,故用**可学习的 patch 查询** + θ 条件注入,
    再由 Transformer 在 patch 间交互,最后每个 patch 解码出 patch_len 个时刻。
    """

    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256, patch=8, **_):
        super().__init__()
        assert n_t % patch == 0, f"n_t={n_t} 必须能被 patch={patch} 整除"
        self.np, self.pl, self.n_ch = n_t // patch, patch, n_ch
        self.cond = nn.Linear(d_in, d)
        self.q = nn.Parameter(torch.randn(1, self.np, d) * 0.02)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0), layers)
        self.head = nn.Linear(d, n_ch * patch)

    def forward(self, x):
        h = self.tr(self.cond(x)[:, None, :] + self.q)       # (B, np, d)
        o = self.head(h).view(-1, self.np, self.n_ch, self.pl)
        return o.permute(0, 2, 1, 3).reshape(-1, self.n_ch, self.np * self.pl)


class ITransformer(nn.Module):
    """Liu et al. ICLR 2024。机制:**倒置** —— 每个变量(通道)整条序列当一个 token,
    注意力作用在**变量之间**,而不是时间步之间。

    本任务有 22 个通道(11 井 × 2 相),井间干扰是真实物理,跨通道注意力天然对口。
    适配:每个通道的 token = 共享的 θ 嵌入 + 该通道自己的可学习通道嵌入。
    """

    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256, **_):
        super().__init__()
        self.n_ch = n_ch
        self.cond = nn.Linear(d_in, d)
        self.ch_emb = nn.Parameter(torch.randn(1, n_ch, d) * 0.02)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0), layers)
        self.head = nn.Linear(d, n_t)

    def forward(self, x):
        h = self.cond(x)[:, None, :] + self.ch_emb           # (B, n_ch, d) —— token 是通道
        return self.head(self.tr(h))                         # (B, n_ch, n_t)


class TimesBlock(nn.Module):
    """TimesNet 核心:FFT 找 top-k 主周期 → 把 1D 序列重排成 (周期 × 周期数) 的 2D →
    用 2D Inception 卷积捕捉**周期内**与**周期间**的变化 → 按振幅加权求和。"""

    def __init__(self, d, n_t, k=3, hid=32):
        super().__init__()
        self.k, self.n_t = k, n_t
        self.conv = nn.Sequential(
            nn.Conv2d(d, hid, 3, padding=1), nn.GELU(), nn.Conv2d(hid, d, 3, padding=1))

    def forward(self, x):                                    # (B, T, d)
        B, T, D = x.shape
        amp = torch.fft.rfft(x, dim=1).abs().mean(-1)        # (B, F)
        amp[:, 0] = 0
        A = amp.mean(0)
        k = min(self.k, max(1, A.shape[0] - 1))
        # 🔴 一次性取回 top-k 索引。此前在循环里逐个 .item(),每次前向做 k 次
        #    GPU→CPU 同步;长跑(数百次训练)下触发过 cudaErrorLaunchFailure。
        top = torch.topk(A, k).indices.tolist()
        outs, ws = [], []
        for f in top:
            f = int(f)
            per = max(1, T // max(f, 1))
            pad = (-T) % per
            xp = F.pad(x.transpose(1, 2), (0, pad)).transpose(1, 2) if pad else x
            L = xp.shape[1]
            z = xp.reshape(B, L // per, per, D).permute(0, 3, 1, 2)   # (B,d,行,周期)
            z = self.conv(z).permute(0, 2, 3, 1).reshape(B, L, D)[:, :T]
            outs.append(z); ws.append(amp[:, f])
        w = torch.softmax(torch.stack(ws, -1), -1)           # (B,k)
        return x + sum(o * w[:, i][:, None, None] for i, o in enumerate(outs))


class TimesNet(nn.Module):
    """Wu et al. ICLR 2023。适配:θ 条件广播成时间序列后过 TimesBlock 堆叠。"""

    def __init__(self, d_in, n_ch, n_t, d=128, layers=2, k=3, **_):
        super().__init__()
        self.cond = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        self.blocks = nn.ModuleList([TimesBlock(d, n_t, k) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = nn.Linear(d, n_ch)

    def forward(self, x):
        h = self.cond(x)[:, None, :] + self.pos
        for b, n in zip(self.blocks, self.norms):
            h = n(b(h))
        return self.head(h).transpose(1, 2)


class NBeats(nn.Module):
    """Oreshkin et al. ICLR 2020。机制:**基展开 + 残差堆叠** ——
    每个 block 从残差里预测一组基系数,乘上基得到一份预测,再把残差减掉,层层累加。

    🔴 本项目唯一成立的改进(傅里叶位置嵌入)就是基函数方法,而 N-BEATS 把基展开
    做成了整个架构。所以它是本批里**先验最相关**的一个。

    适配:无历史 → 无 backcast,残差在 **θ 的隐表示**上做。
    """

    def __init__(self, d_in, n_ch, n_t, width=256, depth=3, blocks=4, basis=32, **_):
        super().__init__()
        self.n_ch, self.n_t, self.nb = n_ch, n_t, blocks
        self.mlps = nn.ModuleList()
        self.coef = nn.ModuleList()
        self.back = nn.ModuleList()
        for _ in range(blocks):
            L = [nn.Linear(d_in if not self.mlps else width, width), nn.GELU()]
            for _ in range(depth - 1):
                L += [nn.Linear(width, width), nn.GELU()]
            self.mlps.append(nn.Sequential(*L))
            self.coef.append(nn.Linear(width, n_ch * basis))
            self.back.append(nn.Linear(width, d_in))
        self.basis = nn.Parameter(torch.randn(basis, n_t) * (1.0 / np.sqrt(basis)))
        self.basis_dim = basis

    def forward(self, x):
        res, out = x, 0
        for i in range(self.nb):
            h = self.mlps[i](res if i == 0 else h_prev)
            h_prev = h
            c = self.coef[i](h).view(-1, self.n_ch, self.basis_dim)
            out = out + torch.einsum("bcp,pt->bct", c, self.basis)
            res = res - self.back[i](h)
        return out


ZOO2 = {"DLinear": DLinear, "PatchTST": PatchTST, "iTransformer": ITransformer,
        "TimesNet": TimesNet, "N-BEATS": NBeats}
