#!/usr/bin/env python3
"""模型库：参数化算子学习的架构对照 + 我们的改进。

## 任务定性(决定该用哪个领域的模型)

输入 24 维**静态**参数 → 输出 11井×2相×40时刻 = 880 维轨迹,**推理时不看历史曲线**。

所以这**不是时序预测任务**。Autoformer / Informer / Chronos / PatchTST 的架构前提是
"给一段历史,预测后一段",核心组件(自相关、序列分解、patch 嵌入)都作用在历史序列上;
本任务编码器端是空的 —— 是**架构错配**,不是调参问题。

正确的名字是**参数化算子学习**(learn G: θ∈R²⁴ → u(t)∈R^(22×40))。
油藏代理模型文献的主流是 **FNO** 与 **DeepONet**。Autoformer 仍作 baseline 报,
因为它是"合理但错配"的对照,审稿人一定会问。

## Gap 分析给出的改进依据(fc_gap.py 实测,非推测)

| 发现 | 数字 | 推论 |
|---|---|---|
| 误差几乎全是**形状**错,不是水平错 | 形状 98.8% / 水平 1.2% | 改的必须是时间轴的建模方式 |
| 误差能量**过度集中在高频** | 误差高频占 18.08%,而信号高频只占 4.63%(4×) | 典型**谱偏差** |
| 逐模态信噪比**单调衰减** | 模态0: 15559 → 模态20: 896(**17×**) | 高频模态学得最差 |
| 物理违背 | 负产 0、累计非单调 0.00%、含水越界 0.00% | 🔴 **物理约束无事可做**,别加 |
| 水突破时刻 | 误差 0 天(全部样本在预测段起点已突破) | 🔴 **相位不是 gap** |
| 异方差 | 误差与产量相关 +0.036 | 🔴 **不是 gap** |

**结论:唯一有证据支持的病灶是谱偏差。** 三条物理/相位/异方差路线全部被实测排除 ——
这省掉了三个方向的无用功,也是为什么先做 Gap 分析而不是直接堆模块。

现有 Transformer 的架构缺陷正对应这一点:
```
h = self.cond(x)[:, None, :] + self.pos      # 单个条件向量广播到 40 个时刻
```
时间结构**完全交给一个自由学习的 self.pos + 自注意力**,没有任何谱/光滑先验。
高频模态没有专属参数,自然学不好。

## 本文件的模型

| 类 | 是什么 | 出处 |
|---|---|---|
| `TFBase` | 现有 Transformer(基线) | — |
| `Autoformer` | 序列分解 + 自相关注意力 | Wu et al. NeurIPS 2021 |
| `DeepONet` | branch(θ) × trunk(t) | Lu et al. Nat. Mach. Intell. 2021 |
| `FNO1d` | 时间轴上的谱卷积,**每个傅里叶模态一组独立权重** | Li et al. ICLR 2021 |
| `OursNet` | 最优骨干 + 傅里叶特征嵌入 + 谱加权损失 | 本文,模块见下 |

`OursNet` 的两个模块都是**直接冲着谱偏差去的**:
- **傅里叶特征位置嵌入**(Tancik et al. NeurIPS 2020):把自由学习的 `pos` 换成
  多尺度 sin/cos 基。原论文证明这能让 MLP 摆脱谱偏差。
- **谱加权损失**:在频域按模态加权,权重随模态号上升,直接补偿 17× 的信噪比衰减。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 基线
class TFBase(nn.Module):
    """现有 Transformer。条件向量广播到各时刻 + 自由学习的位置嵌入。"""

    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256):
        super().__init__()
        self.cond = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0), layers)
        self.head = nn.Linear(d, n_ch)

    def forward(self, x, pos_override=None):
        pos = self.pos if pos_override is None else pos_override
        return self.head(self.tr(self.cond(x)[:, None, :] + pos)).transpose(1, 2)


class SeriesDecomp(nn.Module):
    """Autoformer 的序列分解:滑动平均得趋势,残差为季节项。"""

    def __init__(self, k=13):
        super().__init__()
        self.k = k
        self.avg = nn.AvgPool1d(k, stride=1, padding=0)

    def forward(self, x):                                   # x (B, T, D)
        pad = self.k // 2
        f = x[:, :1].repeat(1, pad, 1); b = x[:, -1:].repeat(1, pad, 1)
        trend = self.avg(torch.cat([f, x, b], 1).transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """Autoformer 的自相关:用 FFT 算时延相关,取 top-k 时延做聚合。"""

    def __init__(self, d, heads, factor=1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.factor = factor

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).permute(0, 2, 3, 1)
        k = self.k(x).view(B, T, self.h, self.dk).permute(0, 2, 3, 1)
        v = self.v(x).view(B, T, self.h, self.dk).permute(0, 2, 3, 1)
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=-1) * torch.conj(torch.fft.rfft(k, dim=-1)),
                               n=T, dim=-1)                 # (B,h,dk,T) 时延相关
        top = max(1, int(self.factor * np.log(max(T, 2))))
        # 🔴 按原文 time_delay_agg_training:时延在 batch 与 head 上取均值后共享,
        #    这样可以整张量 roll。逐 (b,h) 循环在 600 epoch 下慢到不可用。
        mean_corr = corr.mean(dim=(1, 2))                   # (B,T)
        idx = torch.topk(mean_corr.mean(0), top).indices    # (top,)
        p = torch.softmax(torch.stack([mean_corr[:, i] for i in idx], -1), -1)  # (B,top)
        out = torch.zeros_like(v)
        for i in range(top):
            out = out + torch.roll(v, -int(idx[i].item()), dims=-1) * p[:, i][:, None, None, None]
        return self.o(out.permute(0, 3, 1, 2).reshape(B, T, D))


class Autoformer(nn.Module):
    """Autoformer 骨干。🔴 无历史可编码,故为 decoder-only:条件向量广播后过
    (自相关 + 序列分解) 堆叠。这是把时序预测模型搬到本任务的**最合理**改法,
    也正因为如此,它的表现能说明"架构错配"这个论断成不成立。"""

    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=2, ff=256, k=13):
        super().__init__()
        self.cond = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        self.ac = nn.ModuleList([AutoCorrelation(d, heads) for _ in range(layers)])
        self.dc = nn.ModuleList([SeriesDecomp(k) for _ in range(2 * layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
                                  for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = nn.Linear(d, n_ch)
        self.thead = nn.Linear(d, n_ch)

    def forward(self, x, pos_override=None):
        h = self.cond(x)[:, None, :] + (self.pos if pos_override is None else pos_override)
        trend = 0
        for i, (ac, ffn, n1) in enumerate(zip(self.ac, self.ffn, self.n1)):
            h, t1 = self.dc[2 * i](n1(h + ac(h)))
            h, t2 = self.dc[2 * i + 1](h + ffn(h))
            trend = trend + t1 + t2
        return (self.head(h) + self.thead(trend)).transpose(1, 2)


class DeepONet(nn.Module):
    """DeepONet:branch 编码参数 θ,trunk 编码坐标 t,内积得 u(θ)(t)。
    算子学习的两大标准架构之一,天然匹配"参数 → 函数"这个任务形状。"""

    def __init__(self, d_in, n_ch, n_t, p=128, width=256, depth=4):
        super().__init__()
        def mlp(i, o):
            L = [nn.Linear(i, width), nn.GELU()]
            for _ in range(depth - 2):
                L += [nn.Linear(width, width), nn.GELU()]
            return nn.Sequential(*L, nn.Linear(width, o))
        self.branch = mlp(d_in, n_ch * p)
        self.trunk = mlp(1, p)
        self.p, self.n_ch = p, n_ch
        self.register_buffer("t", torch.linspace(0, 1, n_t)[:, None])
        self.b0 = nn.Parameter(torch.zeros(n_ch, n_t))

    def forward(self, x):
        b = self.branch(x).view(-1, self.n_ch, self.p)
        return torch.einsum("bcp,tp->bct", b, self.trunk(self.t)) + self.b0


class SpectralConv1d(nn.Module):
    """FNO 的核心:在傅里叶域对**每个模态**乘一组独立的可学习复权重。
    这正是谱偏差的对症药 —— 高频模态有自己的参数,不必与低频抢容量。"""

    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        s = 1.0 / (cin * cout)
        self.w = nn.Parameter(s * torch.randn(cin, cout, modes, 2))

    def forward(self, x):                                   # x (B, C, T)
        T = x.shape[-1]
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xf.shape[-1])
        w = torch.view_as_complex(self.w[..., :m, :].contiguous())
        out = torch.zeros(x.shape[0], self.w.shape[1], xf.shape[-1],
                          dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bim,iom->bom", xf[..., :m], w)
        return torch.fft.irfft(out, n=T, dim=-1)


class FNO1d(nn.Module):
    """FNO:把 θ 抬升成时间轴上的函数,再过若干谱卷积块。
    油藏代理模型文献的主流骨干(U-FNO 用于 CO₂ 封存等)。"""

    def __init__(self, d_in, n_ch, n_t, width=64, modes=16, layers=4):
        super().__init__()
        self.lift = nn.Linear(d_in + 1, width)              # +1 = 归一化时间坐标
        self.sp = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, n_ch))
        self.register_buffer("t", torch.linspace(0, 1, n_t))

    def forward(self, x):
        B, T = x.shape[0], self.t.shape[0]
        g = torch.cat([x[:, None, :].expand(B, T, x.shape[1]),
                       self.t[None, :, None].expand(B, T, 1)], -1)
        h = self.lift(g).transpose(1, 2)                    # (B, W, T)
        for sp, pw in zip(self.sp, self.pw):
            h = F.gelu(sp(h) + pw(h))
        return self.proj(h.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------- 我们的模型
def fourier_features(t, n_bands):
    """Tancik et al. NeurIPS 2020 的多尺度 sin/cos 基。
    原论文证明它能让网络摆脱谱偏差 —— 这正是 Gap 分析测到的病灶。"""
    f = 2.0 ** torch.arange(n_bands, device=t.device, dtype=t.dtype)   # 1,2,4,...
    a = t[:, None] * f[None, :] * 2 * np.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)                # (T, 2*n_bands)


class OursNet(nn.Module):
    """最优骨干 + 两个针对谱偏差的模块(可逐项消融)。

    ff   : 位置嵌入换成傅里叶特征(而非自由学习的 pos)
    film : 用 θ 对每个时刻做 FiLM 调制(而非单向量相加) —— 让条件能**逐时刻**
           改变响应,是"参数 → 轨迹"这个形状更自然的注入方式
    """

    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff_dim=256,
                 n_bands=8, use_ff=True, use_film=True):
        super().__init__()
        self.use_ff, self.use_film = use_ff, use_film
        self.cond = nn.Linear(d_in, d)
        self.register_buffer("t", torch.linspace(0, 1, n_t))
        if use_ff:
            self.pos_proj = nn.Linear(2 * n_bands, d)
            self.n_bands = n_bands
        else:
            self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        if use_film:
            self.film = nn.Linear(d_in, 2 * d)
            # 🔴 2026-08-27 修:原来用默认初始化,实测 g std 0.583 / b std 0.574,
            #    4.2% 的通道 1+g<0(增益翻号), h 在第 0 步就被放大 1.55× ——
            #    模块起点**不是恒等**,直接把基线砸坏。FiLM 的标准做法是末层置零。
            #    实测(N=1400, 6 种子, 封存 test):
            #      无 FiLM 0.6472% / 默认初始化 1.1291% / 零初始化 0.8873%
            #      → 零初始化回收 50% 的损伤 (p=6e-7);FiLM 本身仍然有害(+0.240pp)。
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff_dim, batch_first=True, dropout=0.0), layers)
        self.head = nn.Linear(d, n_ch)

    def forward(self, x):
        pos = (self.pos_proj(fourier_features(self.t, self.n_bands))[None]
               if self.use_ff else self.pos)
        h = self.cond(x)[:, None, :] + pos
        if self.use_film:
            g, b = self.film(x).chunk(2, -1)
            h = h * (1 + g[:, None, :]) + b[:, None, :]
        return self.head(self.tr(h)).transpose(1, 2)


def spectral_loss(pred, true, alpha=1.0, beta=0.5):
    """MSE + 频域按模态加权的误差。

    Gap 实测:模态 0 信噪比 15559,模态 20 只有 896(衰减 17×)。
    权重 (1+k)^beta 随模态号上升,直接补偿这个衰减。beta=0 时退化为普通频域 MSE。
    """
    mse = F.mse_loss(pred, true)
    fp, ft = torch.fft.rfft(pred, dim=-1), torch.fft.rfft(true, dim=-1)
    k = torch.arange(fp.shape[-1], device=pred.device, dtype=pred.dtype)
    w = (1.0 + k) ** beta
    sp = ((fp - ft).abs() ** 2 * w).mean() / w.mean() / pred.shape[-1]
    return mse + alpha * sp
