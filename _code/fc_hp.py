#!/usr/bin/env python3
"""模块超参搜索：认真检验"六个模块失败是超参没调好"这个假设。

## 用户的质疑，我认下来的部分

我此前**只扫过 `lr ∈ {1e-3, 3e-3}`**,从没调过 weight_decay、dropout、
模块自身的宽度、也没给新模块单独设参数组学习率。
而通道头一加就是 **+20 万参数**,却沿用同一套正则化 —— 过拟合几乎是必然的。
这是真的方法缺陷,不是运气。

## 本轮同时变了两件事，所以两条都要单独看

1. **正经的超参搜索**:模块宽度 × weight_decay × dropout × 模块学习率倍率;
2. **训练样本从 1400 涨到 3400**(扩样本已完成)。
   数据变多后**容量瓶颈可能才浮现**,模块此时才有机会派上用场 ——
   这也是"数据受限"假设的一个可证伪推论:若模块在 3400 样本下仍然全输,
   那它就不是"当时数据不够所以显不出来"。

## 纪律

- 搜索**只看 val**,`test=idx[:400]` 封存;
- 选定后 test 只评一次,与基线/傅里叶同口径对比;
- 模块无效就直说,不因为"这次调了参"而放宽判据。

用法:
    python fc_hp.py --gpu 1
"""
from __future__ import annotations

import argparse
import itertools
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

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_hp"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID


def fourier_pos(n_t, n_bands, device):
    t = torch.linspace(0, 1, n_t, device=device)
    f = 2.0 ** torch.arange(n_bands, device=device, dtype=t.dtype)
    a = t[:, None] * f[None, :] * 2 * np.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class ChannelHead(nn.Module):
    """潜空间通道交互头。相对 fc_block 版新增:可配宽度 dh + dropout。

    dh 是本轮最关键的旋钮 —— 此前固定 dh=d=128,一加就是 +20 万参数;
    dh=32 时只 +5 万,是"模块有用但太大所以过拟合"这个假设的直接检验。
    """

    def __init__(self, n_ch, n_t, d, dh=128, heads=4, drop=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_ch, dh) * 0.02)
        self.kv = nn.Linear(d, dh)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dh) for _ in range(3))
        h = max(1, min(heads, dh // 16))
        self.cross = nn.MultiheadAttention(dh, h, batch_first=True, dropout=drop)
        self.self_ = nn.MultiheadAttention(dh, h, batch_first=True, dropout=drop)
        self.ff = nn.Sequential(nn.Linear(dh, 2 * dh), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(2 * dh, dh))
        self.drop = nn.Dropout(drop)
        self.dec = nn.Linear(dh, n_t)

    def forward(self, h):
        kv = self.kv(h)
        q = self.q.expand(h.shape[0], -1, -1)
        a, _ = self.cross(self.n1(q), kv, kv); q = q + self.drop(a)
        s, _ = self.self_(self.n2(q), self.n2(q), self.n2(q)); q = q + self.drop(s)
        q = q + self.ff(self.n3(q))
        return self.dec(q)


class Net(nn.Module):
    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256,
                 n_bands=16, chan=False, dh=128, drop=0.0):
        super().__init__()
        self.n_bands, self.n_t, self.chan = n_bands, n_t, chan
        self.cond = nn.Linear(d_in, d)
        self.pos_proj = nn.Linear(2 * n_bands, d)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=drop), layers)
        self.head = ChannelHead(n_ch, n_t, d, dh, heads, drop) if chan else nn.Linear(d, n_ch)

    def forward(self, x):
        pos = self.pos_proj(fourier_pos(self.n_t, self.n_bands, x.device))[None]
        h = self.tr(self.cond(x)[:, None, :] + pos)
        return self.head(h) if self.chan else self.head(h).transpose(1, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--ntrain", type=int, default=3400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    TH, Y, IA, FC = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(args.split_seed).permutation(n)
    te, va, pool = idx[:400], idx[400:600], idx[600:]
    tr = pool[:min(args.ntrain, len(pool))]
    Yf = Y.reshape(n, -1); NCH = nw * NPH; D = TH.shape[1]
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    print(f"总样本 {n}  训练 {len(tr)} / val {len(va)} / test {len(te)}(封存)\n")

    def run(ev_rows, *, seed, lr, wd, lrmult=1.0, base=None, **kw):
        ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
        xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
        X = ((TH - xm) / xs).astype(np.float32)
        torch.manual_seed(seed)
        if base == "tf":
            net = M.TFBase(D, NCH, NT).to(dev)
        elif base == "ff":
            net = PosNet(D, NCH, NT, "fourier", n_bands=16).to(dev)
        else:
            net = Net(D, NCH, NT, **kw).to(dev)
        # 模块参数单独一组:此前从未这样做过(修"新模块沿用骨干学习率"这个缺陷)
        if base is None and kw.get("chan") and lrmult != 1.0:
            hp = list(net.head.parameters()); hid = {id(p) for p in hp}
            groups = [{"params": [p for p in net.parameters() if id(p) not in hid], "lr": lr},
                      {"params": hp, "lr": lr * lrmult}]
        else:
            groups = [{"params": list(net.parameters()), "lr": lr}]
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
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
            P = net(torch.tensor(X[ev_rows], device=dev)).cpu().numpy().reshape(len(ev_rows), -1)
        P = P * ys + ym
        ct, cp = cum(Yf[ev_rows]), cum(P)
        rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
        return (float(rel.mean()), float(1 - ((cp-ct)**2).sum() / ((ct-ct.mean())**2).sum()),
                sum(p.numel() for p in net.parameters()))

    t0 = time.time()
    print("=== 1. 模块超参搜索(只看 val;此前只扫过 lr,这是真正的搜索) ===")
    print(f"  {'dh':>4}{'wd':>8}{'drop':>6}{'lr':>7}{'模块lr倍率':>10}{'val':>9}{'参数':>10}")
    rows = []
    for dh, wd, dp, lr, lm in itertools.product(
            (32, 128), (1e-4, 1e-3), (0.0, 0.1), (1e-3, 3e-3), (1.0, 0.3)):
        e, r2, npar = run(va, seed=0, lr=lr, wd=wd, lrmult=lm, chan=True, dh=dh, drop=dp)
        rows.append({"dh": dh, "wd": wd, "drop": dp, "lr": lr, "lrmult": lm,
                     "val": e, "params": npar})
        print(f"  {dh:>4}{wd:>8.0e}{dp:>6.1f}{lr:>7.0e}{lm:>10.1f}{e:>9.3%}{npar:>10,d}"
              f"   ({(time.time()-t0)/60:.0f}min)", flush=True)
    best = min(rows, key=lambda r: r["val"])
    print(f"\n  → 最佳: dh={best['dh']} wd={best['wd']:.0e} drop={best['drop']} "
          f"lr={best['lr']:.0e} 模块lr倍率={best['lrmult']}  val {best['val']:.3%}")

    print(f"\n=== 2. 封存 test({args.seeds} 种子, 与基线/傅里叶同口径) ===")
    R = {}
    CFG = [("基线 Transformer", dict(base="tf", lr=3e-3, wd=1e-4)),
           ("+傅里叶(现最优)", dict(base="ff", lr=3e-3, wd=1e-4)),
           (f"+傅里叶+通道头(调参后 dh={best['dh']})",
            dict(lr=best["lr"], wd=best["wd"], lrmult=best["lrmult"],
                 chan=True, dh=best["dh"], drop=best["drop"]))]
    for tag, kw in CFG:
        a = np.array([run(te, seed=s, **kw) for s in range(args.seeds)])
        R[tag] = {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                  "r2": float(a[:, 1].mean()), "params": int(a[0, 2]), "raw": a[:, 0].tolist()}
        r = R[tag]
        print(f"  {tag:34s} {r['mean']:.3%} ± {r['sd']:.3%}  R² {r['r2']:.4f}  "
              f"参数 {r['params']:>9,d}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    from scipy.stats import ttest_ind
    ref = np.array(R["+傅里叶(现最优)"]["raw"])
    print(f"\n=== 3. 判定(vs 现最优「+傅里叶」) ===")
    sig = {}
    for tag in R:
        if tag == "+傅里叶(现最优)":
            continue
        a = np.array(R[tag]["raw"]); p = float(ttest_ind(a, ref, equal_var=False).pvalue)
        d = (a.mean() - ref.mean()) * 100
        vd = "✔ 显著变好" if p < .05 and d < 0 else "🔴 显著变差" if p < .05 else "— 噪声内"
        sig[tag] = {"delta_pp": d, "p": p, "verdict": vd}
        print(f"  {tag:34s} {d:>+8.3f}pp  p={p:.4f}   {vd}")
    json.dump({"n_train": len(tr), "grid": rows, "best_cfg": best, "results": R,
               "significance": sig, "seeds": args.seeds,
               "note": ("本轮同时变了两件事:(a) 正经的模块超参搜索(此前只扫 lr);"
                        "(b) 训练样本 1400→3400。若模块在 3400 样本 + 调参后仍输,"
                        "则'当时数据不够所以显不出来'这个辩解不成立。")},
              open(OUT / "hp.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/hp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
