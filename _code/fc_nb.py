#!/usr/bin/env python3
"""干净的 n_bands 选型：修掉 test 污染。

Codex 指出:`fc_arch2` 把 n_bands=4/8/16 **直接在 test 上比较**,所以"选 16"这个动作
是 test-set selection bias;`fc_final2` 虽然在 val 上扫了 {16,32,64},但 4 和 8
从未在 val 上比过 —— 16 之所以进入候选集,来源仍可追溯到那次 test 上的比较。

本脚本在 **val** 上扫完整的 {4,8,16,32},冻结选择后**只对 test 评一次**。
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import fc_decision as FD, forecast_gen as FG

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
from fc_mech import PosNet

NPH, NT = 2, FG.FC_N; DAYS = FG.FC_GRID
OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_nb"

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=7); ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--epochs", type=int, default=600)
a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
dev = f"cuda:{a.gpu}"
TH, Y, IA, FC = FD.load(N_PIN)
live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1); Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
idx = np.random.default_rng(0).permutation(n)
te, va, tr = idx[:400], idx[400:600], idx[600:]; trva = np.concatenate([tr, va])
Yf = Y.reshape(n, -1); NCH = nw * NPH
cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)

def go(fit, ev, nb, seed):
    ym, ys = Yf[fit].mean(0), Yf[fit].std(0)+1e-8
    xm, xs = TH[fit].mean(0), TH[fit].std(0)+1e-8
    X = ((TH-xm)/xs).astype(np.float32)
    torch.manual_seed(seed)
    net = PosNet(TH.shape[1], NCH, NT, "learnable" if nb == 0 else "fourier", n_bands=max(nb,1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    A_ = torch.tensor(X[fit], device=dev)
    B_ = torch.tensor(((Yf[fit]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for _ in range(a.epochs):
        pm = torch.randperm(len(A_), device=dev)
        for i in range(0, len(A_), 64):
            b = pm[i:i+64]; opt.zero_grad(); F.mse_loss(net(A_[b]), B_[b]).backward(); opt.step()
        s.step()
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(X[ev], device=dev)).cpu().numpy().reshape(len(ev), -1)*ys+ym
    ct, cp = cum(Yf[ev]), cum(P)
    r = np.abs(cp-ct)/np.abs(ct)
    return float(r.mean()), float(1-((cp-ct)**2).sum()/((ct-ct.mean())**2).sum())

t0 = time.time(); rows = []
print(f"=== val 上扫完整 n_bands(test 封存) {a.seeds} 种子 ===")
for nb in (0, 4, 8, 16, 32):
    e = np.array([go(tr, va, nb, s)[0] for s in range(a.seeds)])*100
    rows.append({"n_bands": nb, "val": float(e.mean()), "sd": float(e.std(ddof=1))})
    print(f"  n_bands={nb if nb else '基线可学习':<12} val {e.mean():.3f}% ± {e.std(ddof=1):.3f}"
          f"  ({(time.time()-t0)/60:.0f}min)", flush=True)
best = min([r for r in rows if r["n_bands"] > 0], key=lambda r: r["val"])["n_bands"]
print(f"\n→ val 选定 n_bands={best}(冻结,不再改)\n")
print(f"=== 封存 test 只评一次 ===")
R = {}
for tag, nb in (("基线可学习 pos", 0), (f"傅里叶 n_bands={best}", best)):
    z = np.array([go(trva, te, nb, s) for s in range(12)])
    R[tag] = {"mean": float(z[:,0].mean()*100), "sd": float(z[:,0].std(ddof=1)*100), "r2": float(z[:,1].mean())}
    print(f"  {tag:22s} {R[tag]['mean']:.3f}% ± {R[tag]['sd']:.3f}  R² {R[tag]['r2']:.4f}", flush=True)
json.dump({"val_sweep": rows, "selected": best, "test": R,
           "protocol": "n_bands 完整在 val 上选;test 只评一次(修掉 fc_arch2 的 test 选型污染)"},
          open(OUT/"nb.json","w"), indent=1, ensure_ascii=False)
print(f"\n已写入 {OUT}/nb.json")
