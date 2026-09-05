#!/usr/bin/env python3
"""fc_confirm 的可追溯性伴随脚本：把聚合数字拆回**每一条 shard**。

`confirm.json` 只有聚合值。本项目撤稿清单 15 条的共同模式是"没检查这个数怎么来的"，
所以这里用与 fc_confirm.py **逐字相同**的配置重训一个种子，把
每个 confirm 样本的场级/每井相对误差落盘到 `confirm_per_sample.npz`，
并打印误差最大的 15 条对应的 shard 文件名 —— 任何聚合数字都能一路查到原始 npz。

用法:
    python fc_confirm_trace.py --gpu 0 --seed 0
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import forecast_gen as FG
import norne_bulk as NB
from fc_mech import PosNet
import fc_confirm as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}" if a.gpu >= 0 else "cpu"

    TH1, Y1, _, _ = C.load_dir(C.V1, "v1_8000")
    THc, Yc, _, _ = C.load_dir(C.CONF, "confirm_500")
    live = ~(np.abs(Y1).max(axis=(0, 3)) < 1e-9).all(1)
    WELLS = [NB.PRODUCERS[i] for i in np.where(live)[0]]
    nw = len(WELLS); NCH = nw * C.NPH; NT = C.NT
    Yf1 = Y1[:, live].reshape(len(TH1), -1); Yfc = Yc[:, live].reshape(len(THc), -1)

    ym, ys = Yf1.mean(0), Yf1.std(0) + 1e-8
    xm, xs = TH1.mean(0), TH1.std(0) + 1e-8
    Xtr = torch.tensor(((TH1 - xm) / xs).astype(np.float32), device=dev)
    Ytr = torch.tensor(((Yf1 - ym) / ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
    torch.manual_seed(a.seed)
    net = PosNet(TH1.shape[1], NCH, NT, "fourier", d=C.D_MODEL, heads=C.HEADS,
                 layers=C.LAYERS, ff=C.FF, n_bands=C.N_BANDS, seed=a.seed).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=C.LR, weight_decay=C.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for _ in range(a.epochs):
        net.train()
        pm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), C.BATCH):
            b = pm[i:i + C.BATCH]; opt.zero_grad()
            F.mse_loss(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(((THc - xm) / xs).astype(np.float32),
                             device=dev)).cpu().numpy().reshape(len(THc), -1) * ys + ym
    cP, cT = C.cum_oil(P, nw), C.cum_oil(Yfc, nw)
    fld = np.abs(cP.sum(1) - cT.sum(1)) / np.abs(cT.sum(1))
    per = np.abs(cP - cT) / np.maximum(np.abs(cT), 1e-9)
    shards = np.array([p.name for p in sorted((C.CONF / "shards").glob("*.npz"))])
    out = C.OUT / "confirm_per_sample.npz"
    np.savez_compressed(out, shard=shards, field_relerr=fld, well_relerr=per,
                        cum_true=cT, cum_pred=cP, wells=np.array(WELLS), seed=a.seed)
    print(f"seed {a.seed}  场级 {fld.mean():.4%}  每井 {per.mean():.4%}  "
          f"p90 {np.quantile(fld,.9):.4%}")
    print(f"\n误差最大的 15 条(可直接打开对应 npz 核对):")
    for i in np.argsort(-fld)[:15]:
        print(f"  {shards[i]}  场级 {fld[i]:.4%}  真值 {cT[i].sum():.4e}  预测 {cP[i].sum():.4e}")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
