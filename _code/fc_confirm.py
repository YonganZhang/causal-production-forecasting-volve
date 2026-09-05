#!/usr/bin/env python3
"""在**从未参与任何训练或模块选择**的 500 个确认集样本上，评一次定版代理模型。

## 为什么必须做

此前报的 0.366%(场级)/0.447%(每井) 是在 `norne_fc` 的 test-400 上得到的。
那 400 条经历了十几轮模块比较、架构选择、学习率选择 —— 它已经不是 sealed test，
而是一个被反复窥视的选择集。任何在它上面取胜的配置都带选择偏差。

`norne_fc_confirm`(种子 20260828, 500 样本)在本脚本之前**从未被任何脚本读过**
(`grep -rl norne_fc_confirm _code/` 空)。它是真正的一次性确认集。

## 三个口径，一次跑完

| 配置 | 训练集 | 评估集 | 回答什么 |
|---|---|---|---|
| MAIN  | v1 全部 8000 | confirm 500 | 定版模型在干净集上到底多准 |
| CTRL  | v1 8000 **去掉旧 test-400** = 7600 | 旧 test-400 **与** confirm 500 | 同一个模型、同一次训练，两个集合的差 = **纯粹的集合效应**(选择偏差的直接证据) |
| TRIV  | 训练集均值曲线 | 同上 | 平凡基线参照 |

CTRL 是关键：MAIN 与历史 0.366% 之间夹着两个变量(训练量 3400→8000、评估集换了)。
CTRL 把训练固定住，只让评估集变，于是差值可以直接归因给集合。

## 纪律
- 归一化统计量(xm/xs/ym/ys)只从各自的训练集算，绝不碰评估集。
- live 井掩码从**训练集**算，再原样套到 confirm 上(并断言 confirm 的 live 集合一致)。
- 6 种子，单模型不集成，配置写死在下面，跑之前不改。
- confirm 里有 1 条 θ 与 v1 的 baseline(全零)重合 —— 单独剔除后再报一次。

用法:
    python fc_confirm.py --gpu 5 --seeds 6
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import forecast_gen as FG
import norne_bulk as NB
from fc_mech import PosNet          # 定版骨干：Transformer + 傅里叶位置嵌入

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_confirm"
CACHE = OUT / "_cache"
V1 = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_fc")
CONF = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_fc_confirm")

NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID
# 定版超参(fc_axial.py / fc_4k.py 已固定，本脚本不再搜索)
D_MODEL, HEADS, LAYERS, FF, N_BANDS = 128, 4, 3, 256, 16
LR, EPOCHS, BATCH, WD = 3e-3, 600, 64, 1e-4


def load_dir(d: Path, tag: str):
    """读一个 shard 目录 → (TH, Y, IA, FC)。带磁盘缓存。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / f"{tag}.npz"
    if c.exists():
        z = np.load(c)
        return z["TH"], z["Y"], z["IA"], z["FC"]
    sh = sorted((d / "shards").glob("*.npz"))
    TH, Y, IA, FC = [], [], [], []
    for p in sh:
        z = np.load(p)
        ob = z["obs"]
        TH.append(z["theta"].ravel())
        IA.append(z["inj_actual"]); FC.append(z["field_cum"])
        Y.append(np.stack([[ob[(i * 3 + k) * NT:(i * 3 + k + 1) * NT] for k in (0, 1)]
                           for i in range(len(NB.PRODUCERS))]))
    TH = np.stack(TH).astype(np.float32); Y = np.stack(Y).astype(np.float32)
    IA = np.stack(IA).astype(np.float32); FC = np.stack(FC).astype(np.float32)
    np.savez_compressed(c, TH=TH, Y=Y, IA=IA, FC=FC)
    print(f"  {tag}: {len(TH)} 样本  Y{Y.shape}  → 缓存 {c.name}", flush=True)
    return TH, Y, IA, FC


def cum_oil(A, nw):
    """(N, nw*NPH*NT) → 每井累计产油 (N, nw)。相位 0 = 产油率。"""
    return np.trapezoid(A.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1)


def metrics(cP, cT):
    """cP/cT: (N, nw) 每井累计产油。返回场级 + 每井两个口径。"""
    fP, fT = cP.sum(1), cT.sum(1)
    fr = np.abs(fP - fT) / np.abs(fT)
    pr = (np.abs(cP - cT) / np.maximum(np.abs(cT), 1e-9)).ravel()
    ss = lambda p, t: float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return {"field_rel": float(fr.mean()), "field_med": float(np.median(fr)),
            "field_p90": float(np.quantile(fr, .9)), "field_max": float(fr.max()),
            "field_r2": ss(fP, fT),
            "well_rel": float(pr.mean()), "well_med": float(np.median(pr)),
            "well_p90": float(np.quantile(pr, .9)), "well_max": float(pr.max()),
            "well_r2": ss(cP.ravel(), cT.ravel())}


def agg(rows):
    """多种子 → 均值 ± 标准差(ddof=1)。"""
    ks = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "sd": float(np.std([r[k] for r in rows], ddof=1)),
                "raw": [float(r[k]) for r in rows]} for k in ks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    t0 = time.time()

    print("=== 读数据 ===", flush=True)
    TH1, Y1, IA1, FC1 = load_dir(V1, "v1_8000")
    THc, Yc, IAc, FCc = load_dir(CONF, "confirm_500")
    n1, nc = len(TH1), len(THc)

    # live 井掩码：只从训练集(v1)算，套到 confirm
    live = ~(np.abs(Y1).max(axis=(0, 3)) < 1e-9).all(1)
    live_c = ~(np.abs(Yc).max(axis=(0, 3)) < 1e-9).all(1)
    assert (live == live_c).all(), f"live 井集合不一致 v1={live.sum()} conf={live_c.sum()}"
    WELLS = [NB.PRODUCERS[i] for i in np.where(live)[0]]
    nw = len(WELLS); NCH = nw * NPH
    Y1 = Y1[:, live]; Yc = Yc[:, live]
    Yf1 = Y1.reshape(n1, -1); Yfc = Yc.reshape(nc, -1)
    print(f"  live 井 {nw}: {WELLS}")

    # 旧 test-400：rng(0).permutation(4000)[:400] —— 与 fc_axial/fc_4k/fc_run 完全一致
    old_te = np.random.default_rng(0).permutation(4000)[:400]

    # confirm 中与 v1 重合的 θ(baseline 全零)
    h1 = {hashlib.md5(r.tobytes()).hexdigest() for r in TH1}
    dup = np.array([hashlib.md5(r.tobytes()).hexdigest() in h1 for r in THc])
    print(f"  θ 与 v1 重合的 confirm 样本: {int(dup.sum())} 条 (索引 {np.where(dup)[0].tolist()})")
    keep = ~dup

    cT_conf = cum_oil(Yfc, nw)
    cT_old = cum_oil(Yf1[old_te], nw)

    def train_eval(tr_idx, seed):
        ym, ys = Yf1[tr_idx].mean(0), Yf1[tr_idx].std(0) + 1e-8
        xm, xs = TH1[tr_idx].mean(0), TH1[tr_idx].std(0) + 1e-8
        Xtr = torch.tensor(((TH1[tr_idx] - xm) / xs).astype(np.float32), device=dev)
        Ytr = torch.tensor(((Yf1[tr_idx] - ym) / ys).reshape(-1, NCH, NT).astype(np.float32),
                           device=dev)
        torch.manual_seed(seed)
        net = PosNet(TH1.shape[1], NCH, NT, "fourier", d=D_MODEL, heads=HEADS,
                     layers=LAYERS, ff=FF, n_bands=N_BANDS, seed=seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            net.train()
            pm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), BATCH):
                b = pm[i:i + BATCH]; opt.zero_grad()
                F.mse_loss(net(Xtr[b]), Ytr[b]).backward(); opt.step()
            sch.step()
        net.eval()

        def pred(THx):
            with torch.no_grad():
                P = net(torch.tensor(((THx - xm) / xs).astype(np.float32),
                                     device=dev)).cpu().numpy().reshape(len(THx), -1)
            return cum_oil(P * ys + ym, nw)
        # 平凡基线：训练集均值曲线
        triv = cum_oil(np.repeat(ym[None], 1, 0), nw)[0]
        return pred, triv, sum(p.numel() for p in net.parameters())

    R = {}
    # ---------- MAIN: 训练 8000 全量, 评 confirm 500 ----------
    print(f"\n=== MAIN: 训练 v1 全部 {n1} → 评 confirm {nc} ({args.seeds} 种子) ===", flush=True)
    tr_all = np.arange(n1)
    rows_c, rows_ck, npar = [], [], 0
    for s in range(args.seeds):
        pred, triv, npar = train_eval(tr_all, s)
        cP = pred(THc)
        rows_c.append(metrics(cP, cT_conf))
        rows_ck.append(metrics(cP[keep], cT_conf[keep]))
        print(f"  seed {s}: confirm 场级 {rows_c[-1]['field_rel']:.4%}  "
              f"每井 {rows_c[-1]['well_rel']:.4%}  R² {rows_c[-1]['field_r2']:.4f}"
              f"   ({(time.time()-t0)/60:.0f}min)", flush=True)
    R["MAIN_confirm500"] = agg(rows_c)
    R["MAIN_confirm499_dedup"] = agg(rows_ck)
    R["params"] = int(npar)
    # 平凡基线(与训练集无关的确定量，单次即可)
    trivP = np.repeat(triv[None], nc, 0)
    R["TRIVIAL_confirm500"] = metrics(trivP, cT_conf)
    R["TRIVIAL_oldtest400"] = metrics(np.repeat(triv[None], 400, 0), cT_old)

    # ---------- CTRL: 训练 8000-旧test400, 同一模型评两个集合 ----------
    tr_ctrl = np.setdiff1d(np.arange(n1), old_te)
    print(f"\n=== CTRL: 训练 {len(tr_ctrl)} (= 8000 − 旧 test 400) → 同时评"
          f" 旧test400 与 confirm500 ({args.seeds} 种子) ===", flush=True)
    rows_o, rows_c2 = [], []
    for s in range(args.seeds):
        pred, _, _ = train_eval(tr_ctrl, s)
        mo = metrics(pred(TH1[old_te]), cT_old)
        mc = metrics(pred(THc), cT_conf)
        rows_o.append(mo); rows_c2.append(mc)
        print(f"  seed {s}: 旧test 场级 {mo['field_rel']:.4%} 每井 {mo['well_rel']:.4%}"
              f" | confirm 场级 {mc['field_rel']:.4%} 每井 {mc['well_rel']:.4%}"
              f"   ({(time.time()-t0)/60:.0f}min)", flush=True)
    R["CTRL_oldtest400"] = agg(rows_o)
    R["CTRL_confirm500"] = agg(rows_c2)

    # ---------- 判定 ----------
    from scipy.stats import ttest_rel
    a = np.array(R["CTRL_confirm500"]["field_rel"]["raw"])
    b = np.array(R["CTRL_oldtest400"]["field_rel"]["raw"])
    pf = float(ttest_rel(a, b).pvalue)
    aw = np.array(R["CTRL_confirm500"]["well_rel"]["raw"])
    bw = np.array(R["CTRL_oldtest400"]["well_rel"]["raw"])
    pw = float(ttest_rel(aw, bw).pvalue)
    R["set_effect"] = {
        "field_delta_pp": float((a.mean() - b.mean()) * 100), "field_p": pf,
        "well_delta_pp": float((aw.mean() - bw.mean()) * 100), "well_p": pw,
        "note": "同一次训练(7600 样本，两个集合都不在训练里)下两个评估集的差 = 纯集合效应"}
    R["meta"] = {"seeds": args.seeds, "epochs": args.epochs, "n_train_main": int(n1),
                 "n_train_ctrl": int(len(tr_ctrl)), "n_confirm": int(nc),
                 "wells": WELLS, "n_live_wells": nw,
                 "arch": {"backbone": "Transformer+傅里叶位置嵌入(PosNet fourier)",
                          "d": D_MODEL, "heads": HEADS, "layers": LAYERS, "ff": FF,
                          "n_bands": N_BANDS, "lr": LR, "batch": BATCH, "wd": WD},
                 "confirm_dir": str(CONF), "v1_dir": str(V1),
                 "historical_claim": {"field": 0.00366, "well": 0.00447, "r2": 0.9960,
                                      "n_train": 3400, "eval": "norne_fc test-400(已被污染)"},
                 "dup_theta_in_confirm": int(dup.sum())}
    json.dump(R, open(OUT / "confirm.json", "w"), indent=1, ensure_ascii=False)

    m, c = R["MAIN_confirm500"], R["CTRL_oldtest400"]
    print(f"\n=== 结果 ===")
    print(f"  MAIN 8000→confirm500  场级 {m['field_rel']['mean']:.4%} ± {m['field_rel']['sd']:.4%}"
          f"  每井 {m['well_rel']['mean']:.4%} ± {m['well_rel']['sd']:.4%}"
          f"  R² {m['field_r2']['mean']:.4f}  p90 {m['field_p90']['mean']:.4%}")
    print(f"  CTRL 7600→旧test400   场级 {c['field_rel']['mean']:.4%}"
          f"  每井 {c['well_rel']['mean']:.4%}")
    print(f"  CTRL 7600→confirm500  场级 {R['CTRL_confirm500']['field_rel']['mean']:.4%}"
          f"  每井 {R['CTRL_confirm500']['well_rel']['mean']:.4%}")
    print(f"  集合效应(场级) {R['set_effect']['field_delta_pp']:+.4f}pp  p={pf:.4f}")
    print(f"  平凡基线 confirm 场级 {R['TRIVIAL_confirm500']['field_rel']:.4%}"
          f"  每井 {R['TRIVIAL_confirm500']['well_rel']:.4%}")
    print(f"\n已写入 {OUT}/confirm.json   总耗时 {(time.time()-t0)/60:.0f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
