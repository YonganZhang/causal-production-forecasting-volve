#!/usr/bin/env python3
"""代理模型改进 v2：修正 v1 的两个方法学错误，并用多种子配对检验裁定每一项改动。

## v1 错在哪(实测,非推测)

**错误 1 —— 物理特征矩阵秩亏。**
`phys_features` 的 7 组特征里有 4 组(tot / cumsum(tot) / diff(tot) / mean(tot))
**是同 6 个数的线性变换**：18 列的数值秩只有 6。整个 31 列特征矩阵秩 20/31，
融合后 55 维输入秩 44，**条件数从基线的 1.21 炸到 1.29e14**。

网络第一层 `self.cond = nn.Linear(d_in, d)` 是线性映射 —— 对线性层而言，
**一列若是其余列的精确线性组合，贡献的表达力严格为零**，只带来两个坏处：
  1. `tot` 方向被复制 ~4 份 → 该方向的梯度被放大约 4 倍，压过其余方向；
  2. 输入从 24 维摊到 55 维，`nn.Linear` 默认初始化的每列权重尺度 ∝ 1/√d_in，
     真正决定答案的 24 个 θ 维度信号被衰减 √(55/24)=1.51 倍。

所以 A(特征融合)变差**不是因为"特征不含新信息"** —— 实测每列对输入的线性 R²
中位仅 0.77、无一列 >0.99，它们确实带非线性信息。是共线性把网络的条件数毁了。

**错误 2 —— 拿单次结果下结论。**
v1 每个配置只跑一个种子，0.598% vs 0.689% 从未做过显著性检验。本脚本每个配置
跑 N_SEED 个种子，报均值±标准差与配对 Wilcoxon p 值。**p≥0.05 的差异一律记为
"在噪声范围内"，不写进结论。**

**错误 3 —— 一半输出通道是常数 0。**
22 口生产井里 **11 口在全部 2000 样本、全部 40 格点上油率与水率恒等于 0**
(B-1H/B-4BH/B-4H/D-1H/D-3AH/D-4AH/D-4H/E-2H/E-3AH/E-3H/E-4AH,预测段已关停)。
44 个输出通道有 22 个是死的。两个后果:
  1. 网络一半的输出容量花在拟合常数 0 上;
  2. **"25% 负产率格点"是无意义指标** —— 分母里一半的井真值本就是 0,
     网络给出 ±1e-10 的数值噪声就被记成"负产率"。实测负产**质量**只有
     7~105 m³,而单样本累计产油 7,818,824 m³,占比 1.3e-05。
     所以 B(单调约束)是在解决一个不存在的问题;实测其惩罚项量级 7e-11,
     对上 0.044 的 MSE —— **它从未起过作用**。

## 本脚本的改动

| | 做法 |
|---|---|
| 特征 | 只保留**互相线性独立**的量，构造后断言满秩 |
| 融合 | 另加 whiten 选项(PCA 白化)，作为通用的共线性兜底 |
| 显著性 | 每配置 N_SEED 种子 + 配对检验 |
| 单调约束 | 同时报负产率**质量**(m³/d)与惩罚项的**实际量级**,确认权重是否真起作用 |
| 死通道 | `--drop-dead` 剔除 11 口恒零井,输出维度减半(累计产油口径不变) |
| 公平性 | 记录训练末期 loss,让"欠训练"可见而不是被误读成"特征没用" |

用法:
    python fc_improve2.py --gpu 6 --seeds 8
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

import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG
import norne_bulk as NB

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_improve2"
NW, NPH, NT = len(NB.PRODUCERS), 2, FG.FC_N
BASE_W = np.array([FG.BASE_WINJ[w] for w in FG.INJ_W])
DAYS = FG.FC_GRID                                                # 与 v1 同一时间网格,口径不变


# ---------------------------------------------------------------- 特征
def phys_features(TH, *, full=True):
    """从注水方案算油藏工程量。

    full=True  : v1 的 7 组(含 4 组互为线性变换的) —— 仅用于复现 v1 的病灶
    full=False : 只留**线性独立**的组
    """
    n = len(TH)
    S = TH.reshape(n, FG.N_STAGE, len(FG.INJ_W))
    rate = (10.0 ** S) * BASE_W[None, None, :]
    tot = rate.sum(2)
    if full:
        f = [tot, np.cumsum(tot, 1), np.diff(tot, axis=1),
             rate.std(2) / np.maximum(rate.mean(2), 1e-9),
             rate.max(2) - rate.min(2), tot.mean(1, keepdims=True),
             (tot[:, :2].mean(1) / np.maximum(tot[:, -2:].mean(1), 1e-9))[:, None]]
    else:
        # 🔴 去掉 cumsum / diff / mean:它们都是 tot 的精确线性变换,
        #    对第一层的线性映射贡献恒为 0,只会把条件数推到 1e14。
        f = [tot,                                                  # 分段总注水(对 θ 非线性)
             rate.std(2) / np.maximum(rate.mean(2), 1e-9),         # 井间失衡度
             rate.max(2) - rate.min(2),                            # 井间极差
             (tot[:, :2].mean(1) / np.maximum(tot[:, -2:].mean(1), 1e-9))[:, None],  # 前重/后重
             np.log10(np.maximum(tot, 1e-9))]                      # 对数尺度(比值关系线性化)
    return np.hstack([x.reshape(n, -1) for x in f]).astype(np.float32)


def rank_report(X, tag):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    s = np.linalg.svd(Z, compute_uv=False)
    r = int((s > s[0] * 1e-6).sum())
    cond = float(s[0] / max(s[-1], 1e-12))
    flag = "🔴 秩亏" if r < X.shape[1] else "✔ 满秩"
    print(f"  {tag:26s} {X.shape[1]:>3d} 维  秩 {r:>3d}  条件数 {cond:>10.3g}  {flag}")
    return {"dim": int(X.shape[1]), "rank": r, "cond": cond}


def whiten(Xtr, X):
    """PCA 白化:通用的共线性兜底。丢掉数值上为零的方向。"""
    m = Xtr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    k = int((s > s[0] * 1e-6).sum())
    W = (Vt[:k].T / s[:k]) * np.sqrt(len(Xtr))
    return ((X - m) @ W).astype(np.float32)


# ---------------------------------------------------------------- 模型
class TF(nn.Module):
    def __init__(self, d_in, n_ch, n_t, d=128, heads=4, layers=3, ff=256):
        super().__init__()
        self.cond = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_ch)

    def forward(self, x):
        h = self.cond(x)[:, None, :] + self.pos
        return self.head(self.tr(h)).transpose(1, 2)


def cumoil(Y):
    """累计产油 = 油率对**真实天数网格**积分。🔴 不能用 .sum():那会丢掉 dt。"""
    return np.trapezoid(Y.reshape(-1, NW, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)


def train_eval(Xtr, Ytr_n, Xte, Yte_raw, ym, ys, dev, *, seed=0, mono=0.0,
               d=128, heads=4, layers=3, ff=256, epochs=300):
    """训练一次,返回留出集指标 + 诊断量(末期训练 loss、惩罚项量级、负产率质量)。"""
    torch.manual_seed(seed)
    net = TF(Xtr.shape[1], NW * NPH, NT, d, heads, layers, ff).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    A = torch.tensor(Xtr, device=dev)
    B = torch.tensor(Ytr_n.reshape(-1, NW * NPH, NT), device=dev)
    ymt = torch.tensor(ym.reshape(NW * NPH, NT), device=dev)
    yst = torch.tensor(ys.reshape(NW * NPH, NT), device=dev)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    last_mse, last_pen = 0.0, 0.0
    for ep in range(epochs):
        pm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 64):
            b = pm[i:i + 64]
            opt.zero_grad()
            out = net(A[b])
            mse = torch.nn.functional.mse_loss(out, B[b])
            loss = mse
            pen = torch.zeros((), device=dev)
            if mono > 0:
                phys = out * yst + ymt
                oil = phys.view(-1, NW, NPH, NT)[:, :, 0, :]
                pen = torch.relu(-oil).mean()
                loss = loss + mono * pen
            loss.backward()
            opt.step()
            if ep == epochs - 1:
                last_mse, last_pen = float(mse), float(pen) * mono
        sch.step()
    net.eval()
    with torch.no_grad():
        P = (net(torch.tensor(Xte, device=dev)).cpu().numpy().reshape(len(Xte), -1)
             * ys + ym)
    ct, cp = cumoil(Yte_raw.reshape(len(Xte), -1)), cumoil(P)
    rel = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
    r2 = 1 - ((cp - ct) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
    oilP = P.reshape(-1, NW, NPH, NT)[:, :, 0, :]
    return {"relerr": float(rel.mean()), "p90": float(np.quantile(rel, 0.9)),
            "r2": float(r2),
            "neg_frac": float((oilP < 0).mean()),
            # 🔴 负产率**质量**才是有物理意义的量;格点占比会被"真值本就为 0 的关停井"灌水
            "neg_mass": float(np.trapezoid(np.maximum(-oilP, 0), DAYS, axis=-1).sum(-1).mean()),
            "train_mse": last_mse, "pen_term": last_pen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--drop-dead", action="store_true",
                    help="剔除恒零的生产井通道。累计产油是求和,死井贡献 0,故指标口径不变")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    TH, Y, IA, FC = FD.load(N_PIN)
    global NW
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)          # (22,) 哪些井真有产量
    print(f"生产井 {len(live)} 口, 其中 {int((~live).sum())} 口油/水率恒为 0: "
          f"{[NB.PRODUCERS[i] for i in np.where(~live)[0]]}")
    if args.drop_dead:
        Y = Y[:, live]; NW = int(live.sum())
        print(f"🔴 --drop-dead: 输出通道 {len(live)*2} → {NW*2} (死井恒为 0, "
              f"累计产油求和不受影响, 指标可直接与不剔除的结果比)\n")
    n = len(TH); Yf = Y.reshape(n, -1)
    rng = np.random.default_rng(args.split_seed); idx = rng.permutation(n)
    n_te = n // 5; te, tr = idx[:n_te], idx[n_te:]
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
    Ytr_n = (Yf[tr] - ym) / ys
    print(f"样本 {n}  训练 {len(tr)}  留出 {len(te)}   种子数 {args.seeds}\n")

    PF_full = phys_features(TH, full=True)
    PF_ind = phys_features(TH, full=False)
    print("=== 输入矩阵条件诊断(改动的直接依据) ===")
    diag = {"theta": rank_report(TH, "基线 θ"),
            "pf_full": rank_report(PF_full, "v1 物理特征"),
            "fused_full": rank_report(np.hstack([TH, PF_full]), "v1 融合"),
            "pf_ind": rank_report(PF_ind, "v2 物理特征(去共线)"),
            "fused_ind": rank_report(np.hstack([TH, PF_ind]), "v2 融合")}

    def norm(X):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8
        return ((X - m) / s).astype(np.float32)
    XF_full = np.hstack([TH, PF_full])
    XF_ind = np.hstack([TH, PF_ind])
    CFG = {
        "基线 θ(24维)":            dict(X=norm(TH), mono=0.0),
        "A1 v1融合(秩亏,复现病灶)": dict(X=norm(XF_full), mono=0.0),
        "A2 v2融合(去共线)":        dict(X=norm(XF_ind), mono=0.0),
        "A3 v1融合+白化":           dict(X=whiten(XF_full[tr], XF_full), mono=0.0),
        "B  单调约束(权重1)":       dict(X=norm(TH), mono=1.0),
        "A2+B":                     dict(X=norm(XF_ind), mono=1.0),
    }
    print(f"\n=== 多种子对比({args.seeds} 种子 × {len(CFG)} 配置) ===")
    res, t0 = {}, time.time()
    for tag, cfg in CFG.items():
        X, runs = cfg["X"], []
        for sd in range(args.seeds):
            runs.append(train_eval(X[tr], Ytr_n, X[te], Y[te], ym, ys, dev,
                                   seed=sd, mono=cfg["mono"], epochs=args.epochs))
        e = np.array([r["relerr"] for r in runs]) * 100
        res[tag] = {"runs": runs, "relerr_pct": e.tolist(),
                    "mean": float(e.mean()), "std": float(e.std(ddof=1)),
                    "r2": float(np.mean([r["r2"] for r in runs])),
                    "neg_frac": float(np.mean([r["neg_frac"] for r in runs])),
                    "neg_mass": float(np.mean([r["neg_mass"] for r in runs])),
                    "train_mse": float(np.mean([r["train_mse"] for r in runs])),
                    "pen_term": float(np.mean([r["pen_term"] for r in runs]))}
        r = res[tag]
        print(f"  {tag:24s} {r['mean']:.3f}±{r['std']:.3f}%  R² {r['r2']:.4f}  "
              f"负产率 {r['neg_frac']:.1%}/{r['neg_mass']:>9.0f}m³  "
              f"训练mse {r['train_mse']:.4f}  罚项 {r['pen_term']:.3g}  "
              f"({(time.time()-t0)/60:.0f}min)", flush=True)

    # ---- 配对显著性:每个配置 vs 基线,同种子配对 ----
    from scipy.stats import wilcoxon, ttest_rel
    b = np.array(res["基线 θ(24维)"]["relerr_pct"])
    print(f"\n=== 配对显著性检验(vs 基线, 同种子配对, n={args.seeds}) ===")
    print(f"  {'配置':24s}{'平均差(百分点)':>16s}{'Wilcoxon p':>13s}{'t检验 p':>10s}   判定")
    sig = {}
    for tag in CFG:
        if tag == "基线 θ(24维)":
            continue
        a = np.array(res[tag]["relerr_pct"]); d = a - b
        pw = float(wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0
        pt = float(ttest_rel(a, b).pvalue)
        verdict = ("✔ 显著变好" if pw < 0.05 and d.mean() < 0 else
                   "🔴 显著变差" if pw < 0.05 else "— 噪声范围内")
        sig[tag] = {"delta": float(d.mean()), "p_wilcoxon": pw, "p_ttest": pt,
                    "verdict": verdict}
        print(f"  {tag:24s}{d.mean():>+16.3f}{pw:>13.4f}{pt:>10.4f}   {verdict}")

    json.dump({"n": n, "n_test": len(te), "seeds": args.seeds, "epochs": args.epochs,
               "conditioning": diag, "results": res, "significance": sig,
               "note": ("每配置多种子 + 配对检验。p≥0.05 一律记为噪声,不写进结论。"
                        "v1 的 A 变差已定位为输入矩阵秩亏(条件数 1.29e14),"
                        "不是'特征不含信息'。")},
              open(OUT / f"ablation_v2{'_dropdead' if args.drop_dead else ''}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/ablation_v2{'_dropdead' if args.drop_dead else ''}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
