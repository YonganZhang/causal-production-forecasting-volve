#!/usr/bin/env python3
"""Norne 代理模型：θ → 4D 场 + 井观测。

🔴 纪律（petro-knowledge/references/08-陷阱清单.md 第 2、6 条）：
  1. **平凡基线必须先跑，且允许它赢。** 三条：
       B0 气候态      —— 预测 = 训练集均值场，完全无视输入
       B1 最近邻      —— 找训练集里 θ 最近的那次运行，原样返回它的场（零训练、零参数）
       B2 PCA + 岭回归 —— 10 行 sklearn
     神经网络若打不过 B2，就该如实报告"深度模型在此无边际价值"。
  2. **必须有分布外测试。** 代理模型天生外推不安全，只报训练分布内的 R²=0.99 是自欺。
     这里把 |θ| 最大的 20% 样本整体留作 OOD 测试集，训练时完全不见。

评估指标（照 Tang 2021 CMA 的口径）：
  δ_P  压力场逐时刻归一化 MAE
  δ_S  含水饱和度场 MAE
  δ_S,front  只在"动区"(|S_T − S_0| > 0.05)上算 —— 否则大量静止格块把误差稀释
  δ_obs 井观测的相对 MAE

用法:
    python norne_surrogate.py baselines      # 先跑平凡基线
    python norne_surrogate.py train          # 训 NN 并与基线对比
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PACK = Path("/mnt/data/yongan-admin-2/datasets/petro/_runs/norne/packed")
OUTDIR = Path(__file__).resolve().parent.parent / "_pipelines" / "norne_surrogate"


def load():
    TH = np.load(PACK / "theta.npy").astype(np.float32)
    OB = np.load(PACK / "obs.npy").astype(np.float32)
    FL = np.load(PACK / "fields.npy")                 # float16 (N,T,2,C)
    return TH, OB, FL


def splits(TH: np.ndarray, ood_frac: float = 0.20, val_frac: float = 0.15, seed: int = 0):
    """OOD = |θ| 最大的一批（整体留出，训练时完全不见）；其余随机分 train/val。"""
    mag = np.linalg.norm(TH, axis=1)
    n_ood = int(len(TH) * ood_frac)
    ood = np.argsort(-mag)[:n_ood]
    rest = np.setdiff1d(np.arange(len(TH)), ood)
    rng = np.random.default_rng(seed); rng.shuffle(rest)
    n_val = int(len(rest) * val_frac)
    return {"train": rest[n_val:], "val": rest[:n_val], "ood": ood}


def field_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """pred/true: (n, T, 2, C) float32。返回归一化 MAE。"""
    p, t = pred.astype(np.float32), true.astype(np.float32)
    out = {}
    for i, name in enumerate(("P", "S")):
        a, b = p[:, :, i], t[:, :, i]
        rng = np.nanmax(b) - np.nanmin(b)
        out[f"delta_{name}"] = float(np.nanmean(np.abs(a - b)) / max(rng, 1e-9))
    # 动区:含水饱和度从首帧到末帧变化超过 0.05 的格块
    moved = np.abs(t[:, -1, 1] - t[:, 0, 1]) > 0.05
    if moved.any():
        d = np.abs(p[:, :, 1] - t[:, :, 1])
        sel = np.broadcast_to(moved[:, None, :], d.shape)
        rng = np.nanmax(t[:, :, 1]) - np.nanmin(t[:, :, 1])
        out["delta_S_front"] = float(np.nanmean(d[sel]) / max(rng, 1e-9))
        out["moved_cell_frac"] = float(moved.mean())
    return out


def obs_metric(pred: np.ndarray, true: np.ndarray) -> float:
    m = np.isfinite(true) & np.isfinite(pred)
    scale = np.nanmean(np.abs(true[m]))
    return float(np.nanmean(np.abs(pred[m] - true[m])) / max(scale, 1e-9))


def run_baselines(TH, OB, FL, sp) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    tr = sp["train"]
    res = {}
    for tag in ("val", "ood"):
        te = sp[tag]
        # B0 气候态
        p_f = np.repeat(FL[tr].astype(np.float32).mean(0)[None], len(te), axis=0)
        p_o = np.repeat(np.nanmean(OB[tr], 0)[None], len(te), axis=0)
        res[f"B0_climatology/{tag}"] = {**field_metrics(p_f, FL[te]), "delta_obs": obs_metric(p_o, OB[te])}
        # B1 最近邻训练样本
        d = ((TH[te][:, None] - TH[tr][None]) ** 2).sum(-1)
        nn = tr[d.argmin(1)]
        res[f"B1_nearest/{tag}"] = {**field_metrics(FL[nn].astype(np.float32), FL[te]),
                                    "delta_obs": obs_metric(OB[nn], OB[te])}
        # B2 PCA + 岭回归
        Ftr = FL[tr].astype(np.float32).reshape(len(tr), -1)
        k = min(40, len(tr) - 1)
        pca = PCA(n_components=k).fit(Ftr)
        Z = pca.transform(Ftr)
        rg = Ridge(alpha=1.0).fit(TH[tr], Z)
        p_f = pca.inverse_transform(rg.predict(TH[te])).reshape((len(te),) + FL.shape[1:])
        ob_tr = np.nan_to_num(OB[tr], nan=0.0)
        rg2 = Ridge(alpha=1.0).fit(TH[tr], ob_tr)
        res[f"B2_pca_ridge/{tag}"] = {**field_metrics(p_f, FL[te]),
                                      "delta_obs": obs_metric(rg2.predict(TH[te]), OB[te])}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "train"])
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    TH, OB, FL = load()
    sp = splits(TH)
    print(f"样本 {len(TH)}   train {len(sp['train'])} / val {len(sp['val'])} / OOD {len(sp['ood'])}")
    print(f"θ {TH.shape}  obs {OB.shape}  fields {FL.shape} ({FL.nbytes/1e9:.2f} GB)")
    print(f"OOD 定义: |θ| 最大的 20%  —— OOD 组 |θ| 范围 "
          f"[{np.linalg.norm(TH[sp['ood']],axis=1).min():.3f}, "
          f"{np.linalg.norm(TH[sp['ood']],axis=1).max():.3f}]  "
          f"训练组 [{np.linalg.norm(TH[sp['train']],axis=1).min():.3f}, "
          f"{np.linalg.norm(TH[sp['train']],axis=1).max():.3f}]")

    if args.cmd == "baselines":
        res = run_baselines(TH, OB, FL, sp)
        print(f"\n{'基线/切分':26s} {'δ_P':>9s} {'δ_S':>9s} {'δ_S前缘':>10s} {'δ_obs':>9s}")
        for k in sorted(res):
            r = res[k]
            print(f"{k:26s} {r['delta_P']:>9.4f} {r['delta_S']:>9.4f} "
                  f"{r.get('delta_S_front', float('nan')):>10.4f} {r['delta_obs']:>9.4f}")
        (OUTDIR / "baselines.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写入 {OUTDIR/'baselines.json'}")
        print("\n🔴 神经网络必须打过 B2_pca_ridge 才有价值; 打不过就如实报告负结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
