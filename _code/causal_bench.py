#!/usr/bin/env python3
"""NORNE-CF：两臂因果基准的分析。

    干预臂 (norne_bulk)     θ ~ 独立随机       →  因果效应【真值】, 天然无混杂
    观测臂 (norne_obs_arm)  θ = π(油藏状态)    →  带混杂的观测数据

两臂同一个模拟器、同一套参数化, 所以真值已知。本脚本回答三个问题:

  Q1  真值是什么      —— 从干预臂估井间因果连通矩阵 ∂产油/∂注入
  Q2  朴素方法错多少  —— 在观测臂上跑朴素回归(=CRM 的做法), 与真值比
  Q3  因果方法救得回吗 —— 在观测臂上跑 DML, 看能否逼近真值

🔴 诚实边界:这里的"因果"是**模拟器内部**的因果, 不是真实 Norne 油田的因果。
   正确表述是"我们建立了带真值的基准并量化了朴素方法的偏差",
   不是"我们发现了 Norne 的真实井间连通性"。

用法:
    python causal_bench.py truth          # Q1
    python causal_bench.py bias           # Q2 + Q3(需要观测臂)
    python causal_bench.py field --inj C-3H   # 3D 因果效应场
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import paths as P

BULK = P.NORNE_BULK
OBS = P.NORNE_OBS_V2
OUTDIR = Path(__file__).resolve().parent.parent / "_pipelines" / "causal_bench"

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")
N_TIMES = 40


def control_names(schema: int) -> list[str]:
    """schema v1 是 9 个混相井名; v2 是 13 个 (井_相态)。见 norne_bulk.scan_injection_controls。"""
    if schema >= 2:
        import norne_bulk as NB
        return [f"{w}_{ph}" for w, ph in NB.CONTROLS]
    return list(INJECTORS)


def cum_oil(obs: np.ndarray) -> np.ndarray:
    """(n, 22) 每口生产井的累计产油(用 WOPR 在 40 个时刻上求和近似)。"""
    return np.stack([obs[:, (i * 3 + 0) * N_TIMES:(i * 3 + 1) * N_TIMES].sum(1)
                     for i in range(len(PRODUCERS))], axis=1)


def load_arm(root: Path, key: str = "theta") -> tuple[np.ndarray, np.ndarray]:
    pk = root / "packed"
    if pk.exists() and (pk / "obs.npy").exists():
        ob = np.load(pk / "obs.npy")
        th = np.load(pk / (f"{key}.npy" if (pk / f"{key}.npy").exists() else "theta_inj.npy"))
        return th, ob
    # 未合并时直接读分片
    sh = sorted((root / "shards").glob("*.npz"))
    if not sh:
        raise FileNotFoundError(f"{root} 下既无 packed 也无 shards")
    TH, OB = [], []
    for p in sh:
        d = np.load(p, allow_pickle=True)
        TH.append(d[key] if key in d else d["theta_inj"])
        OB.append(d["obs"])
    return np.stack(TH), np.stack(OB)


def ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """逐列最小二乘, 返回 (n_x, n_y) 的系数(已标准化 X)。"""
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    A = np.hstack([Xs, np.ones((len(Xs), 1))])
    return np.linalg.lstsq(A, Y, rcond=None)[0][:X.shape[1]]


def truth() -> np.ndarray:
    """Q1:干预臂 → 因果效应真值。θ 独立随机, 所以 OLS 系数就是因果效应。"""
    TH, OB = load_arm(BULK)
    X, Q = TH[:, :len(INJECTORS)], cum_oil(OB)
    B = ols(X, Q)
    print(f"干预臂样本 {len(TH)}   θ 各维两两相关的最大绝对值 "
          f"{np.abs(np.corrcoef(X.T) - np.eye(X.shape[1])).max():.4f}  (应接近 0 = 真随机)")
    return B


def naive_from_obs() -> tuple[np.ndarray, int]:
    """Q2:观测臂上的朴素回归 —— 这正是 CRM 那一派的做法(不做混杂校正)。"""
    TH, OB = load_arm(OBS, key="theta_inj")
    X = TH.reshape(len(TH), -1).mean(1)[:, None] if TH.ndim == 3 else TH
    if TH.ndim == 3:                       # (n, stages, 9) → 各井全期平均乘子
        X = TH.mean(1)
    return ols(X[:, :len(INJECTORS)], cum_oil(OB)), len(TH)


def dml_from_obs() -> np.ndarray:
    """Q3:观测臂上的 DML —— 用状态代理变量做混杂校正, 看能否逼近真值。

    混杂路径是 注入率 ← 油藏状态 → 产量。观测臂里可用的状态代理是**早期**的
    井观测(含水率/压力的前若干时刻), 它们发生在后期注入决策之前。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR

    TH, OB = load_arm(OBS, key="theta_inj")
    X = TH.mean(1)[:, :len(INJECTORS)] if TH.ndim == 3 else TH[:, :len(INJECTORS)]
    Q = cum_oil(OB)
    # 状态代理:每口生产井前 8 个时刻的产水率与井底压力(早于后期决策)
    W = np.concatenate([OB[:, (i * 3 + 1) * N_TIMES:(i * 3 + 1) * N_TIMES + 8]
                        for i in range(len(PRODUCERS))]
                       + [OB[:, (i * 3 + 2) * N_TIMES:(i * 3 + 2) * N_TIMES + 8]
                          for i in range(len(PRODUCERS))], axis=1)
    B = np.zeros((X.shape[1], Q.shape[1]))
    for a in range(X.shape[1]):
        rD = X[:, a] - GBR(max_iter=200, random_state=0).fit(W, X[:, a]).predict(W)
        for b in range(Q.shape[1]):
            rY = Q[:, b] - GBR(max_iter=200, random_state=0).fit(W, Q[:, b]).predict(W)
            B[a, b] = (rD * rY).sum() / max((rD ** 2).sum(), 1e-12)
    # 与 truth 同口径:truth 用标准化后的 X, 这里也换算过去
    return B * X.std(0)[:, None]


def report(B: np.ndarray, title: str, top: int = 10) -> None:
    print(f"\n=== {title} ===")
    print(f"{'注水井':8s}" + "".join(f"{p[:6]:>8s}" for p in PRODUCERS[:top]))
    for j, w in enumerate(INJECTORS):
        print(f"{w:8s}" + "".join(f"{B[j, i]:>8.0f}" for i in range(top)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["truth", "bias", "field"])
    ap.add_argument("--inj", default="C-3H",
                    help="schema v1 用井名(如 C-3H); v2 用 '井_相态'(如 C-3H_WATER)")
    ap.add_argument("--root", default=None, help="分片根目录, 默认 BULK")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "truth":
        B = truth()
        report(B, "Q1 因果效应真值(干预臂, 随机化 → 无混杂)")
        np.save(OUTDIR / "truth_connectivity.npy", B)
        sp = float((np.abs(B) < 0.1 * np.abs(B).max()).mean())
        print(f"\n稀疏度: {sp:.0%} 的井对效应 < 最大值的 10%")
        json.dump({"n_injectors": len(INJECTORS), "n_producers": len(PRODUCERS),
                   "sparsity": sp}, open(OUTDIR / "truth_meta.json", "w"), indent=1)
        return 0

    if args.cmd == "bias":
        T = np.load(OUTDIR / "truth_connectivity.npy") if (OUTDIR / "truth_connectivity.npy").exists() else truth()
        N, n_obs = naive_from_obs()
        report(T, "真值(干预臂)")
        report(N, f"朴素回归(观测臂, n={n_obs}) —— CRM 那一派的做法")
        def err(B):
            return dict(rmse=float(np.sqrt(((B - T) ** 2).mean())),
                        sign_agree=float((np.sign(B) == np.sign(T)).mean()),
                        corr=float(np.corrcoef(B.ravel(), T.ravel())[0, 1]))
        out = {"naive": err(N), "n_obs": n_obs}
        print(f"\n朴素 vs 真值: RMSE {out['naive']['rmse']:.0f}  "
              f"符号一致率 {out['naive']['sign_agree']:.1%}  相关 {out['naive']['corr']:+.3f}")
        try:
            D = dml_from_obs()
            out["dml"] = err(D)
            report(D, "DML(观测臂 + 状态代理校正)")
            print(f"DML  vs 真值: RMSE {out['dml']['rmse']:.0f}  "
                  f"符号一致率 {out['dml']['sign_agree']:.1%}  相关 {out['dml']['corr']:+.3f}")
        except Exception as e:                                # noqa: BLE001
            print(f"DML 未跑成: {type(e).__name__}: {e}")
        json.dump(out, open(OUTDIR / "bias.json", "w"), indent=1, ensure_ascii=False)
        return 0

    # field: 3D 效应场。三处相对旧版的修正见下面注释。
    root = Path(args.root) if args.root else BULK
    sh = sorted((root / "shards").glob("*.npz"))
    if not sh:
        print(f"{root} 下尚无分片"); return 1
    d0 = np.load(sh[0])
    schema = int(d0["schema"]) if "schema" in d0.files else 1
    names = control_names(schema)
    if args.inj not in names:
        print(f"schema v{schema} 的控制变量是 {names}\n  要的 '{args.inj}' 不在其中"); return 1
    a = names.index(args.inj)

    TH, FL = [], []
    for p in sh:
        d = np.load(p)
        if (int(d["schema"]) if "schema" in d.files else 1) != schema:
            print(f"🔴 {p.name} 的 schema 与首个分片不同, 拒绝混用"); return 1
        TH.append(d["theta"]); FL.append(d["fields"][-1, 1])     # 末帧 SWAT
    TH = np.stack(TH).astype(np.float64); S = np.stack(FL).astype(np.float64)
    n, p_ = len(TH), TH.shape[1]

    # 🔴 修正 1:多元回归而不是一元。θ 各维独立随机 → 一元无偏, 但残差里塞着其余 12~16 维
    #    的方差, 白白损失约一半有效样本量(实测 HC1 中位标准误 2.755e-4 → 2.170e-4)。
    Xs = (TH - TH.mean(0)) / np.where(TH.std(0) > 1e-12, TH.std(0), 1.0)
    A = np.hstack([Xs, np.ones((n, 1))])
    B = np.linalg.lstsq(A, S, rcond=None)[0]                     # (p+1, ncell)
    beta = B[a]
    E = S - A @ B

    # 🔴 修正 2:给出 HC1 稳健标准误。旧版只存点估计, 于是把"统计上与 0 无法区分"的
    #    单元和真有效应的单元用同样的颜色画出来 —— 实测约半数单元 |t|<2。
    #    只要一个系数的方差, 所以不必算整个 (p+1)x(p+1) 三明治:
    #        var_a = Σ_i h_i² e_i²,  h = A · (AᵀA)⁻¹e_a
    u = np.linalg.pinv(A.T @ A)[:, a]
    h = A @ u
    se = np.sqrt((h ** 2) @ (E ** 2) * n / max(n - p_ - 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    const = S.std(0) == 0

    stem = OUTDIR / f"effect_field_SWAT_{args.inj}"
    np.save(f"{stem}.npy", beta.astype(np.float32))
    np.save(f"{stem}_se.npy", se.astype(np.float32))
    np.save(f"{stem}_t.npy", t.astype(np.float32))

    # 🔴 修正 3:如实记录估计量。θ 是 log10 乘子且回归前做过标准化, 所以系数**不是**
    #    ∂SWAT/∂(物理注入率), 而是"每 +1 个 θ 标准差"。旧版色标写 d(Sw)/d(inj rate) 是错的。
    sd = float(TH[:, a].std())
    meta = {"schema": schema, "control": args.inj, "n": n, "n_theta": p_,
            "estimand": "OLS coefficient of final SWAT on standardized log10 injection-target "
                        "multiplier (multivariate, all controls included)",
            "unit": f"delta SWAT per +1 SD of theta (1 SD = {sd:.5f} dex = x{10**sd:.4f} rate)",
            "sd_theta_dex": sd, "sd_as_rate_factor": float(10 ** sd),
            "frac_abs_t_lt2": float((np.abs(t) < 2).mean()),
            "n_constant_cells": int(const.sum()),
            "beta_range": [float(beta.min()), float(beta.max())]}
    json.dump(meta, open(f"{stem}_meta.json", "w"), indent=1, ensure_ascii=False)

    print(f"3D 效应场 [{args.inj}]  schema v{schema}  n={n}  θ 维度 {p_}  活动单元 {len(beta)}")
    print(f"  估计量: 每 +1 个 θ 标准差(= {sd:.4f} dex = 注入目标率 ×{10**sd:.3f}) 的末帧 SWAT 变化")
    print(f"  值域 [{beta.min():+.5f}, {beta.max():+.5f}]")
    print(f"  |t|<2(与 0 无法区分)的单元 {int((np.abs(t) < 2).sum()):,} ({(np.abs(t) < 2).mean():.1%})"
          f"   其中所有样本 SWAT 完全恒定的 {int(const.sum()):,}")
    print(f"  已写入 {stem}.npy / _se.npy / _t.npy / _meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
