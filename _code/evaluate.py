"""指标与分层聚合。

轨迹目标含 NaN(缺日), 一律用 mask 跳过, 不做零填补 —— 零填补会把"没记录"
当成"产量为 0", 直接污染 MAE。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _mask(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(yhat)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _mask(y, yhat)
    return float(np.abs(y[m] - yhat[m]).mean()) if m.any() else float("nan")


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _mask(y, yhat)
    return float(np.sqrt(((y[m] - yhat[m]) ** 2).mean())) if m.any() else float("nan")


def per_window_mae(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """每个窗口一个 MAE, 供分层聚合与 bootstrap 用。"""
    m = _mask(y, yhat)
    num = np.where(m, np.abs(y - yhat), 0.0).sum(axis=1)
    den = m.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def naive_scale(hist_oil: np.ndarray, season: int = 1) -> np.ndarray:
    """MASE 的分母: 历史窗内 season 步 naive 的平均绝对误差(逐窗)。"""
    a, b = hist_oil[:, season:], hist_oil[:, :-season]
    m = np.isfinite(a) & np.isfinite(b)
    num = np.where(m, np.abs(a - b), 0.0).sum(axis=1)
    den = m.sum(axis=1)
    scale = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return np.where(scale > EPS, scale, np.nan)


def mase(y: np.ndarray, yhat: np.ndarray, hist_oil: np.ndarray, season: int = 1,
         mode: str = "ratio_of_sums") -> float:
    """MASE。

    🔴 2026-08-07 审计:旧版用「逐窗比值再取均值」, 被重尾窗口主导(实测最大比值 1633.7),
    且静默丢弃 scale 为 NaN 的窗口(8739 个里 193 个, 2.21%), 导致 MASE 与 MAE 的样本集不同、
    数值不可互推。实测三种口径差 2.4 倍:比值均值 6.02 / 求和式 2.52 / 中位数 2.26。
    默认改用求和式(与 MAE 同底), 另提供 mean_of_ratios / median 供对照。
    """
    scale = naive_scale(hist_oil, season)
    per = per_window_mae(y, yhat)
    m = np.isfinite(per) & np.isfinite(scale)
    if not m.any():
        return float("nan")
    if mode == "ratio_of_sums":
        return float(per[m].sum() / scale[m].sum())
    r = per[m] / scale[m]
    return float(np.median(r)) if mode == "median" else float(r.mean())


def crps_from_quantiles(y: np.ndarray, q_pred: np.ndarray, levels: np.ndarray) -> float:
    """分位数损失(pinball)对分位水平取平均 —— GluonTS 口径的 mean weighted
    quantile loss 的未归一化版本, 数值上等价于离散化 CRPS 的 2x。

    q_pred: (N, F, Q); levels: (Q,)
    """
    y_e = y[..., None]
    m = np.isfinite(y_e) & np.isfinite(q_pred)
    diff = np.where(m, y_e - q_pred, 0.0)
    lv = levels.reshape(1, 1, -1)
    loss = np.maximum(lv * diff, (lv - 1.0) * diff)
    n = m.sum() / len(levels)
    if n < 1:                      # 全被 mask 掉 -> NaN, 不是 0.0(那是满分, 会被误读)
        return float("nan")
    return float(2.0 * loss.sum() / n / len(levels))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(((y >= lo) & (y <= hi))[m].mean()) if m.any() else float("nan")


# 🔴 step_days=1 时相邻窗口共享 59/60 个日历日, 逐窗 MAE 的 lag-1 自相关实测 0.77。
# iid bootstrap 会把有效样本量高估约 (history+horizon)=60 倍, CI 窄约 5 倍。
# 2026-08-07 审计发现。默认改用 moving-block bootstrap, 块长 = 窗口重叠长度。
BLOCK_LEN = 60


def _block_indices(size: int, block: int, n: int, rng) -> np.ndarray:
    """moving-block bootstrap 的重采样下标 (n, size)。"""
    n_blocks = int(np.ceil(size / block))
    starts = rng.integers(0, max(size - block + 1, 1), size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)
    return np.clip(idx[:, :size], 0, size - 1)


def bootstrap_ci(values: np.ndarray, n: int = 2000, alpha: float = 0.05,
                 seed: int = 2026, block: int | None = BLOCK_LEN) -> tuple[float, float]:
    """对逐窗指标做 bootstrap, 返回均值的置信区间。

    block=None 时退化为 iid(仅适用于确实独立的样本); 默认 moving-block。
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if block and v.size > block:
        means = v[_block_indices(v.size, block, n, rng)].mean(axis=1)
    else:
        means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000,
                          alpha: float = 0.05, seed: int = 2026,
                          block: int | None = BLOCK_LEN) -> dict:
    """配对 bootstrap: 同一批窗口上 a 与 b 的逐窗指标之差。

    配对是关键 —— 两个模型在同一窗口上的误差高度相关, 独立 bootstrap 会
    大幅高估差值的不确定性。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 2:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(d.size)}
    rng = np.random.default_rng(seed)
    if block and d.size > block:
        means = d[_block_indices(d.size, block, n, rng)].mean(axis=1)
    else:
        means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(d.size),
    }


def stratified_table(per_window: dict[str, np.ndarray], strata: np.ndarray,
                     order: list[str] | None = None) -> pd.DataFrame:
    """逐窗指标按分层聚合, 每个模型一列。"""
    keys = order or sorted(set(np.asarray(strata).tolist()))
    rows = []
    for k in keys:
        sel = np.asarray(strata) == k
        row = {"stratum": k, "n": int(sel.sum())}
        for name, vals in per_window.items():
            v = np.asarray(vals)[sel]
            v = v[np.isfinite(v)]
            row[name] = float(v.mean()) if v.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
