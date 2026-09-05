"""Baseline 阶梯 (30 步轨迹)。

从最笨到有领域先验:
  last_value       持平最后一个观测值
  history_mean     持平历史窗均值 (军伟 T3 用的 B1)
  seasonal_naive7  重复最后 7 天
  arps_exponential 指数递减曲线 —— 油藏工程标准做法, 不是通用时序基线
  lightgbm         需要训练的强表格基线 (可选带干预协变量)

🔴 identity_rate_x_hours —— **强制对照, 不许省略**

    日产油体积 ≈ 每开井小时产油率 x 当日开井小时数。这是近似恒等式:
    数据里开井小时=0 的日子, 产油=0 的比例 99.87%。

    因此任何"把未来开井小时数当已知协变量"的模型, 都能靠这条记账关系刷出巨大
    提升, 而完全不需要学会油嘴或注水的任何因果效应。2026-08-06 本项目正是在这里
    栽过一次: 报出 "干预条件化让 MAE 降 42%", 而这条三行公式的 MAE 反而更低。

    规则: 只要预测目标是 BORE_OIL_VOL 且模型用了未来 ON_STREAM_HRS,
    结果表必须同时列出 identity_rate_x_hours。打不过它 = 没有证据。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def _last_observed(arr: np.ndarray) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i, row in enumerate(arr):
        obs = row[np.isfinite(row)]
        if obs.size:
            out[i] = obs[-1]
    return out


def last_value(ws, idx: np.ndarray) -> np.ndarray:
    v = _last_observed(ws.hist[ws.target_col][idx])
    return np.repeat(v[:, None], ws.horizon, axis=1)


def history_mean(ws, idx: np.ndarray) -> np.ndarray:
    v = np.nanmean(ws.hist[ws.target_col][idx], axis=1)
    return np.repeat(v[:, None], ws.horizon, axis=1)


def identity_rate_x_hours(ws, idx: np.ndarray) -> np.ndarray:
    """强制对照: (历史每开井小时产油率) x (未来开井小时数)。

    只对 BORE_OIL_VOL 目标有意义 —— 对 OIL_RATE 目标它退化成"持平历史平均速率"。
    这个基线**使用了未来的开井小时数**, 因此归入 interventional 组, 与同样用了
    该信息的模型直接对拍。它是那类模型的及格线, 不是可选项。
    """
    # 🔴 2026-08-07 审计:rate 口径下它退化成 history_mean, 完全不读未来信息,
    # 却仍被 run_experiment 当"干预组及格线"用 —— 实测该"及格线"(8.82)比最简单的
    # 纯历史基线 last_value(8.47) 还松。现在显式返回 NaN, 由调用方判定不适用。
    if ws.target_col != "BORE_OIL_VOL":
        return np.full((len(idx), ws.horizon), np.nan)
    ho, hh = ws.hist["BORE_OIL_VOL"][idx], ws.hist["ON_STREAM_HRS"][idx]
    m = np.isfinite(ho) & np.isfinite(hh)
    rate = np.where(m, ho, 0.0).sum(axis=1) / np.maximum(np.where(m, hh, 0.0).sum(axis=1), EPS)
    fh = ws.fut["ON_STREAM_HRS"][idx]
    return np.maximum(rate[:, None] * np.where(np.isfinite(fh), fh, np.nan), 0.0)


def seasonal_naive(ws, idx: np.ndarray, season: int = 7) -> np.ndarray:
    hist = ws.hist[ws.target_col][idx]
    tail = hist[:, -season:]
    # 缺失回落到历史均值, 而不是 0
    fallback = np.nanmean(hist, axis=1)[:, None]
    tail = np.where(np.isfinite(tail), tail, fallback)
    reps = int(np.ceil(ws.horizon / season))
    return np.tile(tail, (1, reps))[:, : ws.horizon]


def arps_exponential(ws, idx: np.ndarray, fit_days: int = 30) -> np.ndarray:
    """对历史窗做 log 线性拟合 q(t)=q0*exp(-D t), 外推 horizon 步。

    只对正产量点拟合; 递减率 D 夹在 [0, 0.05]/天 (负 D = 增产, 对 30 天外推不稳,
    统一截到 0 表示"不外推增长")。拟合点不足或全零时回落到历史均值。
    """
    hist = ws.hist[ws.target_col][idx]
    h = hist.shape[1]
    t = np.arange(h, dtype=np.float64)
    steps = np.arange(1, ws.horizon + 1, dtype=np.float64)
    out = np.empty((len(idx), ws.horizon))
    fallback = np.nanmean(hist, axis=1)
    for i, row in enumerate(hist):
        win = row[-fit_days:]
        tt = t[-fit_days:]
        m = np.isfinite(win) & (win > EPS)
        if m.sum() < 5:
            out[i] = fallback[i]
            continue
        slope, intercept = np.polyfit(tt[m], np.log(win[m]), 1)
        decline = float(np.clip(-slope, 0.0, 0.05))
        q0 = float(np.exp(intercept + slope * tt[m][-1]))
        out[i] = q0 * np.exp(-decline * steps)
    return np.maximum(np.nan_to_num(out, nan=0.0), 0.0)


# ---------------------------------------------------------------- LightGBM

def _tabular_features(ws, idx: np.ndarray, *, use_interventions: bool) -> np.ndarray:
    """每个窗口一行特征; 预测 30 步时把 step 作为额外列展开(见 lightgbm_forecast)。"""
    hist_oil = ws.hist[ws.target_col][idx]
    blocks = [hist_oil]
    for col in ("ON_STREAM_HRS", "AVG_CHOKE_SIZE_P", "AVG_DOWNHOLE_PRESSURE", "AVG_WHP_P"):
        h = ws.hist[col][idx]
        blocks.append(np.stack([
            np.nanmean(h, axis=1), np.nanstd(h, axis=1),
            _last_observed(h), np.nanmean(h[:, -7:], axis=1),
        ], axis=1))
    if use_interventions:
        # 预测窗内的作业计划 —— 与 Chronos-2 的 future_covariates 信息量对齐
        for col in ("ON_STREAM_HRS", "AVG_CHOKE_SIZE_P", "FIELD_WI_VOL"):
            f = ws.fut[col][idx]
            blocks.append(np.stack([
                np.nanmean(f, axis=1), np.nanstd(f, axis=1),
                np.nanmin(f, axis=1), np.nanmax(f, axis=1),
            ], axis=1))
    x = np.concatenate([b if b.ndim == 2 else b[:, None] for b in blocks], axis=1)
    return np.nan_to_num(x, nan=-999.0, posinf=-999.0, neginf=-999.0)


def lightgbm_forecast(ws, train_idx: np.ndarray, valid_idx: np.ndarray, *,
                      use_interventions: bool = False, anchor: bool = True,
                      seed: int = 2026) -> np.ndarray:
    """全局 LightGBM: 行 = (窗口, 预测步), 特征加一列 step。

    anchor=True 时预测的是「相对历史末值的增量」而不是绝对产量。这不是修饰,
    是给树模型一个公平机会: Volve 的产量水平从 2008 年的中位 746 Sm3/d 衰减到
    2016 年的 168, 时间序切分下训练区间与验证区间几乎不重叠, 树模型无法外推到
    训练集值域之外。锚定到 last_value 把问题转到近似平稳的增量空间。
    """
    import lightgbm as lgb

    def expand(idx):
        x = _tabular_features(ws, idx, use_interventions=use_interventions)
        n, h = len(idx), ws.horizon
        step = np.tile(np.arange(h, dtype=np.float64), n)[:, None]
        xr = np.repeat(x, h, axis=0)
        feats = np.concatenate([xr, step], axis=1)
        if use_interventions:
            # 逐步的干预值(而不只是窗口统计量)
            per_step = np.stack([ws.fut[c][idx].reshape(-1) for c in
                                 ("ON_STREAM_HRS", "AVG_CHOKE_SIZE_P", "FIELD_WI_VOL")], axis=1)
            feats = np.concatenate([feats, np.nan_to_num(per_step, nan=-999.0)], axis=1)
        return feats

    def anchor_of(idx):
        a = _last_observed(ws.hist[ws.target_col][idx])
        a = np.where(np.isfinite(a), a, np.nanmean(ws.hist[ws.target_col][idx], axis=1))
        return np.nan_to_num(a, nan=0.0)

    tr_anchor, va_anchor = anchor_of(train_idx), anchor_of(valid_idx)
    xtr = expand(train_idx)
    ytr = ws.y[train_idx].reshape(-1)
    if anchor:
        ytr = ytr - np.repeat(tr_anchor, ws.horizon)
    ok = np.isfinite(ytr)
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=seed, verbose=-1,
    )
    model.fit(xtr[ok], ytr[ok])
    pred = model.predict(expand(valid_idx)).reshape(len(valid_idx), ws.horizon)
    if anchor:
        pred = pred + va_anchor[:, None]
    return np.maximum(pred, 0.0)


# 纯历史组: 只用截止点之前的信息
REGISTRY_PURE = {
    "last_value": last_value,
    "history_mean": history_mean,
    "seasonal_naive7": seasonal_naive,
    "arps_exponential": arps_exponential,
}
# 干预组: 使用了预测窗内的作业信息, 与同类模型对拍
REGISTRY_INTERVENTIONAL = {
    "identity_rate_x_hours": identity_rate_x_hours,
}
REGISTRY = {**REGISTRY_PURE, **REGISTRY_INTERVENTIONAL}

# 撤稿教训的守卫: 恒等式基线不许从 REGISTRY 里摘掉(见模块 docstring 与 _tests)。
IDENTITY_IN_REGISTRY = "identity_rate_x_hours" in REGISTRY
