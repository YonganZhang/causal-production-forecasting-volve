"""滑窗构造 + 干预分层标签。

窗口约定: history H 天 -> forecast F 天, 步长 S。日历连续(缺日为 NaN 行, 不跳过),
与军伟 T3 契约一致, 保证 E0 可复现。

干预分层 (E3 的核心): 标签只看**预测窗内**发生了什么, 不看历史, 因为要问的是
"当预测窗里发生干预时, 纯历史模型是否更容易崩"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from volve_data import (
    ALL_NUMERIC,
    PRODUCERS,
    calendarize_well,
    field_injection_series,
    load_daily,
)

# 分层阈值。SHUTIN 用状态跳变(0 <-> >0)判定, 另两个用相对变化。
STEADY_REL = 0.05
INTERVENTION_REL = 0.20
EPS = 1e-6


@dataclass
class WindowSet:
    """一批窗口。数组均按窗口索引对齐, 第 0 维是样本数。"""

    well: np.ndarray                      # (N,) WELL_BORE_CODE
    cutoff: np.ndarray                    # (N,) 预测窗第一天 (datetime64)
    hist: dict[str, np.ndarray]           # col -> (N, H)
    fut: dict[str, np.ndarray]            # col -> (N, F)
    y: np.ndarray                         # (N, F) 日产油轨迹, 可含 NaN
    y_scalar: np.ndarray                  # (N,) 预测窗日产油均值 (nanmean)
    strata: dict[str, np.ndarray] = field(default_factory=dict)
    history_days: int = 30
    horizon: int = 30
    target_col: str = "BORE_OIL_VOL"

    def __len__(self) -> int:
        return len(self.well)

    def sample_ids(self, prefix: str = "productivity-calendar") -> list[str]:
        """与军伟 T3 一致的 sample_id 形式, E0 的种子子采样依赖它。"""
        return [
            f"{prefix}:{w}:{pd.Timestamp(c).date().isoformat()}"
            for w, c in zip(self.well, self.cutoff)
        ]

    def subset(self, mask: np.ndarray) -> "WindowSet":
        mask = np.asarray(mask)
        return WindowSet(
            well=self.well[mask],
            cutoff=self.cutoff[mask],
            hist={k: v[mask] for k, v in self.hist.items()},
            fut={k: v[mask] for k, v in self.fut.items()},
            y=self.y[mask],
            y_scalar=self.y_scalar[mask],
            strata={k: v[mask] for k, v in self.strata.items()},
            history_days=self.history_days,
            horizon=self.horizon,
            target_col=self.target_col,
        )


def _rel_change(future: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """预测窗内相对参考值的最大相对变化; 参考值或全窗缺失时返回 NaN。"""
    ref = np.where(np.abs(reference) > EPS, reference, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(future - ref[:, None]) / np.abs(ref[:, None])
    allnan = ~np.isfinite(rel).any(axis=1)
    out = np.full(len(rel), np.nan)
    if (~allnan).any():
        out[~allnan] = np.nanmax(rel[~allnan], axis=1)
    return out


def _quartile(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """按经验分位数分箱; NaN 归入 'na' 箱。用分位数而非固定阈值, 避免拍脑袋的
    切点, 并保证各组样本量均衡(30 天预测窗里"几乎不变"的窗口天然极少)。"""
    out = np.full(len(values), "na", dtype=object)
    ok = np.isfinite(values)
    if ok.sum() >= n_bins:
        edges = np.quantile(values[ok], np.linspace(0, 1, n_bins + 1)[1:-1])
        idx = np.searchsorted(edges, values[ok], side="right")
        out[ok] = [f"Q{i + 1}" for i in idx]
    return np.asarray(out)


def _last_observed(arr: np.ndarray) -> np.ndarray:
    """每行最后一个非 NaN 值; 全 NaN 则 NaN。"""
    out = np.full(len(arr), np.nan)
    for i, row in enumerate(arr):
        obs = row[np.isfinite(row)]
        if obs.size:
            out[i] = obs[-1]
    return out


def add_strata(ws: WindowSet) -> WindowSet:
    """给窗口打干预分层标签。

    提供两套口径, E3 主用**分位数剂量-反应**口径:

    1. 离散事件: `shutin_flip` —— 预测窗内 ON_STREAM_HRS 在 "0" 与 ">0" 间跳变。
       这是真正的二值事件, 样本量天然均衡, 不需要人为切点。
    2. 连续强度分位: `choke_q` / `wi_q` —— 油嘴与全场注水的相对变化按经验四分位分箱。
       用分位数是因为 30 天预测窗内"几乎不变"的窗口极稀少(固定 5% 阈值只剩不到 1%),
       固定阈值会得到一个小到无法统计的对照组。分位口径给的是剂量-反应曲线,
       比二元 "有干预/无干预" 更强的证据。

    同时保留固定阈值的 S0..S4 标签作参考, 但因 S0 样本过少, 不作为 E3 主结论口径。
    """
    on_fut = ws.fut["ON_STREAM_HRS"]
    on_last = _last_observed(ws.hist["ON_STREAM_HRS"])
    state = np.concatenate([(on_last > EPS)[:, None], on_fut > EPS], axis=1)
    observed = np.concatenate([np.isfinite(on_last)[:, None], np.isfinite(on_fut)], axis=1)
    # 🔴 2026-08-07 审计:旧版只比较**日历上严格相邻**的两天, 于是转折点上只要插一个缺日,
    # 真实的开井→关井事件就完全漏检, 该窗口掉进 S2_choke/S3_injection, 污染最干净的那个分层。
    # 改为在**相邻的两个有观测日**之间比较(跳过中间的缺日)。
    shutin = np.zeros(len(on_fut), dtype=bool)
    for i in range(len(on_fut)):
        obs = np.flatnonzero(observed[i])
        if obs.size < 2:
            continue
        st = state[i][obs]
        shutin[i] = bool((st[:-1] != st[1:]).any())

    choke_rel = _rel_change(ws.fut["AVG_CHOKE_SIZE_P"], _last_observed(ws.hist["AVG_CHOKE_SIZE_P"]))
    wi_ref = np.nanmean(ws.hist["FIELD_WI_VOL"][:, -7:], axis=1)
    wi_rel = _rel_change(ws.fut["FIELD_WI_VOL"], wi_ref)

    # 🔴 2026-08-07 审计:旧版用 nan_to_num(..., 0.0) 把"参考值不可用/整窗无观测"当成
    # "相对变化为 0", 于是这些**无法判定**的窗口被判进 S0_steady, 对照组混入纯粹的无数据窗口。
    # 改为显式的三态:有干预 / 无干预 / 不可判定(S9_unknown)。
    undecidable = ~np.isfinite(choke_rel) | ~np.isfinite(wi_rel)
    big_choke = np.isfinite(choke_rel) & (choke_rel > INTERVENTION_REL)
    big_wi = np.isfinite(wi_rel) & (wi_rel > INTERVENTION_REL)
    calm = (
        ~shutin & ~undecidable
        & (choke_rel < STEADY_REL)
        & (wi_rel < STEADY_REL)
    )

    label = np.full(len(ws), "S4_mild", dtype=object)
    label[undecidable] = "S9_unknown"
    label[calm] = "S0_steady"
    label[big_wi & ~big_choke & ~shutin] = "S3_injection"
    label[big_choke & ~shutin] = "S2_choke"
    label[shutin] = "S1_shutin"

    ws.strata = {
        "label": np.asarray(label),
        "shutin_flip": shutin,
        "choke_rel_change": choke_rel,
        "wi_rel_change": wi_rel,
        "choke_q": _quartile(choke_rel),
        "wi_q": _quartile(wi_rel),
    }
    return ws


def build_windows(
    daily: pd.DataFrame | None = None,
    *,
    wells: dict[str, str] | None = None,
    history_days: int = 30,
    horizon: int = 30,
    step_days: int = 7,
    min_observed_future: int = 21,
    columns: tuple[str, ...] = ALL_NUMERIC,
    target_col: str = "BORE_OIL_VOL",
) -> WindowSet:
    """在生产井上滑窗。默认参数即军伟 T3 契约 (30/30/7/21)。

    target_col:
      BORE_OIL_VOL — 日产油**体积**。注意它满足近似恒等式 体积 = 速率 x 开井小时,
                     所以"把未来开井小时数当协变量喂进去"会以记账方式刷高分数,
                     必须与 baselines.identity_rate_x_hours 对拍才知道模型有没有真本事。
      OIL_RATE     — 开井期间的日产油**速率** (Sm3/开井小时)。把排产记账项除掉后
                     剩下的油藏物理量, 这才是"油嘴/注水影响产能"该作用的对象。
    """
    daily = load_daily() if daily is None else daily
    wells = PRODUCERS if wells is None else wells
    injection = field_injection_series(daily)
    field_wi = injection.sum(axis=1, min_count=1).rename("FIELD_WI_VOL")

    cols = [c for c in columns if c in daily.columns] + ["FIELD_WI_VOL", "OIL_RATE"]
    if target_col not in cols:
        raise ValueError(f"未知 target_col: {target_col}")
    rec_well, rec_cut = [], []
    rec_hist = {c: [] for c in cols}
    rec_fut = {c: [] for c in cols}
    rec_y, rec_scalar = [], []

    for code in wells:
        well = daily[daily["WELL_BORE_CODE"] == code]
        if well.empty:
            continue
        grid = calendarize_well(well, columns)
        grid["FIELD_WI_VOL"] = field_wi.reindex(grid.index)
        hrs = grid["ON_STREAM_HRS"]
        # 关井日速率无定义(不是 0), 留 NaN; 用 0 会把"没在产"当成"产能为零"
        grid["OIL_RATE"] = np.where(hrs > EPS, grid["BORE_OIL_VOL"] / hrs.where(hrs > EPS), np.nan)
        arrays = {c: grid[c].to_numpy(dtype=np.float64) for c in cols}
        n = len(grid)
        for i in range(history_days, n - horizon + 1, step_days):
            # 入窗条件始终以产油**体积**的可观测性为准, 保证不同 target 下窗口集合一致、可对比
            oil_future = arrays["BORE_OIL_VOL"][i : i + horizon]
            if np.isfinite(oil_future).sum() < min_observed_future:
                continue
            oil_hist = arrays["BORE_OIL_VOL"][i - history_days : i]
            if not np.isfinite(oil_hist).any():
                continue
            if not np.isfinite(np.nanmean(oil_future)):
                continue
            # 🔴 2026-08-07 审计:旧版在这里用 target 相关的条件 continue, 与上方注释
            # "保证不同 target 下窗口集合一致"矛盾(实测 1250 vs 1212), 且剔除的恰是关井
            # 密集的窗口。现在不再因 target 缺失而剔除窗口 —— 目标里的 NaN 由指标层 mask 处理。
            y_future = arrays[target_col][i : i + horizon]
            rec_well.append(code)
            rec_cut.append(grid.index[i])
            for c in cols:
                rec_hist[c].append(arrays[c][i - history_days : i])
                rec_fut[c].append(arrays[c][i : i + horizon])
            rec_y.append(y_future)
            rec_scalar.append(float(np.nanmean(y_future)))

    ws = WindowSet(
        target_col=target_col,
        well=np.asarray(rec_well),
        cutoff=np.asarray(rec_cut, dtype="datetime64[ns]"),
        hist={c: np.asarray(v, dtype=np.float64) for c, v in rec_hist.items()},
        fut={c: np.asarray(v, dtype=np.float64) for c, v in rec_fut.items()},
        y=np.asarray(rec_y, dtype=np.float64),
        y_scalar=np.asarray(rec_scalar, dtype=np.float64),
        history_days=history_days,
        horizon=horizon,
    )
    return add_strata(ws)


if __name__ == "__main__":
    import collections

    for step in (7, 1):
        ws = build_windows(step_days=step)
        print(f"\n=== step={step}d  窗口数={len(ws)} ===")
        for key in ("label", "choke_q", "wi_q"):
            counts = collections.Counter(ws.strata[key])
            line = "  ".join(f"{k}:{counts[k]}" for k in sorted(counts))
            print(f"  {key:8s} {line}")
        print(f"  shutin_flip: {int(ws.strata['shutin_flip'].sum())} / {len(ws)}")
        by_well = collections.Counter(ws.well)
        print("  按井:", {str(w): c for w, c in sorted(by_well.items())})
