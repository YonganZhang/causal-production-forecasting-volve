"""Volve 日度生产数据装载与日历化。

设计约束(来自数据集 README 的「已知坑」):
  - 缺日 (date 完全没有记录) 与关井日 (ON_STREAM_HRS=0 但有记录) 是两回事,
    绝不用 0 填补把两者混成一样。日历化后缺日为 NaN, 关井日为真实的 0。
  - 井名有多套编码; 本模块内部统一用 WELL_BORE_CODE(与军伟 T3 契约的 group_key 一致),
    对外展示用短名 (F-12 等)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = PROJECT_ROOT / "_data" / "volve_causal_v0.2" / "daily_production.csv"

# WELL_BORE_CODE -> 短名。角色依据 _data/volve_causal_v0.2/well_metadata.csv。
PRODUCERS = {
    "NO 15/9-F-1 C": "F-1C",
    "NO 15/9-F-11 H": "F-11",
    "NO 15/9-F-12 H": "F-12",
    "NO 15/9-F-14 H": "F-14",
    "NO 15/9-F-15 D": "F-15D",
}
# F-5 在 2016-04-20 前是注水井、之后转生产(v0.2 修正)。作为邻井干预源时只用其注水段。
INJECTORS = {
    "NO 15/9-F-4 AH": "F-4",
    "NO 15/9-F-5 AH": "F-5",
}
F5_ROLE_SWITCH = pd.Timestamp("2016-04-20")

# 体积/累计类按日求和, 状态类按日求均值 —— 与军伟 T3 契约一致, 保证 E0 可比。
VOLUME_COLUMNS = ("BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "ON_STREAM_HRS", "BORE_WI_VOL")
STATE_COLUMNS = (
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS",
)
ALL_NUMERIC = VOLUME_COLUMNS + STATE_COLUMNS

# 军伟 T3 的 7 通道历史序列, 顺序不可改(E0 复现依赖)。
T3_SEQUENCE_COLUMNS = (
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
)


def load_daily(path: Path | str = DAILY_CSV) -> pd.DataFrame:
    """读日度 CSV, 解析日期与数值列。"""
    frame = pd.read_csv(path, low_memory=False)
    frame["DATEPRD"] = pd.to_datetime(frame["DATEPRD"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["DATEPRD"])
    for column in ALL_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["WELL_BORE_CODE", "DATEPRD"]).reset_index(drop=True)


def calendarize_well(well: pd.DataFrame, columns: tuple[str, ...] = ALL_NUMERIC) -> pd.DataFrame:
    """把单井压到规则日历日网格上; 缺日补 NaN 行(而不是 0)。"""
    wanted = [c for c in columns if c in well.columns]
    # 🔴 体积列必须用 min_count=1:pandas 的 sum() 默认把「整日全 NaN」折成 0.0,
    # 那等于把"缺测"伪造成"产量/注入量为零"—— 正是本模块声称绝不做的事。
    # 2026-08-07 审计发现:F-4 有 337 天、F-5 有 590 天 BORE_WI_VOL 缺测被折成 0,
    # 凭空制造了"全场停注"事件, 直接污染 E3 的注水干预分层。
    daily = well.groupby("DATEPRD", sort=True).agg(
        **{c: pd.NamedAgg(column=c, aggfunc=(
            (lambda x: x.sum(min_count=1)) if c in VOLUME_COLUMNS else "mean"))
           for c in wanted})
    index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(index)
    daily.index.name = "DATEPRD"
    return daily


def calendarize_all(frame: pd.DataFrame, wells: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """按 WELL_BORE_CODE 逐井日历化。"""
    codes = list(wells) if wells is not None else sorted(frame["WELL_BORE_CODE"].unique())
    return {
        code: calendarize_well(frame[frame["WELL_BORE_CODE"] == code])
        for code in codes
        if (frame["WELL_BORE_CODE"] == code).any()
    }


def field_injection_series(frame: pd.DataFrame) -> pd.DataFrame:
    """构造井间干预源: 每口注水井的日注水量, 对齐到统一日历。

    F-5 转生产后(>= 2016-04-20)不再是注水源, 其注水列在此之后置 0 —— 这是 v0.2
    角色修正的直接后果, 用 v0.1 的错误时间线会把这段标反。
    """
    series = {}
    for code, short in INJECTORS.items():
        well = frame[frame["WELL_BORE_CODE"] == code]
        if well.empty:
            continue
        daily = calendarize_well(well, ("BORE_WI_VOL",))
        wi = daily["BORE_WI_VOL"]
        if code == "NO 15/9-F-5 AH":
            wi = wi.copy()
            wi.loc[wi.index >= F5_ROLE_SWITCH] = 0.0
        series[f"WI_{short}"] = wi
    out = pd.DataFrame(series)
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(full)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """快速体检表: 每口生产井的跨度、缺日、关井日、关键列缺失。"""
    rows = []
    for code, short in PRODUCERS.items():
        well = frame[frame["WELL_BORE_CODE"] == code]
        daily = calendarize_well(well)
        on = daily["ON_STREAM_HRS"]
        rows.append({
            "well": short,
            "start": daily.index.min().date(),
            "end": daily.index.max().date(),
            "calendar_days": len(daily),
            "missing_days": int(on.isna().sum()),
            "shut_in_days": int((on == 0).sum()),
            "oil_nan": int(daily["BORE_OIL_VOL"].isna().sum()),
            "choke_nan": int(daily["AVG_CHOKE_SIZE_P"].isna().sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_daily()
    print(summarize(df).to_string(index=False))
    wi = field_injection_series(df)
    print("\n注水源(F-5 转产后已置 0):")
    print(wi.describe().to_string())
