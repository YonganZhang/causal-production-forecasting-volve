"""Chronos-2 封装: 纯历史(零样本) 与 干预条件化 两种模式。

权重固定在本机快照 revision 29ec3766d36d6f73f0696f85560a422f50e8498c, 不自动下载。

⚠️ 两种模式**不是同一个预测问题**:
  - zero-shot / past-only : 只用截止点之前的信息 → pure forecasting
  - with future covariates: 额外给了预测窗内的真实干预值(油嘴、开井时长、邻井注水)
    → interventional forecasting, 回答的是"若按此作业计划, 产量会是多少"。
  报告二者对比时必须显式声明这个信息差, 不得表述为"我们的模型更准"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots" / CHRONOS2_REVISION
)

# 军伟 T3 口径的过去协变量(顺序与 p7_chronos2.py 一致, E0 复现依赖)
T3_PAST_COVARIATES = (
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
)

# 本项目 E2 的干预条件化配置
STATE_COVARIATES = (
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "AVG_DP_TUBING",
)
INTERVENTION_COVARIATES = (
    "ON_STREAM_HRS",       # 开井时长: 作业决策
    "AVG_CHOKE_SIZE_P",    # 油嘴开度: 作业决策
    "FIELD_WI_VOL",        # 邻井注水: 井间外生干预
)

_PIPELINE = None


def get_pipeline(device: str = "cuda"):
    """懒加载并缓存 pipeline。"""
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from chronos import Chronos2Pipeline

        if not (SNAPSHOT / "model.safetensors").is_file():
            raise FileNotFoundError(f"Chronos-2 本地快照缺失: {SNAPSHOT}")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        _PIPELINE = Chronos2Pipeline.from_pretrained(
            str(SNAPSHOT), device_map=device, torch_dtype=dtype
        )
    return _PIPELINE


def _clean(a: np.ndarray) -> np.ndarray:
    """把 NaN 留给 Chronos 自己处理, 但 inf 必须拦掉。"""
    a = np.asarray(a, dtype=np.float32)
    if np.isinf(a).any():
        raise ValueError("Chronos 输入含 inf")
    return a


def forecast(
    ws,
    index: np.ndarray | None = None,
    *,
    past_covariates: tuple[str, ...] = (),
    future_covariates: tuple[str, ...] = (),
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """对 WindowSet 的指定子集做预测。

    返回 (n, horizon, n_quantiles) 的分位预测, 已 clip 到非负。
    future_covariates 必须是 past_covariates 的子集(Chronos-2 的约束)。
    """
    pipeline = get_pipeline(device)
    idx = np.arange(len(ws)) if index is None else np.asarray(index)
    missing = set(future_covariates) - set(past_covariates)
    if missing:
        raise ValueError(f"future_covariates 必须是 past_covariates 的子集, 多出: {sorted(missing)}")

    inputs = []
    for i in idx:
        item = {"target": _clean(ws.hist[ws.target_col][i])}
        if past_covariates:
            item["past_covariates"] = {
                c.lower(): _clean(ws.hist[c][i]) for c in past_covariates
            }
        if future_covariates:
            item["future_covariates"] = {
                c.lower(): _clean(ws.fut[c][i]) for c in future_covariates
            }
        inputs.append(item)

    quantiles, _mean = pipeline.predict_quantiles(
        inputs,
        prediction_length=ws.horizon,
        quantile_levels=list(quantile_levels),
        batch_size=int(batch_size),
        cross_learning=False,
    )
    out = np.empty((len(idx), ws.horizon, len(quantile_levels)), dtype=np.float64)
    for j, q in enumerate(quantiles):
        v = q.detach().float().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        v = np.squeeze(v)                      # -> (horizon, n_quantiles)
        if v.shape != (ws.horizon, len(quantile_levels)):
            v = v.T
        out[j] = np.maximum(v, 0.0)
    return out


def forecast_t3_calendar(sequences: np.ndarray, timestamps: np.ndarray,
                         sample_ids, *, batch_size: int = 196, device: str = "cuda"):
    """军伟 T3 的 predict_df 调用路径, 原样复刻用于 E0。

    sequences: (n, 7, 30), 通道顺序 = volve_data.T3_SEQUENCE_COLUMNS
    返回 (daily (n,30), scalar_mean (n,))
    """
    pipeline = get_pipeline(device)
    array = np.asarray(sequences, dtype=np.float32)
    names = [c.lower() for c in T3_PAST_COVARIATES]
    records = []
    for b, sid in enumerate(sample_ids):
        item_time = pd.DatetimeIndex(pd.to_datetime(timestamps[b]))
        for s, ts in enumerate(item_time):
            rec = {"item_id": str(sid), "timestamp": ts, "target": float(array[b, 0, s])}
            for ch, name in enumerate(names, start=1):
                rec[name] = float(array[b, ch, s])
            records.append(rec)
    frame = pd.DataFrame.from_records(records).sort_values(
        ["item_id", "timestamp"], kind="stable").reset_index(drop=True)

    forecast_df = pipeline.predict_df(
        frame,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=30,
        quantile_levels=[0.5],
        batch_size=int(batch_size),
        context_length=30,
        cross_learning=False,
        validate_inputs=True,
    )
    by_item = {str(k): g for k, g in forecast_df.groupby("item_id", sort=False)}
    daily = []
    for sid in sample_ids:
        g = by_item[str(sid)].sort_values("timestamp")
        daily.append(np.maximum(g["predictions"].to_numpy(dtype=np.float64), 0.0))
    daily = np.asarray(daily, dtype=np.float64)
    return daily, daily.mean(axis=1)
