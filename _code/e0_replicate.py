"""E0 — 复现军伟项目 T3 baseline, 作为本实验台的信任锚点。

目标数字 (来源: 师弟-军伟的比赛-2693e5/_wiki-methodology/_tests/_run_ledger.md:86)
  历史均值 MAE 184.6686
  Chronos-2 MAE 172.3162
契约 (来源: _pipelines/02_task_datasets/sweetspot/p8/calendar_data.py)
  30 天历史 -> 未来 30 天日产油**均值**(标量, Sm3/d)
  HISTORY=30 FORECAST=30 STEP=7 MIN_OBSERVED_FUTURE=21 ROOT_SEED=2693
  development 井 4 口 (F-1C/F-11/F-12/F-14; F-15D 是留出测试集)
  4 折留一井, 每折种子子采样 196 train / 84 validation
验收门: 两个数各落在 ±1% 内。不过门就停下来查, 不往上叠实验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import mae
from volve_data import T3_SEQUENCE_COLUMNS, load_daily
from windows import build_windows

ROOT_SEED = 2693
TARGET_HISTORY_MEAN_MAE = 184.6686
TARGET_CHRONOS_MAE = 172.3162
TOLERANCE = 0.01

DEVELOPMENT_GROUPS = ["NO 15/9-F-1 C", "NO 15/9-F-11 H", "NO 15/9-F-12 H", "NO 15/9-F-14 H"]
FOLDS = [
    {"fold_id": 0, "train_groups": ["NO 15/9-F-11 H", "NO 15/9-F-12 H", "NO 15/9-F-14 H"],
     "validation_groups": ["NO 15/9-F-1 C"]},
    {"fold_id": 1, "train_groups": ["NO 15/9-F-1 C", "NO 15/9-F-11 H", "NO 15/9-F-14 H"],
     "validation_groups": ["NO 15/9-F-12 H"]},
    {"fold_id": 2, "train_groups": ["NO 15/9-F-1 C", "NO 15/9-F-11 H", "NO 15/9-F-12 H"],
     "validation_groups": ["NO 15/9-F-14 H"]},
    {"fold_id": 3, "train_groups": ["NO 15/9-F-1 C", "NO 15/9-F-12 H", "NO 15/9-F-14 H"],
     "validation_groups": ["NO 15/9-F-11 H"]},
]


def seeded_subset(sample_ids: list[str], limit: int, *, fold_id: int, lane: str) -> list[int]:
    """复刻 calendar_data._seeded_subset: 按 sha256 排序取前 limit 个。"""
    keyed = sorted(
        range(len(sample_ids)),
        key=lambda i: hashlib.sha256(
            f"{ROOT_SEED}|T3-calendar|{fold_id}|{lane}|{sample_ids[i]}".encode("utf-8")
        ).hexdigest(),
    )
    return keyed[:limit]


def t3_sequences(ws, idx: np.ndarray) -> np.ndarray:
    """(n, 7, 30) 历史张量, 通道顺序 = T3_SEQUENCE_COLUMNS。"""
    return np.stack([ws.hist[c][idx] for c in T3_SEQUENCE_COLUMNS], axis=1)


def t3_timestamps(ws, idx: np.ndarray) -> np.ndarray:
    """(n, 30) 历史日期。cutoff 是预测窗第一天, 历史 = cutoff-30 .. cutoff-1。"""
    import pandas as pd

    starts = pd.to_datetime(ws.cutoff[idx]) - pd.Timedelta(days=ws.history_days)
    return np.stack([
        pd.date_range(s, periods=ws.history_days, freq="D").to_numpy() for s in starts
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-chronos", action="store_true", help="只跑历史均值基线")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    daily = load_daily()
    wells = {code: code for code in DEVELOPMENT_GROUPS}
    ws = build_windows(daily, wells=wells, history_days=30, horizon=30,
                       step_days=7, min_observed_future=21)
    ids = ws.sample_ids()
    print(f"development 窗口总数: {len(ws)}")

    results = {"folds": [], "config": {
        "history": 30, "horizon": 30, "step": 7, "min_observed_future": 21,
        "root_seed": ROOT_SEED, "development_groups": DEVELOPMENT_GROUPS,
        "train_limit": 196, "validation_limit": 84,
    }}
    hist_maes, chronos_maes = [], []

    for spec in FOLDS:
        fid = spec["fold_id"]
        val_pool = [i for i in range(len(ws)) if ws.well[i] in spec["validation_groups"]]
        trn_pool = [i for i in range(len(ws)) if ws.well[i] in spec["train_groups"]]
        val_idx = np.asarray([val_pool[k] for k in
                              seeded_subset([ids[i] for i in val_pool], 84, fold_id=fid, lane="validation")])
        trn_idx = np.asarray([trn_pool[k] for k in
                              seeded_subset([ids[i] for i in trn_pool], 196, fold_id=fid, lane="train")])

        y = ws.y_scalar[val_idx]
        hist_pred = np.nanmean(ws.hist["BORE_OIL_VOL"][val_idx], axis=1)
        h_mae = mae(y, hist_pred)
        hist_maes.append(h_mae)

        row = {"fold_id": fid, "validation_groups": spec["validation_groups"],
               "train_n": len(trn_idx), "validation_n": len(val_idx),
               "history_mean_mae": h_mae}

        if not args.skip_chronos:
            from models_chronos2 import forecast_t3_calendar

            _, scalar = forecast_t3_calendar(
                t3_sequences(ws, val_idx), t3_timestamps(ws, val_idx),
                [ids[i] for i in val_idx], device=args.device)
            c_mae = mae(y, scalar)
            chronos_maes.append(c_mae)
            row["chronos2_mae"] = c_mae

        results["folds"].append(row)
        print(f"  fold {fid} val={spec['validation_groups'][0]:16s} n={len(val_idx)}"
              f"  history_mean MAE={h_mae:9.4f}"
              + (f"  chronos2 MAE={row['chronos2_mae']:9.4f}" if "chronos2_mae" in row else ""))

    macro_hist = float(np.mean(hist_maes))
    results["macro"] = {"history_mean_mae": macro_hist}
    print(f"\n宏平均 history_mean MAE = {macro_hist:.4f}  (目标 {TARGET_HISTORY_MEAN_MAE})")
    ok = [_gate("history_mean", macro_hist, TARGET_HISTORY_MEAN_MAE)]

    if chronos_maes:
        macro_chronos = float(np.mean(chronos_maes))
        results["macro"]["chronos2_mae"] = macro_chronos
        print(f"宏平均 chronos2     MAE = {macro_chronos:.4f}  (目标 {TARGET_CHRONOS_MAE})")
        ok.append(_gate("chronos2", macro_chronos, TARGET_CHRONOS_MAE))

    results["gate_passed"] = all(ok)
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "_pipelines" / "e0_replicate" / "results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0 if all(ok) else 1


def _gate(name: str, value: float, target: float) -> bool:
    rel = abs(value - target) / target
    status = "PASS" if rel <= TOLERANCE else "FAIL"
    print(f"  [{status}] {name}: 相对偏差 {rel:.3%} (门 ±{TOLERANCE:.0%})")
    return rel <= TOLERANCE


if __name__ == "__main__":
    raise SystemExit(main())
