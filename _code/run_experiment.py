"""E1 / E2 / E3 主实验入口。

E1  30 步日产油轨迹, 全量窗口, rolling-origin 时间序 CV(主口径)
E2  Chronos-2 干预条件化 + 消融
E3  按干预类型/强度分层评估 —— 课题主张的可证伪检验

🔴 两条硬规则(2026-08-06 栽过一次之后加的):

1. 信息差: 带 future_* 的模型使用了预测窗内的真实作业值, 与纯历史模型不是同一个
   预测问题。只能在组内比较, 跨组比较必须声明。

2. **恒等式及格线**: 产油体积 ≈ 速率 x 开井小时。任何用了未来开井小时数的模型,
   必须先打过 baselines.identity_rate_x_hours 这条三行公式, 否则它的"提升"是记账
   不是建模。本脚本会自动判定并在总表打 VERDICT 列, 打不过就写 FAIL。

目标口径:
  --target volume  日产油体积 Sm3/d。含排产记账项, 恒等式及格线在此生效。
  --target rate    开井期间产油速率 Sm3/开井小时。除掉记账项后的油藏物理量。
                   "油嘴/注水影响产能" 这类因果主张必须在这个口径上成立才算数。

用法:
    python run_experiment.py --step 1 --target volume
    python run_experiment.py --step 1 --target rate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
from evaluate import (
    bootstrap_ci,
    coverage,
    crps_from_quantiles,
    mae,
    naive_scale,
    paired_bootstrap_diff,
    per_window_mae,
    rmse,
    stratified_table,
)
from splits import (
    assert_no_calendar_overlap,
    describe,
    leave_one_well_out,
    rolling_origin_folds,
)
from windows import build_windows

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_AT = 4
LO_AT, HI_AT = 0, 8          # 10% / 90% -> 80% 区间

INFO_NOTE = (
    "带 future_* 的模型额外使用了预测窗内的真实作业值, 与纯历史模型不是同一个预测问题; "
    "跨这两组比较 MAE 必须声明信息差。干预组内部一律以 identity_rate_x_hours 为及格线。"
)
TARGET_COL = {"volume": "BORE_OIL_VOL", "rate": "OIL_RATE"}

# 名称 -> (past_covariates, future_covariates)。future 必须是 past 的子集。
STATE = ("BORE_GAS_VOL", "BORE_WAT_VOL", "AVG_DOWNHOLE_PRESSURE", "AVG_WHP_P", "AVG_DP_TUBING")
INTERV = ("ON_STREAM_HRS", "AVG_CHOKE_SIZE_P", "FIELD_WI_VOL")

CHRONOS_CONFIGS = {
    # --- 纯历史组 (pure forecasting) ---
    "chronos2_zeroshot":        ((), ()),
    "chronos2_past_state":      (STATE, ()),
    "chronos2_past_all":        (STATE + INTERV, ()),
    # --- 干预条件化组 (interventional forecasting) ---
    "chronos2_futr_choke":      (STATE + INTERV, ("AVG_CHOKE_SIZE_P",)),
    "chronos2_futr_onstream":   (STATE + INTERV, ("ON_STREAM_HRS",)),
    "chronos2_futr_injection":  (STATE + INTERV, ("FIELD_WI_VOL",)),
    "chronos2_futr_all":        (STATE + INTERV, INTERV),
}
PURE_MODELS = set(baselines.REGISTRY_PURE) | {
    "lightgbm_history", "chronos2_zeroshot", "chronos2_past_state", "chronos2_past_all"}
IDENTITY = "identity_rate_x_hours"


def _metrics(y, pred_q, hist_oil, probabilistic: bool) -> dict:
    """probabilistic=False 的模型只有点预测, CRPS 退化成 MAE、区间覆盖率无意义,
    一律报 NaN 而不是印一个会被误读的数字。"""
    med = pred_q[:, :, MEDIAN_AT]
    per = per_window_mae(y, med)
    scale = naive_scale(hist_oil)
    lo, hi = bootstrap_ci(per)
    return {
        "mae": mae(y, med),
        "mae_ci95": [lo, hi],
        "rmse": rmse(y, med),
        "mase": float(np.nanmean(per / scale)),
        "crps": crps_from_quantiles(y, pred_q, np.asarray(QUANTILES)) if probabilistic else float("nan"),
        "coverage80": coverage(y, pred_q[:, :, LO_AT], pred_q[:, :, HI_AT]) if probabilistic else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1, help="滑窗步长(天)")
    ap.add_argument("--target", choices=["volume", "rate"], default="volume")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=["rolling", "well"], default="rolling",
                    help="rolling=时间序(主口径); well=留一井(第二口径, 折间方差大)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    ws = build_windows(step_days=args.step, target_col=TARGET_COL[args.target])
    if args.split == "rolling":
        folds = rolling_origin_folds(ws, n_folds=args.folds)
        for f in folds:
            assert_no_calendar_overlap(ws, f)
        print(f"窗口 {len(ws)} 个, rolling-origin {len(folds)} 折 (泄漏检查通过)")
    else:
        folds = leave_one_well_out(ws)
        print(f"窗口 {len(ws)} 个, leave-one-well-out {len(folds)} 折")
        print("⚠️  留一井口径每折验证集只有 1 口井, 折间方差极大, 仅作第二口径参考。")
    print(describe(ws, folds).to_string(index=False))

    valid = np.zeros(len(ws), dtype=bool)
    for f in folds:
        valid |= f.valid
    vidx = np.flatnonzero(valid)
    y = ws.y[vidx]
    hist_oil = ws.hist[ws.target_col][vidx]
    print(f"\n评估窗口(5 折验证集并集): {len(vidx)}")

    preds: dict[str, np.ndarray] = {}

    # ---- 无需训练的 baseline ----
    for name, fn in baselines.REGISTRY.items():
        p = fn(ws, vidx)
        preds[name] = np.repeat(p[:, :, None], len(QUANTILES), axis=2)
        print(f"  [baseline] {name:18s} done")

    # ---- LightGBM: 逐折训练 ----
    for tag, use_int in (("lightgbm_history", False), ("lightgbm_intervention", True)):
        acc = np.full((len(vidx), ws.horizon), np.nan)
        pos = {g: i for i, g in enumerate(vidx)}
        for f in folds:
            tr, va = np.flatnonzero(f.train), np.flatnonzero(f.valid)
            p = baselines.lightgbm_forecast(ws, tr, va, use_interventions=use_int)
            for k, g in enumerate(va):
                acc[pos[g]] = p[k]
        preds[tag] = np.repeat(acc[:, :, None], len(QUANTILES), axis=2)
        print(f"  [lgbm]     {tag:18s} done")

    # ---- Chronos-2 各配置 ----
    from models_chronos2 import forecast

    for name, (past, futr) in CHRONOS_CONFIGS.items():
        t = time.time()
        preds[name] = forecast(ws, vidx, past_covariates=past, future_covariates=futr,
                               quantile_levels=QUANTILES, device=args.device)
        print(f"  [chronos]  {name:18s} {time.time() - t:5.1f}s")

    # ---- E1/E2 汇总表 ----
    rows = []
    per_window = {}
    for name, pq in preds.items():
        m = _metrics(y, pq, hist_oil, probabilistic=name.startswith("chronos2"))
        per_window[name] = per_window_mae(y, pq[:, :, MEDIAN_AT])
        rows.append({"model": name,
                     "setting": "pure" if name in PURE_MODELS else "interventional",
                     **{k: v for k, v in m.items() if k != "mae_ci95"},
                     "mae_lo": m["mae_ci95"][0], "mae_hi": m["mae_ci95"][1]})
    table = pd.DataFrame(rows).sort_values(["setting", "mae"])

    # 恒等式及格线判定 —— 干预组任何模型打不过三行公式就是 FAIL
    identity_mae = float(table.loc[table["model"] == IDENTITY, "mae"].iloc[0])
    identity_applicable = np.isfinite(identity_mae)   # rate 口径下恒等式无定义
    best_pure = float(table.loc[table["setting"] == "pure", "mae"].min())
    table["verdict"] = [
        "—" if r.model == IDENTITY else
        ("n/a(恒等式不适用)" if not identity_applicable else
         ("PASS" if r.mae < identity_mae else "FAIL(输给恒等式)")) if r.setting == "interventional"
        else ("PASS" if r.mae <= best_pure else "—")
        for r in table.itertuples()]

    print(f"\n===== E1/E2 总表 (目标={args.target}, 30 步轨迹, {args.split} 验证集并集) =====")
    print(f"⚠️  {INFO_NOTE}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_fail = int((table["verdict"].str.startswith("FAIL")).sum())
    if identity_applicable:
        print(f"\n🔴 恒等式及格线 = {identity_mae:.4f} ({IDENTITY});"
              f" 干预组中打不过它的模型: {n_fail} 个")
    else:
        print("\n⚠️  恒等式及格线在本口径(rate)下无定义 —— 体积=速率x时长 这条恒等式已被除掉。"
              "\n    干预组请与纯历史组最优(%.4f)对照, 不要用一条退化基线冒充门。" % best_pure)

    # ---- E3 分层 ----
    strata = {k: ws.strata[k][vidx] for k in ("label", "choke_q", "wi_q")}
    strata["shutin"] = np.where(ws.strata["shutin_flip"][vidx], "shutin", "no_shutin")

    key_models = ["history_mean", "arps_exponential", "lightgbm_history",
                  "chronos2_zeroshot", "chronos2_past_all",
                  IDENTITY, "lightgbm_intervention", "chronos2_futr_all"]
    key_models = [m for m in key_models if m in per_window]
    sub = {m: per_window[m] for m in key_models}

    e3 = {}
    print("\n===== E3 干预分层 (逐窗 MAE 均值) =====")
    for key, order in (("shutin", ["no_shutin", "shutin"]),
                       ("choke_q", ["Q1", "Q2", "Q3", "Q4"]),
                       ("wi_q", ["Q1", "Q2", "Q3", "Q4"]),
                       ("label", None)):
        tab = stratified_table(sub, strata[key], order)
        e3[key] = tab.to_dict(orient="records")
        print(f"\n-- 分层: {key} --")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # 退化幅度: 最强干预层 vs 最弱干预层的 MAE 比值
    print("\n===== E3 退化幅度 (高干预层 MAE / 低干预层 MAE, 越接近 1 越稳健) =====")
    degradation = {}
    for key, lo_k, hi_k in (("shutin", "no_shutin", "shutin"),
                            ("choke_q", "Q1", "Q4"), ("wi_q", "Q1", "Q4")):
        s = strata[key]
        row = {}
        for m in key_models:
            a = per_window[m][s == hi_k]
            b = per_window[m][s == lo_k]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            row[m] = float(a.mean() / b.mean()) if a.size and b.size else float("nan")
        degradation[key] = row
        print(f"  {key:8s} " + "  ".join(f"{m.split('_', 1)[-1][:14]}={row[m]:.2f}" for m in key_models))

    # 配对 bootstrap: 干预条件化 vs 其纯历史同源对照
    print("\n===== 配对 bootstrap: chronos2_futr_all - identity_rate_x_hours (逐窗 MAE 差) =====")
    print("     对手换成恒等式基线 —— 与 chronos2_past_all 比是不公平的(信息量不同),")
    print("     与恒等式比才公平(信息量相同)。负值 = 模型确实学到了记账之外的东西。")
    pairs = {}
    for key, order in (("shutin", ["no_shutin", "shutin"]), ("choke_q", ["Q1", "Q2", "Q3", "Q4"])):
        for k in order:
            s = strata[key] == k
            d = paired_bootstrap_diff(per_window["chronos2_futr_all"][s],
                                      per_window[IDENTITY][s])
            pairs[f"{key}={k}"] = d
            print(f"  {key}={k:10s} n={d['n']:5d}  Δ={d['diff']:9.2f}  95%CI [{d['lo']:8.2f}, {d['hi']:8.2f}]")

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "_pipelines" / f"e1_e3_{args.target}_{args.split}" / "results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"step_days": args.step, "n_folds": args.folds,
                   "target": args.target, "target_col": ws.target_col,
                   "identity_gate_mae": identity_mae, "n_failing_identity_gate": n_fail,
                   "n_windows": int(len(ws)), "n_eval": int(len(vidx)),
                   "quantiles": list(QUANTILES),
                   "chronos_configs": {k: {"past": list(v[0]), "future": list(v[1])}
                                       for k, v in CHRONOS_CONFIGS.items()}},
        "information_disclosure": INFO_NOTE,
        "folds": describe(ws, folds).to_dict(orient="records"),
        "e1_e2_table": table.to_dict(orient="records"),
        "e3_strata": e3,
        "e3_degradation_ratio": degradation,
        "e3_paired_bootstrap": pairs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    table.to_csv(out.parent / "e1_e2_table.csv", index=False)
    print(f"\n结果已写入 {out}  (耗时 {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
