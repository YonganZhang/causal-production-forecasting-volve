"""实验台的正确性测试。可直接 `python _tests/test_harness.py` 跑, 也兼容 pytest。

覆盖: 窗口构造无泄漏、分层标签与原始 CSV 手工核对、指标实现、切分 embargo。
不覆盖 GPU 推理(那是 e0_replicate.py 的复现门负责)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_code"))

import evaluate  # noqa: E402
from splits import assert_no_calendar_overlap, rolling_origin_folds  # noqa: E402
from volve_data import calendarize_well, load_daily  # noqa: E402
from windows import build_windows  # noqa: E402

_DAILY = None
_WS = None


def daily():
    global _DAILY
    if _DAILY is None:
        _DAILY = load_daily()
    return _DAILY


def ws7():
    global _WS
    if _WS is None:
        _WS = build_windows(daily(), step_days=7)
    return _WS


def test_history_and_future_do_not_overlap():
    """历史窗必须严格是 [cutoff-H, cutoff-1], 未来窗是 [cutoff, cutoff+F-1]。"""
    ws = ws7()
    d = daily()
    rng = np.random.default_rng(7)
    for i in rng.choice(len(ws), size=30, replace=False):
        grid = calendarize_well(d[d["WELL_BORE_CODE"] == ws.well[i]])
        cut = pd.Timestamp(ws.cutoff[i])
        hist = grid.loc[cut - pd.Timedelta(days=ws.history_days): cut - pd.Timedelta(days=1)]
        futr = grid.loc[cut: cut + pd.Timedelta(days=ws.horizon - 1)]
        assert len(hist) == ws.history_days and len(futr) == ws.horizon
        np.testing.assert_allclose(hist["BORE_OIL_VOL"].to_numpy(),
                                   ws.hist["BORE_OIL_VOL"][i], equal_nan=True)
        np.testing.assert_allclose(futr["BORE_OIL_VOL"].to_numpy(),
                                   ws.y[i], equal_nan=True)


def test_missing_days_stay_nan_not_zero():
    """缺日必须是 NaN, 不能被当成产量 0 —— 否则 MAE 被系统性污染。"""
    d = daily()
    grid = calendarize_well(d[d["WELL_BORE_CODE"] == "NO 15/9-F-12 H"])
    recorded = set(d[d["WELL_BORE_CODE"] == "NO 15/9-F-12 H"]["DATEPRD"])
    gaps = [ts for ts in grid.index if ts not in recorded]
    assert len(gaps) == 85, f"F-12 应有 85 个缺日, 实得 {len(gaps)}"
    assert grid.loc[gaps, "BORE_OIL_VOL"].isna().all()
    # 关井日则是真实的 0, 有记录
    shut = grid[(grid["ON_STREAM_HRS"] == 0)]
    assert len(shut) > 0 and shut.index.isin(list(recorded)).all()


def test_shutin_label_matches_raw_data():
    """S1 关井标签逐窗手工核对: 预测窗内 ON_STREAM_HRS 确实在 0 与 >0 间跳变。"""
    ws = ws7()
    flip = ws.strata["shutin_flip"]
    rng = np.random.default_rng(11)
    for i in rng.choice(len(ws), size=40, replace=False):
        on = np.concatenate([ws.hist["ON_STREAM_HRS"][i][-1:], ws.fut["ON_STREAM_HRS"][i]])
        obs = np.isfinite(on)
        state = on > 1e-6
        pair = obs[:-1] & obs[1:]
        expected = bool(((state[:-1] != state[1:]) & pair).any())
        assert bool(flip[i]) == expected, f"窗口 {i} 关井标签与原始序列不符"


def test_quartile_bins_are_balanced():
    ws = ws7()
    for key in ("choke_q", "wi_q"):
        counts = pd.Series(ws.strata[key]).value_counts()
        q = counts[[k for k in counts.index if k.startswith("Q")]]
        assert q.min() > 0.4 * q.max(), f"{key} 分箱严重不均: {dict(q)}"


def test_target_scalar_matches_trajectory_mean():
    ws = ws7()
    np.testing.assert_allclose(ws.y_scalar, np.nanmean(ws.y, axis=1), rtol=1e-12)


def test_rolling_origin_has_no_calendar_leak():
    ws = build_windows(daily(), step_days=1)
    for f in rolling_origin_folds(ws, n_folds=5):
        assert_no_calendar_overlap(ws, f)


def test_identity_baseline_beats_intervention_conditioned_chronos_on_volume():
    """回归测试, 锁住 2026-08-06 的撤稿教训。

    日产油体积满足 体积 ≈ 速率 x 开井小时。任何把未来开井小时当协变量的模型,
    必须先打过这条恒等式公式。这里只断言恒等式基线本身**存在且强**——它比所有
    纯历史基线都好，因为它用了未来信息。若哪天有模型真的打过它，这个测试不会挡路
    (它不断言模型必败)，但 run_experiment.py 的 verdict 列会如实记录。
    """
    import baselines
    from evaluate import mae as _mae

    ws = ws7()
    idx = np.arange(len(ws))
    ident = baselines.identity_rate_x_hours(ws, idx)
    assert _mae(ws.y, ident) < _mae(ws.y, baselines.history_mean(ws, idx)), \
        "恒等式基线应显著优于持平历史均值; 若不成立说明窗口或列对齐坏了"
    assert baselines.IDENTITY_IN_REGISTRY, "恒等式基线必须留在 REGISTRY 里, 不许被摘掉"


def test_rate_target_removes_the_bookkeeping_identity():
    """rate 口径下, 关井日必须是 NaN(速率无定义)而不是 0。"""
    ws = build_windows(daily(), step_days=7, target_col="OIL_RATE")
    assert ws.target_col == "OIL_RATE"
    hrs = ws.fut["ON_STREAM_HRS"]
    shut = np.isfinite(hrs) & (hrs <= 0)
    assert shut.any(), "测试数据里应存在关井日"
    assert np.isnan(ws.y[shut]).all(), "关井日的产油速率必须是 NaN, 不能是 0"
    open_days = np.isfinite(hrs) & (hrs > 0) & np.isfinite(ws.y)
    assert (ws.y[open_days] >= 0).all()


def test_mae_ignores_nan_instead_of_filling_zero():
    y = np.array([[10.0, np.nan, 20.0]])
    p = np.array([[12.0, 999.0, 18.0]])
    assert evaluate.mae(y, p) == 2.0
    np.testing.assert_allclose(evaluate.per_window_mae(y, p), [2.0])


def test_crps_of_point_forecast_equals_mae():
    """退化成点预测时, pinball 平均应等于 MAE —— 用来验证 CRPS 实现没有系数错误。"""
    rng = np.random.default_rng(3)
    y = rng.normal(100, 20, size=(50, 30))
    point = rng.normal(100, 20, size=(50, 30))
    levels = np.arange(0.1, 0.95, 0.1)
    q = np.repeat(point[:, :, None], len(levels), axis=2)
    np.testing.assert_allclose(
        evaluate.crps_from_quantiles(y, q, levels), evaluate.mae(y, point), rtol=1e-9)


def test_paired_bootstrap_detects_no_difference():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 2000)
    out = evaluate.paired_bootstrap_diff(a, a.copy())
    assert out["diff"] == 0.0 and out["lo"] == 0.0 and out["hi"] == 0.0


def test_f5_is_injector_then_producer():
    """v0.2 的角色修正必须体现在数据里: F-5 先注水后产油。"""
    d = daily()
    f5 = d[d["WELL_BORE_CODE"] == "NO 15/9-F-5 AH"]
    first_oil = f5.loc[f5["BORE_OIL_VOL"] > 0, "DATEPRD"].min()
    first_wi = f5.loc[f5["BORE_WI_VOL"] > 0, "DATEPRD"].min()
    assert first_wi < first_oil, "F-5 应是先注水后产油 (injector_to_producer)"
    assert first_wi == pd.Timestamp("2008-08-26") and first_oil == pd.Timestamp("2016-04-20")

    meta = pd.read_csv(ROOT / "_data" / "volve_causal_v0.2" / "well_metadata.csv")
    row = meta[meta["well"] == "15/9-F-5"].iloc[0]
    assert row["role"] == "injector_to_producer"
    assert row["role_switch_date"] == "2016-04-20"


def test_f5_injection_covariate_zeroed_after_conversion():
    """F-5 转生产后不再是注水干预源, 否则 E2/E3 的注水协变量会串味。"""
    from volve_data import F5_ROLE_SWITCH, field_injection_series

    wi = field_injection_series(daily())
    after = wi.loc[wi.index >= F5_ROLE_SWITCH, "WI_F-5"]
    assert (after.fillna(0) == 0).all()




def test_injection_missing_days_are_nan_not_zero():
    """回归测试:锁 2026-08-07 审计的 CRITICAL-1。

    pandas 的 sum() 默认 min_count=0, 会把「整日全 NaN」折成 0.0,
    等于把注水缺测伪造成"停注"。实测 F-4 有 337 天、F-5 有 590 天缺测。
    """
    from volve_data import field_injection_series

    d = daily()
    wi = field_injection_series(d)
    raw_nan = int(d[d.WELL_BORE_CODE == "NO 15/9-F-4 AH"].BORE_WI_VOL.isna().sum())
    assert raw_nan > 300, f"原始数据应有大量注水缺测, 实得 {raw_nan}"
    out_nan = int(wi["WI_F-4"].isna().sum())
    assert out_nan > 300, f"缺测必须保持 NaN, 实得 {out_nan}(被折成 0 了)"


def test_placebo_test_shifts_design_matrix_too():
    """回归测试:锁 2026-08-07 审计的 CRITICAL-2。

    旧实现只平移结果变量 y、不平移设计矩阵 X。而安慰剂位移 (1,3,7) 全在
    LAGS=(1,2,3,7) 里, 平移后的 y 恰好是 X 的一列(R²=0.9999), nuisance 完美拟合,
    θ_placebo 机械趋近 0 —— 检验对任何输入都返回"通过"。

    这里检验**行为**而不是结构:在已知存在混杂的 F-15D 上, 修好后的安慰剂值必须
    明显非零。旧实现在同一口井上给出的是 ~0.02 量级。
    """
    from dml_choke import build_well_frame, placebo_test

    g = build_well_frame(daily(), "NO 15/9-F-15 D")
    out = placebo_test(g, "qo", shifts=(0, 3), k=3)
    plac = abs(out.get(3, 0.0))
    assert np.isfinite(plac), "安慰剂检验应返回有限值"
    assert plac > 0.02, (
        f"安慰剂值 {plac:.4f} 过小, 疑似 placebo_test 又退回"
        f"只平移 y 不平移 X 的实现(那会让检验恒真)")


def test_block_bootstrap_is_wider_than_iid_on_correlated_data():
    """回归测试:锁 2026-08-07 审计的 MAJOR。重叠窗口上 iid bootstrap 会把 CI 收窄数倍。"""
    rng = np.random.default_rng(0)
    x = np.convolve(rng.normal(size=3000), np.ones(60) / 60, "same")
    lo_i, hi_i = evaluate.bootstrap_ci(x, block=None)
    lo_b, hi_b = evaluate.bootstrap_ci(x)
    assert (hi_b - lo_b) > 3 * (hi_i - lo_i), "自相关序列上 block bootstrap 必须显著更宽"


def test_crps_all_masked_returns_nan_not_zero():
    """回归测试:CRPS 全 mask 时曾返回 0.0(满分), 会被误读成完美预测。"""
    y = np.full((3, 5), np.nan)
    q = np.zeros((3, 5, 9))
    assert np.isnan(evaluate.crps_from_quantiles(y, q, np.arange(0.1, 0.95, 0.1)))


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
