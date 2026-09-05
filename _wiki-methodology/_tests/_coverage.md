# COL3 测试覆盖地图

## Gates（`_gates.yml`，每条已实测 exit0 后才登记）

| gate | 覆盖什么 | 需要 GPU | 典型耗时 |
|---|---|---|---|
| `harness` | 13 项实验台正确性 | 否 | ~30s |
| `e0-replication` | 与军伟 T3 契约的可比性锚点（±1%，实测 0.000%） | 是 | ~60s |
| `lit-fetch-smoke` | Elsevier 凭证与网络连通 | 否 | ~10s |

## `harness` 的 13 项覆盖什么

| 测试 | 防什么错 |
|---|---|
| `history_and_future_do_not_overlap` | 窗口构造把未来漏进历史 |
| `missing_days_stay_nan_not_zero` | 把缺日当成"产量 0"，系统性污染 MAE |
| `rate_target_removes_the_bookkeeping_identity` | 关井日速率被填成 0（速率在关井时无定义） |
| `identity_baseline_beats_intervention_conditioned_chronos_on_volume` | **锁 2026-08-06 撤稿教训**：恒等式基线被从 REGISTRY 摘掉 |
| `shutin_label_matches_raw_data` | 干预分层标签与原始 CSV 不符（逐窗手工核对 40 个样本） |
| `quartile_bins_are_balanced` | 分位分箱塌掉导致某层样本过少 |
| `rolling_origin_has_no_calendar_leak` | 时间序切分的日历重叠泄漏 |
| `mae_ignores_nan_instead_of_filling_zero` | 指标实现用零填补 |
| `crps_of_point_forecast_equals_mae` | CRPS 实现的系数错误 |
| `paired_bootstrap_detects_no_difference` | 配对 bootstrap 实现错误 |
| `target_scalar_matches_trajectory_mean` | 标量目标与轨迹目标不一致 |
| `f5_is_injector_then_producer` | F-5 角色方向退回 v0.1 的错误 |
| `f5_injection_covariate_zeroed_after_conversion` | F-5 转产后仍被当注水干预源 |

## 没有覆盖的（已知缺口）

- **模型输出的数值正确性**没有测试守（Chronos-2 是外部权重，只能靠 `e0-replication` 端到端锚定）。
- LightGBM 基线没有回归测试，其结果仅在实验报告中记录。
- 文献工作流（`lit_fetch.py` + Workflow）只有连通性冒烟，无内容质量 gate。

## 运行记录

见 `_run_ledger.md`。
