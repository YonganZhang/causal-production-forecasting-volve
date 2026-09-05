---
title: Volve 产能实验台 — 从原始 xlsx 到可对拍的模型评估
user_workflow: 想验证"某个方法/某条信息能不能提升产能预测"时，用它构造窗口、切分、跑模型、和平凡基线对拍，并自动判定是否只是记账
owner_refs:
- code:volve_data
- code:windows
- code:splits
- code:baselines
- code:models_chronos2
- code:evaluate
- code:run_experiment
- code:e0_replicate
- test:harness
- test:e0-replication
tracked_paths_hint:
- _code/volve_data.py
- _code/windows.py
- _code/splits.py
- _code/baselines.py
- _code/models_chronos2.py
- _code/evaluate.py
- _code/run_experiment.py
- _code/e0_replicate.py
last_verified_hash: 7b4b257437c4646182d75ac67b8962e2f28ef1106f150d3eb5f5501563775c79
validator_version: 1
---

## 它解决什么问题

给一个关于 Volve 产能的假设（"知道未来油嘴能不能预测得更准"），回答两件事：**数字是多少**、**这个数字算不算证据**。

第二件事才是本实验台存在的理由。2026-08-06 本项目报出过"干预条件化让 MAE 降 42%"，随后发现那个提升来自记账恒等式 `日产油体积 ≈ 每小时产油率 × 开井小时数`，被一条三行公式的基线击败。实验台此后内建了防这类错误的机制。

## 怎么用

```bash
VENV=/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python

$VENV _tests/test_harness.py                                  # 先跑 gate:harness
CUDA_VISIBLE_DEVICES=5 $VENV _code/e0_replicate.py            # 信任锚点 gate:e0-replication
CUDA_VISIBLE_DEVICES=5 $VENV _code/run_experiment.py --target volume   # 含恒等式及格线
CUDA_VISIBLE_DEVICES=5 $VENV _code/run_experiment.py --target rate     # 物理口径
```

结果落 `_pipelines/e1_e3_{volume,rate}_{rolling,well}/results.json`。

## 两个口径（用错口径 = 结论无效）

| `--target` | 目标列 | 什么时候用 |
|---|---|---|
| `volume` | `BORE_OIL_VOL` 日产油体积 | 关心"实际产多少油"。**含排产记账项**，恒等式及格线在此生效 |
| `rate` | `OIL_RATE` = 产油 / 开井小时 | 关心油藏物理。**因果类主张必须在此口径成立才算数** |

关井日的速率是 NaN（无定义），不是 0 —— 用 0 会把"没在产"当成"产能为零"。

## 恒等式及格线（本实验台的核心断言）

`baselines.identity_rate_x_hours` 是**强制对照**，不许从 REGISTRY 摘掉（`_tests/test_harness.py` 有回归测试守着）。

`run_experiment.py` 自动给每个模型打 `verdict` 列：用了未来 `ON_STREAM_HRS` 却打不过这条公式的，直接标 `FAIL(输给恒等式)`。当前 volume 口径下干预组 5 个模型全部 FAIL。

## 模块职责

| 文件 | 干嘛 |
|---|---|
| `volve_data.py` | 装载日度 CSV、逐井日历化（**缺日为 NaN，关井日为真 0**，两者不许混）、构造注水干预源（F-5 转产后置 0） |
| `windows.py` | 30+30 滑窗（默认参数即军伟 T3 契约 30/30/7/21）+ 干预分层标签（关井跳变 / 油嘴强度四分位 / 注水强度四分位） |
| `splits.py` | rolling-origin 时间序 CV（主口径，带 60 天 embargo 和**日历重叠断言**）+ 留一井 CV（第二口径） |
| `baselines.py` | 平凡基线阶梯 + 恒等式及格线 + LightGBM（锚定增量，避免树模型无法外推） |
| `models_chronos2.py` | Chronos-2 封装。权重锁 revision `29ec3766`，不自动下载。支持零样本 / past covariates / future covariates |
| `evaluate.py` | MAE/RMSE/MASE/CRPS/覆盖率 + 分层聚合 + **配对** bootstrap（两模型同窗误差高度相关，独立 bootstrap 会高估不确定度） |
| `run_experiment.py` | E1/E2/E3 主入口 |
| `e0_replicate.py` | 复现军伟 T3 baseline 作信任锚点 |

## 三条硬规则

1. **缺日 ≠ 关井**。缺日无记录 → NaN；关井有记录 → `ON_STREAM_HRS=0` 是真实的 0。零填补会系统性污染 MAE。
2. **信息差**。带 `future_*` 的模型用了预测窗内的真实作业值，与纯历史模型不是同一个预测问题（interventional vs pure forecasting）。只能组内比较。
3. **恒等式及格线**。见上。打不过 = 没有证据。

## 相关

- 结论与撤稿记录：[`_wiki-methodology/_top/_findings/E1-E3_intervention_conditioning.md`](../_wiki-methodology/_top/_findings/E1-E3_intervention_conditioning.md)
- 数据清单与陷阱：[`_refs/OUR_DATA_INVENTORY.md`](../_refs/OUR_DATA_INVENTORY.md)
