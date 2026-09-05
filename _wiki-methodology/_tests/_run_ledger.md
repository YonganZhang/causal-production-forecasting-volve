# 运行记录

只记**可核实**的运行结果，不记叙事。

## 2026-08-06

- `gate:harness` — 13/13 通过。
- `gate:e0-replication` — 历史均值 MAE `184.6686`（目标 184.6686，偏差 0.000%）；Chronos-2 MAE `172.3162`（目标 172.3162，偏差 0.000%）。两门均 PASS。
- `run_experiment --target volume --split rolling` — 8,739 窗口，5 折 rolling-origin，评估 6,931 窗口。恒等式及格线 `identity_rate_x_hours` MAE **81.6690**；干预组 5 个模型全部 FAIL（最好的 chronos2_futr_all 85.2436）。纯历史组最优 chronos2_past_all 147.3002。
- `run_experiment --target rate --split rolling` — 恒等式及格线（此口径下退化为持平历史均速率）4.4366；chronos2_past_all **3.9696**（−10.5%）；chronos2_futr_choke 3.8418。配对 bootstrap：未来油嘴 Δ=−0.133 [−0.193,−0.072] 显著；未来注水 Δ=−0.023 [−0.055,+0.011] **不显著**；三者全给 Δ=−0.071 [−0.140,+0.002] **不显著**。
- `run_experiment --split well`（留一井第二口径）— 定性结论与主口径一致，绝对 MAE 高得多（F-12/F-14 早期高产年份进验证集）。
- `gate:lit-fetch-smoke` — 通过，Scopus HTTP 200。

**撤稿**：同日首版 E2/E3 结论（"干预条件化 MAE 降 42%、鲁棒性大幅提升"）已作废，详见 `_top/_findings/E1-E3_intervention_conditioning.md`。
