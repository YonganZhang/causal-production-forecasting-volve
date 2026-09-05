# Causal Production Forecasting on the Volve Field

基于因果神经网络的单油田产能预测研究（Volve, 挪威北海, Equinor 开放数据）。

## 一句话

产能预测现有工作几乎全是相关性时序模型，遇到关井/换油嘴/注水调整等干预就失效。本课题不追求"预测更准"（实测该路径在 Volve 上已饱和），而是回答**"如果这样调作业，产量会怎么变"**，并用 Volve 独有的三件套把答案验证到位：真实干预记录 + F-5 注水启停天然实验（2008-08-26 启注 / 2016-04 停注转产）+ 官方 Eclipse 油藏模型反事实模拟。

## 数据

- 加工版数据集（analysis-ready，2.6MB）：`_data/volve_causal_v0.2/`，字段字典与已知坑见其 [README](_data/volve_causal_v0.2/README.md)
- 直接下载：https://share.yongan.site/causal-production-volve/volve_causal_v0.2.zip
- v0.1 已归档至 `_legacy/2026-08-06-dataset-v0.1/`（`role` 列把 F-5 的方向标反了，见 v0.2 README）
- 原始源头：Equinor Volve Data Village（Equinor Open Data Licence，署名可复用；全量 4.57TB，本仓库只含生产数据加工版）
- 加工脚本：`_code/build_causal_dataset.py`（仅格式转换，不改数值）

## 研究计划

见 [`_wiki-methodology/_top/_task_plan.md`](_wiki-methodology/_top/_task_plan.md)：三层路线（因果发现+预测 → 干预效应估计 → 模拟器反事实验证）。

## 当前结果

⚠️ 首版结论（"干预条件化让 MAE 降 42%"）**已于 2026-08-06 当日撤稿**——那个提升来自
`日产油体积 ≈ 速率 × 开井小时` 这条记账恒等式，一条三行公式的基线（MAE 81.67）反而
打赢了整套模型（85.24）。详见 [`_findings/E1-E3_intervention_conditioning.md`](_wiki-methodology/_top/_findings/E1-E3_intervention_conditioning.md)。

修正后的结论（rate 口径，已除掉记账项）：

- Chronos-2 纯历史比平凡基线好 **10.5%**——真实但不惊人。
- 知道未来油嘴开度再多 **2.7%**（显著但小），且只在油嘴变动最剧烈的 1/4 窗口里有用。
- **未来注水的效应在 30 天日尺度上完全测不出来**（CI 含 0）。这符合物理：注水响应是月到年的尺度。
- 「干预条件化带来鲁棒性」的主张，在当前证据下**不成立**。

方法论产出（本轮真正留下的东西）：**volume / rate 双口径 + 恒等式强制及格线**。
`run_experiment.py` 会自动判定并给每个模型打 `verdict`，用了未来开井小时却打不过恒等式的直接标 `FAIL`。

复现：

```bash
python _tests/test_harness.py                                          # 13 项正确性测试
CUDA_VISIBLE_DEVICES=5 python _code/e0_replicate.py                    # 复现门(±1%)
CUDA_VISIBLE_DEVICES=5 python _code/run_experiment.py --target volume  # 含恒等式及格线
CUDA_VISIBLE_DEVICES=5 python _code/run_experiment.py --target rate    # 物理口径
```

代码与数据登记见 [`_meta/_registry.yml`](_meta/_registry.yml) 与 [`_meta/_data_registry.yml`](_meta/_data_registry.yml)。

## License

- 数据：Equinor Open Data Licence（`_data/volve_causal_v0.2/license.txt`）
- 代码：MIT
