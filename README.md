# Causal Production Forecasting on the Volve Field

基于因果神经网络的单油田产能预测研究（Volve, 挪威北海, Equinor 开放数据）。

## 一句话

产能预测现有工作几乎全是相关性时序模型，遇到关井/换油嘴/转注等干预就失效。本课题用因果机器学习建模干预（油嘴开度、注水）对产量的效应，并用 Volve 独有的三件套做验证：真实干预记录 + F-5 生产转注水天然实验（2008-08-26）+ 官方 Eclipse 油藏模型反事实模拟。

## 数据

- 加工版数据集（analysis-ready，2.6MB）：`_data/volve_causal_v0.1/`，字段字典与已知坑见其 [README](_data/volve_causal_v0.1/README.md)
- 直接下载：https://share.yongan.site/causal-production-volve/volve_causal_v0.1.zip
- 原始源头：Equinor Volve Data Village（Equinor Open Data Licence，署名可复用；全量 4.57TB，本仓库只含生产数据加工版）
- 加工脚本：`_code/build_causal_dataset.py`（仅格式转换，不改数值）

## 研究计划

见 [`_wiki-methodology/_top/_task_plan.md`](_wiki-methodology/_top/_task_plan.md)：三层路线（因果发现+预测 → 干预效应估计 → 模拟器反事实验证）。

## License

- 数据：Equinor Open Data Licence（`_data/volve_causal_v0.1/license.txt`）
- 代码：MIT
