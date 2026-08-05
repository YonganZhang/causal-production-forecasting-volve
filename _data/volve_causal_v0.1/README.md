# Volve Causal Production Forecasting Dataset v0.1

Analysis-ready 加工版 Volve 油田生产数据，面向**因果机器学习产能预测**研究。

## 来源与许可

- 原始数据：Equinor 官方开放数据「Volve Data Village」（挪威北海 Volve 油田，2008–2016 生产，2016 退役后公开）。官方现行渠道为 Databricks Marketplace（旧 data.equinor.com 门户已下线）。
- 许可：**Equinor Open Data Licence**（见 `license.txt`，类 CC BY：允许署名复用与再分发，禁止转售）。使用本数据集须署名 Equinor。
- 本包为衍生加工版：仅做格式转换（xlsx → CSV）与井元数据统计，未修改任何数值。原始 xlsx 原样附带。

## 文件

| 文件 | 内容 |
|---|---|
| `Volve production data.xlsx` | 官方原始文件（日度 + 月度两个 sheet） |
| `daily_production.csv` | 日度生产表，15,634 行 × 24 列，2007-09 ~ 2016-12 |
| `monthly_production.csv` | 月度汇总表，527 行 |
| `well_metadata.csv` | 7 个井眼的角色、记录区间、产油/注水天数、首次注水日期 |
| `license.txt` | Equinor Open Data Licence 原文 |

## 因果角色字段字典（daily_production.csv 关键列）

| 列 | 因果角色 | 说明 |
|---|---|---|
| `BORE_OIL_VOL` / `BORE_GAS_VOL` / `BORE_WAT_VOL` | **结果变量** | 日产油/气/水（Sm³） |
| `AVG_CHOKE_SIZE_P` | **干预变量** | 油嘴开度（人为可操控） |
| `BORE_WI_VOL` | **干预变量** | 日注水量（注水井对邻近生产井的外部干预） |
| `AVG_DOWNHOLE_PRESSURE` / `AVG_DOWNHOLE_TEMPERATURE` | 混杂/状态 | 井底压力/温度 |
| `AVG_WHP_P` / `AVG_WHT_P` / `AVG_DP_TUBING` / `AVG_ANNULUS_PRESS` | 混杂/状态 | 井口压温、油管压差、环空压力 |
| `ON_STREAM_HRS` | 混杂 | 当日开井小时数（0 = 关井） |
| `FLOW_KIND` / `WELL_TYPE` | 井角色 | production/injection；OP/WI 标志 |

## 井眼与天然实验锚点（well_metadata.csv）

- 5 口生产井：F-1C、F-11、F-12、F-14、F-15D（F-12/F-14 有 2008–2016 全程约 3,000 天序列）。
- 2 口实际注水井：F-4（2008-04-23 起）、F-5（**2008-08-26 由生产转注水**——这是数据集内最干净的天然实验：转注前后对邻井产量的因果效应可识别）。
- ⚠️ 诚实备注：F-1C 部分日期 `WELL_TYPE` 标为 WI，但日度/月度注水量均为 0，即**从未实际注水**；不要把 F-1C 当第二个转注实验，真正可用的转注事件只有 F-5。

## 已知坑

- 井名有多套编码（`WELL_BORE_CODE` / `NPD_WELL_BORE_CODE` / `NPD_WELL_BORE_NAME`），跨表关联建议用 NPD 名。
- 缺日与关井日并存：`ON_STREAM_HRS=0` 是"记录了关井"，日期缺失是"没记录"，时序建模不要用零填补混淆两者。
- 压力/温度列在部分早期日期为空。

## 建议引用

Equinor (2018). Volve field data set. Equinor Open Data Licence.
加工脚本：`_code/build_causal_dataset.py`（本研究项目仓库内）。
