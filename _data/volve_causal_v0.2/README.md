# Volve Causal Production Forecasting Dataset v0.2

Analysis-ready 加工版 Volve 油田生产数据，面向**因果机器学习产能预测**研究。

## v0.2 变更（重要更正）

v0.1 把 F-5 标为 `producer_to_injector`（"2008-08-26 生产转注水"）。**方向是反的。**

`well_metadata.csv` 的 `role` 列在 v0.1 只判断"是否同时存在产油日和注水日"，没有比较时间先后。逐行核对原始记录：

- F-5 在 **2008-08-26 之前的 360 天**全部是 `FLOW_KIND=injection` / `WELL_TYPE=WI`，且产油、产气、产水、`ON_STREAM_HRS` **全为 0** —— 是尚未启动的注水井，从未产过油。
- 全油田最早产油日是 2008-02-12（F-12），F-5 在此前后都没有产量记录。
- F-5 **2008-08-26 起实际注水**（2,558 天，至 2016-04-06）。
- F-5 **2016-04-20 转为生产井**（129 天产油，至 2016-08-26）。

正确角色是 **`injector_to_producer`，切换日 2016-04-20**。v0.2 的 role 判定改为比较 `first_oil_date` 与 `first_injection_date` 的先后，并新增 `first_oil_date`、`last_injection_date`、`role_switch_date` 三列供使用者自行核验。

`daily_production.csv` 与 `monthly_production.csv` **与 v0.1 逐字节相同**，本次只修 `well_metadata.csv` 与文档。

### 这对天然实验设计的影响

Volve 可用的干预事件是两个，**都不是"生产转注水"**：

| 事件 | 日期 | 性质 | 优点 | 缺点 |
|---|---|---|---|---|
| **A. F-5 注水启动** | 2008-08-26 | 注水 onset | 后续 7.6 年长窗口 | F-4 已于 2008-04-23 先行注水，前窗口已非"无注水"基线；F-14 首次产油仅早 6 周，几乎无 pre-period |
| **B. F-5 注水停止 + 转生产** | 2016-04（停注 04-06，转产 04-20） | 注水撤除 | 无同期他井干预混杂，识别更干净 | 仅 5 个月，且处于油田末期停产阶段，与退役趋势混淆 |

建议两个都做，互为稳健性检验。

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
| `well_metadata.csv` | 7 个井眼的角色、记录区间、产油/注水起止、角色切换日 |
| `license.txt` | Equinor Open Data Licence 原文 |

## 因果角色字段字典（daily_production.csv 关键列）

| 列 | 因果角色 | 说明 |
|---|---|---|
| `BORE_OIL_VOL` / `BORE_GAS_VOL` / `BORE_WAT_VOL` | **结果变量** | 日产油/气/水（Sm³） |
| `AVG_CHOKE_SIZE_P` | **干预变量** | 油嘴开度（人为可操控） |
| `BORE_WI_VOL` | **干预变量** | 日注水量（注水井对邻近生产井的外部干预） |
| `AVG_DOWNHOLE_PRESSURE` / `AVG_DOWNHOLE_TEMPERATURE` | 混杂/状态 | 井底压力/温度 |
| `AVG_WHP_P` / `AVG_WHT_P` / `AVG_DP_TUBING` / `AVG_ANNULUS_PRESS` | 混杂/状态 | 井口压温、油管压差、环空压力 |
| `ON_STREAM_HRS` | 混杂/干预 | 当日开井小时数（0 = 关井，本身也是作业决策） |
| `FLOW_KIND` / `WELL_TYPE` | 井角色 | production/injection；OP/WI 标志 |

## 井眼时间线（well_metadata.csv）

| 井 | 角色 | 产油区间 | 注水区间 |
|---|---|---|---|
| F-1C | producer | 2014-04-22 ~ 2016-04-06 | — |
| F-11 | producer | 2013-07-24 ~ 2016-09-17 | — |
| F-12 | producer | 2008-02-12 ~ 2016-08-12 | — |
| F-14 | producer | 2008-07-13 ~ 2016-07-13 | — |
| F-15D | producer | 2014-01-16 ~ 2016-07-06 | — |
| F-4 | injector | — | 2008-04-23 ~ 2016-09-19 |
| F-5 | **injector_to_producer** | 2016-04-20 ~ 2016-08-26 | 2008-08-26 ~ 2016-04-06 |

⚠️ F-1C 有 2 天 `WELL_TYPE=WI`，但注水量恒为 0（其中 1 天还在产油）——纯属标签噪声，**不要**当成第二个转注实验。

## 已知坑

- 井名有多套编码（`WELL_BORE_CODE` / `NPD_WELL_BORE_CODE` / `NPD_WELL_BORE_NAME`），跨表关联建议用 NPD 名。
- 缺日与关井日并存：`ON_STREAM_HRS=0` 是"记录了关井"，日期缺失是"没记录"，时序建模不要用零填补混淆两者。F-12/F-14 各有 85 个缺日。
- 压力/温度列在部分早期日期为空；F-14 的 `AVG_CHOKE_SIZE_P` 缺 196 天。
- 关井日占比不低（F-1C 308/746、F-14 332/3056、F-12 218/3056、F-15D 209/978），是干预分析的信号，不是噪声。

## 建议引用

Equinor (2018). Volve field data set. Equinor Open Data Licence.
加工脚本：`_code/build_causal_dataset.py`（本研究项目仓库内）。
