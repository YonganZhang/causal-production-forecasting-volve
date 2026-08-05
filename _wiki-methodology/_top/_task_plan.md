# 自己-产能预测 · 基于因果神经网络的产能预测

创建：2026-08-06。真源：本文件。数据真源：`_data/volve_causal_v0.1/`。

## 当前目标

用因果机器学习/因果神经网络做单油田（Volve）产能预测，把"观测学习 + 模拟器反事实验证 + 天然实验识别"做成课题主打牌。**只做 Volve 单油田，不引入外部油田**（用户 2026-08-06 拍板）。

## 数据来源（源头交代清楚）

- **原始源头**：Equinor 官方开放数据「Volve Data Village」，挪威北海 Volve 油田 2008–2016 全生命周期数据，2016 退役后依 Equinor Open Data Licence（类 CC BY，署名可复用可再分发，禁止转售）公开。官方现行下载渠道为 Databricks Marketplace；全量 14 个 zip 共 4.57TB 已存本机 `师弟-军伟的比赛-2693e5/_sandbox/volve_data/`（2026-07-13 与远端字节级比对无遗漏）。
- **本课题加工版**：`_data/volve_causal_v0.1/`（日度生产 CSV + 月度 CSV + 井元数据 + 原始 xlsx + license），加工脚本 `_code/build_causal_dataset.py`，只做格式转换不改数值。
- **公开下载**：https://share.yongan.site/causal-production-volve/volve_causal_v0.1.zip （2.6MB，已验证 HTTP 200）
- 后续可按需从军伟项目补充：测井 LAS（3 口井 LFP）、层位断层解释、井身结构、Eclipse/RMS 油藏模型（反事实验证用）。

## 数据事实（已逐行核实，两个口径别再混）

- 全油田历史钻井 **24 口**（测井/井身全覆盖）→ 用作静态地质协变量。
- 有产量记录的井眼 **7 个**：5 口生产井（F-1C/F-11/F-12/F-14/F-15D，其中 F-12、F-14 有约 3,000 天连续序列）+ F-4 纯注水 + **F-5 于 2008-08-26 生产转注水**。
- 日度 15,634 井·天 × 24 列：产量（油气水）、油嘴开度、注水量、井底/井口压温等。
- ⚠️ F-1C 有 WI 标志但注水量恒为 0，从未实际注水；**唯一干净的转注天然实验是 F-5**。

## 因果框架设定

| 因果角色 | 变量 |
|---|---|
| 结果 | 日产油/气/水（BORE_OIL/GAS/WAT_VOL） |
| 干预 | 油嘴开度（AVG_CHOKE_SIZE_P）、注水量（BORE_WI_VOL，井间外溢干预） |
| 混杂/状态 | 井底压温、井口压温、油管压差、开井小时数 |
| 结构先验 | 井位/完井/层位连通性（军伟项目解释数据）、24 口井测井 |
| 天然实验 | F-5 转注事件（2008-08-26）前后对邻井 F-12/F-14 产量的因果效应 |

## 方法路线（三层，逐层出成果）

1. **因果发现 + 预测**：在观测数据上学井内/井间因果图（如 neural granger / DYNOTEARS / amortized causal discovery 类），把因果结构注入神经预测器，与非因果基线（军伟项目已有 Chronos-2 基线：MAE 172.3 vs 均值基线 184.7）对比——主张"因果结构提升预测鲁棒性/可迁移性"。
2. **干预效应估计**：油嘴/注水对产量的处理效应（时变混杂下的 g-methods / 反事实回归网络），用 F-5 转注天然实验做半"金标准"校验。
3. **模拟器反事实验证**：用官方 Eclipse 模型（开源 OPM Flow 跑）生成干预场景的反事实产量，验证第 2 层估计——这是多数因果论文没有的验证条件，也是本课题最大卖点。
   - 🔴 前提待验证：OPM Flow 能否跑通 Volve Eclipse deck（半天实验，未做）。

## 为什么有价值

- 产能预测现有工作几乎全是纯相关性时序模型，关井/换油嘴/转注这类干预一来就失效；因果模型主张对干预分布漂移稳健，而 Volve 恰好有真实干预记录 + 转注天然实验 + 官方模拟器三件套，可以把这个主张**验证**而不只是宣称。
- 单油田深做的诚实定位：case study + benchmark 发布（数据集已公开），不做跨油田泛化主张。

## 已完成

1. 数据集 v0.1 加工、打包、公网发布（见上链接）。
2. 井角色/转注事件逐行核实（含 F-1C 假转注坑）。
3. GitHub 仓库与研究窗口见下方登记。

## 下一步

1. 验证 OPM Flow + Volve Eclipse deck（课题成立性前提）。
2. 文献调研：因果时序预测 + 油藏领域已有因果 ML 工作（可用 deep-discover / journal_info.sh 带分区）。
3. 定第一篇论文主线（走 share-sci-write 写主线流程）。

## 登记

- 数据：`_data/volve_causal_v0.1/`（本文件即登记处；schema 见其 README）
- 代码：`_code/build_causal_dataset.py`
- 公网包：https://share.yongan.site/causal-production-volve/volve_causal_v0.1.zip
- GitHub：https://github.com/YonganZhang/causal-production-forecasting-volve （private，公开与否等用户拍板）
