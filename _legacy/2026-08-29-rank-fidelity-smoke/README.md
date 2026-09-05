# 冒烟测试算例（无科学价值）

来源：`_code/fc_rank_fidelity.py --tag SMOKE`，2026-08-29 联调时跑的 12 次 OPM Flow。
θ 来自一个只训练 **3 个 epoch** 的代理模型（场级误差 2.4%），只为验证
「训练 → 多起点 → 注水校正 → 真跑 → 排名统计」整条链路能跑通。

**不要引用其中任何数字。** 正式结果在 `_pipelines/fc_rank_fidelity/rank.json`。
恢复方式：mv 回 `_pipelines/fc_decide/sim/`（但没有理由这么做）。
