# 归档:B4 等预算随机 / BG 贪心短期 —— 退出主线对比

日期 2026-09-06。**用户裁定:唯一的无 Agent 对照是一维缩放(B2)。**

## 退役了什么

| 臂 | 做法 | 成绩 | 用掉的真模拟次数 |
|---|---|---|---|
| B4 等预算随机 | 总水量不变,随机分配 | +247.0M | 3 |
| BG 贪心短期 | 逐时段选当下产油最优,不看长期 | +293.5M | 31 |

参照:一维缩放 +62.6M(7 次);Agent Team +361.3M(每次实验 2.7 次)。

## 我提出过的反对意见与结论

我曾主张保留 BG 一行,理由是"智能体用 1/11 的模拟预算赢了贪心 67.7M
(p=2.2e-03)"是最能说明方法价值的对照。用户两次明确要求只留一维缩放,
**这是用户的决定,已执行**。

风险仍然记录在此:若审稿人问「与贪心/进化类搜索相比如何」,
本目录的 `baselines_v2_full.json` 与 `sim/` 可即时恢复应答,不需重跑。

## 恢复

    cp _legacy/2026-09-06-baselines-B4-BG-retired/baselines_v2_full.json \
       _pipelines/fc_baselines/baselines_v2.json
    mv _legacy/2026-09-06-baselines-B4-BG-retired/sim/*.npz _pipelines/fc_decide/sim/

生成代码未删除:`_code/fc_baselines.py` 的 `b4_random` / `bg_greedy` 仍在,
但 `--method` 默认只跑 B2。
