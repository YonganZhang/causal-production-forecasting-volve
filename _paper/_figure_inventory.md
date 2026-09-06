# Figure 清单(share-results-craft 第 2 段)

按论证顺序排,不按实验时间序。

| # | 图/表 | 一句话 takeaway (leaf claim) | 证据强度 | 数据源 |
|---|---|---|---|---|
| Table 1 | 算例与经济设定 | 交代 Norne 模型、6×4 决策变量、油价与水价口径 | — | `_pipelines/fc_decide/sim/baseline.npz` |
| Fig. 2 | 代理精度 | 代理在**随机 θ 上**几乎无偏:场级误差 0.369%,R²=0.9959,快 3.3×10⁶ 倍 | **strong** | `fc_final/final.json`, `fc_rank_fidelity/rank.json:control_random_theta_full` |
| **Fig. 3** | **代理的失效边界** | **同一个代理,在优化器自己的候选区里排名崩塌**(ρ 0.998→−0.21,top-1 命中 100%→0),且它选中的正是自己高估最狠的候选 → **裁判不可省** | **strong** | `fc_rank_fidelity/rank.json` |
| **Fig. 4** | **三臂经济表现 + 真模拟预算** | Agent 臂 +361.3 / +347.4 M$,一维缩放 +62.6 M$;且 Agent 每次实验只用 2.7 次真模拟,一维缩放用 7 次 | **strong** (p=1.7e-08 / 6.0e-07) | `python _code/gaia.py --json` |
| Fig. 5 | 闭环轨迹 | 20/20 次实验末轮 ≥ 首轮,闭环不发散 | **strong** | 同上 |
| Fig. 6 | tilt–ΔNPV | 智能体优解偏向一个可解释的调度方向(前期加注),ρ=+0.714 | **moderate**(相关,非因果) | `fc_team/tilt_bins.json`(275 随机算例) |
| Table 2 | 多角色消融 | 七角色相对单角色**未检出**均值差异(+13.8 M$,p=0.63,95% CI [−45.2,+72.8]) | **negative result** | 同 Fig. 4 |

## 第 4 问:负面结果怎么处理

**报**,并且定位为「消融定位了真正起作用的机制」:主要收益来自
**代理加速 + 智能体闭环 + 模拟器裁定**这一整套,而非角色数量。
不写"我们的多智能体没用",也不写"多角色更稳健"(方差差异不可检验)。
