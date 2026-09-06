# 待拍板草稿:Title / Abstract / Introduction

🔴 **状态 `proposed`。** 按 `research-objective-contract.md`:
「没有用户确认的一句话主线,不得写标题、摘要、Introduction 贡献」。
主线三候选见 `_wiki-methodology/_wiki/paper-storyline/v3-proposed.md`。

下面按**候选 A** 起草,给用户看实物而非抽象选项。拍板后移入
`_paper/01_introduction.md` 与 `_paper/02_abstract.md`,并把 v3 转为
`current-user-approved`。

---

## Title(按候选 A,三个具体措辞)

**T1** — Surrogate-accelerated, simulator-adjudicated closed-loop optimisation of water-injection
scheduling: a Norne field case study

**T2** — When a surrogate may screen but not decide: a physics-adjudicated agent framework for
waterflood scheduling

**T3** — Keeping decision authority with the physics: surrogate-accelerated agent proposal and
full-physics adjudication for water-injection scheduling

取舍:T1 最保守、最像 Petroleum Science 常见标题,但把 §4.1 埋掉了;
T2 辨识度最高,风险是被读成代理可靠性论文;
T3 折中,"decision authority" 是全文真正的落点。**我倾向 T3,其次 T1。**

---

## Abstract(≈250 词,按咨询建议的权重:先 gap,再框架,再经济,最后消融)

Water-injection scheduling over multiple periods and multiple wells requires many full-physics
simulations, and learned surrogates are the standard means of affording them. Whether a surrogate
validated for prediction can be trusted to *choose* among optimiser-generated candidates has
received less attention. Using the Norne field model, we show that it cannot be assumed: an
operator surrogate reproducing the simulator's ranking of 500 independently sampled schedules at
Spearman ρ = 0.998 with a top-1 hit rate of 1.0 and a field-level error of 0.369 % ranked the ten
schedules produced by optimising against itself at ρ = −0.21 with a top-1 hit rate of 0, selecting
the candidate whose objective it overstated most (by 10.6× against a shortlist median of 1.66×);
two controls place the cause in the optimiser rather than in top-*k* selection. We therefore
organise surrogate screening, multi-role language-agent proposal, an arithmetic feasibility gate
and full-physics adjudication into a closed loop in which the simulator alone assigns value. Over
ten repetitions per arm, the loop produced simulator-verified improvements of 361.3 and
347.4 M US$ in NPV at 8 % over the historical schedule, against 62.6 M US$ for one-dimensional
uniform scaling, using 2.7 and 2.6 full-physics evaluations per run against 7 for the reference
sweep; all twenty runs finished at or above their starting point. Decomposing the reasoning into
seven specialised roles gave no detectable gain over a single role (+13.8 M US$, p = 0.63),
locating the improvement in the closed loop rather than in the number of roles. Accepted schedules
concentrate on one interpretable direction, earlier rather than later injection (ρ = +0.714).

**自查**:Abstract 最强 claim 不超过 Results/Conclusion ✅;数字与正文逐位一致 ✅;
无 equivalence / robustness / causal / expert-level / monotonic ✅;
无"优于传统优化方法" ✅。

---

## Introduction(按咨询建议 (a) 为主轴、吸收 (b);四层 gap)

### 第 1 层 — 工程问题:全物理优化很贵

开场落在**注水决策**,不落在 LLM。逐井、逐阶段的注水调度是油藏管理的常规
决策,而每次评估需要一次全物理模拟(本文模型约 1 分钟)。决策空间随井数与
阶段数相乘增长,任何需要大量试错的方法在模拟器自身的开销下不可行。
→ 引文:注水优化、井控优化、闭环油藏管理。

### 第 2 层 — 通行解法与其**隐含跳跃**:代理加速

代理模型是降低这一开销的标准手段,通常用独立采样的测试集验证,报告
R²、相对误差或排名相关。**这里存在一个未被检验的跳跃**:
从「代理预测得准」推到「用代理做优化是可信的」。
→ 引文:油藏代理模型、算子学习、代理辅助优化。

### 第 3 层 — 真正的 gap(本文最原创的一层)

一个为**预测**验证过的代理,未必能在**优化器实际产生的候选**中做出可靠选择。
优化器不会随机采样代理的误差分布,它主动搜索代理乐观的区域,于是
「按代理选」退化为「按代理的误差选」。据我们所知,这一边界在油藏代理优化
文献中很少被直接量化——通常做的是预测验证,而不是决策验证。

### 第 4 层 — 本文做什么

若代理不能做最终判断,那么谁做?我们把权限拆开:
**代理提供规模,智能体提供有结构的候选,全物理模拟器保留唯一的赋值权。**
智能体消化异质工程证据(连通性、压力/含水史、生产监视、经济、地力学类比、
4D 地震、实测注入)并提出候选;算术闸门滤掉不可执行者;模拟器裁定;
实测结果回流。

**贡献(三条)**
1. 在同一个代理上量化筛选与决策两种可靠性的分离(ρ 0.998 → −0.21),
   并用两个对照把病因定位在优化器而非 top-k 选择;
2. 据此构建物理裁定的闭环决策框架,在 Norne 上给出模拟器实测的经济结果
   (+361.3 M US$,每次实验 2.7 次全物理评估);
3. 以角色消融定位收益来源:七角色相对单角色未检出均值差异,
   收益来自闭环而非角色数量。

### 第 5 层 — 边界(放 Introduction 末尾一句,不放大)

结论限于本 Norne 模型与本经济设定;对照为一维缩放这一**刻意受限**的参考策略,
不构成对传统优化方法的比较。

---

## 待办

- [ ] 用户拍板主线候选(A / B / C)
- [ ] 拍板后填 `_current.yml`,v3 转 `current-user-approved`
- [ ] Introduction 补引文(走 `tools/academic-search`)
- [ ] Fig. 1 框架示意图(计划见 `_figure_plan.md`)
- [ ] Table 1 / Table 2
