---
title: Gaia 注水决策智能体团队 —— 代理当模拟器替身、多角色提案、模拟器唯一裁定的闭环框架
user_workflow: 给定注入成本与折现率，运行团队环路产出注水方案；代理毫秒级筛选，OPM Flow 裁定真值，红队按硬约束否决，结果回流下一轮
owner_refs:
- code:fc_team
- code:fc_team_loop
- code:fc_team_proxy
- code:fc_water_econ
- test:team-framework
tracked_paths_hint:
- _code/fc_team.py
- _code/fc_team_loop.py
- _code/fc_team_proxy.py
- _code/fc_team_ablate.py
last_verified_hash: 1d4a54c15629cb55188e4b0f4ec31c8a53f49bac8b1346e8581480a9f1170771
validator_version: 1
---

# 它是什么

一个把**领域知识、快速代理评估、全物理裁定**组织成闭环的注水决策框架。
架构语义借自本机知识发现项目 `deep_discover` 的
Discovery → Cartographer → Readers → adaptive Auditor → Synthesizer 范式（只借语义，不跨项目 import）。

Norne 油田是**台架**，不是交付对象——历史已经发生，任何"多采 X%"都是反事实。
论文贡献是框架本身与它提炼的可迁移规律。

# 怎么跑

```bash
VENV=/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python
CUDA_VISIBLE_DEVICES=1 $VENV _code/fc_team.py            # 1. 建连通性图谱
CUDA_VISIBLE_DEVICES=1 $VENV _code/fc_team_loop.py \
    --c-inj 2.0 --c-prod 1.0 --w-max-dev 0.25 --tag _x   # 2. 跑闭环
$VENV _code/fc_team_ablate.py                            # 3. 消融（先测噪声底）
```

`--drop <role_id>` 剔除任意角色做消融，不需改骨架。`--drop` 全部角色即"裸 LLM"对照臂。

# 四个部件各管什么

| 部件 | owner | 职责 | 关键约束 |
|---|---|---|---|
| 消息协议 | `_code/fc_team.py` | 每条断言带 `agent_id/confidence/provenance(含 64 位 SHA)` | 无据可引的角色**发不出合法消息** |
| Cartographer | `_code/fc_team.py:cartography` | 从已有算例偏回归反推注水井→生产井影响矩阵 | 条件数 ≥30 自动挂 `high` flag |
| 代理模型 | `_code/fc_team_proxy.py` | 模拟器的快速替身，批量评估候选 | **只用于排序**，绝对值以模拟器为准 |
| 主环路 | `_code/fc_team_loop.py` | Readers→总工→代理→红队→模拟器→回流 | 终止＝连续两轮零 high/medium |

# 角色视角:为什么不派方向性立场

角色之间要有真实分歧，但分歧必须来自**不同的数据**，不能是编排者派的偏见。

演进史（三版，每版都由实测推翻前一版）：
1. **无立场** → 七个角色建议全部撞车（还叠加了固定提示词里的结论剧透）。
2. **派方向性立场** → 分歧确实出现，但配置歪了：1 个增产派对 6 个减注/设限派，
   团队被系统性拖向保守；而单 Agent 对照臂恰好只保留那个唯一的增产派，等于发 buff。
   证据：full7 三次差结果产油 −0.07% / +0.04% / −2.70%（只省水不产油），
   总工日志原文出现「reservoir_engineer **少数派**压力保全」。
3. **无方向性「专业视角」（当前）** → 每个角色只声明盯什么、对什么负责；
   方向由它从自己的数据读出。岩石力学产出可行区间上限、4D 地震产出风险敞口排序、
   约束核查产出可行性判定 —— 三者都与「该增该减」无关。

# 臂间差异清单（做对比前必须列全）

比较任意两臂时，除名义自变量外的系统性差异必须先列清单再下结论。已识别四项：

| 差异 | 处理 |
|---|---|
| 立场倾向分布 | 已消除（取消方向性立场） |
| 执行时间顺序 | 已消除（交错随机化，固定种子可复现） |
| LLM 调用预算（实测 48 / 9 / 6 次每运行） | **不等化**，已入库 `llm_calls` 字段作为声明的限制 |
| 质疑轮仅在多角色臂触发 | **不等化**，属「有团队」这一处理的组成部分 → 论文中必须表述为**复合处理** |

🔴 后两项不得在论文中被表述为「仅角色数不同」。

# 🔴 数据泄露红线（2026-09-04 事故，已由 gate 锁住）

**给智能体的资料必须区分「解题先验」与「解题结果」。**
判据只有一条:**这条信息是否只有在解完题之后才写得出来?**

    ❌ 结果:"最好的方案是 +0.62 +0.62 +0.62 +0.62 +0.03 -0.63"
             "顶级方案统计:前 4 段 +0.469、F-4H 锁 +0.40"
    ✅ 先验:"含水率上升前注入的水驱油效率最高"
             "受最低井底流压约束的注水井，提高目标速率不改变实际注入量"

事故经过:为让智能体找到高收益策略，我把方案库里收益最高的 8 个 θ 逐格打印
并附顶级配方，写进角色切片与总工提示词，且**发给了全部三个臂**。
后果:三臂分数全挤在 +480~+488(相差<2%)，因为都在抄同一份答案;
`_W`/`_X`/`_Y` 三批 100+ 次运行的对比结论全部作废。
量化证据:与实例表的最小 L2 距离 —— 无 Agent 0.81(最贴)、单 Agent 0.91、
Agent Team 1.34。**"无 Agent 最好"正是因为它抄得最像。**

滑坡路径:找不到好方案 → 总结规律 → 算成线性表(与收益相关 -0.29，越听越亏)
→ **干脆直接贴最优方案** → 分数上去了，判定为"教得好"。
**每一步单看都像在修 bug，合起来就是作弊。**

守护:`_tests/test_team.py::test_no_solution_leakage_in_role_slices` 与
`::test_chief_prompt_has_no_solution_leakage`，出现禁用字样即红。

# 🔴 五类结构性缺陷与防线（2026-09-05 Workflow 五维审计，45 条归并）

审计的 45 条发现归并后是 5 类，每类配一条 `_tests/test_truth.py` 的结构性防线。
**逐条打补丁修不完 —— 必须按类修。**

| 类 | 症状举例 | 防线 |
|---|---|---|
| ① 同一个量有多套算法 | 实测注水量在 6 个文件用 trapezoid，与 FWIT 差最大 3.12pp、**14 例可行性结论相反**；提示词硬编码基准产油 7,469,000（真值 7,493,840） | `fc_truth.py` 唯一真源；`test_no_second_algorithm_for_measured_water` |
| ② 标识符用位置而非内容 | 模拟缓存键不含 θ，重跑静默复用别的方案的结果；提示词「候选#N」跨轮复用制造伪造证据 | 缓存键并入 θ 指纹；`test_simulation_cache_key_is_content_addressed` |
| ③ 防线是黑名单而非白名单 | tilt 分箱表命中黑名单 **0 条**，却是第三处泄露（最优两箱 234 个算例里 228 个来自智能体自解） | 改为来源白名单；`test_role_slices_read_no_forbidden_source` |
| ④ 状态更新与控制流顺序错 | incumbent 更新写在 break 之后，终止轮成绩被静默丢弃 | `test_state_updated_before_early_termination` 断言行号顺序 |
| ⑤ 注释断言与代码不符 | 我"修复" ④ 时**只改了注释没验证顺序** —— 注释写"已在上文更新"而代码在下文 | 关键注释配可执行测试 |

另修:`_run_sim` 抛 `SystemExit`（继承 `BaseException`），`except Exception` 接不住，
模拟一失败整次运行崩溃且不落盘。

# 对等性设计（用户 2026-09-05 拍板:对单 Agent 严格一点）

两处曾对单 Agent 有利，已拆:

1. **共享背景厚此薄彼**:tilt 先验 + 领域规则此前只发给 7 个角色中的 4 个，
   而没拿到的三个恰好是 PERSPECTIVE 限定「只能给上限/风险/否决」的角色 ——
   既不给数据又限定只能唱反调，纯稀释;更关键的是单 Agent 用的
   `economics_analyst` 恰在「拿到」组。现提升为**全体共享背景**（7 角色完全一致）。
2. **单 Agent 的角色选择**:目标函数是 NPV，而七个角色里只有 `economics_analyst`
   直接读到各年油价与水成本，其余六个只看物理量。用它当单 Agent 等于挑了最强的。
   改用 `reservoir_engineer`（核心领域角色、只读物理量、非文献边缘角色）——
   不选 `constraint_auditor`(293 字) 或 `geomechanics_expert`(156 字纯文献)，
   是为了「靠近严格但不刻意削弱」。

# 对照臂设计（三组只差「有无 Agent、有几个」）

| 臂 | 内容 |
|---|---|
| 无 Agent | **不含 LLM** 的平凡基线，见 `_code/fc_baselines.py`:B2 一维均匀缩放 / B4 等预算随机搜索 / BG 贪心短期最优 |
| 单 Agent | 1 个领域角色 + 真实油藏数据 + 文献规则 |
| Agent Team | 7 个角色各读一块数据 + 文献规则 + 互相质疑 |

🔴 领域知识**归角色所有**:删掉角色就同时失去知识。
   知识经角色的 `evidence` 字段**逐字转呈**给总工，不做有损概括;
   无 Agent 臂该字段为空，总工提示词明写"这是有意的对照设置，不是遗漏"。

# 三个必须知道的坑（都已由 gate 锁住）

1. **经济排序不能用纯产油。** 第一版按产油排序，团队一路加水 +35%→+40%→+58%，
   在高水价下全部方案反而不如维持基准。那是目标函数缺水成本项，不是智能体的错。
2. **红队必须能否决，不能只报警。** `hard_check` 是**程序化**判定（注水预算、θ 越界、能力饱和），
   不交给 LLM——LLM 会被说服，数字不会。
3. **常量从真源 import。** `fc_water_econ` 曾自建 `T_START=2006-01-01`，
   而 Norne 1997 投产，油价整体错位，基准 NPV8 算成 1798M（真值 2857M）。

# 已知的方法学结论

- 代理在闭环路径上误差稳定 ±1.3%，GPU 批量 18.1 µs/方案（约 3.3e6 倍于模拟）。
- **消融必须先测重复性**：完整臂重复 5 次，ΔNPV@8% 在 +158~+249 M$ 间飘，
  合并 sd 51.3 M$。n=1 的消融表会给出完全错误的结论（曾把 +102.6M 的"显著贡献"
  在 n=6 时塌成 +15.6M）。
- 四口注水井**全部饱和**：目标率横跨 227~98,510（430 倍），实测最大约 20,000。
  这是"名义控制自由度 ≠ 有效控制自由度"的量化形式。

# 当前缺口

固定提示词中曾写入已知结论，把全部角色锚成同一意见（诊断见 `_pipelines/fc_team/diag_round.json`：
6 个候选挤在 θ∈[-0.20,+0.20]，硬约束允许 ±1.0）。角色之间亦不可见彼此输出。
修复与重测是 plan 的 P4.9 / P4.10。

## 读数接口(2026-09-06)

**结果只有一个入口:`_code/gaia.py`。**

    python _code/gaia.py            # 正式指标表
    python _code/gaia.py --json     # 机器可读

它不实现算法,只把已固化的四个环节接到一个稳定表面上:

| 归属 | 实现在 | gaia 暴露 |
|---|---|---|
| 实测口径 | `fc_truth` | `measured_winj / measured_oil / measured_wprod` |
| 经济 | `fc_water_econ.econ` | `econ / baseline_npv` |
| 智能体臂 | `_pipelines/fc_team/loop_F2*` | `arm() / ARMS` |
| 无 Agent | `fc_baselines` | `nonllm()` |
| 汇总 | — | `summary() / report()` |

### 为什么要有这一层

`fc_truth` 解决的是「同一个量两套算法」——实测水量曾同时存在梯形积分和 FWIT
两套口径,**14 个算例给出相反的可行性判定**。ΔNPV、基准、正式批次是同一类风险:
每写一个临时分析脚本就重抄一遍口径,迟早抄歪。`gaia.py` 把这三个量收成常量与函数,
`_tests/test_gaia.py` 守住它们:

- `test_baseline_is_not_hardcoded` — 基准必须现算,写死会在换水价后静默失效
- `test_delta_matches_raw_json` — 接口的 ΔNPV 必须能由原始 json 独立复算
- `test_vetoed_solutions_excluded` — 被模拟器否决的方案不得计入成绩
- `test_only_canonical_batch_is_active` — 正式批次唯一,历史批次必须已归档
- `test_all_nonllm_baselines_reported` — **强基线不得被藏起来**
- `test_arms_are_balanced` — 臂间样本量必须相等

### 正式批次

`gaia.BATCH = "F2"`(证据强度加权后)。A2(对等化前)、E2(加权前)及更早的
B2/D2/S/T/U/V/W/X/Y 迭代已归档到 `_legacy/2026-09-06-batches-A2-E2-superseded/`,
其中 **W/X/Y 因数据泄露作废**。归档不改变任何结论:E2 与 F2 方向一致
(有 Agent ≫ 无 Agent;Team 与单 Agent 未检出差异)。

### 主对照用哪条基线

主表用 **B2 一维缩放**(唯一决策变量,最朴素)。但报告**必须同时列出**
B4 等预算随机与 BG 贪心短期:BG 是三条里最强的(+293.5M),
只报 B2(+62.6M)会让智能体优势虚高 4 倍以上。这条由测试强制。
