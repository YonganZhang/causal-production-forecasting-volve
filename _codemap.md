# Codemap — 自己-产能预测 · Gaia 注水决策智能体框架

> L0 路由表。寻路靠 CodeBook，核实靠源码与 gate；本文件只负责"该去哪找"，不复制状态。
> 结构事实以 live CodeGraph 为准：`codegraph explore "<符号或问题>"`（索引 2,143 节点 / 4,155 边）。

## 主线一句话

代理模型当油藏模拟器的**快速替身**（18 µs/方案），多角色智能体团队据领域数据与实时反馈提方案，
**OPM Flow 全物理模拟器唯一裁定**，结果回流形成闭环；Norne 油田是 case study，不是交付对象。

## 防混乱入口表

| 我想…… | 去哪 |
|---|---|
| 搞懂 Gaia 团队怎么运转、代理/裁判/红队各管什么 | **[`_codebook/gaia-agent-team.md`](_codebook/gaia-agent-team.md)** |
| 搞懂实验台两个口径与恒等式及格线 | [`_codebook/experiment-harness.md`](_codebook/experiment-harness.md) |
| 知道当前坐标、下一步、已完成 | `plan show` → [`_wiki-methodology/_top/_task_plan.md`](_wiki-methodology/_top/_task_plan.md) |
| 看实验结论（含多次撤稿） | [`_wiki-methodology/_top/_findings/`](_wiki-methodology/_top/_findings/) |
| 找脚本入口与运行命令 | [`_meta/_registry.yml`](_meta/_registry.yml) |
| 找数据资产、许可、归档 | [`_meta/_data_registry.yml`](_meta/_data_registry.yml) |
| 跑校验闸门 | [`_wiki-methodology/_tests/_gates.yml`](_wiki-methodology/_tests/_gates.yml) |
| **看正式指标表** | **`python _code/gaia.py`**（勿再写临时统计脚本） |
| 找某个符号的调用面 | `codegraph explore "<符号>"`（勿用 grep 猜） |

## 代码分层（`_code/` 共 60+ 脚本，按角色而非字母序）

```
读数接口（对外唯一表面）
  gaia.py           ★★ **结果的唯一入口**。不实现算法，只把实测口径/经济/三臂/无 Agent
                       接到一个稳定表面并装配正式指标表。正式批次常量 gaia.BATCH。
                       🔴 新增读数点必须调它，不得在临时脚本里重抄 ΔNPV/基准/批次口径。
                       `python _code/gaia.py [--json]`

真源与地基
  fc_truth.py       ★ 模拟结果的**唯一**读取口径:FWIT/FOPT/FWPT，禁梯形积分、禁硬编码基准
  forecast_gen.py      预测段 deck 生成 + FC_GRID + 井表 + harvest（θ→模拟输入的唯一入口）
  norne_bulk.py        Norne 井名、观测键、Docker 镜像常量
  fc_decide.py         _run_sim：唯一的 OPM Flow 调用点；以及代理梯度优化器

代理模型
  fc_mech.py           PosNet（傅里叶位置嵌入）
  fc_rank_fidelity.py  定版代理训练 + 排名保真度实验
  fc_team_proxy.py     ★ 代理包装：给团队做毫秒级批量评估

Gaia 团队（本轮主线）
  fc_team.py           ★ 消息协议 + Cartographer + 角色注册表
  fc_team_loop.py      ★ 主环路 Readers→Synthesizer→代理→红队→模拟器→回流
  fc_team_diag.py      ★ 诊断：把一轮的全部原文落盘
  fc_team_ablate.py    ★ 消融分析（先测噪声底再判边际）
  fc_baselines.py      ★ 无 Agent 对照。**主线只用 B2 一维缩放**（唯一决策变量，不含 LLM）;
                       B4 等预算随机 / BG 贪心短期 2026-09-06 退役，代码仍在但 --method 默认 B2
  run_coarse_queue.sh  ★ 三臂批次驱动：交错随机化、按内容指纹跳过已完成，可反复调用补样本量

经济
  fc_fix.py            主线口径：field_cum 读数 + 真实 Brent NPV（勿用梯形积分）
  fc_water_econ.py     ★ 含注水/采出水成本的经济重评价与水价扫描

历史/旁支（结论已归档，勿当主线）
  fc_agent.py fc_gaia_loop.py fc_ms.py fc_gap.py fc_leak.py fc_noise.py … 见 registry
```

★ = 本轮主线，有 `gate:team-framework` 覆盖。

## 目录职责

```
_code/         源码（角色见上）
_data/         volve_causal_v0.2/（方向一遗留，当前主线不依赖）
_pipelines/    46 个实验产物目录；主线在 fc_team/、fc_decide/sim/、fc_water_econ/
_tests/        test_harness.py / test_team.py / test_surrogate.py / test_causal_arms.py
_codebook/     用户视角手册（盖章制）
_meta/         代码与数据登记
_paper/        论文章节 + manifest（🔴 03/04 仍是旧主线，待 P4.11 重写）
_wiki-methodology/_top/    计划、发现、决策
_legacy/       归档
```

## 五类结构性缺陷（2026-09-05 Workflow 审计，45 条归并；防线在 `_tests/test_truth.py`）

| 类 | 判据 |
|---|---|
| ① 同一个量有多套算法 | 这个量在项目里是否只有一种算法? |
| ② 标识符用位置而非内容 | 这个 key 换个上下文还指同一个东西吗? |
| ③ 防线是黑名单而非白名单 | 新出现的同类问题，现有防线能发现吗? |
| ④ 状态更新与控制流顺序错 | 提前退出时，本轮成绩会不会丢? |
| ⑤ 注释断言与代码不符 | 这条注释有测试守着吗? |

数据泄露的唯一判据:**这条信息是否只有在解完题之后才写得出来?**
本项目已发生三次泄露(实例表 / sweep.json / tilt 分箱表由智能体自解算例支撑)。

## 三条不要踩的线

1. **取值一律用模拟器自报 `field_cum`**，不要对井曲线做梯形积分（实测差 0.58%，是容差的数十倍）。
2. **代理只用于筛选与排序**，绝对值一律以 OPM Flow 为准。
3. **常量从既有真源 import**，不要自建（`fc_water_econ` 曾自建 `T_START` 致基准 NPV 错 37%）。
