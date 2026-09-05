# CodeBook — 目录

用户视角的手册层。结构事实以 live codegraph 与源码为准,本层只讲「怎么用、为什么这样设计」。

| 节点 | 讲什么 |
|---|---|
| [gaia-agent-team.md](gaia-agent-team.md) | Gaia 注水决策框架:多角色团队、模拟器裁定、三臂对比、数据泄露红线与五类结构性缺陷防线 |
| [experiment-harness.md](experiment-harness.md) | 实验 harness:批次驱动、内容寻址缓存、消融开关 |

## 入口

    python _code/gaia.py          # 正式指标表(唯一读取接口)
    python _code/gaia.py --json   # 机器可读
    bash   _code/run_coarse_queue.sh   # 补齐三臂样本量,跳过已完成
