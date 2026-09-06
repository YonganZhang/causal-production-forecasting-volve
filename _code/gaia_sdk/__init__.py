"""Gaia Agent Team —— 面向油藏决策的分层多智能体框架(SDK)。

本包**不实现新算法**。它把这套框架里已经跑通、已经产出论文结果的各个环节,
包装成一组可调用、可复现的 Agent,使「架构」不再只是论文里的一张图。

    from gaia_sdk import GaiaTeam
    team = GaiaTeam()
    print(team.describe())          # 分层架构与每层的真实开销
    kb = team.knowledge.curate()    # 复现 20 次实验实际消费的知识库

🔴 **忠实性约束**:SDK 各 Agent 的输出必须与已发表实验实际消费的内容一致。
   `_tests/test_gaia_sdk.py` 对此做逐字节比对 —— SDK 是对已跑实验的**打包**,
   不是另一个系统。任何让 SDK 与 fc_team_loop 产生分歧的改动都会让测试变红。

分层
----
L1 数据层   DataEngineerAgent    原始 deck → 结构化算例库
            CartographerAgent    算例库 → 注采影响矩阵
L2 知识层   KnowledgeCuratorAgent  deck 审计 + 文献检索 → 各角色知识库
L3 推理层   DomainRoleAgent × 7  各读各的证据 → 互相质疑
            SynthesizerAgent     按证据强度裁决,合成候选 θ
L4 评估层   SurrogateAgent       18.1 µs/方案的批量筛选
            FeasibilityAuditor   算术闸门(不交给语言模型)
L5 裁决层   SimulatorAdjudicator OPM Flow —— 唯一有权赋值的环节,**不是 Agent**
"""
from .agents import (  # noqa: F401
    CartographerAgent,
    DataEngineerAgent,
    DomainRoleAgent,
    FeasibilityAuditor,
    KnowledgeCuratorAgent,
    SimulatorAdjudicator,
    SurrogateAgent,
    SynthesizerAgent,
)
from .team import GaiaTeam  # noqa: F401

__all__ = [
    "GaiaTeam", "DataEngineerAgent", "CartographerAgent", "KnowledgeCuratorAgent",
    "DomainRoleAgent", "SynthesizerAgent", "SurrogateAgent", "FeasibilityAuditor",
    "SimulatorAdjudicator",
]
__version__ = "0.1.0"
