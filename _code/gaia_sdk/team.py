"""GaiaTeam:把五层装配成一个可自述、可调用的整体。

    python -m gaia_sdk          # 打印架构与实测开销
"""
from __future__ import annotations

from typing import Any

from .agents import (
    CartographerAgent,
    DataEngineerAgent,
    DomainRoleAgent,
    FeasibilityAuditor,
    KnowledgeCuratorAgent,
    SimulatorAdjudicator,
    SurrogateAgent,
    SynthesizerAgent,
)
from .spec import LAYERS


class GaiaTeam:
    """分层多智能体框架。

    权限是分开的,而且是**有证据支撑的分开**:
      · 代理模型有速度,没有赋值权(它在优化区的排名 ρ=−0.21);
      · 语言模型有推理,没有否决权(可行性由算术闸门判);
      · 全物理模拟器有赋值权,是全文所有经济数字的唯一来源。
    """

    def __init__(self) -> None:
        self.data = DataEngineerAgent()
        self.cartographer = CartographerAgent()
        self.knowledge = KnowledgeCuratorAgent()
        self.roles = [DomainRoleAgent(r["id"]) for r in DomainRoleAgent.roster()]
        self.synthesizer = SynthesizerAgent()
        self.surrogate = SurrogateAgent()
        self.auditor = FeasibilityAuditor()
        self.adjudicator = SimulatorAdjudicator()

    # ---------------------------------------------------------------- 自述
    def agents(self) -> list:
        return [self.data, self.cartographer, self.knowledge, *self.roles,
                self.synthesizer, self.surrogate, self.auditor, self.adjudicator]

    def describe(self) -> str:
        out = [f"Gaia Agent Team —— {len(self.agents())} 个 Agent,5 层", "=" * 72]
        for layer in LAYERS:
            members = [a for a in self.agents()
                       if (a.spec.layer == layer if not isinstance(a, DomainRoleAgent)
                           else layer == "L3 推理")]
            if not members:
                continue
            out.append(f"\n### {layer}")
            for a in members:
                out.append(a.describe())
                out.append("")
        return "\n".join(out)

    # ---------------------------------------------------------------- 事实
    def evidence(self) -> dict[str, Any]:
        """框架的关键实测事实。全部现算,不写死。"""
        return {
            "case_library": self.data.case_library(),
            "connectivity": self.cartographer.influence(),
            "deck_audit": self.knowledge.deck_audit(),
            "literature": self.knowledge.literature(),
            "surrogate_fidelity": self.surrogate.fidelity(),
            "verified_outcome": self.adjudicator.verified_outcome(),
            "n_roles": len(self.roles),
        }
