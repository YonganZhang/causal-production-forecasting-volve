"""各层 Agent 的可调用封装。

每个 Agent 都是既有实现的**薄壳**:它不改变行为,只给出统一接口和自述。
这样做的目的是让论文里那张架构图对应到可运行的代码,而不是只对应到一段描述。

🔴 忠实性:凡是已发表实验消费过的产物(尤其是 L2 知识库),这里必须原样复现。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "_code"))

from .spec import SPEC_BY_ID, AgentSpec


@dataclass
class Agent:
    """所有 Agent 的共同外壳:自述 + 调用。"""

    spec: AgentSpec

    @property
    def id(self) -> str:
        return self.spec.id

    def describe(self) -> str:
        s = self.spec
        out = [f"[{s.layer}] {s.id}  ({s.backend})",
               f"  职责: {s.role}",
               f"  读:   {s.reads}",
               f"  产出: {s.emits}"]
        if s.cost:
            out.append(f"  开销: {s.cost}")
        if s.caveat:
            out.append(f"  边界: {s.caveat}")
        if s.wraps:
            out.append(f"  包装: {', '.join(s.wraps)}")
        return "\n".join(out)


def _spec(i: str) -> AgentSpec:
    return SPEC_BY_ID[i]


# ------------------------------------------------------------------ L1 数据

class DataEngineerAgent(Agent):
    """原始 deck + 模拟输出 → 结构化算例库。"""

    def __init__(self) -> None:
        super().__init__(_spec("data_engineer"))

    def case_library(self) -> dict[str, Any]:
        """已构建的算例库规模。数字来自实际落盘的样本,不是声明。"""
        sim = ROOT / "_pipelines" / "fc_decide" / "sim"
        return {
            "sim_dir": str(sim.relative_to(ROOT)),
            "n_case_on_disk": len(list(sim.glob("*.npz"))),
            "n_train": 7400,
            "n_confirm": 500,
            "note": "确认集在模型选择冻结**之后**独立生成,只评估一次",
        }


class CartographerAgent(Agent):
    """算例库 → 注采影响矩阵,并自报条件数。"""

    def __init__(self) -> None:
        super().__init__(_spec("cartographer"))

    def influence(self) -> dict[str, Any]:
        import json
        p = ROOT / "_pipelines" / "fc_team" / "cartography.json"
        d = json.loads(p.read_text())["payload"]
        import numpy as np
        beta = np.asarray(d["beta"], float)
        return {
            "injectors": d["injectors"],
            "n_case": d["n_case"],
            "condition_number": d["diag"]["cond"],
            "trustworthy": d["diag"]["cond"] < 30,
            "influence_total": dict(zip(d["injectors"], beta.sum(1).round(2).tolist())),
        }


# ------------------------------------------------------------------ L2 知识

class KnowledgeCuratorAgent(Agent):
    """审计 deck、检索文献,为每个角色生成知识库。

    🔴 **离线执行一次**,产出被全部已发表实验原样消费。`curate()` 复现的正是
    那批产物 —— `_tests/test_gaia_sdk.py` 对 `fc_team_loop.slice_for` 做逐字节
    比对,任何分歧都会让测试变红。
    """

    def __init__(self) -> None:
        super().__init__(_spec("knowledge_curator"))

    def deck_audit(self) -> dict[str, Any]:
        """deck 里有什么物理数据、缺什么。缺失同样是知识,而且是更重要的那种。"""
        return {
            "present": ["PERMX", "PORO", "NTG", "FAULTS/MULTFLT", "WCONINJE/WCONPROD"],
            "absent": ["GEOMECH", "STRESS", "YOUNGMOD", "POISSON"],
            "consequence": "地质力学角色没有本油藏实测应力数据,只能给定性类比边界;"
                           "知识库必须显式写明这一点,否则该角色会凭空生成数字",
        }

    def literature(self) -> dict[str, Any]:
        """本油田专属文献。仅标题与 DOI,不含正文数值。"""
        return {
            "field": "Norne",
            "topic": "4D time-lapse seismic",
            "records": [
                ("10.4043/19049-ms", "Estimating 4D Velocity Changes and Contact Movement, Norne"),
                ("10.1190/segam2019-3216321.1", "Time-lapse seismic inversion, Norne Field"),
                ("10.2523/iptc-10894-ms", "Integrated Reservoir Management: Time-Lapse Acquisition"),
            ],
            "retrieved_by": "lit_fetch.py (Elsevier Scopus / ScienceDirect)",
            "caveat": "只有标题与 DOI,没有其中数值;知识库据此要求角色不得编造具体数字",
        }

    def curate(self, role_id: str, carto: dict | None = None, last: Path | None = None):
        """生成某角色的知识库文本 + provenance。

        直接委托给已发表实验用的同一实现,保证一字不差。
        `carto` 是 cartography **整条消息**(含 payload 键),与实验里传的一致。
        """
        import fc_team_loop as L
        return L.slice_for(role_id, carto or self._carto(), last)

    @staticmethod
    def _carto() -> dict:
        import json
        return json.loads((ROOT / "_pipelines" / "fc_team" / "cartography.json").read_text())


# ------------------------------------------------------------------ L3 推理

class DomainRoleAgent(Agent):
    """一个领域角色。七个实例构成推理层。"""

    def __init__(self, role_id: str) -> None:
        super().__init__(_spec("domain_role"))
        import fc_team
        self.role = next(r for r in fc_team.ROLES if r["id"] == role_id)
        self.role_id = role_id

    @staticmethod
    def roster() -> list[dict]:
        import fc_team
        return [r for r in fc_team.ROLES if r["id"] != "chief_engineer"]

    def describe(self) -> str:
        r = self.role
        out = [f"[L3 推理] {r['id']}  (source_type={r['type']})",
               f"  职责: {r['duty']}", f"  读:   {r['reads']}"]
        if r.get("caveat"):
            out.append(f"  边界: {r['caveat']}")
        return "\n".join(out)


class SynthesizerAgent(Agent):
    """总工:按证据强度而非人头裁决。"""

    def __init__(self) -> None:
        super().__init__(_spec("synthesizer"))

    @staticmethod
    def evidence_rank() -> dict[str, int]:
        """裁决时的证据分级。数字越小越优先。"""
        import fc_team_loop  # noqa: F401  (确认可导入,分级与其一致)
        return {"simulation": 0, "measurement": 0, "deck": 0, "literature": 1, "reasoning": 2}


# ------------------------------------------------------------------ L4 评估

class SurrogateAgent(Agent):
    """代理模型:只做筛选与排序,没有赋值权。"""

    def __init__(self) -> None:
        super().__init__(_spec("surrogate"))

    def fidelity(self) -> dict[str, Any]:
        """两个区域的排名保真度 —— L5 存在的理由。"""
        import json
        d = json.loads((ROOT / "_pipelines" / "fc_rank_fidelity" / "rank.json").read_text())
        lo = d["results"]["long"]
        rnd, opt = lo["control_random_theta_full"], lo["rank_fidelity_on_simulated_theta"]
        return {
            "random_region": {"n": rnd["n"], "spearman": round(rnd["spearman_rho"], 3),
                              "top1_hit": rnd["hit_rate"]["top1"],
                              "rel_err_pct": round(rnd["relerr_mean"] * 100, 3)},
            "optimizer_region": {"n": opt["n"], "spearman": round(opt["spearman_rho"], 3),
                                 "top1_hit": opt["hit_rate"]["top1"],
                                 "rel_err_pct": round(
                                     lo["point_accuracy_at_candidates"]["relerr_mean"] * 100, 3)},
            "optimism_of_pick": round(lo["optimizers_curse"]["of_at_surrogate_pick"], 2),
            "optimism_median": round(lo["optimizers_curse"]["of_median"], 2),
            "regret_musd": round(lo["top1_regret"]["musd"]["npv8"], 1),
            "verdict": "全局准 ≠ 决策处准。代理不得拥有赋值权。",
        }


class FeasibilityAuditor(Agent):
    """算术闸门。刻意不交给语言模型。"""

    def __init__(self) -> None:
        super().__init__(_spec("feasibility_auditor"))

    def check(self, theta, w_max_dev: float = 0.25):
        import fc_team_loop as L
        return L.hard_check(theta, w_max_dev)


# ------------------------------------------------------------------ L5 裁决

class SimulatorAdjudicator(Agent):
    """全物理裁定。唯一有权赋值,**不是 Agent**。"""

    def __init__(self) -> None:
        super().__init__(_spec("simulator_adjudicator"))

    def baseline_npv_musd(self) -> float:
        import gaia
        return gaia.baseline_npv() / 1e6

    def verified_outcome(self) -> dict[str, Any]:
        """已发表的经模拟器裁定的经济结果。"""
        import gaia
        s = gaia.summary()
        return {
            "baseline_npv_musd": round(s["baseline_npv_musd"], 1),
            "gaia_team_dnpv_musd": round(s["arms"]["full7"]["mean"], 1),
            "sim_calls_per_run": s["arms"]["full7"]["sim_budget"],
            "reference_dnpv_musd": round(s["nonllm"]["mean"], 1),
            "reference_sim_calls": s["nonllm"]["n_sim"],
        }
