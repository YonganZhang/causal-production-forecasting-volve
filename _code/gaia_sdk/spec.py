"""架构规格:每个 Agent 是什么、读什么、产出什么、实测开销多少。

这里的数字全部来自已跑实验,由 `_code/fc_paper_check.py` 与
`_tests/test_gaia_sdk.py` 守住;不得手写未经核对的值。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    id: str
    layer: str
    role: str                     # 这个 Agent 干什么
    reads: str                    # 输入
    emits: str                    # 输出
    backend: str                  # 谁在执行:llm / deterministic / simulator
    cost: str = ""                # 实测开销
    caveat: str = ""              # 能力边界(写进论文的诚实声明)
    wraps: tuple[str, ...] = field(default_factory=tuple)   # 包装了哪些既有实现


LAYERS = ("L1 数据", "L2 知识", "L3 推理", "L4 评估", "L5 裁决")

SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        id="data_engineer", layer="L1 数据", backend="deterministic",
        role="把原始油藏 deck 与模拟输出整理成结构化算例库",
        reads="Norne Eclipse deck(46×112×22,44,431 活跃网格)+ OPM Flow 输出",
        emits="7,400 训练样本 + 500 独立确认样本(θ, 多井多相轨迹)",
        cost="每算例一次全物理模拟,约 1 min",
        wraps=("norne_sampler.py", "forecast_gen.py", "patch_eclipse_deck.py", "norne_pack.py"),
    ),
    AgentSpec(
        id="cartographer", layer="L1 数据", backend="deterministic",
        role="从算例库反演注采井间影响,并自报可信度",
        reads="471 个已模拟算例的 inj_actual 与各产油井累计",
        emits="4×22 影响矩阵 + 设计矩阵条件数(实测 2.05,阈值 30)",
        cost="秒级",
        caveat="偏回归,不是示踪实测;条件数超阈值时自动升高severity",
        wraps=("fc_team.cartography",),
    ),
    AgentSpec(
        id="knowledge_curator", layer="L2 知识", backend="llm",
        role="审计 deck 里有什么物理数据、缺什么;检索本油田文献;"
             "为每个领域角色生成知识库,并**显式写明该角色没有什么证据**",
        reads="Eclipse deck 关键字清单 + 文献检索(Elsevier Scopus/ScienceDirect)",
        emits="各角色的知识库文本 + provenance(source_type 区分实测/文献)",
        cost="离线执行一次;产出被全部已发表实验原样消费",
        caveat="🔴 **离线策展,不在决策回路内运行**。这样做的副作用是有益的:"
               "知识库在所有重复实验中固定不变,run-to-run 差异不被知识漂移混杂。",
        wraps=("fc_team_loop.slice_for", "lit_fetch.py"),
    ),
    AgentSpec(
        id="domain_role", layer="L3 推理", backend="llm",
        role="六个领域角色各读各的证据切片,先独立表态,再质疑分歧最大的同伴",
        reads="本角色专属知识库 + 上一轮实测在本域内的表现",
        emits="带 provenance 的类型化消息(agent_id, confidence, source_*)",
        cost="实测约 42 次 LLM 调用 / 实验(六角色配置)",
        caveat="没有可引用证据的角色**发不出合法消息** —— 由协议判定,不靠争论。"
               "geomechanics_expert 已按消融裁定剔除:无本油藏实测却给硬上限,"
               "自报置信度 0.29 vs 实测类 0.65。见 gaia_sdk.agents.EXCLUDED。",
        wraps=("fc_team.ROLES", "fc_team_loop.slice_for"),
    ),
    AgentSpec(
        id="synthesizer", layer="L3 推理", backend="llm",
        role="按**证据强度**而非人头裁决,合成候选注入方案 θ",
        reads="全部角色消息,按 source_type 排序(实测类先出现)",
        emits="候选 θ ∈ R^(6×4) + 每条采纳建议的来源 agent_id",
        cost="每轮一次",
        caveat="一条 literature 类顾虑不足以否决一条 simulation 类正向证据;"
               "多角色同一顾虑若依据都是 literature,仍只算一份",
        wraps=("fc_team_loop.ASK_CHIEF",),
    ),
    AgentSpec(
        id="surrogate", layer="L4 评估", backend="deterministic",
        role="全物理模拟器的快速替身,让候选能被大批量筛选",
        reads="θ(24 维静态控制向量)",
        emits="多井多相产量轨迹 (W×P×T);**只做筛选与排序,不给最终取值**",
        cost="18.1 µs/方案,约 3.3×10⁶ 倍加速",
        caveat="🔴 随机方案上排名 ρ=0.998,**优化器自己的候选区 ρ=−0.21**、"
               "top-1 命中 0,且选中的正是自己高估最狠的候选 —— "
               "因此它没有赋值权。这条边界是 L5 存在的理由。",
        wraps=("fc_team_proxy.py", "fc_mech.py", "fc_rank_fidelity.py"),
    ),
    AgentSpec(
        id="feasibility_auditor", layer="L4 评估", backend="deterministic",
        role="算术可行性闸门:注入预算偏离、θ 箱约束、按实测井筒能力的饱和检查",
        reads="候选 θ + 各井实测注入能力",
        emits="通过 / 否决 + severity",
        cost="微秒级",
        caveat="🔴 **刻意不交给语言模型**:语言模型可以被说服,算术边界不能",
        wraps=("fc_team_loop.hard_check",),
    ),
    AgentSpec(
        id="simulator_adjudicator", layer="L5 裁决", backend="simulator",
        role="全物理裁定。**唯一有权给出数值的环节,不是 Agent**",
        reads="通过闸门的候选 θ",
        emits="模拟器自报累计量 FOPT/FWIT/FWPT → NPV;被否决者不计入成绩",
        cost="实测 2.7 次调用 / 实验(团队臂),约 1 min/次",
        caveat="全文所有经济数字均出自此环节;代理模型的预测值一个都不报",
        wraps=("fc_decide._run_sim", "fc_truth.py", "fc_water_econ.econ"),
    ),
)

SPEC_BY_ID = {s.id: s for s in SPECS}
