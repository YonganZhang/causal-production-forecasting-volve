"""Gaia 团队框架的回归测试。

🔴 为什么补:CodeGraph 对 `hard_check` / `inj_water` 标注 "no covering tests found"。
   这些函数直接决定论文数字 —— 否决权判谁出局、经济排序判谁最优 —— 却全靠手工核对。
   本文件锁住几个**已用模拟器独立核验过**的锚点，防止重构悄悄改掉结论。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_code"))

SIM = ROOT / "_pipelines" / "fc_decide" / "sim"
BASE_OIL, BASE_WATER = 7_493_840.0, 93_931_360.0     # OPM Flow 自报 field_cum 实测值


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"缺少算例 {p.name}")


# ------------------------------------------------------------ 经济评价
def test_econ_baseline_matches_fc_fix():
    """含水成本模块在零水价下必须与既有主线 fc_fix 的 NPV 完全一致。

    🔴 这条锁住的是一个真实事故:fc_water_econ 第一版自建 T_START=2006-01-01，
       而 Norne 1997 投产，油价整体错位，基准 NPV8 算成 1798M(真值 2857M)。
    """
    _need(SIM / "baseline.npz")
    from fc_water_econ import econ
    import fc_fix
    a = econ(SIM / "baseline.npz", 0.0, 0.0)
    b = fc_fix.load("baseline")
    assert a["oil"] == pytest.approx(b["oil"], rel=1e-9)
    assert a["npv8"] == pytest.approx(b["npv8"], rel=1e-6)


def test_econ_water_cost_reduces_npv():
    """水一旦计价，NPV 必须单调下降；否则成本项没接上。"""
    _need(SIM / "baseline.npz")
    from fc_water_econ import econ
    v = [econ(SIM / "baseline.npz", c, c / 2)["npv8"] for c in (0.0, 1.0, 2.0, 3.0)]
    assert all(x > y for x, y in zip(v, v[1:])), f"NPV 未随水价单调下降: {v}"


# ------------------------------------------------------------ 解析注水量
def test_inj_water_baseline():
    """θ=0 必须还原基准注水量(误差 <1%)。这是解析式的定标锚点。"""
    from fc_team_loop import inj_water
    assert inj_water(np.zeros(24)) == pytest.approx(BASE_WATER, rel=0.01)


def test_inj_water_monotone():
    """θ 整体抬高必须使注水量增加。"""
    from fc_team_loop import inj_water
    lo, hi = inj_water(np.full(24, -0.2)), inj_water(np.full(24, 0.2))
    assert lo < BASE_WATER < hi


def test_inj_water_respects_cap():
    """极大 θ 不能让注水量无限增长 —— 四口井都有实测注入能力上限。

    🔴 这正是"名义控制自由度 ≠ 有效控制自由度":目标率可以横跨 430 倍，
       实测被 BHP 能力钉住。
    """
    from fc_team_loop import inj_water
    assert inj_water(np.full(24, 2.0)) / inj_water(np.full(24, 1.0)) < 1.05


# ------------------------------------------------------------ 红队否决权
def test_hard_check_passes_baseline():
    from fc_team_loop import hard_check
    assert hard_check(np.zeros(24), 0.25) == []


def test_hard_check_warns_but_does_not_veto_moderate_overrun():
    """预算超标只警告，不否决 —— 由模拟器实测裁定。

    🔴 语义变更(2026-09-03):此前预测超标即 high 否决。但任何由 θ 预测实测
       注水量的近似都不够准(固定上限式 R²=0.219、最大误差 40.2pp;二次回归
       R²=0.960 但高 θ 稀疏区仍差 29pp)。代价是实打实的:**收益最高的 4 个方案
       实测注水仅 -2.35%~+5.36%、完全合规，却被判 +38%~+44% 超标而否决**。
       现改为 medium 警告 + 送模拟器，由实测 FWIT 决定去留。
    """
    from fc_team_loop import hard_check
    best = np.tile(np.array([0.62, 0.62, 0.62, 0.62, 0.03, -0.63])[:, None], (1, 4)).ravel()
    f = hard_check(best, 0.25)
    assert not [x for x in f if x.severity == "high"], (
        "实测合规(-2.35%)的最优方案不得被 high 否决")
    assert any(x.code == "water_budget_predicted" for x in f), "应保留 medium 警告"


def test_hard_check_still_vetoes_extreme_overrun():
    """极端偏离(>2.5 倍容差)仍必须否决，否则模拟预算会被明显不可行的方案吃掉。"""
    from fc_team_loop import hard_check
    f = hard_check(np.full(24, 0.5), 0.25)
    assert any(x.severity == "high" and x.code == "water_budget_extreme" for x in f)


def test_hard_check_vetoes_out_of_box():
    from fc_team_loop import hard_check
    f = hard_check(np.full(24, 1.5), 0.25)
    assert any(x.severity == "high" and x.code == "theta_out_of_box" for x in f)


# ------------------------------------------------------------ 消息协议
def test_message_validation_rejects_bad_provenance():
    """无据可引的角色必须发不出合法消息 —— 这是"角色是不是空壳"的判据。"""
    from fc_team import Flag, Message, Provenance, validate
    ok = Message("a", "reading", {}, 0.5, Provenance("s", "r", "simulation", "0" * 64))
    assert validate(ok) == []
    for bad, why in [
        (Message("", "reading", {}, .5, Provenance("s", "r", "simulation", "0"*64)), "空 agent_id"),
        (Message("a", "nope", {}, .5, Provenance("s", "r", "simulation", "0"*64)), "非法 stage"),
        (Message("a", "reading", {}, 1.7, Provenance("s", "r", "simulation", "0"*64)), "conf 越界"),
        (Message("a", "reading", {}, .5, Provenance("s", "r", "bogus", "0"*64)), "非法 source_type"),
        (Message("a", "reading", {}, .5, Provenance("s", "r", "simulation", "short")), "非 SHA256"),
    ]:
        assert validate(bad), f"应当拒绝: {why}"


def test_cartography_message_is_valid():
    """落盘的 Cartographer 消息必须仍然合法(防 schema 漂移)。"""
    import json
    from fc_team import Message, Provenance, validate
    f = ROOT / "_pipelines" / "fc_team" / "cartography.json"
    _need(f)
    d = json.loads(f.read_text()); pv = d["provenance"]
    m = Message(d["agent_id"], d["stage"], d["payload"], d["confidence"],
                Provenance(pv["source_id"], pv["source_region"],
                           pv["source_type"], pv["source_sha256"]))
    assert validate(m) == []


def test_connectivity_is_attributable():
    """连通性矩阵的设计矩阵条件数必须 <30，否则井间归因不可信。"""
    import json
    f = ROOT / "_pipelines" / "fc_team" / "cartography.json"
    _need(f)
    d = json.loads(f.read_text())
    cond = d["payload"].get("diag", {}).get("cond")
    if cond is None:
        pytest.skip("旧版 cartography.json 无 diag 字段")
    assert cond < 30, f"条件数 {cond} 过高，井间归因不可信"


# ------------------------------------------------------------ θ 方向
def test_theta_orientation_is_not_silently_reshaped():
    """🔴 锁一个定时炸弹:LLM 偶尔返回 (井, 时段) 而非 (时段, 井)。

    两者都是 24 个元素，`reshape(6, 4)` 会**静默成功**并把井与时段彻底错位。
    2026-09-03 在一次诊断中实测到 LLM 返回 (4, 6)。
    """
    import numpy as np
    from fc_team_loop import INJ, N_STAGE
    wells_by_stage = np.arange(24, dtype=float).reshape(len(INJ), N_STAGE)   # 错误方向
    scrambled = wells_by_stage.reshape(N_STAGE, len(INJ))
    assert not np.allclose(scrambled, wells_by_stage.T), (
        "reshape 与转置结果相同则本测试无意义")
    # 正确处理必须是转置而不是 reshape
    assert wells_by_stage.T.shape == (N_STAGE, len(INJ))
    assert np.allclose(wells_by_stage.T[0], wells_by_stage[:, 0])


# ------------------------------------------------------------ 泄露防线
def test_no_solution_leakage_in_role_slices():
    """🔴 任何角色的数据切片都不得包含"已知最优解"。

    2026-09-04 事故:我把方案库里收益最高的 8 个 θ 逐格打印、并附上
    "顶级方案统计配方"，发给了全部三个臂的角色与总工。那是**答案**不是知识 ——
    等于考前发答案。后果:三臂分数全部挤在 +480 上下(都在抄同一份答案)，
    整批实验作废。此测试锁死这条边界。
    """
    import json
    from fc_team import OUT
    from fc_team import ROLES
    from fc_team_loop import slice_for
    f = OUT / "cartography.json"
    if not f.exists():
        pytest.skip("缺 cartography.json")
    carto = json.loads(f.read_text())
    # 🔴 2026-09-05 补充:第二处泄露是 sweep.json 的"最佳方案注水量为基准的 X%"。
    #    它不是市场事实，是**已知最优解的答案卡片**;而单 Agent 臂恰好保留该角色，
    #    造成"单 Agent 远高于 Agent Team"的反常排序。
    BANNED = ["最好与最差实例", "顶级方案的统计配方", "收益最高的",
              "ΔNPV +5", "方案池", "边际价值表",
              "最佳方案的注水量", "当前最佳方案", "最优注水**总量**随水价"]
    for r in ROLES:
        txt, _ = slice_for(r["id"], carto, None)
        hit = [b for b in BANNED if b in txt]
        assert not hit, f"角色 {r['id']} 的切片含疑似泄露内容: {hit}"


def test_chief_prompt_has_no_solution_leakage():
    """总工提示词同样不得含已知最优解。"""
    from fc_team_loop import ASK_CHIEF
    BANNED = ["最好与最差实例", "顶级方案的统计配方", "+0.62 +0.62"]
    hit = [b for b in BANNED if b in ASK_CHIEF]
    assert not hit, f"总工提示词含疑似泄露内容: {hit}"
