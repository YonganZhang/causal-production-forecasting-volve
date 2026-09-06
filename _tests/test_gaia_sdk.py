"""`_code/gaia_sdk` 的忠实性测试。

SDK 是对**已跑实验**的打包,不是另一个系统。守两件事:
  ① SDK 复现的知识库与已发表实验实际消费的**逐字节一致**;
  ② SDK 自述的每个数字都是现算的,不是写死的文案。

任何让 SDK 与 fc_team_loop 产生分歧的改动都会让本文件变红 —— 这正是目的。
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_code"))

import fc_team                      # noqa: E402
import fc_team_loop as L            # noqa: E402
from gaia_sdk import GaiaTeam       # noqa: E402
from gaia_sdk.spec import SPECS     # noqa: E402

# slice_for 收的是 cartography **整条消息**(含 payload 键),与实验里传的一致
CARTO = json.loads((ROOT / "_pipelines" / "fc_team" / "cartography.json").read_text())
TEAM = GaiaTeam()


@pytest.mark.parametrize("role_id", [r["id"] for r in fc_team.ROLES if r["id"] != "chief_engineer"])
def test_curated_knowledge_is_byte_identical(role_id):
    """SDK 的知识层产出必须与实验实际消费的一字不差。"""
    got_txt, got_prov = TEAM.knowledge.curate(role_id, CARTO)
    want_txt, want_prov = L.slice_for(role_id, CARTO, None)
    assert got_txt == want_txt, f"{role_id}: SDK 知识库与实验消费的不一致"
    assert got_prov.source_type == want_prov.source_type
    assert got_prov.source_id == want_prov.source_id


def test_roster_matches_published_run():
    """角色名单必须与已发表实验的名单一致。"""
    loops = sorted((ROOT / "_pipelines" / "fc_team").glob("loop_F2full7_*.json"))
    ran = json.loads(loops[0].read_text())["roles"]
    assert [r.role_id for r in TEAM.roles] == ran, "SDK 角色名单与实验跑的不同"


def test_literature_dois_appear_in_the_prompt_actually_used():
    """L2 声称检索到的 DOI,必须真的出现在角色实际读到的知识库里。"""
    txt, _ = TEAM.knowledge.curate("seismic_4d_analyst", CARTO)
    for doi, _title in TEAM.knowledge.literature()["records"]:
        assert doi in txt, f"SDK 声称检索到 {doi},但角色实际没读到它"


def test_deck_audit_absences_are_stated_to_the_role():
    """deck 缺什么,必须在角色知识库里明说 —— 缺失是更重要的那种知识。"""
    txt, prov = TEAM.knowledge.curate("geomechanics_expert", CARTO)
    for kw in TEAM.knowledge.deck_audit()["absent"]:
        assert kw in txt, f"deck 缺 {kw},但没告诉地质力学角色"
    assert prov.source_type == "literature", "无本油藏实测的角色不得标成 simulation/measurement"


def test_evidence_rank_matches_the_synthesizer_actually_used():
    """证据分级必须与总工提示词里用的一致。"""
    src = (ROOT / "_code" / "fc_team_loop.py").read_text()
    m = re.search(r"_RANK\s*=\s*(\{[^}]*\})", src)
    assert m, "fc_team_loop 里找不到 _RANK"
    assert eval(m.group(1)) == TEAM.synthesizer.evidence_rank()


def test_surrogate_has_no_pricing_authority():
    """代理只筛不判 —— SDK 必须报出它在优化区失效,否则这个架构没有理由。"""
    f = TEAM.surrogate.fidelity()
    assert f["random_region"]["spearman"] > 0.9
    assert f["optimizer_region"]["spearman"] < 0.5
    assert f["optimizer_region"]["top1_hit"] == 0.0
    assert f["optimism_of_pick"] > f["optimism_median"], "代理选中的应是它高估更狠的"


def test_numbers_are_computed_not_hardcoded():
    """自述里的关键数字必须现算。改了底层数据,SDK 必须跟着变。"""
    import gaia
    out = TEAM.adjudicator.verified_outcome()
    s = gaia.summary()
    assert out["gaia_team_dnpv_musd"] == round(s["arms"]["full7"]["mean"], 1)
    assert out["reference_dnpv_musd"] == round(s["nonllm"]["mean"], 1)
    assert out["baseline_npv_musd"] == round(s["baseline_npv_musd"], 1)
    assert TEAM.cartographer.influence()["condition_number"] == CARTO["payload"]["diag"]["cond"]


def test_every_spec_declares_what_it_wraps_or_is_a_role():
    """每个 Agent 都要能追到既有实现,不许有只存在于文档里的部件。"""
    for s in SPECS:
        assert s.wraps, f"{s.id} 没声明它包装了什么 —— 只存在于文档里的 Agent 不算数"
        for w in s.wraps:
            mod = w.split(".")[0].replace(".py", "")
            assert (ROOT / "_code" / f"{mod}.py").exists(), f"{s.id} 声称包装 {w},但文件不存在"


def test_offline_curation_is_declared_not_hidden():
    """知识层是离线执行的,这一点必须写在 caveat 里,不能含糊过去。"""
    kc = next(s for s in SPECS if s.id == "knowledge_curator")
    assert "离线" in kc.caveat and "不在决策回路内" in kc.caveat
