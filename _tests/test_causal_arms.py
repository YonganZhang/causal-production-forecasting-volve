"""把 2026-08-09 那轮审计(Codex 跨模型复核 + 13 agent Workflow)查出的 bug 锁死。

每个测试对应一个**真实发生过**的错误，不是假想场景。测试不跑模拟器，只验参数化逻辑。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_code"))

import norne_bulk as NB          # noqa: E402
import norne_obs_arm as OA       # noqa: E402

# 发布包不含 Norne deck(ODbL 上游自取), 但下游仍应能跑其余测试。
# 需要 deck 的用例明确 skip 并说明原因 —— 不静默通过, 也不显示成失败。
needs_deck = pytest.mark.skipif(
    not (NB.BASE / NB.SCH).exists(),
    reason=f"需要 Norne deck ({NB.BASE}); 设 NORNE_DECK 环境变量, 或 {NB.P.DECK_HOWTO}")


# ---------------------------------------------------------------- 相态分离

def test_mixed_phase_wells_get_separate_controls():
    """🔴 v1 的致命 bug:θ 只按井名匹配, 把同一口井的注水和注气乘以同一个乘子。

    实测 9 口"注水井"里有 4 口是水气混注, 而 C-3H 末期水率是 0、气率 143344 ——
    它到历史末段其实是注气井。于是 ∂SWAT/∂θ 被当成"注水波及"讲, 物理解释是错的。
    """
    by_well: dict[str, set[str]] = {}
    for w, ph in NB.CONTROLS:
        by_well.setdefault(w, set()).add(ph)
    assert by_well["C-3H"] == {"WATER", "GAS"}, "C-3H 是水气混注, 必须拆成两个控制变量"
    mixed = {w for w, ph in by_well.items() if len(ph) > 1}
    assert mixed == {"C-1H", "C-3H", "C-4AH", "C-4H"}, f"混相井集合变了: {mixed}"
    assert len(NB.CONTROLS) == 13 and NB.N_THETA == 17


@needs_deck
def test_build_only_scales_the_matching_phase(tmp_path):
    """给 C-3H 的 WATER 一个大乘子、GAS 乘子为 0, 改写后只有 WATER 记录该变。"""
    theta = np.zeros(NB.N_THETA)
    iw = NB.CONTROLS.index(("C-3H", "WATER"))
    theta[iw] = 1.0                                    # ×10
    work = tmp_path / "case"
    NB.build(theta, work)
    txt = (work / NB.SCH).read_text(errors="ignore")

    def rates(phase: str) -> list[float]:
        out = []
        for ln in txt.splitlines():
            b = ln.split("--")[0]
            t = b.replace("/", " ").split()
            if len(t) >= 2 and t[0].strip("'") == "C-3H" and t[1].strip("'").upper() == phase:
                g = re.search(r"'RATE'\s+([\d.eE+-]+)", b)
                if g:
                    out.append(float(g.group(1)))
        return out

    base = (NB.BASE / NB.SCH).read_text(errors="ignore")
    orig_gas = [float(re.search(r"'RATE'\s+([\d.eE+-]+)", l.split("--")[0]).group(1))
                for l in base.splitlines()
                if (t := l.split("--")[0].replace("/", " ").split()) and len(t) >= 2
                and t[0].strip("'") == "C-3H" and t[1].strip("'").upper() == "GAS"
                and re.search(r"'RATE'\s+([\d.eE+-]+)", l.split("--")[0])]
    assert rates("GAS") == pytest.approx(orig_gas, rel=1e-6), "GAS 乘子为 0, 气率不该变"
    assert max(rates("WATER")) > 0, "WATER 记录应存在"


@needs_deck
def test_build_refuses_to_silently_produce_baseline(tmp_path, monkeypatch):
    """相态匹配失效时必须炸, 不能静默跑出一个与基线相同的 deck。"""
    monkeypatch.setattr(NB, "CONTROLS", (("NO-SUCH-WELL", "WATER"),))
    with pytest.raises(RuntimeError, match="没有任何 WCONINJE"):
        NB.build(np.zeros(1 + len(NB.PERM_REGIONS)), tmp_path / "c")


# ---------------------------------------------------------------- 观测臂混杂

def test_policy_refuses_to_run_without_state():
    """🔴 旧版 state.get("wct", 0.3) / ("pres_ratio", 1.0) 的默认值恰好让策略项归零。

    配合 read_state 里两个 `except Exception: pass`, "读不到状态"被静默变成"无混杂",
    观测臂退化成第二个干预臂 —— 整个"量化朴素方法偏差"的设计当场失效, 且不报错。
    """
    rng = np.random.default_rng(0)
    g = dict(gain_wct=1.0, gain_pres=1.0, noise=0.0)
    with pytest.raises(KeyError):
        OA.policy({}, 0, rng, **g)
    with pytest.raises(KeyError):
        OA.policy({"wct": 0.5}, 0, rng, **g)           # 缺 pres_ratio 也必须炸


def test_policy_actually_responds_to_state():
    """混杂的前提:不同状态必须给出不同动作。"""
    rng = np.random.default_rng(0)
    g = dict(gain_wct=1.0, gain_pres=1.0, noise=0.0)
    dry = OA.policy({"wct": 0.05, "pres_ratio": 0.90}, 0, rng, **g)
    wet = OA.policy({"wct": 0.70, "pres_ratio": 1.05}, 0, rng, **g)
    assert dry > wet + 0.3, f"含水率高应显著减注, 实得 dry={dry:.3f} wet={wet:.3f}"


def test_read_state_fails_loud_when_summary_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        OA.read_state(tmp_path, 100.0)


# ---------------------------------------------------------------- 分段日期

def test_stage_dates_are_parsed_not_averaged():
    """🔴 旧版按 `day += 3312 / DATES 行数` 平均推进, 而 Norne 报告步间隔极不均匀。

    策略动作会被贴到错误的时间段, 打乱"策略响应状态"的因果结构。
    """
    assert OA._to_day(6, "NOV", 1997) == 0.0
    assert OA._to_day(1, "DEC", 2006) == pytest.approx(3312.0, abs=1.0)
    assert OA._to_day(1, "JAN", 2000) == pytest.approx(786.0, abs=1.0)
    # 段号必须随真实日期单调不减
    days = [OA._to_day(1, "JAN", y) for y in range(1998, 2007)]
    rows = [min(int(np.searchsorted(OA.STAGE_DAYS, d, "right") - 1), OA.N_STAGES - 1)
            for d in days]
    assert rows == sorted(rows) and rows[0] == 0 and rows[-1] == OA.N_STAGES - 1


# ---------------------------------------------------------------- schema 隔离

def test_v1_and_v2_outputs_do_not_share_a_directory():
    """v1/v2 的 theta 语义不同(是否分相态), 混在一起 --merge 会静默出错。"""
    assert NB.SCHEMA == 2
    assert NB.OUT != NB.OUT_V1
    assert OA.OUT != Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_obs")


def test_theta_names_expose_phase():
    names = NB.theta_names()
    assert "inj_C-3H_WATER" in names and "inj_C-3H_GAS" in names
    assert len(names) == NB.N_THETA
    # 旧的无相态命名不该再出现, 否则下游会误以为是纯注水
    assert "inj_C-3H" not in names


# ---------------------------------------------------------------- 两臂接口一致

def test_harvest_contract_is_a_dict_with_required_keys():
    """🔴 harvest() 从返回元组改成返回 dict 时, 观测臂还在 `obs, fld = got` 解包,
    探路时两个样本全报 "too many values to unpack"。两臂共用 norne_bulk,
    改了一边必须同步另一边 —— 这里锁住契约。

    用 AST 而不是字符串匹配:本测试第一版去文件里 grep 元组解包的字面量,
    结果搜到了**自己写的注释**, 假阳性。注释里出现的代码片段不是代码。
    """
    import ast
    import inspect

    for mod in (NB, OA):
        tree = ast.parse(Path(inspect.getfile(mod)).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            is_harvest = (isinstance(call, ast.Call)
                          and getattr(call.func, "attr", getattr(call.func, "id", "")) == "harvest")
            if is_harvest:
                assert not isinstance(node.targets[0], ast.Tuple), (
                    f"{mod.__name__} 在按元组解包 harvest(), 但它返回 dict")

    src = inspect.getsource(NB.harvest)
    for k in ("obs", "fields", "inj_actual", "field_days", "n_restart"):
        assert f'"{k}"' in src, f"harvest 的返回里缺 {k}"


def test_both_arms_write_schema_and_rc():
    """两臂的分片都必须带 schema 与 rc, 否则下游无法拒绝混用或未收敛的样本。"""
    import ast
    import inspect

    for mod in (NB, OA):
        tree = ast.parse(Path(inspect.getfile(mod)).read_text())
        saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "savez_compressed"]
        assert saves, f"{mod.__name__} 没有落盘调用"
        for call in saves:
            kw = {k.arg for k in call.keywords if k.arg}
            assert "schema" in kw, f"{mod.__name__} 落盘时没写 schema"
            assert "rc" in kw, f"{mod.__name__} 落盘时没写 rc"


# ---------------------------------------------------------------- 显著性

def test_hc1_se_matches_textbook_formula():
    """效应场用的捷径 var_a = Σ h_i² e_i² 必须等于完整三明治的对角元。"""
    rng = np.random.default_rng(7)
    n, p = 200, 5
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(size=n) * (0.5 + np.abs(X[:, 0]))
    A = np.hstack([X, np.ones((n, 1))])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ b
    a = 2
    XtXi = np.linalg.pinv(A.T @ A)
    full = XtXi @ (A.T @ (A * (e ** 2)[:, None])) @ XtXi          # 完整三明治
    h = A @ XtXi[:, a]
    quick = float((h ** 2) @ (e ** 2))                            # 脚本里用的捷径
    assert quick == pytest.approx(float(full[a, a]), rel=1e-9)


def test_nan_aware_layer_pick_is_not_hijacked_by_masked_cells():
    """🔴 显著性掩码把 eff 变成含 NaN, 用 sum/argmax 会让含 NaN 的层排到最前。

    实测把自动选层从 K=10 带偏到 K=1。必须用 nan* 版本, 且用均值消除各层单元数差异。
    """
    k = np.repeat(np.arange(5), 10)
    eff = np.zeros(50)
    eff[k == 3] = 1.0                       # 真正最强的层
    eff[0] = np.nan                         # K=0 里塞一个被掩掉的单元
    naive = [np.abs(eff[k == i]).sum() for i in range(5)]
    assert int(np.argmax(naive)) == 0, "复现旧 bug:argmax 遇 NaN 直接返回该下标"
    fixed = [np.nanmean(np.abs(eff[k == i])) if np.isfinite(eff[k == i]).any() else -np.inf
             for i in range(5)]
    assert int(np.argmax(fixed)) == 3
