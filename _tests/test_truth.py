"""类层防线:单一真源 + 内容寻址 + 泄露白名单。

🔴 这三条测试不是查"某处写错了"，是查**一整类问题**能否复发。
   来源:2026-09-05 Workflow 五维审计的 45 条发现，归并后是 5 个类:
     ① 同一个量有多套算法(双真源)
     ② 标识符用位置而非内容
     ③ 防线是黑名单而非白名单
     ④ 状态更新与控制流顺序错
     ⑤ 注释断言与代码不符
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_code"))


# ---------------------------------------------------------------- 类 ① 单一真源
def test_no_second_algorithm_for_measured_water():
    """项目里不得存在第二种"实测注水量"算法。

    审计实测:trapezoid(inj_actual) 与模拟器自报 FWIT 平均差 0.99pp、最大 3.12pp，
    按 ±25% 判可行性有 14 例结论相反。
    """
    bad = []
    for f in (ROOT / "_code").glob("*.py"):
        if f.name in ("fc_truth.py",):
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(txt.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"trapezoid\(\s*d?\[?[\"']?inj_actual", line):
                bad.append(f"{f.name}:{i}")
    assert not bad, ("这些位置仍在用 trapezoid(inj_actual) 算注水量，"
                     f"应改用 fc_truth.measured_winj(): {bad}")


def test_baseline_constants_not_hardcoded_in_prompts():
    """提示词里不得硬编码基准产油/注水量 —— 必须从 fc_truth 取。

    审计实测:曾硬编码 7,469,000 而真值 7,493,840(差 0.33%)，已导致审计员误报 high。
    """
    src = (ROOT / "_code" / "fc_team_loop.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    for lit in ("7,469,000", "7469000", "93_931_360.0", "93,931,360"):
        assert lit not in body, f"提示词/代码中硬编码了基准常量 {lit}"


# ---------------------------------------------------------------- 类 ② 内容寻址
def test_simulation_cache_key_is_content_addressed():
    """模拟缓存键必须由 θ 内容决定，否则重跑会静默复用别的方案的结果。"""
    from fc_team_loop import _theta_key
    import numpy as np
    a = _theta_key(np.zeros(24))
    b = _theta_key(np.zeros(24))
    c = _theta_key(np.r_[0.5, np.zeros(23)])
    assert a == b, "同一 θ 必须得到同一 key"
    assert a != c, "不同 θ 必须得到不同 key"


def test_prompt_refers_to_solutions_by_global_key():
    """提示词引用历史方案必须用全局 key，不得用"候选#N"这种轮内序号。"""
    src = (ROOT / "_code" / "fc_team_loop.py").read_text(encoding="utf-8")
    m = re.search(r"上一轮.*?模拟器实测最佳[^\n]*", src)
    assert m, "未找到 hist 的实测最佳表述"
    assert "候选#{best_sim[0]}" not in m.group(0), (
        "hist 仍用轮内序号引用方案;候选编号每轮重编，跨轮引用会指向别的方案")


# ---------------------------------------------------------------- 类 ③ 白名单防线
_ALLOWED_SOURCES = {
    "sim/baseline.npz", "baseline.npz",          # 基准算例的实测量
    "NORNE_ATW2013.DATA", "deck",                # 油藏静态属性
    "EIA-RBRTE", "BRENT",                        # 市场油价
    "cartography.json",                          # 井间连通性(物理性质)
    "crossref", "doi:",                          # 文献
}
_FORBIDDEN_SOURCES = [
    "exemplars.json",        # 已知最优解的实例
    "stage_well_marginal",   # 由已优化算例回归
    "stage_marginal",
    "unit_value",
    "sweep.json",            # 每行含"最佳方案=<算例名>,ΔNPV=..."
]


def test_role_slices_read_no_forbidden_source():
    """角色切片不得读取任何"跨算例目标函数聚合量"。

    🔴 黑名单不够(审计实证:tilt 分箱表命中黑名单 0 条却是第三处泄露)。
       改为**来源白名单**:凡读取由已解算例聚合而来的目标函数量，一律 fail。
    """
    src = (ROOT / "_code" / "fc_team_loop.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    hit = [k for k in _FORBIDDEN_SOURCES if k in body]
    assert not hit, (f"提示词构造路径读取了已解算例的聚合量: {hit}。"
                     "这类信息只有解完题之后才写得出来，属泄露。")


def test_tilt_prior_excludes_agent_solved_cases():
    """tilt 先验必须由**非智能体**算例导出。

    审计实证:原表 tilt>=0.6 的 234 个算例里 228 个来自 team_*(智能体自己解出的)，
    等于把已删的答案洗一遍再发回去。现要求其 source 字段显式声明排除范围。
    """
    import json
    f = ROOT / "_pipelines" / "fc_team" / "tilt_bins.json"
    if not f.exists():
        pytest.skip("tilt_bins.json 不存在")
    d = json.loads(f.read_text())
    src = d.get("source", "")
    assert "排除" in src and "team" in src, (
        "tilt_bins.json 未声明排除智能体解出的算例;"
        f"当前 source = {src!r}")


# ---------------------------------------------------------------- 类 ④ 先结算后跳转
def test_state_updated_before_early_termination():
    """自适应终止的 break 必须在状态更新之后。

    🔴 本项目在此处翻过两次车:第一次是 incumbent 更新写在 break 之后;
       第二次我"修复"时只改了注释、没验证顺序，注释断言"已在上文更新"而代码在下文。
    """
    src = (ROOT / "_code" / "fc_team_loop.py").read_text(encoding="utf-8").splitlines()
    upd = next(i for i, l in enumerate(src) if "incumbent, incumbent_th = _m" in l)
    brk = next(i for i, l in enumerate(src)
               if "clean_streak >= 2" in l and not l.lstrip().startswith("#"))
    assert upd < brk, (f"incumbent 更新在第 {upd+1} 行，终止判断在第 {brk+1} 行 —— "
                       "终止轮的成绩会被静默丢弃")
