"""`_code/gaia.py` 读取接口的类层防线。

守的是**同一个量不能有第二套口径**这一类缺陷 —— 与 `test_truth.py` 同源:
那里防的是"实测水量两套算法",这里防的是"ΔNPV / 批次 / 基准三处各自为政"。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_code"))

import gaia  # noqa: E402


def test_baseline_is_not_hardcoded():
    """基准 NPV 必须现算,不得在接口里写死 —— 写死会在换水价后静默失效。"""
    src = (ROOT / "_code" / "gaia.py").read_text()
    for m in re.findall(r"\b1[67]\d\d(?:\.\d+)?\b", src):
        pytest.fail(f"gaia.py 里出现疑似写死的基准 NPV: {m}")
    assert 1000 < gaia.baseline_npv() / 1e6 < 3000


def test_delta_matches_raw_json():
    """接口给出的 ΔNPV 必须能由原始 loop json 独立复算出来。"""
    base = gaia.baseline_npv()
    f = sorted(gaia.LOOP_DIR.glob(f"loop_{gaia.BATCH}full7_*.json"))[0]
    d = json.loads(f.read_text())
    hand = []
    for r in d["rounds"]:
        vetoed = set(r.get("vetoed") or [])
        hand += [(m["npv8"] - base) / 1e6
                 for k, m in (r.get("adjudicated") or {}).items() if int(k) not in vetoed]
    got = [r for r in gaia._runs("full7", gaia.BATCH) if r["file"] == f.name][0]["traj"]
    assert np.allclose(sorted(hand), sorted(got))


def test_vetoed_solutions_excluded():
    """被模拟器实测否决的方案不得计入成绩 —— 否则等于用未验证方案报数。"""
    for f in gaia.LOOP_DIR.glob(f"loop_{gaia.BATCH}*.json"):
        d = json.loads(f.read_text())
        for r in d["rounds"]:
            if r.get("vetoed"):
                traj = [x["traj"] for x in gaia._runs(
                    "full7" if "full7" in f.name else "one1", gaia.BATCH) if x["file"] == f.name]
                if not traj:
                    continue
                base = gaia.baseline_npv()
                bad = [(r["adjudicated"][str(k)]["npv8"] - base) / 1e6
                       for k in r["vetoed"] if str(k) in (r.get("adjudicated") or {})]
                for b in bad:
                    assert not any(abs(b - t) < 1e-9 for t in traj[0]), \
                        f"{f.name} 把被否决的方案计入了成绩"
                return


def test_only_canonical_batch_is_active():
    """正式批次唯一。历史批次必须已归档,不能与 F2 混在同一目录里被误汇总。"""
    live = {re.sub(r"_\d+\.json$", "", p.name)[5:]
            for p in gaia.LOOP_DIR.glob("loop_*full7_*.json")}
    assert live == {f"{gaia.BATCH}full7"}, f"三臂对比目录里混有非正式批次: {live}"


def test_all_nonllm_baselines_reported():
    """强基线不得被藏起来:报告必须同时列出三条非 LLM 基线。"""
    txt = gaia.report()
    for k in ("B2", "B4", "BG"):
        assert gaia.label(k) in txt, f"报告漏掉了非 LLM 基线 {k}"
    s = gaia.summary()
    strongest = max(s["nonllm_all"].values(), key=lambda v: v["mean"])
    assert strongest["mean"] <= s["arms"]["full7"]["mean"], \
        "Agent Team 未打赢最强的非 LLM 基线 —— 结论不成立,不是报表问题"


def test_arms_are_balanced():
    """两个智能体臂样本量必须相同,否则均值比较受重复次数混杂。"""
    n = {k: gaia.arm(k)["n"] for k in gaia.ARMS}
    assert len(set(n.values())) == 1, (
        f"臂间样本量不等: {n} —— 批次未跑完时本项必红,这是设计如此:"
        f"样本量不等的均值比较受重复次数混杂,不得据以报数。")


def test_cli_runs():
    """CLI 是论文取数的入口,必须可跑且输出可解析。"""
    r = subprocess.run([sys.executable, str(ROOT / "_code" / "gaia.py"), "--json"],
                       capture_output=True, text=True, cwd=ROOT / "_code", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(r.stdout)
    assert d["batch"] == gaia.BATCH and "tests" in d
