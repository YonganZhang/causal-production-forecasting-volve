#!/usr/bin/env python3
"""闭环 G0（无知识库）—— 闭环盖亚的对照臂。

§6.1「一次实验只切一个变量」：本文件**不复制任何编排逻辑**，而是直接 import
`fc_loop`（闭环盖亚的实现），只把输出落点改成 G0 专属，避免两臂互相覆盖。

    唯一变量 = --arm 决定读哪个提示词
        闭环盖亚 : _pipelines/fc_agent/brief_G2.txt  (L1 通用知识 + L2 本油藏诊断事实)
        闭环 G0  : _pipelines/fc_agent/brief_G0.txt  (无任何知识库，只有任务描述)

    完全相同的部分：LOOP_NOTE 反馈模板、注水量标量校正器 solve_s/predict_water、
    cap 常数（**共用同一个 water_cap.npy**）、OPM 预算、终止判据、裁判(OPM Flow)、
    NPV 口径(fc_npv.BRENT 的 EIA 真实历史 Brent)、LLM 调用方式与隔离方式。

被改掉的只有落点：
    fc_loop.OUT   → _pipelines/fc_loop_G0/   (prompt_r*.txt / ans_r*.json / loop.log)
    _dump 的路径  → _pipelines/fc_agent/loop_G0.json   (原实现里硬编码成 loop_G2.json)

用法:
    python fc_loop_g0.py --budget 5 --threads 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import fc_loop as FL

ROOT = Path(__file__).resolve().parent.parent
G0OUT = ROOT / "_pipelines" / "fc_loop_G0"
TARGET_S1 = ROOT / "_pipelines" / "fc_agent" / "loop_G0.json"
TARGET = TARGET_S1        # 由 main() 按 --seed-tag 改写
SHARED_CAP = ROOT / "_pipelines" / "fc_loop" / "water_cap.npy"

_orig_dump = FL._dump


def _dump_g0(path, a, base, hist, used):
    """强制写到 G0 的落点，并补上可追溯的出处字段。"""
    _orig_dump(TARGET, a, base, hist, used)
    j = json.loads(TARGET.read_text(encoding="utf-8"))
    j["experiment"] = "闭环 G0（无知识库）—— 闭环盖亚的对照臂"
    j["single_variable_vs_gaia_loop"] = (
        "唯一变量 = 提示词里的知识库。G0 用 brief_G0.txt（无知识库），"
        "闭环盖亚用 brief_G2.txt（L1 通用 + L2 本油藏诊断事实）。"
        "编排模板、注水量校正器、cap 常数、OPM 预算、终止判据、裁判、NPV 口径全部相同。")
    j["code"] = {"driver": "_code/fc_loop.py (import 复用，未改一行逻辑)",
                 "wrapper": "_code/fc_loop_g0.py",
                 "fc_loop_sha256": hashlib.sha256(
                     (ROOT / "_code" / "fc_loop.py").read_bytes()).hexdigest(),
                 "water_cap_npy": str(FL.CAPF)}
    j["brief"] = {"file": "_pipelines/fc_agent/brief_G0.txt",
                  "chars": len((ROOT / "_pipelines" / "fc_agent" /
                                f"brief_{a.arm}.txt").read_text(encoding="utf-8"))}
    j["sim_dir"] = str(FL.SIMDIR)
    j["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    TARGET.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    (G0OUT / TARGET.name).write_text(json.dumps(j, ensure_ascii=False, indent=1),
                                        encoding="utf-8")


def main() -> int:
    """直接借用 fc_loop 自己的 argparse —— 参数与默认值必须与闭环盖亚逐字相同,
    所以不在这里重新声明一遍(重声明会漂移,已经踩过一次: 漏了 --max-cal)。"""
    import sys
    argv = ["run", "--arm", "G0"] + sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fitcap", "check", "run"])
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--max-cal", type=int, default=2)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--llm-timeout", type=int, default=900)
    ap.add_argument("--seed-tag", default="s1")
    a = ap.parse_args(argv)
    global TARGET
    TARGET = (TARGET_S1 if a.seed_tag == "s1"
              else ROOT / "_pipelines" / "fc_agent" / f"loop_G0_{a.seed_tag}.json")
    a.arm = "G0"
    G0OUT.mkdir(parents=True, exist_ok=True)
    FL.OUT = G0OUT
    # 共用闭环盖亚拟合的同一份 cap（约束模型必须两臂一致，否则就不止一个变量了）
    FL.CAPF = SHARED_CAP if SHARED_CAP.exists() else G0OUT / "water_cap.npy"
    FL._dump = _dump_g0
    return FL.run(a)


if __name__ == "__main__":
    raise SystemExit(main())
