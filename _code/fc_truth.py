"""模拟结果的**唯一**读取口径。

🔴 类层修复 #1「同一个量有多套算法」(2026-09-05 Workflow 审计):
   审计在 45 条发现里查出至少 5 条同源问题 ——
     · "实测注水量"在 6 个文件里用 np.trapezoid(inj_actual)，
       与模拟器自报 FWIT 平均差 0.99pp、最大 3.12pp，
       按 ±25% 判可行性有 **14 例结论相反**;
     · 代理排序分与最终裁定用两套目标函数(平价不折现 vs 逐年 Brent+8% 折现);
     · npv8 这个 key 同时被 fc_fix.load()(仅油收入) 与 fc_water_econ.econ()(已扣水成本) 使用，
       基准值分别是 2,856.8M 与 1,773.4M;
     · 提示词里硬编码基准产油 7,469,000，而真值是 7,493,840(差 0.33%)，已导致审计员误报;
     · 提示词里硬编码水价 2.0/1.0，绕过 --c-inj/--c-prod。

   根因不是"某处写错了"，是**没有单一真源**:同一个概念在多处各自实现，
   于是每加一处调用就多一次漂移机会。本模块把这些量收成唯一入口，
   并由 _tests/test_truth.py 断言项目里不存在第二种算法。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIMDIR = ROOT / "_pipelines" / "fc_decide" / "sim"


def _fc(npz) -> np.ndarray:
    d = npz if isinstance(npz, np.lib.npyio.NpzFile) else np.load(npz)
    return d["field_cum"]


def measured_winj(npz) -> float:
    """实测累计注水量。**唯一口径 = 模拟器自报 FWIT**，禁止梯形积分。"""
    fc = _fc(npz)
    return float(fc[1][-1] - fc[1][0])


def measured_oil(npz) -> float:
    """实测累计产油量。唯一口径 = 模拟器自报 FOPT。"""
    fc = _fc(npz)
    return float(fc[0][-1] - fc[0][0])


def measured_wprod(npz) -> float:
    """实测累计采出水量。唯一口径 = 模拟器自报 FWPT。"""
    fc = _fc(npz)
    return float(fc[2][-1] - fc[2][0])


# 基准值:全项目唯一来源，不得在任何提示词或脚本里硬编码
BASELINE_NPZ = SIMDIR / "baseline.npz"


def baseline() -> dict:
    """基准算例的实测量。提示词与脚本一律从这里取，不得写死数字。"""
    return {"oil": measured_oil(BASELINE_NPZ),
            "winj": measured_winj(BASELINE_NPZ),
            "wprod": measured_wprod(BASELINE_NPZ)}
