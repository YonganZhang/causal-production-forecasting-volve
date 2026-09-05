#!/usr/bin/env python3
"""A. 模拟器噪声地板：0.88% 这个效应到底测不测得出来？

## 为什么必须做

主线裁定「只顾眼前的代价 = +0.88%」。但我**从没量过 OPM Flow 的 run-to-run 波动**。
若同一方案重复跑的波动就有 ±0.5%,那 0.88% 不可宣称。

## 两个来源分开量

1. **数值/线程噪声**:同一个 deck、不同 `--threads-per-process`。
   并行归约顺序变了,浮点舍入就变 —— 这是模拟器自身的不确定度。
2. **条件数(灵敏度)**:θ 加极小扰动(相对 1e-3),看产油变多少。
   若微小扰动就引起 ~1% 变化,说明响应面陡峭,优化结果本身不稳。

用法:
    python fc_noise.py --threads-list 4 8 16 24 --eps 1e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import forecast_gen as FG
import fc_decide as FD

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_decide"
DAYS = FG.FC_GRID


class _A:                                    # _run_sim 需要的最小参数对象
    def __init__(self, threads):
        self.threads = threads


def oil_of(key):
    d = np.load(OUT / "sim" / f"{key}.npz")
    ob = d["obs"]
    import norne_bulk as NB
    n = len(NB.PRODUCERS); T = FG.FC_N
    oil = np.stack([ob[(i*3)*T:(i*3+1)*T] for i in range(n)])
    return float(np.trapezoid(oil, DAYS, axis=-1).sum()), \
           float(d["field_cum"][1][-1] - d["field_cum"][1][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads-list", type=int, nargs="+", default=[4, 8, 16, 24])
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--n-eps", type=int, default=3)
    a = ap.parse_args()
    pj = json.loads((OUT / "plans_calibrated.json").read_text())
    th0 = np.asarray(pj["plans"]["long"], dtype=np.float32).reshape(FG.N_STAGE, len(FG.INJ_W))

    print("=== 1. 数值/线程噪声(同一 deck,不同线程数) ===")
    vals = []
    for t in a.threads_list:
        key = f"noise_t{t}"
        if not (OUT / "sim" / f"{key}.npz").exists():
            FD._run_sim(key, th0, _A(t))
        o, w = oil_of(key)
        vals.append(o)
        print(f"  threads={t:<3d}  全期产油 {o:,.2f}   注水 {w:,.0f}", flush=True)
    v = np.array(vals)
    rel = (v.max() - v.min()) / v.mean()
    print(f"\n  极差/均值 = {rel:.3%}   标准差/均值 = {v.std(ddof=1)/v.mean():.3%}")
    print(f"  → 数值噪声地板 ≈ {rel:.3%}"
          f"  {'🔴 与 0.88% 同量级,效应不可靠' if rel > 0.004 else '✔ 远小于 0.88%,效应可分辨'}")

    print(f"\n=== 2. 条件数:θ 加 ±{a.eps:.0e} 相对扰动 ===")
    pv = []
    rng = np.random.default_rng(0)
    for k in range(a.n_eps):
        d = rng.normal(0, a.eps, th0.shape).astype(np.float32)
        key = f"noise_eps{k}"
        if not (OUT / "sim" / f"{key}.npz").exists():
            FD._run_sim(key, th0 + d, _A(24))
        o, _ = oil_of(key)
        pv.append(o)
        print(f"  扰动 {k}  全期产油 {o:,.2f}   相对基准 {o/v.mean()-1:+.4%}", flush=True)
    pv = np.array(pv)
    sens = (pv.max() - pv.min()) / pv.mean()
    print(f"\n  θ 扰动 {a.eps:.0e} 引起的产油极差 = {sens:.3%}")
    print(f"  → {'🔴 响应面陡峭,优化结果不稳' if sens > 0.004 else '✔ 响应面平滑'}")

    json.dump({"threads": a.threads_list, "oil_by_threads": vals,
               "numeric_noise_range_rel": float(rel),
               "numeric_noise_sd_rel": float(v.std(ddof=1)/v.mean()),
               "eps": a.eps, "oil_perturbed": pv.tolist(),
               "sensitivity_range_rel": float(sens),
               "headline_effect": 0.0088,
               "verdict": ("效应可分辨" if max(rel, sens) < 0.004 else "🔴 效应落在噪声内")},
              open(OUT / "noise.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/noise.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
