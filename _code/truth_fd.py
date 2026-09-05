#!/usr/bin/env python3
"""真值实验：直接测「改注入率，产量变多少」，用它裁定流诊与因果谁更接近。

## 为什么必须做

流动诊断与因果回归在同口径下仍然不一致（Spearman +0.452、top1 0/6）。
但**对拍不能裁定谁对** —— 两者算的本来就不是同一个量。来回三轮都在口径上打转，
唯一能终结的办法是拿真值。

真值定义得毫不含糊：**同一地质实现**上跑三次（基准 / 注入率 ×(1+ε) / ×(1−ε)），
中心差分给出

    dq_j/dw_i = [q_j(w_i·(1+ε)) − q_j(w_i·(1−ε))] / [w_i(1+ε) − w_i(1−ε)]

分母用**实测**注入率（WWIR），不用 deck 目标率 —— 后者会被 600 bar BHP 上限截断。
成对扰动共用同一 θ_perm，所以地质、初始状态、其余井控完全一致，差分里只剩目标井。

## 三方对照

| | 是什么 | 本脚本怎么比 |
|---|---|---|
| **真值** | 中心差分 dq/dw | 基准 |
| 因果回归 | 跨集合的 OLS 斜率 dq/dw | 与真值同量纲，直接比 |
| 流动诊断 | 当前流场里 j 的液量有多少来自 i（份额→注入井口径） | 量纲相同，但含义不同 |

🔴 诚实边界：中心差分测的是**当前工况附近的局部导数**。因果回归的 OLS 系数是整个
   随机化分布（θ~N(0,0.35)，即 ±2.2 倍）上的最佳线性投影，两者只在响应近线性时相等。
   若真值与因果回归不符，先查是不是这个原因，别急着说谁错。

用法:
    python truth_fd.py --eps 0.2 --wells F-1H F-2H C-2H --jobs 9
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import norne_bulk as NB

OUT = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_truth")
PRODUCERS = NB.PRODUCERS
N_TIMES = NB.N_TIMES
DAY_GRID = NB.DAY_GRID


def run_case(tag: str, theta: np.ndarray, threads: int) -> dict | None:
    """跑一个算例并抽出末期(与场同刻)的产液率与实测注入率。"""
    shard = OUT / "cases" / f"{tag}.npz"
    if shard.exists():
        d = np.load(shard)
        return {k: d[k] for k in d.files}
    work = OUT / "_work" / tag
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    try:
        NB.build(theta, work)
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{work}:/data", "-w", "/data", NB.IMAGE, "flow", NB.DECK,
               "--output-dir=/data/out", f"--threads-per-process={threads}"]
        with open(work / "run.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
        if rc != 0:
            print(f"  {tag}: flow rc={rc}"); return None
        got = NB.harvest(work)
        if got is None:
            print(f"  {tag}: harvest 失败"); return None
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, theta=theta.astype(np.float32), **got)
        return got
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.2, help="相对扰动幅度, ±eps")
    ap.add_argument("--wells", nargs="+", default=["F-1H", "F-2H", "F-3H", "C-2H"])
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)

    ctrl = [f"{w}_{ph}" for w, ph in NB.CONTROLS]
    # 地质固定为基准(θ_perm=0), 这样差分里只剩目标井的注入率
    base = np.zeros(NB.N_THETA)
    jobs = [("base", base.copy())]
    for w in args.wells:
        key = f"{w}_WATER"
        if key not in ctrl:
            print(f"⚠ {key} 不在控制变量里, 跳过"); continue
        k = ctrl.index(key)
        for sgn, nm in ((+1, "up"), (-1, "dn")):
            th = base.copy()
            th[k] = np.log10(1.0 + sgn * args.eps)      # θ 是 log10 乘子
            jobs.append((f"{w}_{nm}", th))
    print(f"共 {len(jobs)} 个算例(1 基准 + {len(jobs)-1} 扰动), ±{args.eps:.0%}, {args.jobs} 并发")

    res, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_case, tag, th, args.threads): tag for tag, th in jobs}
        for f in as_completed(futs):
            tag = futs[f]
            res[tag] = f.result()
            print(f"  完成 {tag}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    if res.get("base") is None:
        print("🔴 基准算例失败"); return 1

    jt = int(np.argmin(np.abs(DAY_GRID - 3260.0)))       # 与场快照同刻
    def liq(g):
        return np.array([g["obs"][(i * 3 + 0) * N_TIMES + jt] + g["obs"][(i * 3 + 1) * N_TIMES + jt]
                         for i in range(len(PRODUCERS))])
    def winj(g, k):
        return float(g["inj_actual"][k, jt])

    truth = {}
    print(f"\n=== 真值:中心差分 dq_liq/dw_inj (第 {DAY_GRID[jt]:.0f} 天) ===")
    for w in args.wells:
        key = f"{w}_WATER"
        if key not in ctrl:
            continue
        k = ctrl.index(key)
        up, dn = res.get(f"{w}_up"), res.get(f"{w}_dn")
        if up is None or dn is None:
            print(f"  {w}: 扰动算例缺失, 跳过"); continue
        dw = winj(up, k) - winj(dn, k)
        if abs(dw) < 1e-9:
            print(f"  {w}: 实测注入率没变(dw={dw:.3e}) —— 可能被 BHP 上限完全截断"); continue
        dq = liq(up) - liq(dn)
        g = dq / dw
        truth[w] = g
        top = int(np.argmax(np.abs(g)))
        print(f"  {w:6s} dw={dw:8.1f} m3/d   最强响应 {PRODUCERS[top]:8s} {g[top]:+.4f}"
              f"   Σ|g|={np.abs(g).sum():.4f}")

    if not truth:
        print("没有可用真值"); return 1

    # ---- 三方对照 ----
    import flow_diagnostics as FD
    fdj = Path(__file__).resolve().parent.parent / "_pipelines" / "flow_diagnostics" / "allocation.json"
    A_fd = {}
    if fdj.exists():
        jd = json.loads(fdj.read_text())
        for pw, by in jd.get("injector_fraction", {}).items():
            for iw, v in by.items():
                A_fd.setdefault(iw, {})[pw] = v

    cbf = Path(__file__).resolve().parent.parent / "_pipelines" / "crm_bridge" / "crm_allocation_f.npy"
    A_cf = np.load(cbf) if cbf.exists() else None

    from scipy.stats import spearmanr
    print(f"\n=== 三方对照(与真值的相关) ===")
    print(f"{'注入井':8s}{'真值top1':>10s}{'流诊top1':>10s}{'因果top1':>10s}"
          f"{'流诊r':>9s}{'因果r':>9s}")
    rows = []
    for w, g in truth.items():
        fd_v = np.array([A_fd.get(w, {}).get(p, np.nan) for p in PRODUCERS])
        cf_v = np.full(len(PRODUCERS), np.nan)
        if A_cf is not None and f"{w}_WATER" in ctrl:
            cf_v = A_cf[ctrl.index(f"{w}_WATER")]
        m1 = np.isfinite(fd_v) & np.isfinite(g)
        m2 = np.isfinite(cf_v) & np.isfinite(g)
        r1 = float(spearmanr(fd_v[m1], g[m1]).statistic) if m1.sum() > 3 else float("nan")
        r2 = float(spearmanr(cf_v[m2], g[m2]).statistic) if m2.sum() > 3 else float("nan")
        t_t = PRODUCERS[int(np.argmax(np.abs(g)))]
        t_f = PRODUCERS[int(np.nanargmax(fd_v))] if m1.any() else "-"
        t_c = PRODUCERS[int(np.nanargmax(cf_v))] if m2.any() else "-"
        rows.append({"well": w, "truth_top": t_t, "fd_top": t_f, "causal_top": t_c,
                     "spearman_fd": r1, "spearman_causal": r2,
                     "truth": {PRODUCERS[i]: float(g[i]) for i in range(len(PRODUCERS))}})
        print(f"{w:8s}{t_t:>10s}{t_f:>10s}{t_c:>10s}{r1:>+9.3f}{r2:>+9.3f}")

    ok_fd = np.nanmean([r["spearman_fd"] for r in rows])
    ok_cf = np.nanmean([r["spearman_causal"] for r in rows])
    print(f"\n平均 Spearman: 流诊 {ok_fd:+.3f}   因果 {ok_cf:+.3f}")
    print(f"top1 命中: 流诊 {sum(r['fd_top']==r['truth_top'] for r in rows)}/{len(rows)}"
          f"   因果 {sum(r['causal_top']==r['truth_top'] for r in rows)}/{len(rows)}")

    dst = Path(__file__).resolve().parent.parent / "_pipelines" / "truth_fd"
    dst.mkdir(parents=True, exist_ok=True)
    json.dump({"eps": args.eps, "day": float(DAY_GRID[jt]), "rows": rows,
               "mean_spearman_fd": float(ok_fd), "mean_spearman_causal": float(ok_cf),
               "caveat": ("中心差分是**局部**导数; 因果回归是 θ~N(0,0.35)(±2.2倍)全分布上的"
                          "最佳线性投影。不符时先查是不是这个原因, 不要直接判谁错。")},
              open(dst / "truth.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
