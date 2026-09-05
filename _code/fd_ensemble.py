#!/usr/bin/env python3
"""在**同一批 v2 样本**上跑流动诊断，让对拍口径真正对齐。

## 为什么要这个

此前的对拍是拿「单次基准算例的末帧瞬时流场」去比「跨 3000 个扰动样本的回归」——
两者的注入率、地质乘子、时刻全都不对等，Spearman 只有 +0.465、top1 一致 1/5，
但那个数字**没有解释价值**，因为口径本身就是错的。

本脚本对选定的 v2 样本逐个重跑（deck 打上 `RPTRST ... FLOWS FLORES` 以输出通量），
算出**该样本自己的**分配因子。于是可以做两件此前做不到的事：

  1. **同口径对拍**：用同一批样本的流诊均值 vs 因果回归系数。
  2. **直接检验"冻结流场"假设**：流诊的分配因子随 θ 怎么变？把 α_ij 对 θ 回归，
     得到 ∂(流诊分配)/∂(注入率)，再与因果侧的 ∂产量/∂注入率 比。
     两者若一致，说明冻结流场在这个扰动幅度下够用；若不一致，**分歧幅度就是
     线性化假设的失效程度** —— 这正是我们相对流线唯一能站住的增量。

🔴 v2 分片没存通量场（当时没想到要做流诊），所以必须重跑。单样本约 250 秒。

用法:
    python fd_ensemble.py --n 40 --jobs 12
    python fd_ensemble.py --merge
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
import flow_diagnostics as FD

OUT = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_fd")
THETA = NB.OUT / "theta_all.npy"


def patch_flows(work: Path) -> None:
    """给 deck 打开通量输出。不加这个 UNRST 里没有 FLO*/FLR* 关键字。"""
    d = work / NB.DECK
    s = d.read_text(errors="ignore")
    if "FLOWS" in s.split("RPTRST")[1][:200] if "RPTRST" in s else False:
        return
    s2 = s.replace("BASIC=2 KRO KRW KRG /", "BASIC=2 KRO KRW KRG FLOWS FLORES /", 1)
    if s2 == s:
        raise RuntimeError("没找到 RPTRST 行，无法打开通量输出")
    d.write_text(s2)


def one(j: int, theta: np.ndarray, threads: int) -> dict:
    shard = OUT / "alloc" / f"{j:05d}.json"
    if shard.exists():
        return {"j": j, "status": "cached"}
    work = OUT / "_work" / f"w{j:05d}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    try:
        NB.build(theta, work)
        patch_flows(work)
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{work}:/data", "-w", "/data", NB.IMAGE, "flow", NB.DECK,
               "--output-dir=/data/out", f"--threads-per-process={threads}"]
        with open(work / "run.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
        if rc != 0:
            return {"j": j, "status": "failed", "rc": rc}

        from resdata.grid import Grid
        unrst = work / "out" / "NORNE_ATW2013.UNRST"
        grid = Grid(str(work / "out" / "NORNE_ATW2013.EGRID"))
        flux, step = FD.read_fluxes(unrst, -1)
        src, dst, rate = FD.build_graph(grid, flux)
        wr, qw = FD.well_connection_rates(unrst, step, grid)
        inj = {w: a for w, a in wr.items() if w in FD.INJECTORS and (-a).sum() > 0}
        conc, q, rec = FD.solve_tracers(grid.getNumActive(), src, dst, rate, inj, qw)

        sch = NB.BASE / NB.SCH
        comp = FD.completions(sch, set(FD.PRODUCERS))
        lut = {tuple(int(x) for x in grid.get_ijk(active_index=a)): a
               for a in range(grid.getNumActive())}
        pc = {w: [lut[c] for c in v if c in lut] for w, v in comp.items()}
        alloc = FD.allocation(conc, q, pc)

        q_prod = {w: float(np.maximum(a, 0).sum()) for w, a in wr.items() if w in FD.PRODUCERS}
        q_inj = {w: float(np.maximum(-a, 0).sum()) for w, a in wr.items() if w in FD.INJECTORS}
        inj_frac = {pw: {iw: fr * q_prod.get(pw, 0.0) / q_inj[iw]
                         for iw, fr in by.items() if q_inj.get(iw, 0) > 0}
                    for pw, by in alloc.items()}
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(
            {"j": j, "theta": theta.tolist(), "step": step,
             "injector_fraction": inj_frac, "producer_fraction": alloc,
             "q_prod": q_prod, "q_inj": q_inj, "reconstruction": rec},
            ensure_ascii=False), encoding="utf-8")
        return {"j": j, "status": "ok", "seconds": round(time.time() - t0, 1)}
    except Exception as e:                                   # noqa: BLE001
        return {"j": j, "status": "error", "err": f"{type(e).__name__}: {e}"[:200]}
    finally:
        shutil.rmtree(work, ignore_errors=True)              # 抽完立刻删, 单例约 350 MB


def merge() -> int:
    fs = sorted((OUT / "alloc").glob("*.json"))
    if not fs:
        print("没有分片"); return 1
    recs = [json.loads(f.read_text()) for f in fs]
    ctrl = [f"{w}_{ph}" for w, ph in NB.CONTROLS]
    TH = np.asarray([r["theta"] for r in recs], float)
    A = np.full((len(recs), len(FD.INJECTORS), len(FD.PRODUCERS)), np.nan)
    for k, r in enumerate(recs):
        for pw, by in r["injector_fraction"].items():
            if pw not in FD.PRODUCERS:
                continue
            for iw, v in by.items():
                if iw in FD.INJECTORS:
                    A[k, FD.INJECTORS.index(iw), FD.PRODUCERS.index(pw)] = v
    print(f"样本 {len(recs)}   分配因子张量 {A.shape}   有效元素 {np.isfinite(A).mean():.1%}")
    print(f"值域 [{np.nanmin(A):.4f}, {np.nanmax(A):.4f}]  (应全在 [0,1])")

    # 分配因子随 θ 怎么变 —— 直接检验"冻结流场"假设
    W = np.array([[TH[k, ctrl.index(f"{iw}_WATER")] if f"{iw}_WATER" in ctrl else np.nan
                   for iw in FD.INJECTORS] for k in range(len(recs))])
    sens = np.full((len(FD.INJECTORS), len(FD.PRODUCERS)), np.nan)
    for i in range(len(FD.INJECTORS)):
        if not np.isfinite(W[:, i]).all():
            continue
        x = W[:, i] - W[:, i].mean()
        if x.std() < 1e-9:
            continue
        for jx in range(len(FD.PRODUCERS)):
            y = A[:, i, jx]
            m = np.isfinite(y)
            if m.sum() > max(10, len(recs) // 4) and y[m].std() > 0:
                sens[i, jx] = float(np.polyfit(x[m], y[m], 1)[0])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "ensemble.npz", alloc=A, theta=TH, sens=sens)
    mean = np.nanmean(A, axis=0)
    json.dump({"n": len(recs), "controls": ctrl,
               "mean_injector_fraction": {FD.INJECTORS[i]: {FD.PRODUCERS[j]: float(mean[i, j])
                                          for j in range(len(FD.PRODUCERS)) if np.isfinite(mean[i, j])}
                                          for i in range(len(FD.INJECTORS)) if np.isfinite(mean[i]).any()},
               "note": "sens = d(流诊分配因子)/d(log10 注入乘子), 用于检验冻结流场假设"},
              open(OUT / "ensemble.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {OUT}/ensemble.npz 与 ensemble.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alloc").mkdir(exist_ok=True)
    if args.merge:
        return merge()

    if not THETA.exists():
        print(f"缺 {THETA}"); return 1
    TH = np.load(THETA)
    idx = np.linspace(0, len(TH) - 1, args.n).astype(int)   # 等距抽, 覆盖整个 θ 分布
    print(f"从 v2 的 {len(TH)} 个 θ 里等距抽 {len(idx)} 个重跑(带通量输出)")
    t0, done, bad = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one, int(j), TH[j], args.threads): int(j) for j in idx}
        for f in as_completed(futs):
            r = f.result(); done += 1
            if r["status"] not in ("ok", "cached"):
                bad.append(r)
            if done % 5 == 0 or done == len(idx):
                el = (time.time() - t0) / 3600
                print(f"[{done}/{len(idx)}] 失败 {len(bad)}  已用 {el:.2f}h  "
                      f"预计总 {el/max(done,1)*len(idx):.1f}h", flush=True)
    if bad:
        (OUT / "failures.json").write_text(json.dumps(bad, indent=1, ensure_ascii=False))
        print(f"失败 {len(bad)} 个, 已登记 failures.json")
    return merge()


if __name__ == "__main__":
    raise SystemExit(main())
