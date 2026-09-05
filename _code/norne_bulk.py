#!/usr/bin/env python3
"""Norne 大规模标签工厂：边跑边抽、抽完删原始文件。

为什么要重写而不是复用 norne_sampler.py：
  旧版是"跑完 300 个再统一打包"，于是 300 个 case 就占了 104 GB。
  10,000 个样本按旧法要 3 TB —— 这台是共享机器，不能这么占。
  本脚本每跑完一个 case 就立刻抽出 (obs 2640 维 + 8 快照 × 2 场)，
  写成 1.4 MB 的分片，然后**删掉 350 MB 的原始输出**。10,000 个只要 14 GB。

分片落盘而不是最后一次性 np.save：中途挂掉也不丢已完成的样本，可断点续跑。

用法:
    python norne_bulk.py --n 10000 --jobs 32 --threads 4
    python norne_bulk.py --merge          # 把分片合成训练张量
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import paths as P

BASE = P.NORNE_DECK
# 🔴 schema v2 换了 theta 语义(注入相态分离), 与 v1 分片**不可混用**。
#    换目录而不是原地追加, 否则 --merge 会把两种语义的 theta 静默拼在一起。
SCHEMA = 2
OUT = P.NORNE_BULK_V2
OUT_V1 = P.NORNE_BULK                                    # 旧数据, 只读
IMAGE = "openporousmedia/opmreleases:latest"
DECK = "NORNE_ATW2013.DATA"
SCH = "INCLUDE/BC0407_HIST01122006.SCH"

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")
PERM_REGIONS = (1, 2, 3, 4)
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")
OBS_KEYS = ("WOPR", "WWPR", "WBHP")
FIELDS = ("PRESSURE", "SWAT")
N_TIMES, N_SNAP = 40, 8
DAY_GRID = np.linspace(1.0, 3312.0, N_TIMES)
# 🔴 场快照也必须按**天数**取, 不能按重启步序号。参考算例 65 个重启步落在
#    第 0~3260 天上, 间隔极不均匀:按序号等距抽出来的 8 个快照实测落在
#    [0, 424, 1272, 1425, 2105, 2265, 2726, 3260] 天, 相邻间隔从 153 天到 848 天,
#    差 5.5 倍。这与此前修过的 obs 时间轴 bug 是同一类 —— 当时只改了 obs 没改 fields。
#    另外 UNRST 末帧是第 3260 天而 obs 末点是第 3312 天, 直接拿来配对会差 118 天。
SNAP_DAYS = np.linspace(0.0, 3260.0, N_SNAP)
INJ_RATE_KEYS = {"WATER": "WWIR", "GAS": "WGIR"}     # 实测注入率, 用来核对处理变量


def scan_injection_controls() -> tuple[tuple[str, str], ...]:
    """从 SCHEDULE 扫出每口注入井实际用到的**相态**, 返回 (井, 相态) 控制变量清单。

    🔴 v1 的致命 bug 就在这:build() 当年只按 `'井名' in line and 'RATE' in line` 匹配,
       不看相态, 于是把同一口井的注水记录和注气记录乘以了**同一个乘子**。
       实测 9 口"注水井"里有 4 口是水气混注(C-1H 133W/92G、C-3H 82W/114G、
       C-4AH 44W/20G、C-4H 53W/94G), 而 C-3H 末期水率是 0、气率 143344 ——
       它到历史末段其实是注气井。于是 ∂SWAT/∂θ 被当成"注水波及"讲, 物理解释是错的。
       现在按 (井, 相态) 分开控制, 每个相态一个独立随机的 θ。
    """
    sch = BASE / SCH
    if not sch.exists():
        # 🔴 别在 import 期崩。发布包里没有 deck(ODbL 上游自取), 但下游只是想读数据、
        #    跑分析、看图 —— 这时用打包时冻结的控制清单就够了。真要构造 deck 时,
        #    build() 会通过 P.require 明确报错并告诉对方怎么拿 deck。
        if P.CONTROLS_JSON.exists():
            return tuple(tuple(c) for c in json.loads(P.CONTROLS_JSON.read_text())["controls"])
        raise FileNotFoundError(
            f"既找不到 deck({sch}) 也找不到 {P.CONTROLS_JSON}\n  {P.DECK_HOWTO}")
    txt = sch.read_text(errors="ignore")
    seen: dict[tuple[str, str], int] = {}
    for m in re.finditer(r"^WCONINJE(.*?)^/\s*$", txt, re.S | re.M):
        for ln in m.group(1).splitlines():
            ln = ln.split("--")[0].strip()
            if not ln or ln == "/":
                continue
            t = ln.replace("/", " ").split()
            if len(t) < 2:
                continue
            w, ph = t[0].strip("'"), t[1].strip("'").upper()
            if w in INJECTORS and ph in INJ_RATE_KEYS:
                seen[(w, ph)] = seen.get((w, ph), 0) + 1
    return tuple(sorted(seen, key=lambda k: (INJECTORS.index(k[0]), k[1])))


CONTROLS = scan_injection_controls()
N_THETA = len(CONTROLS) + len(PERM_REGIONS)


def theta_names() -> list[str]:
    return [f"inj_{w}_{ph}" for w, ph in CONTROLS] + [f"perm_fluxnum{r}" for r in PERM_REGIONS]


def build(theta: np.ndarray, work: Path) -> None:
    P.require(BASE, "Norne deck", P.DECK_HOWTO)
    shutil.copytree(BASE, work, ignore=shutil.ignore_patterns(
        "out", "*.UNRST", "*.UNSMRY", "*.EGRID", "*.INIT", "*.PRT", "*.DBG", "*.ESMRY"))
    m = 10.0 ** np.asarray(theta, float)
    deck = work / DECK
    L = deck.read_text(errors="ignore").split("\n")
    i = next(k for k, l in enumerate(L) if l.strip().upper().startswith("EDIT"))
    blk = ["-- [bulk] 渗透率分区乘子", "MULTIREG"]
    for j, r in enumerate(PERM_REGIONS):
        v = m[len(CONTROLS) + j]
        blk += [f"  {kw}  {v:.6f}  {r}  F /" for kw in ("PERMX", "PERMY", "PERMZ")]
    deck.write_text("\n".join(L[:i] + blk + ["/", ""] + L[i:]))

    sch = work / SCH
    T = sch.read_text(errors="ignore").split("\n")
    mult = {c: m[k] for k, c in enumerate(CONTROLS)}
    hits = 0
    for k, l in enumerate(T):
        body = l.split("--")[0]
        if "'RATE'" not in body:
            continue
        t = body.replace("/", " ").split()
        if len(t) < 2:
            continue
        # 🔴 必须同时匹配井名**和相态**。只匹配井名会把注水和注气一起乘 —— 见
        #    scan_injection_controls() 的说明, 这是 v1 最严重的错误。
        key = (t[0].strip("'"), t[1].strip("'").upper())
        if key not in mult:
            continue
        g = re.search(r"'RATE'\s+([\d.eE+-]+)", body)
        if g:
            T[k] = l.replace(g.group(0), f"'RATE' {float(g.group(1)) * mult[key]:.4f}")
            hits += 1
    if hits == 0:
        raise RuntimeError("没有任何 WCONINJE 记录被改写 —— 相态匹配逻辑坏了, 不能静默跑成基线")
    sch.write_text("\n".join(T))
    (work / "out").mkdir(exist_ok=True)


def harvest(work: Path) -> dict | None:
    """抽出 obs / fields / **实测注入率** / 场快照的时间戳。抽完由调用方删原始文件。

    相对 v1 的三处修正:
      · fields 从 float16 改 float32 —— float16 的量化会把小的 SWAT 变化抹平,
        v1 实测 15,789/44,431 (35.5%) 个单元在全部样本上 SWAT 完全相同, 斜率恒为 0。
      · 存 WWIR/WGIR **实测**注入率 —— θ 缩放的是 WCONINJE 的**目标**率, 但 deck 里
        1463 条注入记录带 600 bar 的 BHP 上限, 顶到上限后实际注入量会被截断。
        不存实测量就永远无法核查处理变量是否名副其实。
      · 存 field_days / n_restart —— v1 只按重启步序号取 [-1], 分片里没有时间戳,
        无法证明各样本的"末帧"是同一个模拟时刻。
    """
    from resdata.resfile import ResdataFile
    from resdata.summary import Summary

    smry = work / "out" / "NORNE_ATW2013.UNSMRY"
    unrst = work / "out" / "NORNE_ATW2013.UNRST"
    if not smry.exists() or not unrst.exists():
        return None
    s = Summary(str(smry))
    days = np.asarray(s.numpy_vector("TIME"), float)

    def series(key: str) -> np.ndarray:
        try:
            v = np.asarray(s.numpy_vector(key), float)
        except Exception:                                    # noqa: BLE001  该 key 不存在
            return np.full(N_TIMES, np.nan)
        mk = np.isfinite(days) & np.isfinite(v)
        return (np.interp(DAY_GRID, days[mk], v[mk]) if mk.sum() >= 2
                else np.full(N_TIMES, np.nan))

    cols = [series(f"{k}:{w}") for w in PRODUCERS for k in OBS_KEYS]
    inj = np.stack([series(f"{INJ_RATE_KEYS[ph]}:{w}") for w, ph in CONTROLS])

    f = ResdataFile(str(unrst))
    n = f.num_named_kw(FIELDS[0])
    if n < N_SNAP:
        return None
    rd = np.asarray([(d - f.dates[0]).total_seconds() / 86400.0 for d in f.dates], np.float64)
    if len(rd) != n:
        return None
    if rd[-1] < SNAP_DAYS[-1] - 1.0:
        return None            # 跑早停了, 末帧不在约定时刻上 —— 拒收而不是外推
    # 逐单元沿时间插值到固定的 SNAP_DAYS, 保证**所有样本的第 k 帧是同一个模拟时刻**
    # 插值权重只依赖时间轴, 与单元无关 —— 算一次, 对 44431 个单元一起用。
    # (逐单元 np.interp 要循环 4.4 万次 × 1 万个样本, 慢到不可接受。)
    j = np.clip(np.searchsorted(rd, SNAP_DAYS, side="right") - 1, 0, n - 2)
    span = np.where(rd[j + 1] - rd[j] > 0, rd[j + 1] - rd[j], 1.0)
    w = np.clip((SNAP_DAYS - rd[j]) / span, 0.0, 1.0)[:, None]     # (N_SNAP, 1)
    # 只读夹住各目标时刻的那两个重启步。读全部 65 步实测 5.5s/场, 只读需要的 16 步是
    # 0.7s —— 每样本省约 8s。(旧版为省事读全量, 是我引入的性能退化。)
    need = sorted(set(j.tolist()) | set((j + 1).tolist()))
    pos = {g: r for r, g in enumerate(need)}
    lo = np.asarray([pos[g] for g in j]); hi = np.asarray([pos[g] for g in (j + 1)])
    stacks = []
    for k in FIELDS:
        cube = np.stack([np.asarray(f[k][i], np.float32) for i in need])       # (len(need), ncell)
        stacks.append((cube[lo] * (1.0 - w) + cube[hi] * w).astype(np.float32))
    fl = np.stack(stacks, axis=1)                            # (N_SNAP, n_field, ncell)
    fdays = SNAP_DAYS.copy()
    return {"obs": np.concatenate(cols).astype(np.float32),
            "fields": fl.astype(np.float32),
            "inj_actual": inj.astype(np.float32),
            "field_days": fdays,
            "restart_days_last": np.float64(rd[-1]),
            "n_restart": np.int32(n),
            "sim_days_end": np.float64(days[np.isfinite(days)].max() if np.isfinite(days).any() else np.nan)}


def one(j: int, theta: np.ndarray, threads: int, keep_raw: bool) -> dict:
    work = OUT / "_work" / f"w{j:05d}"
    shard = OUT / "shards" / f"{j:05d}.npz"
    if shard.exists():
        return {"j": j, "status": "cached"}
    if work.exists():
        shutil.rmtree(work)
    t0 = time.time()
    try:
        build(theta, work)
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{work}:/data", "-w", "/data", IMAGE, "flow", DECK,
               "--output-dir=/data/out", f"--threads-per-process={threads}"]
        with open(work / "run.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
        # 🔴 v1 不看 rc 就 harvest, 于是"模拟器报错但写出了 ≥8 个重启步"的样本
        #    会被当成正常样本混进训练集。收敛失败的样本物理上是错的, 必须拒绝。
        if rc != 0:
            return {"j": j, "status": "failed", "rc": rc, "why": "flow 返回非零",
                    "seconds": round(time.time() - t0, 1)}
        got = harvest(work)
        if got is None:
            return {"j": j, "status": "failed", "rc": rc, "why": "harvest 拿不到输出",
                    "seconds": round(time.time() - t0, 1)}
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, theta=theta.astype(np.float32), rc=np.int32(rc),
                            schema=np.int32(SCHEMA), **got)
        return {"j": j, "status": "ok", "rc": rc, "seconds": round(time.time() - t0, 1)}
    except Exception as e:                                   # noqa: BLE001
        return {"j": j, "status": "error", "err": f"{type(e).__name__}: {e}"[:200]}
    finally:
        if not keep_raw and work.exists():
            shutil.rmtree(work, ignore_errors=True)          # 🔴 抽完立刻删 350 MB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--sigma-inj", type=float, default=0.35)
    ap.add_argument("--sigma-perm", type=float, default=0.40)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shards").mkdir(exist_ok=True)

    if args.merge:
        sh = sorted((OUT / "shards").glob("*.npz"))
        TH, OB, FL, IJ, FD = [], [], [], [], []
        for p in sh:
            d = np.load(p)
            got = int(d["schema"]) if "schema" in d else 1
            if got != SCHEMA:
                # 🔴 v1 与 v2 的 theta 语义不同(v1 未分相态), 混在一起会静默出错
                raise SystemExit(f"{p.name} 是 schema v{got}, 本脚本是 v{SCHEMA}; 不混用, 请分开处理")
            if int(d["rc"]) != 0:
                raise SystemExit(f"{p.name} 的 rc={int(d['rc'])} 却落了盘 —— 生成端把关漏了")
            TH.append(d["theta"]); OB.append(d["obs"]); FL.append(d["fields"])
            IJ.append(d["inj_actual"]); FD.append(d["field_days"])
        TH, OB, FL = np.stack(TH), np.stack(OB), np.stack(FL)
        IJ, FD = np.stack(IJ), np.stack(FD)

        # 断言各样本的场快照落在同一组模拟时刻 —— 之前踩过时间轴错位的坑, 这次显式挡住
        spread = float(np.nanmax(FD.max(0) - FD.min(0))) if np.isfinite(FD).any() else float("nan")
        if np.isfinite(spread) and spread > 1.0:
            print(f"⚠ 场快照时间跨样本最大相差 {spread:.1f} 天 —— 不是同一时刻, 下游按天数插值再用")
        # 处理变量是否名副其实:θ(目标率乘子) 与实测注入率的相关性
        with np.errstate(invalid="ignore"):
            tot = np.nanmean(IJ, axis=2)                       # (n, n_ctrl) 各控制的全期平均实测率
        corr = [float(np.corrcoef(TH[:, k], tot[:, k])[0, 1]) if np.isfinite(tot[:, k]).all()
                and tot[:, k].std() > 0 else float("nan") for k in range(len(CONTROLS))]
        weak = [f"{CONTROLS[k][0]}/{CONTROLS[k][1]}={corr[k]:.2f}" for k in range(len(CONTROLS))
                if np.isfinite(corr[k]) and corr[k] < 0.8]
        if weak:
            print(f"⚠ θ 与实测注入率相关性偏低(可能被 BHP 上限截断): {', '.join(weak)}")

        pk = OUT / "packed"; pk.mkdir(exist_ok=True)
        np.save(pk / "theta.npy", TH); np.save(pk / "obs.npy", OB); np.save(pk / "fields.npy", FL)
        np.save(pk / "inj_actual.npy", IJ); np.save(pk / "field_days.npy", FD)
        (pk / "meta.json").write_text(json.dumps({
            "schema": SCHEMA, "n": len(TH), "theta_names": theta_names(),
            "controls": [list(c) for c in CONTROLS],
            "theta_shape": list(TH.shape), "obs_shape": list(OB.shape),
            "fields_shape": list(FL.shape), "field_names": list(FIELDS),
            "case_ids": [p.stem for p in sh],
            "obs_nan_frac": float(np.isnan(OB).mean()),
            "theta_range": [float(TH.min()), float(TH.max())],
            "field_days_median": [float(x) for x in np.nanmedian(FD, axis=0)],
            "field_days_max_spread": spread,
            "theta_vs_actual_inj_corr": corr,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"合并 {len(TH)} 个样本  theta{TH.shape} obs{OB.shape} fields{FL.shape} "
              f"({FL.nbytes/1e9:.2f} GB)  场快照时刻最大离散 {spread:.2f} 天")
        return 0

    rng = np.random.default_rng(args.seed)
    TH = np.empty((args.n, N_THETA), np.float32)
    TH[:, :len(CONTROLS)] = rng.normal(0, args.sigma_inj, (args.n, len(CONTROLS)))
    TH[:, len(CONTROLS):] = rng.normal(0, args.sigma_perm, (args.n, len(PERM_REGIONS)))
    np.save(OUT / "theta_all.npy", TH)
    (OUT / "config.json").write_text(json.dumps(vars(args) | {
        "theta_names": theta_names(), "n_theta": N_THETA}, indent=1), encoding="utf-8")

    t0, done, bad = time.time(), 0, []
    log = (OUT / "progress.log").open("a")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, j, TH[j], args.threads, args.keep_raw) for j in range(args.n)]
        for fu in as_completed(futs):
            r = fu.result()
            done += r["status"] in ("ok", "cached")
            if r["status"] not in ("ok", "cached"):
                bad.append(r)
            if done % 25 == 0 or r["status"] not in ("ok", "cached"):
                el = time.time() - t0
                msg = (f"[{done}/{args.n}] 失败 {len(bad)}  已用 {el/3600:.2f}h  "
                       f"预计总 {el/max(done,1)*args.n/3600:.1f}h  "
                       f"分片 {sum(1 for _ in (OUT/'shards').glob('*.npz'))}")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
    (OUT / "failures.json").write_text(json.dumps(bad, indent=1), encoding="utf-8")
    print(f"\n完成 {done}/{args.n}, 失败 {len(bad)}(已登记 failures.json, 未静默丢弃)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
