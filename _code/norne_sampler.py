#!/usr/bin/env python3
"""Norne 标签工厂：批量跑模拟，为代理模型生成 (输入, 输出) 样本对。

为什么换到 Norne（依据 petro-knowledge/references/10-数据资产.md 的实测）：
  - 22 产 + 9 注 = 36 口井（Volve 只有 7）→ 井间盲区大幅缓解
  - 单次 251 秒（Volve 24 分钟）→ 同样算力能跑 6 倍样本
  - 开箱即跑，不需要任何 deck 修补

一次运行 = 一对训练样本，两个方向都能用：
  正向代理：  (地质参数, 井控)  →  4D 场 + 产量        用于加速优化
  反向代理：  井产量历史        →  4D 场               用于状态推断/数字孪生

采样什么（13 维）：
  9 维 = 每口注水井的注入率乘子（真实的决策变量）
  4 维 = 渗透率分区乘子（地质不确定性，按 EQLNUM 的 5 区取前 4 个大区）

用法:
    python norne_sampler.go.py probe            # 只造 2 个 case 验证参数化生效
    python norne_sampler.go.py run --n 200 --jobs 12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = Path("/mnt/data/yongan-admin-2/datasets/petro/opm-data/norne")
WORK = Path(os.environ.get("NORNE_RUNS", "/mnt/data/yongan-admin-2/datasets/petro/_runs/norne"))
IMAGE = "openporousmedia/opmreleases:latest"
DECK = "NORNE_ATW2013.DATA"
SCH = "INCLUDE/BC0407_HIST01122006.SCH"

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")
PERM_REGIONS = (1, 2, 3, 4)                  # EQLNUM 前 4 区
N_THETA = len(INJECTORS) + len(PERM_REGIONS)  # 13

# 观测:22 口生产井的产油/产水/井底压力
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")
OBS_KEYS = ("WOPR", "WWPR", "WBHP")
N_TIMES = 40


def theta_names() -> list[str]:
    return [f"inj_{w}" for w in INJECTORS] + [f"perm_eq{r}" for r in PERM_REGIONS]


def make_case(theta: np.ndarray, name: str) -> Path:
    """theta: 前 9 个是注入率的 log10 乘子, 后 4 个是渗透率分区 log10 乘子。"""
    case = WORK / name
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(BASE, case, ignore=shutil.ignore_patterns("out", "*.UNRST", "*.UNSMRY",
                                                             "*.EGRID", "*.INIT", "*.PRT", "*.DBG"))
    m = 10.0 ** np.asarray(theta, dtype=float)

    # --- 1. 渗透率:按 EQLNUM 区乘 ---
    deck = case / DECK
    lines = deck.read_text(errors="ignore").split("\n")
    idx = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("EDIT"))
    blk = ["-- [sampler] 渗透率分区乘子", "MULTIREG"]
    for j, r in enumerate(PERM_REGIONS):
        v = m[len(INJECTORS) + j]
        for kw in ("PERMX", "PERMY", "PERMZ"):
            blk.append(f"  {kw}  {v:.6f}  {r}  F /")
    blk += ["/", ""]
    deck.write_text("\n".join(lines[:idx] + blk + lines[idx:]))

    # --- 2. 注入率:逐条 WCONINJE 记录乘 ---
    sch = case / SCH
    txt = sch.read_text(errors="ignore").split("\n")
    inj_mult = {w: m[i] for i, w in enumerate(INJECTORS)}
    n_hit = 0
    for i, l in enumerate(txt):
        body = l.split("--")[0]
        for w, mult in inj_mult.items():
            if f"'{w}'" in body and "'RATE'" in body:
                mm = re.search(r"'RATE'\s+([\d.eE+-]+)", body)
                if mm:
                    txt[i] = l.replace(mm.group(0), f"'RATE' {float(mm.group(1)) * mult:.4f}")
                    n_hit += 1
                break
    sch.write_text("\n".join(txt))

    (case / "out").mkdir(exist_ok=True)
    np.save(case / "theta.npy", np.asarray(theta, dtype=float))
    (case / "meta.json").write_text(json.dumps(
        {"theta": list(map(float, theta)), "names": theta_names(),
         "n_inj_records_modified": n_hit}, indent=1), encoding="utf-8")
    return case


def run(case: Path, threads: int = 4, timeout: int = 3600) -> int:
    cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{case}:/data", "-w", "/data", IMAGE,
           "flow", DECK, "--output-dir=/data/out", f"--threads-per-process={threads}"]
    with open(case / "run.log", "w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


# 🔴 2026-08-08 修:旧版按**数组索引**抽稀 summary 向量, 但 summary 的长度是求解器
# ministep 数(实测 300 个 case 里 348~360 不等), 于是同一个时刻槽位在不同 case 对应
# 不同的天 —— 实测最大差 457 天。更坏的是 ministep 数与 θ 弱相关(θ 越极端越难收敛),
# 所以错位是一个**与输入相关的伪信号**, 代理模型会去学它。
# 改为按**真实天数**插值到公共网格(所有 case 的 schedule 相同, 末天都是 3312 天)。
DAY_GRID = np.linspace(1.0, 3312.0, N_TIMES)


def extract(case: Path) -> dict | None:
    """抽出 (井观测向量, 4D 场路径)。3D 场留在盘上按需读, 不进内存。"""
    from resdata.summary import Summary

    smry = case / "out" / "NORNE_ATW2013.UNSMRY"
    if not smry.exists():
        return None
    s = Summary(str(smry))
    days = np.asarray(s.numpy_vector("TIME"), dtype=float)
    cols = []
    for w in PRODUCERS:
        for k in OBS_KEYS:
            try:
                v = np.asarray(s.numpy_vector(f"{k}:{w}"), dtype=float)
            except Exception:
                cols.append(np.full(N_TIMES, np.nan)); continue
            m = np.isfinite(days) & np.isfinite(v)
            cols.append(np.interp(DAY_GRID, days[m], v[m]) if m.sum() >= 2
                        else np.full(N_TIMES, np.nan))
    return {"obs": np.concatenate(cols),
            "fopt": float(s.numpy_vector("FOPT")[-1]),
            "unrst": str(case / "out" / "NORNE_ATW2013.UNRST")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sigma-inj", type=float, default=0.20)
    ap.add_argument("--sigma-perm", type=float, default=0.25)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def draw(k):
        t = np.empty((k, N_THETA))
        t[:, :len(INJECTORS)] = rng.normal(0, args.sigma_inj, (k, len(INJECTORS)))
        t[:, len(INJECTORS):] = rng.normal(0, args.sigma_perm, (k, len(PERM_REGIONS)))
        return t

    if args.cmd == "probe":
        th = np.vstack([np.zeros(N_THETA), draw(1)[0]])
        for j in range(2):
            c = make_case(th[j], f"probe_{j}")
            meta = json.loads((c / "meta.json").read_text())
            print(f"probe_{j}: 改了 {meta['n_inj_records_modified']} 条注入记录")
            rc = run(c, args.threads)
            d = extract(c)
            print(f"  exit={rc}  观测 shape={d['obs'].shape if d else None}  "
                  f"FOPT={d['fopt']/1e6:.2f} 百万 Sm³" if d else f"  exit={rc} 失败")
        return 0

    TH = draw(args.n)
    np.save(WORK / "theta_all.npy", TH)

    def one(j):
        c = make_case(TH[j], f"s{j:04d}")
        rc = run(c, args.threads)
        return j, rc, extract(c)

    OBS = np.full((args.n, len(PRODUCERS) * len(OBS_KEYS) * N_TIMES), np.nan)
    FOPT = np.full(args.n, np.nan)
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for j, rc, d in ex.map(one, range(args.n)):
            if d:
                OBS[j] = d["obs"]; FOPT[j] = d["fopt"]; done += 1
            if j % 10 == 0 or not d:
                print(f"  s{j:04d} exit={rc} {'ok' if d else 'FAILED'}  累计成功 {done}", flush=True)
    np.save(WORK / "obs_all.npy", OBS)
    np.save(WORK / "fopt_all.npy", FOPT)
    print(f"\n完成 {done}/{args.n}  观测张量 {OBS.shape}  FOPT 范围 "
          f"{np.nanmin(FOPT)/1e6:.1f}~{np.nanmax(FOPT)/1e6:.1f} 百万 Sm³")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
