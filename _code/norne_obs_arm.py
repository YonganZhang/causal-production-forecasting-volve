#!/usr/bin/env python3
"""Norne 观测臂：注入率由「作业策略」决定，而不是随机抽。

这是因果基准 NORNE-CF 的另一半。两臂来自**同一个生成器**，所以真值已知：

    干预臂 (norne_bulk.py)  θ ~ 独立随机        →  ∂产油/∂注入 的【真值】, 天然无混杂
    观测臂 (本脚本)          θ = π(油藏状态)     →  带混杂的观测数据

为什么这一半才是关键：现有因果基准(IHDP / ACIC / Twins)的混杂都是**作者手写的
propensity 函数**，审稿人一直批评这点。这里的混杂是**真实偏微分方程**——未观测
混杂就是油藏状态本身，不是编出来的。

## 作业策略 π

模拟真实作业者的反应式决策，只用**当时可观测**的量（不许用未来、不许用全场真值）：

    含水率高 → 减注（水窜了，多注也是产水）
    压力低   → 加注（保压）
    末期     → 整体减注（经济性）
    + 有界随机扰动（真实作业不是确定性的，也保证 positivity 不被完全破坏）

🔴 正是这个策略制造了混杂：注入率与油藏状态相关，而油藏状态又直接决定产量。
   于是观测数据里的 corr(注入, 产油) 混合了「注水的真实效应」与「状态的反向选择」。

策略是**分段常数**的：把 9.1 年切成 K 段，每段开头读一次状态、定一次注入率。
这样既有反馈又不会让 deck 变得无法生成。

用法:
    python norne_obs_arm.py probe            # 造 2 个 case, 打印策略轨迹与混杂强度
    python norne_obs_arm.py run --n 2000 --jobs 32
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import norne_bulk as NB          # 复用 deck 构造、抽取、常量, 保证两臂口径一致

# 与干预臂同步升 v2:CONTROLS 变了(相态分离), 旧观测臂分片不可混用
OUT = NB.P.NORNE_OBS_V2
N_STAGES = 6                     # 策略分段数(9.1 年 → 每段约 1.5 年)
STAGE_DAYS = np.linspace(0.0, 3312.0, N_STAGES + 1)
CTRL_INDEX = {c: i for i, c in enumerate(NB.CONTROLS)}   # (井, 相态) → θ 列号
_MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_START = _dt.date(1997, 11, 6)   # Norne deck 的 START;与 DAY_GRID 的 0 点一致


def _to_day(d: int, mon: str, y: int) -> float:
    """DATES 记录 → 距 START 的天数。段边界必须按真实日期切, 不能按报告步平均。"""
    return float((_dt.date(y, _MON[mon], d) - _START).days)


def policy(state: dict, stage: int, rng: np.random.Generator,
           gain_wct: float, gain_pres: float, noise: float) -> float:
    """作业策略 π：返回本段的注入率 log10 乘子。

    state 是**该段开始时**从上一段模拟结果读到的、现场真能测到的量:
        wct   全场含水率
        pres  全场平均压力(归一化到初始值)
    """
    # 不给默认值:缺键必须炸。默认 wct=0.3/pres=1.0 恰好让策略项归零 = 静默取消混杂。
    if "wct" not in state or "pres_ratio" not in state:
        raise KeyError(f"策略拿不到状态 {sorted(state)} —— 拒绝退化成无混杂的常数策略")
    wct = float(np.clip(state["wct"], 0.0, 1.0))
    pres = float(np.clip(state["pres_ratio"], 0.5, 1.5))
    # 含水率越高越减注; 压力越低越加注; 末期整体减注
    a = -gain_wct * (wct - 0.3) + gain_pres * (1.0 - pres) - 0.04 * stage
    return float(np.clip(a + rng.normal(0, noise), -0.9, 0.9))


def read_state(work: Path, day: float) -> dict:
    """从已完成的模拟里读 day 时刻的全场状态。策略只能看这些量。"""
    from resdata.summary import Summary

    smry = work / "out" / "NORNE_ATW2013.UNSMRY"
    if not smry.exists():
        raise FileNotFoundError(f"读不到状态: {smry} 不存在")
    s = Summary(str(smry))
    t = np.asarray(s.numpy_vector("TIME"), float)
    # 🔴 这里原来是两个 `except Exception: pass`。读状态失败时字典会缺键,
    #    而 policy 的默认值恰好是 wct=0.3、pres_ratio=1.0 —— 正好让策略项归零。
    #    于是"读不到状态"被静默变成"无混杂", 观测臂退化成第二个干预臂,
    #    整个"量化朴素方法偏差"的设计当场失效, 而且不报任何错。必须 fail loud。
    wct = np.asarray(s.numpy_vector("FWCT"), float)
    p = np.asarray(s.numpy_vector("FPR"), float)
    m = np.isfinite(p) & (p > 0)
    if not np.isfinite(wct).any() or m.sum() < 2:
        raise ValueError(f"状态序列不可用: FWCT 有限值 {int(np.isfinite(wct).sum())}, FPR 正值 {int(m.sum())}")
    return {"wct": float(np.interp(day, t, wct)),
            "pres_ratio": float(np.interp(day, t[m], p[m]) / p[m][0])}


def build_with_stages(theta_inj: np.ndarray, theta_perm: np.ndarray, work: Path) -> None:
    """按段写入不同的注入率乘子。theta_inj shape (N_STAGES, len(NB.CONTROLS))。"""
    shutil.copytree(NB.BASE, work, ignore=shutil.ignore_patterns(
        "out", "*.UNRST", "*.UNSMRY", "*.EGRID", "*.INIT", "*.PRT", "*.DBG", "*.ESMRY"))
    # 渗透率:与干预臂同一套参数化
    deck = work / NB.DECK
    L = deck.read_text(errors="ignore").split("\n")
    i = next(k for k, l in enumerate(L) if l.strip().upper().startswith("EDIT"))
    m = 10.0 ** np.asarray(theta_perm, float)
    blk = ["-- [obs-arm] 渗透率分区乘子", "MULTIREG"]
    for j, r in enumerate(NB.PERM_REGIONS):
        blk += [f"  {kw}  {m[j]:.6f}  {r}  F /" for kw in ("PERMX", "PERMY", "PERMZ")]
    deck.write_text("\n".join(L[:i] + blk + ["/", ""] + L[i:]))

    # 注入率:按 DATES 所处的段选乘子 —— 这是与干预臂的唯一结构差别
    #
    # 🔴 旧版这里的日期指针是假的:它按 `day += 3312 / DATES 行数` 平均推进, 而 Norne 的
    #    报告步间隔极不均匀(早期按月、后期按季)。于是策略动作会被贴到错误的时间段,
    #    "策略响应状态"这个因果结构就被打乱了。现在直接解析 DATES 记录里的真实日期。
    sch = work / NB.SCH
    T = sch.read_text(errors="ignore").split("\n")
    mult_row, in_dates = 0, False
    for k, l in enumerate(T):
        body = l.split("--")[0]
        if body.strip().upper().startswith("DATES"):
            in_dates = True
            continue
        if in_dates:
            g2 = re.match(r"\s*(\d+)\s+'(\w{3})\w*'\s+(\d{4})", body)
            if g2:
                d = _to_day(int(g2.group(1)), g2.group(2).upper(), int(g2.group(3)))
                mult_row = min(int(np.searchsorted(STAGE_DAYS, d, "right") - 1), N_STAGES - 1)
                mult_row = max(mult_row, 0)
            if body.strip().startswith("/"):
                in_dates = False
        if "'RATE'" not in body:
            continue
        t = body.replace("/", " ").split()
        if len(t) < 2:
            continue
        key = (t[0].strip("'"), t[1].strip("'").upper())     # 与干预臂同口径:井 + 相态
        if key not in CTRL_INDEX:
            continue
        g = re.search(r"'RATE'\s+([\d.eE+-]+)", body)
        if g:
            f = 10.0 ** float(theta_inj[mult_row, CTRL_INDEX[key]])
            T[k] = l.replace(g.group(0), f"'RATE' {float(g.group(1)) * f:.4f}")
    sch.write_text("\n".join(T))
    (work / "out").mkdir(exist_ok=True)


def one(j: int, seed: int, threads: int, gains: dict) -> dict:
    """两遍法:先用先验中值跑一遍拿状态轨迹, 再按策略定各段注入率跑正式的一遍。

    两遍是为了让策略"看得到"状态。真作业者是在线决策, 这里用离线两遍近似,
    代价是多一倍算力, 换来的是策略确实依赖状态(否则就没有混杂)。
    """
    rng = np.random.default_rng(seed + j)
    shard = OUT / "shards" / f"{j:05d}.npz"
    if shard.exists():
        return {"j": j, "status": "cached"}
    w0 = OUT / "_work" / f"p{j:05d}"
    w1 = OUT / "_work" / f"f{j:05d}"
    for w in (w0, w1):
        if w.exists():
            shutil.rmtree(w)
    t0 = time.time()
    try:
        theta_perm = rng.normal(0, 0.40, len(NB.PERM_REGIONS))
        # ---- 第一遍:注入率全为基准值, 只为拿状态轨迹 ----
        build_with_stages(np.zeros((N_STAGES, len(NB.CONTROLS))), theta_perm, w0)
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{w0}:/data", "-w", "/data", NB.IMAGE, "flow", NB.DECK,
               "--output-dir=/data/out", f"--threads-per-process={threads}"]
        with open(w0 / "run.log", "w") as lf:
            rc0 = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
        if rc0 != 0:      # 第一遍失败 → 状态不可信 → 策略无意义, 不能带着跑第二遍
            return {"j": j, "status": "failed", "rc": rc0, "why": "第一遍(取状态)未收敛"}
        # ---- 按策略定各段注入率 ----
        th = np.zeros((N_STAGES, len(NB.CONTROLS)), np.float32)
        traj = []
        for st in range(N_STAGES):
            state = read_state(w0, STAGE_DAYS[st])
            a = policy(state, st, rng, **gains)
            th[st, :] = a + rng.normal(0, 0.08, len(NB.CONTROLS))   # 井间小差异
            traj.append({"stage": st, "day": float(STAGE_DAYS[st]), **state, "action": a})
        # ---- 第二遍:正式运行 ----
        build_with_stages(th, theta_perm, w1)
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{w1}:/data", "-w", "/data", NB.IMAGE, "flow", NB.DECK,
               "--output-dir=/data/out", f"--threads-per-process={threads}"]
        with open(w1 / "run.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
        # 🔴 harvest() 在 v2 里改成返回 dict(多了 inj_actual / field_days / n_restart),
        #    这里原来是 `obs, fld = got` 的元组解包 —— 探路时全部样本报
        #    "too many values to unpack"。两臂共用 norne_bulk, 改了一边必须同步另一边。
        if rc != 0:
            return {"j": j, "status": "failed", "rc": rc, "why": "第二遍(正式)未收敛"}
        got = NB.harvest(w1)
        if got is None:
            return {"j": j, "status": "failed", "rc": rc, "why": "harvest 拿不到输出"}
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, theta_inj=th, theta_perm=theta_perm.astype(np.float32),
                            schema=np.int32(NB.SCHEMA), rc=np.int32(rc),
                            traj=json.dumps(traj), **got)
        return {"j": j, "status": "ok", "seconds": round(time.time() - t0, 1)}
    except Exception as e:                                       # noqa: BLE001
        return {"j": j, "status": "error", "err": f"{type(e).__name__}: {e}"[:200]}
    finally:
        for w in (w0, w1):
            shutil.rmtree(w, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run", "merge"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--gain-wct", type=float, default=1.2)
    ap.add_argument("--gain-pres", type=float, default=1.5)
    ap.add_argument("--noise", type=float, default=0.15)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shards").mkdir(exist_ok=True)
    gains = {"gain_wct": args.gain_wct, "gain_pres": args.gain_pres, "noise": args.noise}

    if args.cmd == "merge":
        sh = sorted((OUT / "shards").glob("*.npz"))
        TI, TP, OB, FL = [], [], [], []
        for p in sh:
            d = np.load(p, allow_pickle=True)
            TI.append(d["theta_inj"]); TP.append(d["theta_perm"])
            OB.append(d["obs"]); FL.append(d["fields"])
        pk = OUT / "packed"; pk.mkdir(exist_ok=True)
        np.save(pk / "theta_inj.npy", np.stack(TI)); np.save(pk / "theta_perm.npy", np.stack(TP))
        np.save(pk / "obs.npy", np.stack(OB)); np.save(pk / "fields.npy", np.stack(FL))
        print(f"合并 {len(sh)} 个观测臂样本  theta_inj{np.stack(TI).shape}")
        return 0

    if args.cmd == "probe":
        for j in (0, 1):
            r = one(j, args.seed, 8, gains)
            print(f"probe {j}: {r}")
            p = OUT / "shards" / f"{j:05d}.npz"
            if p.exists():
                d = np.load(p, allow_pickle=True)
                traj = json.loads(str(d["traj"]))
                print("  策略轨迹(状态 → 动作):")
                for t in traj:
                    print(f"    段{t['stage']} 第{t['day']:.0f}天  含水率={t.get('wct',float('nan')):.3f} "
                          f"压力比={t.get('pres_ratio',float('nan')):.3f}  → 注入乘子 log10={t['action']:+.3f}")
                print(f"  obs{d['obs'].shape} fields{d['fields'].shape}")
        return 0

    t0, done, bad = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, j, args.seed, args.threads, gains) for j in range(args.n)]
        for fu in as_completed(futs):
            r = fu.result()
            done += r["status"] in ("ok", "cached")
            if r["status"] not in ("ok", "cached"):
                bad.append(r)
            if done % 25 == 0 or r["status"] not in ("ok", "cached"):
                el = time.time() - t0
                print(f"[{done}/{args.n}] 失败 {len(bad)}  已用 {el/3600:.2f}h  "
                      f"预计总 {el/max(done,1)*args.n/3600:.1f}h", flush=True)
    (OUT / "failures.json").write_text(json.dumps(bad, indent=1), encoding="utf-8")
    print(f"\n完成 {done}/{args.n}, 失败 {len(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
