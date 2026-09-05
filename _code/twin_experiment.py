#!/usr/bin/env python3
"""D1 历史拟合：合成孪生实验（twin experiment）。

为什么必须是孪生实验而不是直接拟合真实数据：
  1. Equinor 已经拟合过一次，直接再拟合没有"正确答案"可判对错。
  2. 真值由我们自己定义后，合成真值与集合成员**用同一个正演算子**，
     于是 `--parsing-strictness=low` 忽略 ADDZCORN 造成的保真度问题自动消失
     —— 结论是"关于反演算法"的，不是"关于 Volve 绝对保真度"的。

参数化（依据 G2 门实测：有效观测自由度只有 6~57，参数必须是 O(10)）：
  θ = [11 个 FIPNUM 区的 log10 渗透率乘子, 4 个垂向层组乘子, 1 个全局 kv/kh 乘子]  → 16 维
  不做 100 维 PCA —— 超出数据能支撑的范围，只会拟合噪声。

观测向量（依据 G0 门实测）：
  WWCT / WWPR / WBHP / WGOR，**不含 WOPR**
  （WCONHIST ORAT 下 WOPR 在井能达标时恒等于目标值，零信息）

用法:
    python twin_experiment.py init                      # 建基准 + 检查参数化生效
    python twin_experiment.py truth --seed 42           # 生成合成真值并跑
    python twin_experiment.py ensemble --ne 20          # 跑先验集合
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
BASE = PROJ / "_sandbox" / "opm_run"
WORK = PROJ / "_sandbox" / "twin"
IMAGE = "openporousmedia/opmreleases:latest"

N_FIP = 11
LAYER_GROUPS = ((1, 15), (16, 30), (31, 45), (46, 63))
NX, NY, NZ = 108, 100, 63
N_THETA = N_FIP + len(LAYER_GROUPS) + 1        # 16

PRODUCERS = ("P-F-12", "P-F-14", "P-F-11B", "P-F-15D", "P-F-1C")
OBS_KEYS = ("WWCT", "WWPR", "WBHP", "WGOR")    # 刻意不含 WOPR，见模块 docstring
N_OBS_TIMES = 24                                # 季度采样


def theta_names() -> list[str]:
    return ([f"fip{i}" for i in range(1, N_FIP + 1)]
            + [f"layer{a}-{b}" for a, b in LAYER_GROUPS] + ["kvkh"])


def multiply_block(theta: np.ndarray) -> list[str]:
    """θ 是 log10 乘子。生成 MULTIREG(按 FIPNUM 区) + MULTIPLY(按层组/全局) 块。"""
    m = 10.0 ** np.asarray(theta, dtype=float)
    out = ["-- [twin] 参数化渗透率乘子", "MULTIREG"]
    for i in range(N_FIP):
        # 第 4 项是**区族简写**不是列名:'F'=FIPNUM/FLUXNUM 族(deck 里两者同分区)。
        # 写成 'FIPNUM' 会报 "not a valid region set name. Expected 'O'/'F'/'M'"。
        out.append(f"  PERMX  {m[i]:.6f}  {i + 1}  F /")
        out.append(f"  PERMY  {m[i]:.6f}  {i + 1}  F /")
        out.append(f"  PERMZ  {m[i]:.6f}  {i + 1}  F /")
    out.append("/")
    out.append("")
    out.append("MULTIPLY")
    for j, (k1, k2) in enumerate(LAYER_GROUPS):
        v = m[N_FIP + j]
        out += [f"  PERMX  {v:.6f}  1 {NX}  1 {NY}  {k1} {k2} /",
                f"  PERMY  {v:.6f}  1 {NX}  1 {NY}  {k1} {k2} /",
                f"  PERMZ  {v:.6f}  1 {NX}  1 {NY}  {k1} {k2} /"]
    out.append(f"  PERMZ  {m[-1]:.6f}  1 {NX}  1 {NY}  1 {NZ} /   -- 全局 kv/kh")
    out += ["/", ""]
    # 自带截断:原 deck 的 MAXVALUE 在我们的块之前,不会再对乘完的值生效
    out += ["-- [twin] 乘完后重新截断到与原 deck 相同的物理上限",
            "MAXVALUE",
            f"  PERMX  20000.0  1 {NX}  1 {NY}  1 {NZ} /",
            f"  PERMY  20000.0  1 {NX}  1 {NY}  1 {NZ} /",
            f"  PERMZ   2000.0  1 {NX}  1 {NY}  1 {NZ} /",
            "/", ""]
    return out


def make_case(theta: np.ndarray, name: str) -> Path:
    case = WORK / name
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(BASE, case, ignore=shutil.ignore_patterns("out", "*.orig", "*.DBG", "*.PRT"))
    deck = case / "VOLVE_2016.DATA"
    lines = deck.read_text(errors="ignore").split("\n")
    # 必须插在 FLUXNUM 被 INCLUDE **之后** —— MULTIREG 的 'F' 区族解析到 FLUXNUM,
    # 在它定义之前用会报 "Trying to work with invalid region: FLUXNUM"。
    # 代价是原 deck 的 MAXVALUE 已经过去了, 所以 multiply_block 自带截断。
    idx = next(i for i, l in enumerate(lines) if "FLUXNUM_2013" in l) + 1
    deck.write_text("\n".join(lines[:idx] + multiply_block(theta) + lines[idx:]))
    (case / "out").mkdir(exist_ok=True)
    np.save(case / "theta.npy", np.asarray(theta, dtype=float))
    return case


def run(case: Path, threads: int = 8, timeout: int = 7200) -> int:
    cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{case}:/data", "-w", "/data", IMAGE,
           "flow", "VOLVE_2016.DATA", "--output-dir=/data/out",
           "--parsing-strictness=low", f"--threads-per-process={threads}"]
    with open(case / "run.log", "w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def extract_obs(case: Path) -> np.ndarray | None:
    """季度采样的观测向量。缺失/关井时刻置 NaN（掩码，不填 0）。"""
    from resdata.summary import Summary

    smry = case / "out" / "VOLVE_2016.UNSMRY"
    if not smry.exists():
        return None
    s = Summary(str(smry))
    cols = []
    for w in PRODUCERS:
        for k in OBS_KEYS:
            try:
                v = np.asarray(s.numpy_vector(f"{k}:{w}"), dtype=float)
            except Exception:
                v = np.full(1, np.nan)
            step = max(1, len(v) // N_OBS_TIMES)
            q = v[::step][:N_OBS_TIMES]
            if len(q) < N_OBS_TIMES:
                q = np.concatenate([q, np.full(N_OBS_TIMES - len(q), np.nan)])
            cols.append(q)
    return np.concatenate(cols)                 # (5 井 × 4 量 × 24 时刻,) = 480


def obs_labels() -> list[str]:
    return [f"{k}:{w}@t{t}" for w in PRODUCERS for k in OBS_KEYS for t in range(N_OBS_TIMES)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "truth", "ensemble"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ne", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.30, help="先验 log10 乘子标准差")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6, help="并行 case 数")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.cmd == "init":
        theta0 = np.zeros(N_THETA)
        case = make_case(theta0, "check_identity")
        print(f"参数维度 N_THETA = {N_THETA}")
        print("参数名:", theta_names())
        print(f"\n生成的 deck: {case}/VOLVE_2016.DATA")
        blk = multiply_block(np.array([0.3] + [0.0] * (N_THETA - 1)))
        print("\n乘子块样例(θ_fip1 = 0.3 → 乘子 2.0):")
        print("\n".join(blk[:6] + ["  ..."] + blk[-6:]))
        print(f"\n下一步: python twin_experiment.py truth --seed {args.seed}")
        return 0

    if args.cmd == "truth":
        # 真值不是先验均值, 也不刻意是先验的一个抽样 —— 用 1.5 倍先验尺度制造"先验略微失准"
        theta_true = rng.normal(0, args.sigma * 1.5, N_THETA)
        case = make_case(theta_true, "truth")
        print("θ_true (log10 乘子):")
        for n, v in zip(theta_names(), theta_true):
            print(f"  {n:12s} {v:+.4f}  (乘子 {10**v:.3f})")
        print("\n跑正演...", flush=True)
        rc = run(case, args.threads)
        print(f"exit={rc}")
        d = extract_obs(case)
        if d is None:
            print("🔴 正演失败, 没有 UNSMRY")
            return 1
        finite = np.isfinite(d)
        print(f"观测向量 shape={d.shape} 有效={finite.sum()} NaN={(~finite).sum()}")
        np.save(WORK / "theta_true.npy", theta_true)
        np.save(WORK / "d_true.npy", d)
        json.dump({"theta_names": theta_names(), "obs_labels_n": len(obs_labels()),
                   "n_finite": int(finite.sum())},
                  open(WORK / "truth_meta.json", "w"), indent=1, ensure_ascii=False)
        return 0

    # ensemble
    theta = rng.normal(0, args.sigma, (args.ne, N_THETA))
    np.save(WORK / "theta_prior.npy", theta)
    from concurrent.futures import ThreadPoolExecutor

    def one(j):
        c = make_case(theta[j], f"prior_{j:03d}")
        rc = run(c, args.threads)
        return j, rc, extract_obs(c)

    D = np.full((args.ne, 20 * N_OBS_TIMES), np.nan)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for j, rc, d in ex.map(one, range(args.ne)):
            if d is not None:
                D[j] = d
            print(f"  prior_{j:03d} exit={rc} obs={'ok' if d is not None else 'FAILED'}", flush=True)
    np.save(WORK / "D_prior.npy", D)
    good = np.isfinite(D).any(axis=1)
    print(f"\n集合完成: {good.sum()}/{args.ne} 成功  D shape={D.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
