#!/usr/bin/env python3
"""可断点续跑的实验执行器：把"一次跑完一整轮"改成"一格一格跑，随时可续"。

## 为什么需要它

本机 GPU 已两次整体故障(2026-08-27 七张、08-28 八张,均报 `GPU requires reset`),
每次都把跑了几小时的实验整轮打掉。此前的脚本是"一个进程跑完全部配置×种子",
**中途崩溃 = 全部重来**。

## 四条抗崩措施

1. **一格一存**:每个 (配置, 种子) 跑完立刻写独立 JSON。崩溃只丢正在跑的那一格。
2. **自动跳过**:重启时已存在的格子直接跳过,天然续跑,重启多少次都不重复算。
3. **健康探测 + 重试**:每格开跑前实测 GPU 能否做矩阵乘;CUDA 故障时换卡重试,
   全部不可用则休眠等待,不做任何重置动作(共享机器,重置需 root 且影响他人)。
4. **子进程隔离**:每格在**独立子进程**里跑。CUDA context 一旦损坏无法在进程内恢复,
   只有换进程才干净。

## 用法

    python fc_run.py plug   --ntrain 7400     # 即插即用消融
    python fc_run.py curve  --ntrain 7400     # 学习曲线
    python fc_run.py report                   # 汇总已完成的格子(随时可看)

反复执行同一条命令即可续跑；全部格子齐了会自动出统计。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "_pipelines" / "fc_run" / "cells"
PY = "/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python"

PLUG = {
    "base":       {},
    "preln":      {"preln": True},
    "rmsnorm":    {"rms": True},
    "swiglu":     {"swiglu": True},
    "layerscale": {"layerscale": True},
    "rezero":     {"rezero": True},
    "rope":       {"use_rope": True},
    "droppath":   {"droppath": 0.1},
    "levelshape": {"levelshape": True},
}
LRS = (1e-3, 3e-3, 1e-2)


MAX_GPUS = 3          # 🔴 并发上限。本机 60 用户共用,不独占整机 GPU。


def healthy_gpus() -> list[int]:
    """实测能否做矩阵乘才算健康 —— nvidia-smi 的状态字不够。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    cand = []
    for line in out.strip().split("\n"):
        if "requires reset" in line or not line.strip():
            continue
        try:
            i, _, mem = [x.strip() for x in line.split(",")]
            # 🔴 共享机器(60 用户):只用**真正空闲**的卡(<1000 MiB),
            #    不再用 12000 这种宽松阈值去蹭别人半空的卡。
            if int(mem.split()[0]) < 1000:
                cand.append(int(i))
        except Exception:
            continue
    ok = []
    for i in cand:
        r = subprocess.run(
            [PY, "-c", "import torch;x=torch.randn(1500,1500,device='cuda:0');"
                       "print(float((x@x).sum()))"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)},
            capture_output=True, text=True, timeout=90)
        if r.returncode == 0:
            ok.append(i)
    return ok[:MAX_GPUS]


def cell_path(kind, tag, lr, seed, split, ntrain):
    return CELLS / f"{kind}__{tag}__lr{lr:g}__s{seed}__{split}__n{ntrain}.json"


# ------------------------------------------------------------------ 单格执行
def run_one(kind, tag, lr, seed, split, ntrain, gpu):
    """在**独立子进程**里跑一格。CUDA context 损坏后无法进程内恢复,只能换进程。"""
    code = f'''
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, {str(Path(__file__).parent)!r})
import fc_decision as FD, forecast_gen as FG
from fc_plug import Net
N_PIN = {ntrain} + 600
TH, Y, IA, FC = FD.load(N_PIN)
live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
idx = np.random.default_rng(0).permutation(n)
te, va, pool = idx[:400], idx[400:600], idx[600:]
tr = pool[:{ntrain}]
Yf = Y.reshape(n, -1); NCH = nw*2; NT = FG.FC_N; DAYS = FG.FC_GRID
ev = te if {split!r} == "test" else va
ym, ys = Yf[tr].mean(0), Yf[tr].std(0)+1e-8
xm, xs = TH[tr].mean(0), TH[tr].std(0)+1e-8
X = ((TH-xm)/xs).astype(np.float32)
torch.manual_seed({seed})
net = Net(TH.shape[1], NCH, NT, **{PLUG[tag]!r}).to("cuda:0")
opt = torch.optim.AdamW(net.parameters(), lr={lr}, weight_decay=1e-4)
A = torch.tensor(X[tr], device="cuda:0")
B = torch.tensor(((Yf[tr]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device="cuda:0")
s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
for _ in range(600):
    net.train()
    pm = torch.randperm(len(A), device="cuda:0")
    for i in range(0, len(A), 64):
        b = pm[i:i+64]; opt.zero_grad()
        F.mse_loss(net(A[b]), B[b]).backward(); opt.step()
    s.step()
net.eval()
with torch.no_grad():
    P = net(torch.tensor(X[ev], device="cuda:0")).cpu().numpy().reshape(len(ev), -1)*ys+ym
cum = lambda a: np.trapezoid(a.reshape(-1, nw, 2, NT)[:, :, 0, :], DAYS, axis=-1)
cP, cT = cum(P), cum(Yf[ev])
fld = np.abs(cP.sum(1)-cT.sum(1))/np.abs(cT.sum(1))
per = np.abs(cP-cT)/np.maximum(np.abs(cT), 1e-9)
r2 = 1-((cP.sum(1)-cT.sum(1))**2).sum()/((cT.sum(1)-cT.sum(1).mean())**2).sum()
json.dump({{"field": float(fld.mean()), "per_well": float(per.mean()),
           "r2": float(r2), "params": sum(p.numel() for p in net.parameters()),
           "tag": {tag!r}, "lr": {lr}, "seed": {seed}, "split": {split!r},
           "ntrain": {ntrain}}}, open({str(cell_path(kind,tag,lr,seed,split,ntrain))!r}, "w"))
print("CELL_OK")
'''
    r = subprocess.run([PY, "-W", "ignore", "-c", code],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                       capture_output=True, text=True, timeout=7200)
    return "CELL_OK" in r.stdout, (r.stderr or "")[-300:]


def pick_best_lr(kind, ntrain):
    """从已完成的 val 格子里逐配置选最优 lr。"""
    from collections import defaultdict
    g = defaultdict(list)
    for f in CELLS.glob(f"{kind}__*__val__n{ntrain}.json"):
        r = json.loads(f.read_text()); g[(r["tag"], r["lr"])].append(r["field"])
    best = {}
    for tag in PLUG:
        rows = [(lr, float(np.mean(g[(tag, lr)]))) for lr in LRS if g[(tag, lr)]]
        if rows:
            best[tag] = min(rows, key=lambda r: r[1])[0]
    return best


def drive(kind, jobs, ntrain):
    """多卡并行 + 一格一存 + 崩溃自动续跑。"""
    from concurrent.futures import ThreadPoolExecutor
    CELLS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not cell_path(kind, *j, ntrain).exists()]
    print(f"总格子 {len(jobs)}   已完成 {len(jobs)-len(todo)}   待跑 {len(todo)}", flush=True)
    t0, done, fail = time.time(), 0, 0
    while todo:
        gpus = healthy_gpus()
        if not gpus:
            print(f"  [{time.strftime('%H:%M:%S')}] 无健康 GPU,休眠 3 分钟后重探"
                  f"(不做任何重置动作)", flush=True)
            time.sleep(180); continue
        batch = todo[:len(gpus)]
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futs = {ex.submit(run_one, kind, *j, ntrain, gpus[i]): j
                    for i, j in enumerate(batch)}
            for f, j in futs.items():
                try:
                    ok, err = f.result()
                except Exception as e:                          # noqa: BLE001
                    ok, err = False, f"{type(e).__name__}: {e}"
                tag, lr, seed, split = j
                if ok and cell_path(kind, *j, ntrain).exists():
                    todo.remove(j); done += 1
                    d = json.loads(cell_path(kind, *j, ntrain).read_text())
                    print(f"  ✔ {tag:11s} lr={lr:<6g} s{seed} {split:4s}  "
                          f"场级 {d['field']:.3%} 每井 {d['per_well']:.3%}  "
                          f"[{done}/{done+len(todo)}] {(time.time()-t0)/60:.0f}min", flush=True)
                else:
                    fail += 1
                    tail = err.splitlines()[-1] if err.strip() else "无 stderr"
                    print(f"  ✘ {tag} lr={lr:g} s{seed} {split} 失败({fail}): {tail}", flush=True)
        if fail and not any(cell_path(kind, *j, ntrain).exists() for j in batch):
            time.sleep(30)                                      # 整批都挂,多半是 GPU 又崩了
    print(f"\n全部完成。失败重试 {fail} 次。")


def report(kind, ntrain):
    fs = sorted(CELLS.glob(f"{kind}__*__n{ntrain}.json"))
    if not fs:
        print("还没有任何格子"); return
    rec = [json.loads(f.read_text()) for f in fs]
    from collections import defaultdict
    g = defaultdict(list)
    for r in rec:
        g[(r["tag"], r["lr"], r["split"])].append(r)
    print(f"已完成 {len(rec)} 格\n")
    print("=== val（逐配置选最优 lr）===")
    best = {}
    for tag in PLUG:
        rows = [(lr, np.mean([x["field"] for x in g[(tag, lr, "val")]]), len(g[(tag, lr, "val")]))
                for lr in LRS if g[(tag, lr, "val")]]
        if not rows:
            continue
        lr, m, k = min(rows, key=lambda r: r[1])
        best[tag] = lr
        print(f"  {tag:11s} 最优 lr={lr:<6g} val {m:.3%} ({k} 种子)"
              + "   " + " ".join(f"lr{l:g}:{v:.3%}" for l, v, _ in rows))
    print("\n=== 封存 test ===")
    T = {}
    for tag in PLUG:
        lr = best.get(tag)
        if lr is None or not g[(tag, lr, "test")]:
            continue
        a = np.array([x["field"] for x in g[(tag, lr, "test")]])
        p = np.array([x["per_well"] for x in g[(tag, lr, "test")]])
        T[tag] = a
        print(f"  {tag:11s} 场级 {a.mean():.3%} ± {a.std(ddof=1) if len(a)>1 else 0:.3%}  "
              f"每井 {p.mean():.3%}  ({len(a)} 种子, lr={lr:g})")
    if "base" in T and len(T["base"]) > 1:
        from scipy.stats import ttest_ind
        print("\n=== 判定(vs base, Welch;可分辨阈值 0.0214pp) ===")
        for tag, a in T.items():
            if tag == "base" or len(a) < 2:
                continue
            pv = float(ttest_ind(a, T["base"], equal_var=False).pvalue)
            d = (a.mean() - T["base"].mean()) * 100
            vd = ("✔ 显著变好" if pv < .05 and d < 0 else
                  "🔴 显著变差" if pv < .05 else "— 噪声内")
            print(f"  {tag:11s} {d:>+8.3f}pp  p={pv:.4f}   {vd}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plug", "report"])
    ap.add_argument("--ntrain", type=int, default=7400)
    ap.add_argument("--screen-seeds", type=int, default=3)
    ap.add_argument("--test-seeds", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "report":
        report("plug", a.ntrain); return 0
    # 阶段一:val 全扫(逐配置 × 逐 lr)
    val_jobs = [(t, lr, s, "val") for t in PLUG for lr in LRS for s in range(a.screen_seeds)]
    drive("plug", val_jobs, a.ntrain)
    # 阶段二:test 只在 val 选出的**最优 lr** 上跑 —— 否则 test 要 216 格,浪费 3 倍算力
    best = pick_best_lr("plug", a.ntrain)
    print(f"\nval 选出的最优 lr: {best}\n")
    test_jobs = [(t, best[t], s, "test") for t in PLUG if t in best
                 for s in range(a.test_seeds)]
    drive("plug", test_jobs, a.ntrain)
    report("plug", a.ntrain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
