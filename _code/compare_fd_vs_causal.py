#!/usr/bin/env python3
"""流动诊断 vs 因果效应：井对连通性的两种算法对拍。

这是论文的核心对照。审稿人一定会问「这跟流线有什么区别」，本脚本就是回答。

## 两者算的**不是同一个东西**，这一点必须说清楚

| | 流动诊断（本领域既有方法） | 我们的因果效应 |
|---|---|---|
| 输入 | **一次**模拟的通量场 | **一组**随机化模拟（θ 独立随机） |
| 含义 | 当前流场里，生产井 j 采出的流体有多少来自注入井 i | 把注入井 i 的注入率**改变**一点，生产井 j 的产量变多少 |
| 数学 | 稳态示踪剂的份额 | ∂q_j / ∂w_i（回归斜率） |
| 假设 | 冻结流场（Thiele 原话 "assumes fixed streamline paths for all time (a linear model)"） | 无（真的把流场重跑了 N 遍） |
| 成本 | 一次模拟 + 一次线性解（秒级） | N 次全模拟（本项目约 10⁴ 次） |

**"现状分配"与"改变的导数"在非线性系统里不等价**。两者相关但不应完全一致——
一致说明流场近似线性，不一致的地方正是冻结流场假设失效之处。**分歧本身就是结果。**

🔴 诚实边界：本对拍**不能**用来说"我们比流线准"。真值需要专门的反事实实验
（改注入率重跑，看产量实际变多少）。本脚本只回答"两者一致到什么程度、在哪不一致"。

用法:
    python compare_fd_vs_causal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "_pipelines" / "flow_diagnostics" / "allocation.json"
CB = ROOT / "_pipelines" / "crm_bridge"
OUT = ROOT / "_pipelines" / "compare_fd_causal"

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")


def main() -> int:
    if not FD.exists():
        print(f"缺 {FD}；先跑 flow_diagnostics.py"); return 1
    if not (CB / "crm_allocation_f.npy").exists():
        print(f"缺 {CB}/crm_allocation_f.npy；先跑 crm_bridge.py"); return 1
    OUT.mkdir(parents=True, exist_ok=True)

    fd = json.loads(FD.read_text())
    # 🔴 必须用 injector_fraction, 不能用 producer_fraction。
    #    两者分母不同(生产井产液量 vs 注入井注入量), 是两个不同的条件概率。
    #    实测:直接比 producer_fraction 时 Spearman +0.266、top1 0/3;
    #    换成 injector_fraction 后 +0.784、top1 1/3。
    #    更要命的是 producer_fraction 的 top1 会被**将死的井**劫持 ——
    #    "F-1H 有 50% 流向 E-2AH"听着很强, 但 E-2AH 末帧只产 11.37 m3/d,
    #    而 E-3CH 产 3,983.52。份额 top1 ≠ 注入量去向 top1。
    src = fd.get("injector_fraction")
    if not src:
        print("🔴 allocation.json 里没有 injector_fraction —— 先重跑 flow_diagnostics.py"); return 1
    A_fd = np.full((len(INJECTORS), len(PRODUCERS)), np.nan)
    for pw, by in src.items():
        if pw not in PRODUCERS:
            continue
        p = PRODUCERS.index(pw)
        for inj, v in by.items():
            if inj in INJECTORS:
                A_fd[INJECTORS.index(inj), p] = v
    A_cf_raw = np.load(CB / "crm_allocation_f.npy")
    cb = json.loads((CB / "f1_bridge.json").read_text())
    names = [r.get("control", r.get("injector")) for r in cb.get("crm_allocation", [])]
    if A_cf_raw.shape[0] == len(INJECTORS):
        A_cf = A_cf_raw                                  # schema v1:行就是 9 口井
    else:                                                # schema v2:行是「井×相态」, 取水相
        A_cf = np.full((len(INJECTORS), len(PRODUCERS)), np.nan)
        import norne_bulk as NB
        ctrl = [f"{w}_{ph}" for w, ph in NB.CONTROLS]
        for k, c in enumerate(ctrl):
            w, ph = c.rsplit("_", 1)
            if ph == "WATER" and w in INJECTORS:
                A_cf[INJECTORS.index(w)] = A_cf_raw[k]
        print(f"因果侧 schema v2:取 {sum(1 for c in ctrl if c.endswith('_WATER'))} 个水相控制与流诊对齐")

    # 只比两边**都有数**的井对。因果侧的 v1 数据里混注井取不到水相记录 → 只有 3 口纯注水井。
    both = np.isfinite(A_fd) & np.isfinite(A_cf)
    rows_ok = [i for i in range(len(INJECTORS)) if both[i].any()]
    print(f"两边都有数的注入井 {len(rows_ok)}: {[INJECTORS[i] for i in rows_ok]}")
    print(f"可比井对 {int(both.sum())} 个\n")

    x = A_fd[both]; y = A_cf[both]
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    r_p = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    r_s = spearman(x, y) if len(x) > 2 else float("nan")
    print(f"两法相关: Pearson {r_p:+.3f}   Spearman(秩) {r_s:+.3f}   n={len(x)}")
    print(f"  流动诊断 值域 [{x.min():.4f}, {x.max():.4f}]  均值 {x.mean():.4f}")
    print(f"  因果效应 值域 [{y.min():.4f}, {y.max():.4f}]  均值 {y.mean():.4f}")
    print(f"  量级比(因果/流诊) 中位 {np.median(y / np.maximum(x, 1e-9)):.3f}")

    # 各自最强的那口生产井是否一致 —— 工程上最关心这个
    print(f"\n{'注入井':8s}{'流诊 top1':>12s}{'份额':>8s}{'因果 top1':>12s}{'份额':>8s}{'一致?':>7s}")
    agree = 0
    detail = []
    for i in rows_ok:
        m = both[i]
        pj = np.where(m)[0]
        a = pj[np.argmax(A_fd[i, m])]; b = pj[np.argmax(A_cf[i, m])]
        ok = a == b
        agree += ok
        detail.append({"injector": INJECTORS[i], "fd_top": PRODUCERS[a],
                       "fd_val": float(A_fd[i, a]), "causal_top": PRODUCERS[b],
                       "causal_val": float(A_cf[i, b]), "agree": bool(ok)})
        print(f"{INJECTORS[i]:8s}{PRODUCERS[a]:>12s}{A_fd[i,a]:>8.3f}"
              f"{PRODUCERS[b]:>12s}{A_cf[i,b]:>8.3f}{'✔' if ok else '✘':>7s}")
    print(f"\n最强连通井对一致率 {agree}/{len(rows_ok)}")

    # 分歧最大的井对 —— 冻结流场假设失效的候选位置
    d = np.full_like(A_fd, np.nan)
    d[both] = y - x
    flat = [(INJECTORS[i], PRODUCERS[j], A_fd[i, j], A_cf[i, j], d[i, j])
            for i in range(len(INJECTORS)) for j in range(len(PRODUCERS)) if both[i, j]]
    flat.sort(key=lambda t: -abs(t[4]))
    print(f"\n分歧最大的 5 个井对(因果 − 流诊):")
    for iw, pw, a, b, dd in flat[:5]:
        print(f"  {iw:7s}→{pw:8s} 流诊 {a:.3f}  因果 {b:.3f}  差 {dd:+.3f}")

    out = {"n_pairs": int(both.sum()), "injectors_compared": [INJECTORS[i] for i in rows_ok],
           "pearson": r_p, "spearman": r_s, "top1_agreement": f"{agree}/{len(rows_ok)}",
           "per_injector": detail,
           "largest_disagreements": [{"injector": a, "producer": b, "fd": c,
                                      "causal": dd, "diff": e} for a, b, c, dd, e in flat[:10]],
           "caveat": ("两法算的不是同一个量:流诊=当前流场的份额(冻结流场), "
                      "因果=改变注入率的导数(重跑 N 遍)。不一致不等于谁错, "
                      "分歧位置正是冻结流场假设失效的候选。本对拍不能证明谁更准 —— "
                      "那需要专门的反事实实验。")}
    (OUT / "compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "matrices.npz", fd=A_fd, causal=A_cf)
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
