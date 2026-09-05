#!/usr/bin/env python3
"""把 3D 因果效应场画出来。

画的是 ∂SWAT/∂(某口注水井的注入率)：每个网格单元一个数，表示
「这口注水井多注一点，这个位置的含水饱和度会变多少」。

正值 = 这口井的水会到这里；负值 = 加注反而把水挤到别处去了。

🔴 为什么这张图观测数据画不出来：它是**因果效应**不是相关性。只有当注入率被
   随机化（干预臂）时，回归系数才等于因果效应。真实油田的注入率由作业者根据
   油藏状态决定，算出来的是混杂后的相关性。

用法:
    python plot_effect_field.py --inj C-3H
    python plot_effect_field.py --inj C-3H --compare C-4H
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID = Path("/mnt/data/yongan-admin-2/datasets/petro/opm-data/norne/out/NORNE_ATW2013.EGRID")
EFFDIR = Path(__file__).resolve().parent.parent / "_pipelines" / "causal_bench"
FIGDIR = Path(__file__).resolve().parent.parent / "_figures"
# 井位从 deck 的 COMPDAT 读不方便, 用 summary 里的井名 + 已知的 Norne 分区大致标注
INJ_LABEL = {"C-1H": "C 区", "C-2H": "C 区", "C-3H": "C 区", "C-4AH": "C 区", "C-4H": "C 区",
             "F-1H": "F 区", "F-2H": "F 区", "F-3H": "F 区", "F-4H": "F 区"}


def load_geometry():
    from resdata.grid import Grid
    g = Grid(str(GRID))
    n = g.getNumActive()
    xyz = np.empty((n, 3), np.float64)
    ijk = np.empty((n, 3), np.int32)
    for i in range(n):
        xyz[i] = g.get_xyz(active_index=i)
        ijk[i] = g.get_ijk(active_index=i)
    return g, xyz, ijk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inj", default="C-3H")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 本机有 93 个中文字体, 挑一个可用的, 否则中文标题变方框
    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [f]; break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib.colors import TwoSlopeNorm

    wells = [args.inj] + ([args.compare] if args.compare else [])
    eff = {}
    for w in wells:
        p = EFFDIR / f"effect_field_SWAT_{w}.npy"
        if not p.exists():
            print(f"缺 {p}；先跑 causal_bench.py field --inj {w}")
            return 1
        eff[w] = np.load(p)

    g, xyz, ijk = load_geometry()
    print(f"网格 {g.getNX()}x{g.getNY()}x{g.getNZ()}  活动单元 {len(xyz):,}")
    for w, e in eff.items():
        print(f"  {w}: 值域 [{e.min():+.4f}, {e.max():+.4f}]  "
              f"|效应|>1e-3 的单元 {(np.abs(e) > 1e-3).sum():,} ({(np.abs(e) > 1e-3).mean():.1%})")

    vmax = max(np.percentile(np.abs(e), 99.5) for e in eff.values())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = "RdBu_r"                                    # 红=水更多, 蓝=水更少

    nrow = len(wells)
    fig = plt.figure(figsize=(15, 4.6 * nrow))
    for r, w in enumerate(wells):
        e = eff[w]
        # --- (1) 3D 散点:只画有明显效应的单元, 否则被大量零值淹没 ---
        ax = fig.add_subplot(nrow, 3, r * 3 + 1, projection="3d")
        m = np.abs(e) > np.percentile(np.abs(e), 90)
        s = ax.scatter(xyz[m, 0], xyz[m, 1], -xyz[m, 2], c=e[m], cmap=cmap, norm=norm,
                       s=1.2, alpha=0.55, linewidths=0)
        ax.set_title(f"{w}：三维因果效应场\n(只显示效应最强的 10% 单元)", fontsize=10)
        ax.set_xlabel("东 (m)", fontsize=8); ax.set_ylabel("北 (m)", fontsize=8)
        ax.set_zlabel("深度 (m)", fontsize=8)
        ax.tick_params(labelsize=6); ax.view_init(elev=22, azim=-58)

        # --- (2) 俯视图:沿深度求和, 看平面波及范围 ---
        ax2 = fig.add_subplot(nrow, 3, r * 3 + 2)
        nx, ny = g.getNX(), g.getNY()
        plane = np.full((ny, nx), np.nan)
        acc = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
        np.add.at(acc, (ijk[:, 1], ijk[:, 0]), e)
        np.add.at(cnt, (ijk[:, 1], ijk[:, 0]), 1)
        plane = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        im = ax2.imshow(plane, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        ax2.set_title(f"{w}：俯视图(沿深度平均)", fontsize=10)
        ax2.set_xlabel("I 方向", fontsize=8); ax2.set_ylabel("J 方向", fontsize=8)
        ax2.tick_params(labelsize=7)

        # --- (3) 逐层剖面:效应在垂向怎么分布 ---
        ax3 = fig.add_subplot(nrow, 3, r * 3 + 3)
        nz = g.getNZ()
        pos = [e[(ijk[:, 2] == k) & (e > 0)].sum() for k in range(nz)]
        neg = [e[(ijk[:, 2] == k) & (e < 0)].sum() for k in range(nz)]
        ax3.barh(range(nz), pos, color="#c0392b", label="正效应(水到这里)")
        ax3.barh(range(nz), neg, color="#2471a3", label="负效应(水被挤走)")
        ax3.invert_yaxis(); ax3.set_ylabel("层号 K(浅→深)", fontsize=8)
        ax3.set_xlabel("该层效应总和", fontsize=8)
        ax3.set_title(f"{w}：垂向分布", fontsize=10)
        ax3.legend(fontsize=7); ax3.tick_params(labelsize=7)
        ax3.axvline(0, color="k", lw=0.6)

    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(s, cax=cax, label="∂含水饱和度 / ∂注入率")
    fig.suptitle("Norne 油田：注水井的三维因果效应场\n"
                 "（干预臂 n=%d，注入率随机化 → 回归系数即因果效应，非相关性）"
                 % 2030, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    out = FIGDIR / f"causal_effect_field_{'_vs_'.join(wells)}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"\n已保存 {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
