#!/usr/bin/env python3
"""按油藏工业软件(Petrel / ResInsight / Eclipse Office)的惯例渲染三维因果效应场。

画的是 ∂SWAT/∂(某口注水井的注入率):每个网格单元一个数,表示"这口井多注一点,
这个位置的含水饱和度变多少"。正值=水会到这里,负值=加注反而把水挤走。

🔴 为什么观测数据画不出这张图:它是**因果效应**不是相关性。只有注入率被随机化
   (干预臂)时回归系数才等于因果效应。真实油田注入率由作业者按油藏状态决定,
   算出来的是混杂后的相关性。

--------------------------------------------------------------------------
本脚本选用的呈现设定, 及其在 ResInsight 源码里的对照(ResInsight 是 OPM 生态的官方
后处理器, 比商业软件截图更可辩护)。⚠️ 这些是**我们为本图选的设定**, 不是"工业强制惯例"
—— 下面逐条注明 ResInsight 的真实默认值, 有几条其实和我们的选择相反:

  1. **离散色带 11 级**。⚠️ ResInsight 的默认其实是**连续渐变**(LINEAR_CONTINUOUS,
     RimRegularLegendConfig.cpp:113-118), 默认级数 8。我们选离散是为了让读者能把
     颜色对回具体数值区间; 这是取舍, 不是规范。
  2. **每个单元画深色边线**(#2b2b2b)。⚠️ ResInsight 默认确实画全网格边线
     (MeshModeType 默认 FULL_MESH), 但默认边线色是**浅灰 (0.92,0.92,0.92)**,
     近黑 (0.08) 是留给**断层线**的。我们用深色是为了在缩图里仍看得见网格。
  3. **实心不透明, 不做阈值筛选**(ResInsight 也支持过滤器, 所以这同样是选择)。我早先按 |effect|>阈值 抠出"效应体"再半透明
     叠加, 结果是满屏漂浮碎片(confetti), 既不像工业图也读不出空间连续性。
     工业软件是**每个单元都上色**, 近零值自然落在色标中间的浅色带。
  4. 画**差值/效应**用发散色且零点居中 —— 这条**与 ResInsight 默认一致**:
     `isFlowResultWithBothPosAndNegValues` 为真时它自动选 BLUE_WHITE_RED 且
     centerLegendAroundZero=true (RimRegularLegendConfig.cpp:1194-1198)。
     绝对量另有公约: SWAT 用**反向**彩虹(高含水=蓝), 三相色是 SWAT 蓝/SGAS 绿/SOIL 红。
     ⚠️ 注释里原先写的"Petrel 风格彩虹"没有一手依据, 已删 —— SLB/CMG 的默认值
     检索不到任何官方公开文档。

另外三个已修的坑, 别再犯:
  - **角点绕序**: resdata 的 0..3 是**顶面**的 (i,j)(i+1,j)(i,j+1)(i+1,j+1),
    绕一圈是 0→1→3→2 而不是 0→1→2→3; 又因渲染时深度取负, 顶面在图上朝"上",
    要与 VTK"先底面后顶面"的约定对调。直接按 0..7 送 VTK_HEXAHEDRON 会让每个
    六面体拧成自相交的蝴蝶结(实测 2000 单元里 954 个负体积)。
    修好后负体积 0 个; VTK 体积中位 55,761 m³, 与 resdata 的 cell_volume 独立算的
    55,784 m³ 相关 0.9999986(中位相对误差 0.096%)。按原始 0..7 绕序则有 22,336 个负体积。
  - **垂向拉伸**: 实测单元中位边长 I 87.0 m / J 91.6 m / K 9.3 m ≈ 9.4:1,
    构造倾角只有 4.4°(7.7 km 落差 594 m)。早先 VE=12 把缓倾角拉成 45° 陡坡。
    VE=4 是**显式的显示选择**(把 4.4° 画成约 17°), 不是真实几何 —— 图注必须写明。
    参考: ResInsight 的默认 Z Scale 就是 5.0, 且默认在视图里显示 "Z: n" 标签,
    所以"垂向夸张 + 标注出来"本身是这一行的常规做法。
  - **坐标框**: PyVista 的 show_bounds 在这个尺度下把刻度标签挤成一团乱码,
    而且轴名会错位。工业软件本来也不画满框, 所以只留方位标 + 图注写尺度。

VTK 渲染不了中文, 图内文字一律 ASCII。
--------------------------------------------------------------------------

用法:
    python plot_reservoir_3d.py --inj C-3H
    python plot_reservoir_3d.py --inj C-3H --views solid,layer,section --layer 11
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import paths as P

GRID = P.NORNE_DECK / "out" / "NORNE_ATW2013.EGRID"
SCH = P.NORNE_DECK / "INCLUDE" / "BC0407_HIST01122006.SCH"
EFFDIR = ROOT / "_pipelines" / "causal_bench"
FIGDIR = ROOT / "_figures"
CACHE = P.NORNE_GRID_CACHE

RESDATA_TO_VTK = [4, 5, 7, 6, 0, 1, 3, 2]     # 见文件头; 改这行前先看负体积断言
VE = 4.0                                       # 垂向拉伸; 真实倾角 4.4°, 拉太狠伪造陡构造
N_BANDS = 11                                   # 工业软件典型 10~11 级离散色带

# 发散色标(效应类, 零点居中)。冷=水被挤走, 暖=水到这里。
GEO_DIVERGING = ["#1E466E", "#376795", "#528FAD", "#72BCD5", "#AADCE0",
                 "#F2F2F2",
                 "#FFE6B7", "#FFD06F", "#F7AA58", "#EF8A47", "#C50032"]
# 绝对量类(PRESSURE 等)的正向彩虹, 对齐 ResInsight 的 NORMAL 色带锚点。
# 注意 SWAT 在 ResInsight 里用的是**反向**彩虹(高含水=蓝), 画绝对饱和度时要翻转。
GEO_RAINBOW = ["#2B2C8C", "#2E6FD0", "#39B7E8", "#4FD08A", "#A8DC49",
               "#F2E31C", "#F7A93B", "#EE5C2B", "#C1122A"]

INJECTORS = ("C-1H", "C-2H", "C-3H", "C-4AH", "C-4H", "F-1H", "F-2H", "F-3H", "F-4H")


# ---------------------------------------------------------------- 几何

def build_grid_cache() -> dict:
    """从 EGRID 抽每个活动单元的 8 个角点。缓存, 因为逐单元取角点很慢。"""
    if CACHE.exists():
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    from resdata.grid import Grid

    g = Grid(str(GRID))
    n = g.getNumActive()
    corners = np.empty((n, 8, 3), np.float32)
    ijk = np.empty((n, 3), np.int32)
    for a in range(n):
        gi = g.get_global_index(active_index=a)
        for c in range(8):
            corners[a, c] = g.get_cell_corner(c, global_index=gi)
        ijk[a] = g.get_ijk(active_index=a)
    out = {"corners": corners, "ijk": ijk,
           "dims": np.array([g.getNX(), g.getNY(), g.getNZ()], np.int32)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def make_mesh(corners: np.ndarray, ijk: np.ndarray, values: np.ndarray, ve: float):
    """角点 → PyVista 六面体非结构网格 + 原点平移 + 垂向放大。

    返回 (mesh, origin)。origin 用来把井轨迹搬到同一坐标系里。
    """
    import pyvista as pv

    n = len(corners)
    pts = corners[:, RESDATA_TO_VTK].reshape(-1, 3).astype(np.float64)
    origin = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 2].min()])
    pts[:, 0] -= origin[0]
    pts[:, 1] -= origin[1]
    pts[:, 2] = -(pts[:, 2] - origin[2]) * ve            # 深度向下为负 + 垂向放大
    cells = np.hstack([np.full((n, 1), 8, np.int64),
                       np.arange(n * 8, dtype=np.int64).reshape(n, 8)]).ravel()
    mesh = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.HEXAHEDRON, np.uint8), pts)
    mesh.cell_data["effect"] = values.astype(np.float32)
    for a, name in enumerate("IJK"):
        mesh.cell_data[name] = ijk[:, a].astype(np.float32)

    vol = mesh.compute_cell_sizes(length=False, area=False).cell_data["Volume"]
    neg = int((vol < 0).sum())
    print(f"活动单元 {n:,}  负体积 {neg}  体积中位 {np.median(np.abs(vol)) / ve:,.0f} m3")
    if neg:
        raise SystemExit(f"🔴 {neg} 个单元体积为负 —— RESDATA_TO_VTK 绕序错了, 先修再画")
    return mesh, origin


def well_paths(corners: np.ndarray, ijk: np.ndarray, origin: np.ndarray,
               ve: float, names: set[str]) -> dict[str, np.ndarray]:
    """从 SCHEDULE 的 COMPDAT 取射孔单元中心, 串成井轨迹折线(与 mesh 同坐标系)。"""
    if not SCH.exists():
        print("⚠ 找不到 SCHEDULE, 不画井轨迹")
        return {}
    lut = {(int(a), int(b), int(c)): k for k, (a, b, c) in enumerate(ijk)}
    center = corners.mean(axis=1)                        # (n, 3) 单元中心

    txt = SCH.read_text(errors="ignore")
    comp: dict[str, set] = {}
    for blk in re.findall(r"^COMPDAT(.*?)^/\s*$", txt, re.S | re.M):
        for line in blk.splitlines():
            line = line.split("--")[0].strip()
            if not line or line == "/":
                continue
            t = line.replace("/", "").split()
            if len(t) < 5:
                continue
            try:
                w, i, j, k1, k2 = t[0].strip("'"), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            except ValueError:
                continue
            if w in names:
                comp.setdefault(w, set()).update((i - 1, j - 1, k) for k in range(k1 - 1, k2))

    out = {}
    for w, cells in comp.items():
        keep = [c for c in sorted(cells, key=lambda c: c[2]) if c in lut]
        if len(keep) < 2:
            continue
        p = center[[lut[c] for c in keep]].astype(np.float64)
        p[:, 0] -= origin[0]
        p[:, 1] -= origin[1]
        p[:, 2] = -(p[:, 2] - origin[2]) * ve
        out[w] = (p, np.array(keep, np.int32))
    return out


# ---------------------------------------------------------------- 渲染

def render(mesh, wells, inj, view, lim, ve, out, layer, bar_title, title, note=None):
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("geo_div", GEO_DIVERGING, N=N_BANDS)
    # 工业软件必画单元边线; 44k 单元时线宽要细, 否则边线糊掉填色
    style = dict(cmap=cmap, clim=(-lim, lim), n_colors=N_BANDS, show_scalar_bar=False,
                 show_edges=True, edge_color="#2b2b2b", line_width=0.2,
                 ambient=0.34, diffuse=0.76, specular=0.05, lighting=True,
                 nan_color="#b9bcc0")     # |t|<2 的单元:中性灰, 与"零效应"的浅色区分开
    bar = dict(title=bar_title, title_font_size=19, label_font_size=15,
               n_labels=6, fmt="%+.3f", color="black", vertical=True,
               position_x=0.885, position_y=0.17, height=0.66, width=0.036,
               n_colors=N_BANDS, interactive=False)

    p = pv.Plotter(off_screen=True, window_size=(1900, 1250))
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")

    I, J, K = (mesh.cell_data[c] for c in "IJK")
    if view == "layer":
        body = mesh.extract_cells(np.flatnonzero(K == layer))
        cam, zoom, sub = "xy", 1.0, f"single layer K={layer + 1}, map view"
    elif view == "section":
        # 🔴 必须切在**目标井自己所在的 J** 上。早先切在 J=median(=54), 而 C-3H 在
        #    J=13, 等于切了一刀这口井根本没穿过的岩石, 图上还有一堆悬空的别的井。
        # 🔴 只取**单个 J 面**, 不再取 ±1 的三层薄片。三层时相机侧是 J=j0+1,
        #    目标井在中间列, 21 个射孔层全被前列单元挡住 —— 图题写"穿过 C-3H 的剖面",
        #    井却一个像素都看不见。单面才名副其实。
        j0 = float(np.median(wells[inj][1][:, 1])) if inj in wells else float(np.median(J))
        body = mesh.extract_cells(np.flatnonzero(J == j0))
        cam, zoom, sub = "slab", 1.45, f"cross-section through {inj} at J={int(j0) + 1}, vert. exagg. x{ve:g}"
        wells = {w: v for w, v in wells.items() if abs(np.median(v[1][:, 1]) - j0) <= 2.0}
    elif view == "solid":
        body, cam = mesh, None
        zoom, sub = 1.3, f"full model, all {mesh.n_cells:,} cells, vert. exagg. x{ve:g}"
    else:
        # cutaway:把远端一半的**浅层**削掉, 让效应最强的那层露成一级台地。
        # 这是 Petrel/ResInsight 的 "cut box" 标准做法 —— 全实心模型只看得到外皮,
        # 而效应集中在 K≈layer 的一个薄层里。
        # (早先我改成按 |effect|>阈值 抠"效应体", 结果是满屏碎片, 见文件头;
        #  改成挖角块又只露出大片无效应的围岩, 都不如按深度削。)
        # 🔴 切除侧必须由**目标井位置**决定, 不能用 median(J)。C-3H 在 J=13、median 是 54,
        #    旧写法把远端切开、井所在的半区完全没动:实测距井 800 m 内强效应像素占比为 0,
        #    整片是白色外皮。按井在哪半边就切哪半边后, 该比例升到 73%。
        jc = np.median(J)
        jw = float(np.median(wells[inj][1][:, 1])) if inj in wells else jc
        far = (J > jc) if jw > jc else (J < jc)
        keep = ~(far & (K < layer))
        body, cam = mesh.extract_cells(np.flatnonzero(keep)), None
        side = ">" if jw > jc else "<"
        zoom, sub = 1.3, (f"cut-box view (J{side}{int(jc)+1} shallower than K={layer+1} removed), "
                          f"vert. exagg. x{ve:g}")

    p.add_mesh(body, scalars="effect", **style)
    p.add_scalar_bar(**bar)

    # 井轨迹:目标注水井粗黑管 + 名字, 其余注水井细灰管
    for w, (path, _wijk) in wells.items():
        hot = (w == inj)
        line = pv.Spline(path, max(len(path) * 3, 12))
        p.add_mesh(line.tube(radius=55 if hot else 26),
                   color="#000000" if hot else "#8a8a8a", lighting=False,
                   show_scalar_bar=False)
        if hot or view == "layer":
            p.add_point_labels([path[0]], [w], font_size=16, text_color="black",
                               shape=None, always_visible=True, show_points=False)

    if cam:
        # 平面图和剖面图在工业软件里都是**正交投影**;透视投影会把远端网格拉歪,
        # 剖面看上去像一堵斜墙而不是一刀切面。三维视图仍保留透视。
        p.enable_parallel_projection()
        if cam == "slab":
            # 🔴 Norne 的 J 轴走向是 NE-SW, 不与全局 Y 轴平行, 所以等 J 薄片的切面
            #    **不垂直于 Y**。早先直接用 camera_position="xz" 沿 Y 看, 剖面就是歪的。
            #    正确做法:对薄片点云做 PCA, 最小方差方向就是切面法向, 相机沿它看过去。
            # 🔴 法向必须**约束在水平面内**。等 J 面近似铅垂, 其法向本就该是水平的;
            #    不约束时 PCA 的最小奇异方向会随 VE 变化 —— 实测 ve=1 时 9 口井的法向
            #    全部偏离真实 J 面法向 51.7°~89.8°, 剖面静默变成俯视图。
            pts = body.points
            c = pts.mean(0)
            n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
            n[2] = 0.0
            nn = np.linalg.norm(n)
            if nn < 1e-6:          # 退化: 该面近水平, 无法定义水平法向
                n = np.array([1.0, 0.0, 0.0]); nn = 1.0
            n = n / nn
            if n[:2] @ np.array([1.0, 1.0]) < 0:
                n = -n                                   # 固定从同一侧看, 避免左右翻转
            span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            p.camera_position = [tuple(c + n * span), tuple(c), (0.0, 0.0, 1.0)]
            p.reset_camera()          # 🔴 正交投影下改机位不会自动设 parallel scale,
                                      #    不 reset 就一头扎进网格内部, 只剩一片灰。
        else:
            p.camera_position = cam
    else:
        p.camera_position = "iso"
        p.camera.azimuth, p.camera.elevation = -55, 20
    p.camera.zoom(zoom)                               # 不 zoom 四周全是白边
    p.add_text(title, position="upper_left", font_size=18, color="black")
    p.add_text(sub if not note else f"{sub}   |   {note}",
               position=(16, 18), font_size=13, color="#555555")
    p.show_axes()                                     # 方位标, 工业软件标配
    p.screenshot(str(out), transparent_background=False)
    p.close()
    print(f"  {view:8s} → {out.name}  ({out.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inj", default="C-3H")
    ap.add_argument("--views", default="cutaway,layer,section")
    ap.add_argument("--layer", type=int, default=None, help="layer 视图第几层(0 基), 默认取效应最强层")
    ap.add_argument("--ve", type=float, default=VE)
    ap.add_argument("--clip", type=float, default=99.0, help="色标截断分位, 防极值吃掉对比度")
    ap.add_argument("--no-mask", action="store_true", help="不把 |t|<2 的单元灰掉(默认会灰掉)")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    f = EFFDIR / f"effect_field_SWAT_{args.inj}.npy"
    if not f.exists():
        print(f"缺 {f}；先跑 causal_bench.py field --inj {args.inj}")
        return 1
    eff = np.load(f)
    meta_f = EFFDIR / f"effect_field_SWAT_{args.inj}_meta.json"
    em = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    # 🔴 色标标题必须写真实估计量。旧版写 "d(Sw)/d(inj rate)" 是错的:θ 是 log10 乘子
    #    且回归前标准化过, 系数是"每 +1 个 θ 标准差"而不是对物理注入率求导。
    fac = em.get("sd_as_rate_factor")
    bar_title = (f"dSWAT per +1 SD of log10 inj-target mult\n(1 SD = x{fac:.2f} target rate)"
                 if fac else "SWAT OLS coefficient (standardized theta)")

    # 🔴 标题不能笼统写 "causal effect on water saturation"。schema v1 的 θ 没分相态,
    #    而 C-1H/C-3H/C-4AH/C-4H 是水气混注 —— 对这些井, θ 同时放大了注水和注气,
    #    效应场是"复合干预"的效应, 说成注水波及是错的(C-3H 末期水率甚至是 0)。
    import norne_bulk as NB
    phases = sorted({ph for w, ph in NB.CONTROLS if w == args.inj.split("_")[0]})
    if em.get("schema", 1) >= 2 or "_" in args.inj:
        title = f"{args.inj}  effect on final SWAT  (randomised intervention arm)"
    else:
        tag = "+".join(phases) if phases else "?"
        title = f"{args.inj}  effect on final SWAT  [intervention scales {tag} together]"
        if len(phases) > 1:
            print(f"⚠ {args.inj} 是{tag}混注, schema v1 的 θ 未分相态 —— "
                  f"这张图是复合干预的效应, 不能讲成注水波及")

    tf = EFFDIR / f"effect_field_SWAT_{args.inj}_t.npy"
    note = None
    if tf.exists() and not args.no_mask:
        tst = np.load(tf)
        ins = np.abs(tst) < 2.0
        eff = eff.copy(); eff[ins] = np.nan       # 交给 nan_color 画成中性灰
        note = f"{ins.sum():,} cells ({ins.mean():.0%}) with |t|<2 shown in grey"
        print(f"不显著(|t|<2)单元 {ins.sum():,} ({ins.mean():.1%}) 以中性灰绘制, 不参与色标")
    elif not tf.exists():
        print("⚠ 找不到 t 统计量, 全部单元一律上色 —— 读图时不能假定都显著")

    G = build_grid_cache()
    corners, ijk, dims = G["corners"], G["ijk"], G["dims"]
    print(f"网格 {dims[0]}x{dims[1]}x{dims[2]}")
    if len(eff) != len(corners):
        print(f"🔴 效应场 {len(eff)} 值 vs 网格 {len(corners)} 活动单元, 对不上")
        return 1

    mesh, origin = make_mesh(corners, ijk, eff, args.ve)
    wells = well_paths(corners, ijk, origin, args.ve, set(INJECTORS) | {args.inj})
    print(f"井轨迹 {len(wells)} 条: {', '.join(sorted(wells))}")

    fin = eff[np.isfinite(eff)]
    lim = float(np.percentile(np.abs(fin), args.clip)) or float(np.abs(fin).max())
    nclip = int((np.abs(fin) > lim).sum())
    print(f"效应值域 [{np.nanmin(eff):+.4f}, {np.nanmax(eff):+.4f}]  "
          f"色标截到 ±{lim:.4f} ({args.clip:g} 分位), 饱和 {nclip:,} 个单元")
    note = ((note + f"; {nclip:,} clipped") if note else f"{nclip:,} cells clipped at colour limits")

    layer = args.layer
    if layer is None:                                 # 效应最强的那层最有信息量
        # 🔴 掩码后 eff 含 NaN, 必须用 nan* 版本。用 sum/argmax 会让含 NaN 的层排到
        #    最前面(argmax 遇 NaN 直接返回该下标), 实测把自动选层从 K=10 带偏到 K=1。
        #    另外改用**均值**:各层活动单元数不同(2019~2263), 用总和会偏向单元多的层。
        s = [np.nanmean(np.abs(eff[ijk[:, 2] == k])) if np.isfinite(eff[ijk[:, 2] == k]).any()
             else -np.inf for k in range(int(ijk[:, 2].max()) + 1)]
        layer = int(np.argmax(s))
        print(f"layer 视图自动选 K={layer + 1} (该层显著单元的平均 |效应| 最大)")

    outs = []
    for v in [s.strip() for s in args.views.split(",") if s.strip()]:
        o = FIGDIR / f"reservoir3d_{args.inj}_{v}.png"
        render(mesh, wells, args.inj, v, lim, args.ve, o, layer, bar_title, title, note)
        outs.append(o)

    if len(outs) > 1:
        from PIL import Image
        ims = [Image.open(o) for o in outs]
        w = max(i.width for i in ims)
        canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
        y = 0
        for i in ims:
            canvas.paste(i, ((w - i.width) // 2, y)); y += i.height
        comb = FIGDIR / f"reservoir3d_{args.inj}_combined.png"
        canvas.save(comb)
        print(f"\n合并 → {comb}  ({comb.stat().st_size / 1e6:.2f} MB)")

    json.dump({"inj": args.inj, "ve": args.ve, "n_bands": N_BANDS, "clim": lim,
               "estimand": em.get("estimand"), "sd_as_rate_factor": fac,
               "masked_insignificant": (not args.no_mask) and tf.exists(), "n_clipped": nclip,
               "layer": layer, "views": args.views, "n_cells": int(mesh.n_cells),
               "n_wells": len(wells)},
              open(FIGDIR / f"reservoir3d_{args.inj}_meta.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
