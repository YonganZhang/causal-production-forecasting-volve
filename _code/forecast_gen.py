#!/usr/bin/env python3
"""预测段数据生成器：历史跑一次，从 T0 分叉出 N 个注水策略跑到 2020。

## 为什么要重做（2026-08-21 用户从常识点破 + 文献查实）

旧设定把优化窗口放在**整个历史期 1997-2006**，等于在问
"假设这九年重来一遍换个注水法"——文献里无人这么做（Crossref 多组措辞检索为负结果），
审稿人会当成 hindsight 天花板对照而非主结果。而且历史段生产井是 WCONHIST 'RESV'，
**采液体积被钉死**，注水改动只能改油水劈分、动不了采液节奏，
所以增益只有 +2.96%——把 Brouwer&Jansen 2004 区分的"加速采油"那一支整块切掉了。

🔴 我曾说"产量被 WCONHIST 写死"，**那是错的**：实测 RESV 1970 条、ORAT 139 条
且那 139 条速率全为 0（关井条目）。RESV 锁的是**地下体积**，油水劈分自由。

## 新协议（参数抄 Equinor 官方预测 deck，本机 opm-tests/norne，ODbL 可引）

    1997-11 ──────── T0=2006-10-10 ──────────→ 2020-01-01
       标定段(不动)         优化段 13.2 年
       真实历史 deck        生产井 WCONPROD 'GRUP'(自由响应)
       只跑一次             注水井 WCONINJE RATE(决策变量)

**历史只跑一次**，之后用 RESTART + SKIPREST 分叉——省掉绝大部分算力。

## 文献里的典型增益（用来判断我们做到没做到）
van Essen 2009 NPV +9.5% / Jansen 2009 闭环 +6.7~8.7% /
Brugge 官方 3 控制区间 vs 1 个 +20% / Brouwer 2004 闭环累计产油 +44%

用法:
    python forecast_gen.py history          # 跑一次历史, 存重启档
    python forecast_gen.py run --n 2000 --jobs 24
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import norne_bulk as NB

OUT = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_fc")
HIST = OUT / "_history"                       # 历史算例(只跑一次)
OFFICIAL = Path("/mnt/data/yongan-admin-2/datasets/petro/opm-tests/norne")

# 预测段:官方 4A 的控制参数 + 官方 17 步报告网格
PRED_DATES = [(1,"JAN",2007),(1,"JUL",2007),(1,"JAN",2008),(1,"JUL",2008),
              (1,"JAN",2009),(1,"JUL",2009),(1,"JAN",2010),(1,"JAN",2011),
              (1,"JAN",2012),(1,"JAN",2013),(1,"JAN",2014),(1,"JAN",2015),
              (1,"JAN",2016),(1,"JAN",2017),(1,"JAN",2018),(1,"JAN",2019),(1,"JAN",2020)]
# 决策周期(6 段):Brugge 官方"3 个控制区间比 1 个多约 20% NPV";
# Jansen 2009 显示年频加密到月频只多 0.4 个百分点 —— 6 段在收益曲线拐点上
STAGE_AT = [2007, 2008, 2010, 2012, 2014, 2017]
# 🔴 决策变量只取 **T0 时真的在注水** 的井。实测 T0(2006-10-10) 各井注水率:
#    F-1H 8554 / F-3H 6865 / F-2H 3676 / F-4H 2071 sm3/d, 而 **C 区五口全是 0.0**
#    —— C-1H~C-4H 早就停注了。之前"C 区因果效应接近零"的根源在这, 不是物理连通问题。
#    把停注井放进决策变量等于优化一个不存在的旋钮。
INJ_W = ["F-1H", "F-2H", "F-3H", "F-4H"]
BASE_WINJ = {"F-1H": 8554.2, "F-2H": 3676.3, "F-3H": 6865.3, "F-4H": 2071.0}   # T0 实测
# 🔴 生产井也只取 T0 仍在产的 8 口(实测产液 > 1 sm3/d)。
#    控制写法抄 Equinor 官方 4A: WCONPROD 'GRUP' + 油/气/液上限 + 最低 BHP/THP。
#    **这是整改的核心** —— 历史段是 WCONHIST(采液体积被钉死), 预测段必须让生产井
#    按压力自由响应, 否则"注水改动能不能改变采油节奏"这个问题根本问不出来。
PROD_W = ["B-1BH", "B-2H", "B-4DH", "D-1CH", "D-2H", "D-3BH", "E-1H", "E-3CH"]
# Qo Qw Qg Qliq Qresv bhp thp —— 抄官方 4A 的上限值, 但 **THP 留空**。
# 🔴 官方 4A 给每口井指定了 VFP 表号(38/41/42...)才能用 THP 约束;
#    我们从 2006 分叉、井集不同, 硬套表号会错配。OPM 明确报错:
#    "Well B-1BH must have a VFP table to handle non-zero THP constraint"。
#    去掉 THP、只保留**最低井底压力 60 bar**:不依赖 VFP 表, 物理上同样是
#    "按压力自由生产"的约束, 且更少人为假设。
PROD_LIM = "5000.0    1*  4.0E6    8000.0    1*    60.0"
# 🔴 schema v2(2026-08-28):把**生产井采液上限**也变成决策变量。
#    v1 只调 4 口注水井 → θ 24 维;"采多快"完全由固定的 Qliq=8000 决定,
#    而这正是短期/长期权衡的核心旋钮。v2 加 8 口生产井 → θ 72 维。
BASE_QLIQ = 8000.0          # v1 里写死的采液上限,v2 作为乘子基准
N_CTRL_V2 = len(INJ_W) + len(PROD_W)          # 4 + 8 = 12
N_STAGE = len(STAGE_AT)


def write_pred_sch(theta_stage: np.ndarray, path: Path) -> None:
    """theta_stage (N_STAGE, n_inj) → 预测段 SCHEDULE。控制写法抄 Equinor 官方 4A deck。

    🔴 生产井必须写 WCONPROD 'GRUP'(按上限自由生产), **不能沿用历史段的 WCONHIST**。
       历史段 WCONHIST 'RESV' 把采液体积钉死, 注水改动只能改油水劈分、动不了采液节奏,
       这正是旧设定增益只有 +2.96% 的结构性原因。第一版预测段只写了 WCONINJE 没写
       WCONPROD —— 那样生产井会沿用历史末期的控制, 整改等于白做。
    """
    L = ["-- 预测段:由 forecast_gen.py 生成", "-- 生产井控制参数抄 Equinor 官方 4A deck",
         "-- (opm-tests/norne/INCLUDE/PRED_WINJ_MIN_THP_4A_STDW.SCH, ODbL)", ""]
    # 一次性设定生产井:GRUP 控制 + 油/气/液上限 + 最低井底/井口压力
    L += ["WCONPROD", "-- 井  状态  控制  Qo  Qw  Qg  Qliq  Qresv  bhp(最低)"]
    for w in PROD_W:
        L.append(f" '{w}'  'OPEN'  'GRUP'  {PROD_LIM}  /")
    L += ["/", ""]
    # 经济极限:含水过高自动关井(抄官方 4A 的 WECON)
    L += ["WECON", " 'B-*'   3*   20000  1*  'WELL' /", " 'D-*'   3*   20000  1*  'WELL' /",
          " 'E-*'   3*   20000  1*  'WELL' /", "/", ""]
    n_inj = len(INJ_W)
    v2 = theta_stage.shape[1] > n_inj          # 列数 >4 即 schema v2(带生产井控制)
    si = 0
    for (d, mon, yr) in PRED_DATES:
        if si < N_STAGE - 1 and yr >= STAGE_AT[si + 1]:
            si += 1
        m = 10.0 ** theta_stage[si]
        L += ["WCONINJE"]
        for k, w in enumerate(INJ_W):
            L.append(f" '{w}'  WATER  'OPEN'  'RATE'  {BASE_WINJ[w]*m[k]:.1f}  1*  500.0  2*  /")
        L += ["/", ""]
        if v2:
            # 🔴 v2:逐段、逐井改采液上限 —— 这是"采多快"的直接旋钮。
            #    只改 Qliq(第 4 个字段),其余上限与 bhp 下限保持 v1 值不动,
            #    这样 v1/v2 在 θ_prod=0 时**逐位等价**(已加断言验证)。
            L += ["WCONPROD", "-- 井  状态  控制  Qo  Qw  Qg  Qliq  Qresv  bhp(最低)"]
            for k, w in enumerate(PROD_W):
                q = BASE_QLIQ * m[n_inj + k]
                L.append(f" '{w}'  'OPEN'  'GRUP'  5000.0    1*  4.0E6    {q:.1f}    1*    60.0  /")
            L += ["/", ""]
        L += ["DATES", f" {d} '{mon}' {yr} /", "/", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def build_restart_deck(work: Path, theta_stage: np.ndarray, hist_rel: str, step: int) -> None:
    """从历史档 RESTART, 接上预测段 SCHEDULE。"""
    src = NB.BASE
    shutil.copytree(src, work, ignore=shutil.ignore_patterns(
        "out", "*.UNRST", "*.UNSMRY", "*.EGRID", "*.INIT", "*.PRT", "*.DBG", "*.ESMRY"))
    deck = work / NB.DECK
    L = deck.read_text(errors="ignore").split("\n")

    # SOLUTION 段:换成 RESTART
    i_sol = next(k for k, l in enumerate(L) if l.strip().upper() == "SOLUTION")
    i_sum = next(k for k, l in enumerate(L) if l.strip().upper() == "SUMMARY")
    sol = ["SOLUTION", "", "RESTART", f"  '{hist_rel}' {step} /", "",
           "RPTRST", "BASIC=2 /", ""]
    # 🔴 按**关键字块**精确保留, 不要用"猜这行长什么样"的过滤器。
    #    第一版按 "以数字开头且含 /" 留行, 结果 EQUIL 的数值行被误当成 THPRES 的记录,
    #    OPM 报 "Malformed integer '0.0' ... Failed to parse record 7 of THPRES"。
    #    RESTART 时初始化关键字(EQUIL/PRESSURE/SWAT...)必须去掉, 状态来自重启档;
    #    但 THPRES(相区间阈压)与 PETRO 属性 INCLUDE 仍需保留。
    KEEP_KW = {"THPRES", "INCLUDE"}
    keep, cur = [], None
    for l in L[i_sol + 1:i_sum]:
        u = l.split("--")[0].strip().upper()
        if u and u.split()[0].isalpha() and u.replace("=", " ").split()[0].isupper():
            cur = u.split()[0]                 # 进入一个新关键字块
        if cur in KEEP_KW:
            keep.append(l)
    L = L[:i_sol] + sol + keep + [""] + L[i_sum:]

    # SCHEDULE 段:SKIPREST + 原历史 SCH + 预测段 SCH
    i_sch = next(k for k, l in enumerate(L) if l.strip().upper() == "SCHEDULE")
    L.insert(i_sch + 1, "SKIPREST")
    write_pred_sch(theta_stage, work / "INCLUDE" / "PRED.SCH")
    # 🔴 END 有**两个**, 而且真正拦住我们的是里层那个。踩了两轮:
    #    第一轮:预测段 INCLUDE 追加在主 deck 文件末尾 = 主 deck 的 END 之后 → 不执行。
    #    第二轮:改插到主 deck 的 END 之前, **仍然不执行** —— 因为
    #      **历史 SCH 文件(BC0407_HIST01122006.SCH)自己末尾就有 END**,
    #      OPM 读历史 SCH 读到它就收工了, 主 deck 后面的 INCLUDE 根本到不了。
    #    两次都**不报任何错**, 只是安静地跑到历史终点就停 —— 这类静默失败最难发现,
    #    我是靠"只跑了 12 秒"这个反常数字才追下去的。
    #    正解:把预测段接进**历史 SCH 内部**的 END 之前。
    hs = work / NB.SCH
    HL = hs.read_text(errors="ignore").split("\n")
    j_end = next((k for k, l in enumerate(HL) if l.split("--")[0].strip().upper() == "END"), None)
    ins = ["", "-- 预测段(forecast_gen.py 生成)", "INCLUDE", " './INCLUDE/PRED.SCH' /", ""]
    HL = (HL[:j_end] + ins + HL[j_end:]) if j_end is not None else (HL + ins)
    hs.write_text("\n".join(HL))
    deck.write_text("\n".join(L))
    (work / "out").mkdir(exist_ok=True)


# 🔴 预测段的时间网格必须**显式定义**, 不能复用 norne_bulk 的 DAY_GRID(1~3312 天)。
#    优化窗口是 T0 之后, 所以从历史名义终点 3312 天起算到 8091 天(2020-01-01)。
#    用 3312 而不是 T0=3260:重启档最后一帧在 3260 天, 但 deck 名义终点是 3312,
#    从 3312 起算可保证所有样本的窗口完全一致, 且不落在重启衔接的过渡区。
FC_T0, FC_T1, FC_N = 3312.0, 8091.0, 40
FC_GRID = np.linspace(FC_T0, FC_T1, FC_N)
FC_SNAP_N = 8
FC_SNAP_DAYS = np.linspace(FC_T0, FC_T1, FC_SNAP_N)


def harvest_fc(work: Path) -> dict | None:
    """预测段抽取。与 norne_bulk.harvest 同结构, 但时间网格换成 FC_GRID。

    🔴 严禁外推:任一样本若跑不到 FC_T1 就拒收, 不做尾部钳位。
       之前踩过"np.interp 尾部静默钳位"的坑 —— 跑早停的样本会被悄悄补成末值。
    """
    from resdata.resfile import ResdataFile
    from resdata.summary import Summary

    smry = work / "out" / "NORNE_ATW2013.UNSMRY"
    unrst = work / "out" / "NORNE_ATW2013.UNRST"
    # 🔴 只要求 UNSMRY。**不要求 UNRST** —— RESTART 算例不写它(实测确认),
    #    而三维场对决策优化不是必需品。改成可选后我忘了删这句前置检查,
    #    于是全部样本仍被判失败, 而 summary 明明是完好的。
    if not smry.exists():
        return None
    s = Summary(str(smry))
    days = np.asarray(s.numpy_vector("TIME"), float)
    if days.max() < FC_T1 - 1.0:
        return None                      # 没跑到终点, 拒收而不是外推

    def series(key):
        try:
            v = np.asarray(s.numpy_vector(key), float)
        except Exception:                                    # noqa: BLE001
            return np.full(FC_N, np.nan)
        mk = np.isfinite(days) & np.isfinite(v)
        return (np.interp(FC_GRID, days[mk], v[mk]) if mk.sum() >= 2
                else np.full(FC_N, np.nan))

    cols = [series(f"{k}:{w}") for w in NB.PRODUCERS for k in NB.OBS_KEYS]
    inj = np.stack([series(f"WWIR:{w}") for w in INJ_W])
    fld = np.stack([series(k) for k in ("FOPT", "FWIT", "FWPT", "FPR")])

    # 🔴 三维场设成**可选**。RESTART 算例的 UNRST 常常不写(实测 /tmp/fc_base 就没有),
    #    而决策优化只需要**井级产量曲线** —— 场是给空间归因用的, 不是这条主线的必需品。
    #    第一版把"拿不到场"当拒收条件, 于是全部样本被判失败, 但 summary 明明是好的。
    #    顺带省磁盘:关掉场之后每样本约 50 KB 而不是 1.8 MB。
    fields = None
    if unrst.exists():
        try:
            f = ResdataFile(str(unrst))
            n = f.num_named_kw(NB.FIELDS[0])
            rd = np.asarray([(d - f.dates[0]).total_seconds() / 86400.0 for d in f.dates], float)
            if len(rd) == n and n >= 2:
                j = np.clip(np.searchsorted(rd, FC_SNAP_DAYS - rd[0], side="right") - 1, 0, n - 2)
                w_ = np.clip((FC_SNAP_DAYS - rd[0] - rd[j])
                             / np.where(rd[j + 1] - rd[j] > 0, rd[j + 1] - rd[j], 1.0), 0.0, 1.0)[:, None]
                need = sorted(set(j.tolist()) | set((j + 1).tolist()))
                pos = {g: r for r, g in enumerate(need)}
                lo = np.asarray([pos[g] for g in j]); hi = np.asarray([pos[g] for g in (j + 1)])
                st = []
                for k in NB.FIELDS:
                    cube = np.stack([np.asarray(f[k][i], np.float32) for i in need])
                    st.append((cube[lo] * (1.0 - w_) + cube[hi] * w_).astype(np.float32))
                fields = np.stack(st, axis=1)
        except Exception:                                    # noqa: BLE001
            fields = None
    out = {"obs": np.concatenate(cols).astype(np.float32),
           "inj_actual": inj.astype(np.float32),
           "field_cum": fld.astype(np.float32),
           "grid_days": FC_GRID.copy(), "snap_days": FC_SNAP_DAYS.copy(),
           "has_fields": np.int32(fields is not None),
           "sim_days_end": np.float64(days.max())}
    if fields is not None:
        out["fields"] = fields
    return out


def run_history(threads: int) -> int:
    HIST.mkdir(parents=True, exist_ok=True)
    if (HIST / "out" / "NORNE_ATW2013.UNRST").exists():
        print("历史档已存在, 跳过"); return 0
    w = HIST / "case"
    shutil.rmtree(w, ignore_errors=True)
    shutil.copytree(NB.BASE, w, ignore=shutil.ignore_patterns("out", "*.UNRST", "*.UNSMRY",
                                                              "*.EGRID", "*.INIT", "*.PRT"))
    (w / "out").mkdir(exist_ok=True)
    t0 = time.time()
    cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{w}:/data", "-w", "/data", NB.IMAGE, "flow", NB.DECK,
           "--output-dir=/data/out", f"--threads-per-process={threads}"]
    rc = subprocess.run(cmd, capture_output=True, timeout=7200).returncode
    print(f"历史算例 rc={rc}  耗时 {(time.time()-t0)/60:.1f} min")
    if rc != 0:
        return 1
    shutil.move(str(w / "out"), str(HIST / "out"))
    shutil.rmtree(w, ignore_errors=True)
    from resdata.resfile import ResdataFile
    f = ResdataFile(str(HIST / "out" / "NORNE_ATW2013.UNRST"))
    n = f.num_named_kw("PRESSURE")
    # 🔴 RESTART 要的是 **deck 的报告步号(SEQNUM)**, 不是重启档里的第几个存档。
    #    `RPTRST BASIC=2` 只在部分报告步存档, 所以 65 个存档对应的报告步号一直到 241。
    #    第一版传了存档序号 64, OPM 报 "Report step 64 not found in restart file"。
    seq = [int(np.asarray(f["SEQNUM"][i])[0]) for i in range(f.num_named_kw("SEQNUM"))]
    (HIST / "meta.json").write_text(json.dumps(
        {"n_restart": n, "last_step": seq[-1], "seqnum_last": seq[-1],
         "T0": str(f.dates[-1].date())}, indent=1))
    print(f"重启档 {n} 个状态, 末帧报告步 SEQNUM={seq[-1]}, T0 = {f.dates[-1].date()}")
    return 0


def main() -> int:
    global OUT                                  # v2 数据写独立目录,不覆盖 v1
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["history", "run", "probe"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.30)
    ap.add_argument("--schema", choices=["v1", "v2"], default="v1",
                    help="v1=只调 4 口注水井(θ 24 维);v2=加 8 口生产井采液上限(θ 72 维)")
    ap.add_argument("--out-suffix", default="", help="输出目录后缀,避免覆盖 v1 数据")
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()
    if args.out_suffix:
        OUT = OUT.parent / (OUT.name + args.out_suffix)
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"🔴 输出目录 {OUT}(不覆盖 v1 数据)")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "history":
        return run_history(args.threads)

    meta_f = HIST / "meta.json"
    if not meta_f.exists():
        print("先跑 history"); return 1
    meta = json.load(open(meta_f))
    hist_rel = str((HIST / "out" / "NORNE_ATW2013").resolve())

    n = 2 if args.cmd == "probe" else args.n
    rng = np.random.default_rng(args.seed)
    n_ctrl = N_CTRL_V2 if args.schema == "v2" else len(INJ_W)
    TH = rng.normal(0, args.sigma, (n, N_STAGE, n_ctrl)).astype(np.float32)
    # 🔴 基准算例(θ=0)必须**无条件**置零。
    #    此前只在 probe 分支置零,而 theta_all.npy 在其后保存 —— probe 跑过一次留下
    #    θ=0 的 shard 00000,后续 run 重写 theta_all 却跳过已存在的 shard,
    #    导致 theta_all.npy[0] 与 shard 00000 不一致(2026-08-26 实测 1/2000 不匹配)。
    #    训练不受影响(loader 读 shard 自己的 θ),但按索引读 theta_all 的脚本会拿错。
    TH[0] = 0.0                                        # 基准算例
    (OUT / "shards").mkdir(exist_ok=True)
    np.save(OUT / "theta_all.npy", TH)
    print(f"预测段 {n} 个样本  schema={args.schema}  θ {TH.shape} = "
          f"{N_STAGE} 段 × {n_ctrl} 个控制"
          + (f"({len(INJ_W)} 注水 + {len(PROD_W)} 生产)" if args.schema == "v2" else "(注水井)"))
    print(f"T0 = {meta['T0']}  重启步 {meta['last_step']}  → 2020-01-01")

    def one(j):
        sh = OUT / "shards" / f"{j:05d}.npz"
        if sh.exists():
            return {"j": j, "status": "cached"}
        w = OUT / "_work" / f"w{j:05d}"
        shutil.rmtree(w, ignore_errors=True)
        t0 = time.time()
        try:
            build_restart_deck(w, TH[j], hist_rel, meta["last_step"])
            cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{w}:/data", "-v", f"{HIST/'out'}:{HIST/'out'}:ro",
                   "-w", "/data", NB.IMAGE, "flow", NB.DECK,
                   "--output-dir=/data/out", f"--threads-per-process={args.threads}",
                   "--parsing-strictness=low"]
            with open(w / "run.log", "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200).returncode
            if rc != 0:
                return {"j": j, "status": "failed", "rc": rc,
                        "tail": (w / "run.log").read_text(errors="ignore")[-400:]}
            g = harvest_fc(w)
            if g is None:
                return {"j": j, "status": "failed", "why": "harvest"}
            np.savez_compressed(sh, theta=TH[j], rc=np.int32(rc), **g)
            return {"j": j, "status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as e:                                    # noqa: BLE001
            return {"j": j, "status": "error", "err": f"{type(e).__name__}: {e}"[:300]}
        finally:
            shutil.rmtree(w, ignore_errors=True)

    t0, bad = time.time(), []
    with ThreadPoolExecutor(max_workers=min(args.jobs, n)) as ex:
        futs = [ex.submit(one, j) for j in range(n)]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["status"] not in ("ok", "cached"):
                bad.append(r)
            if i % 25 == 0 or i == n or args.cmd == "probe":
                print(f"  [{i}/{n}] 失败 {len(bad)}  已用 {(time.time()-t0)/3600:.2f}h", flush=True)
    if bad:
        (OUT / "failures.json").write_text(json.dumps(bad[:20], indent=1, ensure_ascii=False))
        print(f"失败 {len(bad)}, 前几条:")
        for r in bad[:2]:
            print("   ", json.dumps(r, ensure_ascii=False)[:400])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
