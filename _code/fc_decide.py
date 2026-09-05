#!/usr/bin/env python3
"""主线实验：预测段的注水策略决策 —— "一直短期采 vs 一直长期采，能差多少油"。

这是本项目最初就定下的头条问题。代理模型是**手段**，本脚本是**目的**。

## 问题

预测段 2006-12 → 2020-01（13 年）分 6 个决策时点，每个时点给 4 口注水井各定一个注水率。
**同样多的水，换一种分法**，能多采多少油？

| 策略 | 目标函数 |
|---|---|
| **短期最优** | 最大化**前 3 年**累计产油（"急需油"） |
| **长期最优** | 最大化**全 13 年**累计产油（"不急"） |
| **切换** | 前段按短期目标、后段按长期目标（"急的时候短期，不急了转长期"） |
| 基准 | deck 原方案（θ=0） |
| 均匀 | 四口井平均分（对照，说明"随便改"不等于"改好"） |

**🔴 硬约束：全期总注水量与基准一致。** 否则就成了"多注水多采油"这种废话结论。
用 softmax 重参数化把总量做成**结构性守恒**，不靠惩罚项凑。

## 复用历史段版(decision_net.py)踩过的坑

1. **softmax 份额而非直接优化速率** —— 直接优化时投影会把梯度抹掉（实测掉 20%）；
   softmax 让总量天然守恒。
2. **惩罚必须相对化** —— 第一版写成绝对量纲，pen=28.8 对上 obj≈7e7，比值 4e-07，等于没有。
   本版乘 `base_ref` 变成无量纲。
3. **只在满足约束的迭代里选最优** —— 否则会选中"偷偷多注水"的那一步。
4. **信赖域** —— tanh 限制在 ±trust_r·σ 内，不外推到训练分布之外。

## 🔴 必须与结果一起声明的三条限制（都是本项目实测出来的）

1. **地质维度是死的**：渗透率乘子对十年产油 R² 只有 **0.0008**（注入控制是 0.676）。
   所以"在多个留出地质上验证稳健性"这条**当前参数化下是空转的**，不能宣称。
2. **F-4H 是假旋钮**：BHP 上限 500 bar 截断，目标注水率只实现 **27.6%**。
   优化器给它的份额与实际注入量不成比例。
3. **预测段真实工况是水气交替(WAG)**：C-1H 注气 784,660、C-3H 938,575 sm³/d，
   我们只优化水，是**简化**。

## 用法

    python fc_decide.py plan     --gpu 0        # 用代理模型求三套方案
    python fc_decide.py verify   --jobs 12      # 🔴 用 OPM Flow 真跑一遍当裁判(纯 CPU)
    python fc_decide.py report
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

import forecast_gen as FG
import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_decide"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID
N_STAGE, INJ = FG.N_STAGE, FG.INJ_W
BASE_W = np.array([FG.BASE_WINJ[w] for w in INJ])
SHORT_YEARS = 3.0                                   # "短期"的定义:前 3 年


def horizon_mask(years):
    """前 `years` 年对应的时间格点掩码。"""
    return (DAYS - DAYS[0]) <= years * 365.25


# ---------------------------------------------------------------- 求方案
def plan(args) -> int:
    import torch
    import fc_decision as FD
    from fc_mech import PosNet
    import torch.nn.functional as F

    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    TH, Y, IA, FC = FD.load(args.n_pin)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    te, va, pool = idx[:400], idx[400:600], idx[600:]
    tr = pool[:args.ntrain]
    Yf = Y.reshape(n, -1); NCH = nw * NPH
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
    xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
    if len(tr) < 50:
        raise SystemExit(f"🔴 训练集只有 {len(tr)} 条(n_pin={args.n_pin} 太小,"
                         f"前 600 条被 test/val 占满)。请增大 --n-pin。")
    print(f"训练代理模型:{len(tr)} 样本  生产井 {nw}  θ {TH.shape[1]} 维\n")

    X = ((TH - xm) / xs).astype(np.float32)
    torch.manual_seed(0)
    net = PosNet(TH.shape[1], NCH, NT, "fourier", n_bands=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    A = torch.tensor(X[tr], device=dev)
    B = torch.tensor(((Yf[tr]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        pm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 64):
            b = pm[i:i+64]; opt.zero_grad()
            F.mse_loss(net(A[b]), B[b]).backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(X[te], device=dev)).cpu().numpy().reshape(len(te), -1)*ys+ym
    cum = lambda a: np.trapezoid(a.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    ct, cp = cum(Yf[te]), cum(P)
    acc = float(np.mean(np.abs(cp-ct)/np.abs(ct)))
    print(f"代理模型封存 test 误差 {acc:.3%}   ← 后面所有决策都建立在这个精度上\n")

    ymT = torch.tensor(ym, device=dev, dtype=torch.float32)
    ysT = torch.tensor(ys, device=dev, dtype=torch.float32)
    xmT = torch.tensor(xm, device=dev, dtype=torch.float32)
    xsT = torch.tensor(xs, device=dev, dtype=torch.float32)
    BW_t = torch.tensor(BASE_W, device=dev, dtype=torch.float32)
    m_short = torch.tensor(horizon_mask(SHORT_YEARS).astype(np.float32), device=dev)
    m_long = torch.ones(NT, device=dev)
    dT = torch.tensor(DAYS, device=dev, dtype=torch.float32)
    sig = float(TH.std())

    def oil_curve(theta_flat):
        """θ(24维, 物理量纲) → 各时刻总产油率 (NT,)"""
        z = (theta_flat - xmT) / xsT
        o = net(z[None])[0].reshape(-1) * ysT + ymT
        return o.reshape(nw, NPH, NT)[:, 0, :].sum(0)

    def integ(curve, mask):
        c = curve * mask
        return torch.trapezoid(c, dT)

    def water_total(theta_flat):
        """全期总注水量(与基准比)。θ 是 log10 乘子。"""
        r = (10.0 ** theta_flat.reshape(N_STAGE, len(INJ))) * torch.tensor(
            BASE_W, device=dev, dtype=torch.float32)[None]
        return r.sum()

    w0 = float(water_total(torch.zeros(N_STAGE*len(INJ), device=dev)))
    base_curve = oil_curve(torch.zeros(N_STAGE*len(INJ), device=dev))
    base_long = float(integ(base_curve, m_long))
    base_short = float(integ(base_curve, m_short))
    print(f"基准(θ=0):  全期 {base_long:,.0f}   前 {SHORT_YEARS:.0f} 年 {base_short:,.0f}")
    print(f"基准总注水(相对量) {w0:,.1f}\n")

    def solve(mask, ref, tag, stage_mask=None):
        """softmax 重参数化 + 信赖域。总量结构性守恒,不靠惩罚凑。"""
        z = torch.zeros(N_STAGE * len(INJ), device=dev, requires_grad=True)
        o = torch.optim.Adam([z], lr=args.lr)
        best, best_th = -1e30, None
        for it in range(args.steps):
            o.zero_grad()
            th = args.trust_r * sig * torch.tanh(z)
            if stage_mask is not None:                 # 切换策略:只放开指定阶段
                th = th * torch.tensor(stage_mask, device=dev, dtype=torch.float32)
            # 🔴 修正(冒烟测试抓到):此前对**乘子**做 softmax,守恒的是 Σq=4,
            #    而不是 Σ(q·BASE_W)。四口井基准率差 4.1 倍(8554 vs 2071),
            #    换份额会直接改变总注水量,"同样多的水换分法"这个前提被破坏。
            #    正确做法:对**水量份额**做 softmax,每段总水量钉死在基准值,
            #    优化器只决定这些水**怎么分**给四口井。
            frac = torch.softmax(th.reshape(N_STAGE, len(INJ)) * np.log(10.0), 1)
            rate = frac * BW_t.sum()                    # 每段总水量 ≡ 基准,结构性守恒
            th_eff = torch.log10(rate / BW_t[None] + 1e-12).reshape(-1)
            obj = integ(oil_curve(th_eff), mask)
            wr = (water_total(th_eff) - w0) / w0
            pen = args.water_pen * wr ** 2 * ref       # 🔴 相对化,乘 ref 变无量纲
            (-(obj - pen)).backward(); o.step()
            with torch.no_grad():
                v, wv = float(obj), float(abs(wr))
                if wv <= args.water_tol and v > best:  # 🔴 只在满足约束的迭代里选最优
                    best, best_th = v, th_eff.detach().cpu().numpy().copy()
        if best_th is None:
            raise SystemExit(f"🔴 {tag}: 没有任何迭代满足注水量约束(tol={args.water_tol}),拒绝输出")
        wfin = float(abs((water_total(torch.tensor(best_th, device=dev)) - w0) / w0))
        print(f"  {tag:10s} 目标 {best:,.0f}  vs 基准 {(best/ref-1)*100:+.2f}%   "
              f"注水量偏离 {wfin:.2%}")
        return best_th, best, wfin

    print(f"=== 求解(信赖域 {args.trust_r}σ, {args.steps} 步) ===")
    th_short, v_short, w_short = solve(m_short, base_short, "短期最优")
    th_long, v_long, w_long = solve(m_long, base_long, "长期最优")
    # 切换:前 3 个阶段按短期解、后 3 个按长期解
    sw = th_short.reshape(N_STAGE, -1).copy()
    sw[3:] = th_long.reshape(N_STAGE, -1)[3:]
    th_switch = sw.reshape(-1)
    th_unif = np.zeros(N_STAGE * len(INJ), dtype=np.float32)   # 基准即均匀乘子

    plans = {"baseline": np.zeros(N_STAGE*len(INJ)).tolist(),
             "short": th_short.tolist(), "long": th_long.tolist(),
             "switch": th_switch.tolist()}
    with torch.no_grad():
        pred = {k: {"long": float(integ(oil_curve(torch.tensor(v, device=dev, dtype=torch.float32)), m_long)),
                    "short": float(integ(oil_curve(torch.tensor(v, device=dev, dtype=torch.float32)), m_short))}
                for k, v in plans.items()}
    print(f"\n=== 代理模型的预测(还没有裁判) ===")
    for k, v in pred.items():
        print(f"  {k:9s} 全期 {v['long']:,.0f} ({v['long']/base_long-1:+.2%})   "
              f"前{SHORT_YEARS:.0f}年 {v['short']:,.0f} ({v['short']/base_short-1:+.2%})")
    gap = pred["long"]["long"] / pred["short"]["long"] - 1
    print(f"\n  🔴 代理模型预测的「只顾眼前的代价」= {gap:+.2%}")
    print(f"     ⚠️ 这只是代理模型说的。**必须用 OPM Flow 真跑一遍才算数** → fc_decide.py verify")
    json.dump({"surrogate_test_err": acc, "n_train": len(tr), "short_years": SHORT_YEARS,
               "plans": plans, "surrogate_pred": pred, "base_water": w0,
               "water_dev": {"short": w_short, "long": w_long},
               "caveats": [
                   "地质维度失效:渗透率乘子对十年产油 R²=0.0008,多地质稳健性当前参数化下无法宣称",
                   "F-4H 是假旋钮:BHP 500 bar 截断,目标注水率只实现 27.6%",
                   "预测段真实工况是 WAG(水气交替),本工作只优化水,是简化",
               ]},
              open(OUT / "plans.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/plans.json")
    return 0


# ---------------------------------------------------------------- 模拟器裁判
def verify(args) -> int:
    """🔴 用 OPM Flow 真跑,不是用代理模型自己验收自己。纯 CPU 任务。"""
    src = "plans_calibrated.json" if args.calibrated else "plans.json"
    pj = json.loads((OUT / src).read_text())
    print(f"用方案文件: {src}")
    work_root = FG.OUT / "_decide"
    work_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((FG.HIST / "meta.json").read_text())   # 🔴 meta 在 HIST 下,不在 OUT 下
    # 🔴 与 forecast_gen.py:283 保持一致:绝对路径 + deck 名前缀(不含扩展名)。
    #    我此前写成 os.path.relpath(HIST/"out", ...),既是相对路径又漏了 deck 名,
    #    RESTART 会找不到重启文件 —— 8000 个样本就是用下面这种写法跑出来的。
    hist_rel = str((FG.HIST / "out" / "NORNE_ATW2013").resolve())

    # 🔴 缓存键必须带**方案内容**的哈希。此前只按方案名,
    #    3σ 那轮求出新方案后 verify 全部 cached(0.0 min),用的还是 2σ 的旧模拟结果 ——
    #    这种 bug 不报错,最阴。
    import hashlib
    def _h(tag):
        b = np.asarray(pj["plans"][tag], dtype=np.float64).tobytes()
        return hashlib.sha256(b).hexdigest()[:8]
    sfx = "_cal" if args.calibrated else ""
    def one(tag):
        sh = OUT / "sim" / f"{tag}{sfx}_{_h(tag)}.npz"
        if sh.exists():
            return tag, "cached"
        th = np.asarray(pj["plans"][tag], dtype=np.float32).reshape(N_STAGE, len(INJ))
        w = work_root / f"{tag}{sfx}_{_h(tag)}"
        shutil.rmtree(w, ignore_errors=True)
        try:
            FG.build_restart_deck(w, th, hist_rel, meta["last_step"])
            cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{w}:/data", "-v", f"{FG.HIST/'out'}:{FG.HIST/'out'}:ro",
                   "-w", "/data", NB.IMAGE, "flow", NB.DECK,
                   "--output-dir=/data/out", f"--threads-per-process={args.threads}",
                   "--parsing-strictness=low"]
            with open(w / "run.log", "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=7200).returncode
            if rc != 0:
                return tag, f"flow rc={rc}"
            g = FG.harvest_fc(w)
            if g is None:
                return tag, "harvest 失败"
            sh.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sh, theta=th, **g)
            return tag, "ok"
        finally:
            shutil.rmtree(w, ignore_errors=True)

    tags = list(pj["plans"])
    print(f"用 OPM Flow 真跑 {len(tags)} 个方案(纯 CPU,{args.jobs} 并发)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for f in as_completed({ex.submit(one, t): t for t in tags}):
            tag, st = f.result()
            print(f"  {tag:9s} {st}   ({(time.time()-t0)/60:.1f} min)", flush=True)
    return report(args)


def report(args) -> int:
    src = "plans_calibrated.json" if args.calibrated else "plans.json"
    pj = json.loads((OUT / src).read_text())
    ms = horizon_mask(SHORT_YEARS)
    R = {}
    for tag in pj["plans"]:
        import hashlib
        h = hashlib.sha256(np.asarray(pj["plans"][tag], dtype=np.float64).tobytes()).hexdigest()[:8]
        f = OUT / "sim" / f"{tag}{'_cal' if args.calibrated else ''}_{h}.npz"
        if not f.exists():
            print(f"  {tag}: 还没跑"); continue
        d = np.load(f)
        ob = d["obs"]
        n_prod = len(NB.PRODUCERS)
        oil = np.stack([ob[(i*3)*NT:(i*3+1)*NT] for i in range(n_prod)])   # (井, 时刻)
        R[tag] = {"long": float(np.trapezoid(oil, DAYS, axis=-1).sum()),
                  "short": float(np.trapezoid(oil[:, ms], DAYS[ms], axis=-1).sum()),
                  "winj": float(d["field_cum"][1][-1] - d["field_cum"][1][0])}
    if "baseline" not in R:
        print("缺基准算例"); return 1
    b = R["baseline"]
    print(f"\n{'='*74}\n=== 🔴 OPM Flow 裁判结果(不是代理模型自己说的) ===\n")
    print(f"  {'方案':10s}{'全期产油':>16s}{'vs基准':>9s}{'前3年产油':>15s}{'vs基准':>9s}{'注水量偏离':>11s}")
    for tag in ("baseline", "short", "long", "switch"):
        if tag not in R:
            continue
        r = R[tag]
        print(f"  {tag:10s}{r['long']:>16,.0f}{(r['long']/b['long']-1)*100:>+8.2f}%"
              f"{r['short']:>15,.0f}{(r['short']/b['short']-1)*100:>+8.2f}%"
              f"{(r['winj']/b['winj']-1)*100:>+10.2f}%")
    if "short" in R and "long" in R:
        gap = R["long"]["long"] / R["short"]["long"] - 1
        print(f"\n  🔴 **只顾眼前的代价 = {gap:+.2%}**")
        print(f"     (一直按短期最优采 vs 一直按长期最优采,同样多的水)")
        sp = pj["surrogate_pred"]
        gp = sp["long"]["long"] / sp["short"]["long"] - 1
        print(f"     代理模型预测 {gp:+.2%}  →  代理误差 {abs(gap-gp)*100:.2f} 个百分点")
    print(f"\n  ⚠️ 必须与结果一起声明的限制:")
    for c in pj["caveats"]:
        print(f"     · {c}")
    json.dump({"sim": R, "surrogate_pred": pj["surrogate_pred"],
               "caveats": pj["caveats"]},
              open(OUT / "verify.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/verify.json")
    return 0


# ---------------------------------------------------------------- 实测注水量校正
def calib(args) -> int:
    """把**实测**注水量校回基准 —— 目标守恒 ≠ 实测守恒。

    ## 为什么必须做

    第一轮裁判结果:long +7.40%、switch +7.46% 的**实测**注水量超标。
    根因是 **F-4H 是假旋钮**:BHP 500 bar 上限截断,实测注水率几乎不随目标率变
    (实测 baseline→long 只变 −3.9%,而 F-2H 变了 +79.5%)。
    优化器"从 F-4H 拿水"拿不动、"给 F-2H 加水"加得上,净效果就是总量上涨。
    于是"同样多的水换分法"这个前提被击穿,`+4.73%` 那个数不成立。

    ## 做法

    给每套方案的**全部**目标率乘一个标量 s(等价于 θ 整体平移 log10 s),
    用**割线法**在 s 上求根,使实测总注水量 = 基准。每次评估 = 跑一次全模拟。
    因为 F-4H 不响应,s 的实际作用主要落在其余三口井上,但函数仍单调,割线法收敛快。

    🔴 关键:s 只是整体缩放,**不改变井间份额**,所以"换分法"这个研究对象没被动过。
    """
    import copy
    pj = json.loads((OUT / "plans.json").read_text())
    W0 = _sim_water("baseline", 0.0, args)
    print(f"基准实测总注水 {W0:,.0f}\n")
    out = {"baseline": {"s": 0.0, "water_dev": 0.0}}
    for tag in ("short", "long", "switch"):
        th0 = np.asarray(pj["plans"][tag], dtype=np.float64)
        pts = []                                            # (log10 s, 相对偏差)
        for it in range(args.max_iter):
            if it == 0:
                ls = 0.0
            elif it == 1:
                ls = -np.log10(1.0 + pts[0][1])             # 一阶猜测
            else:                                           # 割线法
                (x1, y1), (x2, y2) = pts[-2], pts[-1]
                ls = x2 - y2 * (x2 - x1) / (y2 - y1) if abs(y2 - y1) > 1e-12 else x2
                ls = float(np.clip(ls, -0.3, 0.3))
            W = _sim_water(tag, ls, args, theta=th0 + ls)
            dev = W / W0 - 1.0
            pts.append((ls, dev))
            print(f"  {tag:8s} iter{it}  s=10^{ls:+.4f}={10**ls:.4f}  "
                  f"实测注水 {W:,.0f}  偏离 {dev:+.2%}", flush=True)
            if abs(dev) <= args.water_tol:
                break
        best = min(pts, key=lambda t: abs(t[1]))
        out[tag] = {"s": best[0], "water_dev": best[1]}
        print(f"  → {tag} 选 s=10^{best[0]:+.4f},实测偏离 {best[1]:+.2%}\n")
    # 落盘校正后的方案
    pj2 = copy.deepcopy(pj)
    pj2["plans"] = {k: (np.asarray(v, dtype=np.float64) + out[k]["s"]).tolist()
                    for k, v in pj["plans"].items()}
    pj2["calibration"] = out
    pj2["caveats"].append(
        "方案经**实测注水量**标量校正(整体缩放,不改井间份额),"
        "因为 F-4H 被 BHP 截断导致目标守恒≠实测守恒")
    (OUT / "plans_calibrated.json").write_text(
        json.dumps(pj2, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {OUT}/plans_calibrated.json  →  下一步 verify --calibrated")
    return 0


def _sim_water(tag, ls, args, theta=None):
    """跑一次模拟,返回实测总注水量。结果按 (tag, s) 缓存。"""
    key = f"{tag}_s{ls:+.4f}"
    sh = OUT / "sim" / f"{key}.npz"
    if tag == "baseline" and abs(ls) < 1e-12 and (OUT / "sim" / "baseline.npz").exists():
        sh = OUT / "sim" / "baseline.npz"
    if not sh.exists():
        pj = json.loads((OUT / "plans.json").read_text())
        th = (np.asarray(pj["plans"][tag], dtype=np.float32) if theta is None
              else np.asarray(theta, dtype=np.float32)).reshape(N_STAGE, len(INJ))
        _run_sim(key, th, args)
        sh = OUT / "sim" / f"{key}.npz"
    d = np.load(sh)
    return float(d["field_cum"][1][-1] - d["field_cum"][1][0])


def _run_sim(key, th, args):
    work_root = FG.OUT / "_decide"
    work_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((FG.HIST / "meta.json").read_text())
    hist_rel = str((FG.HIST / "out" / "NORNE_ATW2013").resolve())
    w = work_root / key
    shutil.rmtree(w, ignore_errors=True)
    try:
        FG.build_restart_deck(w, th, hist_rel, meta["last_step"])
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{w}:/data", "-v", f"{FG.HIST/'out'}:{FG.HIST/'out'}:ro",
               "-w", "/data", NB.IMAGE, "flow", NB.DECK,
               "--output-dir=/data/out", f"--threads-per-process={args.threads}",
               "--parsing-strictness=low"]
        with open(w / "run.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=7200).returncode
        if rc != 0:
            raise SystemExit(f"🔴 {key}: flow rc={rc}")
        g = FG.harvest_fc(w)
        if g is None:
            raise SystemExit(f"🔴 {key}: harvest 失败")
        (OUT / "sim").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "sim" / f"{key}.npz", theta=th, **g)
    finally:
        shutil.rmtree(w, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "calib", "verify", "report"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-pin", type=int, default=8000)
    ap.add_argument("--ntrain", type=int, default=7400)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--trust-r", type=float, default=2.0)
    ap.add_argument("--water-pen", type=float, default=50.0)
    ap.add_argument("--water-tol", type=float, default=0.02)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--calibrated", action="store_true",
                    help="verify/report 使用 plans_calibrated.json")
    a = ap.parse_args()
    return {"plan": plan, "calib": calib, "verify": verify, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
