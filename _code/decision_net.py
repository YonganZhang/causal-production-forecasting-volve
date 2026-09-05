#!/usr/bin/env python3
"""方向二主模型：注水方案 → 产量曲线，用它做短期/长期决策优化。

## 这是什么

一个网络：**输入注水方案 + 地质，输出 22 口生产井的完整产量曲线**。
曲线前段 = 短期产量，整条积分 = 长期累计产油。所以一个网络同时给短期和长期，
不必训两个（若训不稳再拆）。

## 为什么它的梯度能用来优化

训练数据来自**干预臂**——注入量是独立随机抽的，不是作业者定的。
所以 ∂产量/∂注入 是**因果导数**，沿它做梯度上升就是在优化真实的因果响应。
拿历史数据训的代理模型，梯度混杂了"作业者为什么这么注"，沿它优化会被误导。
（观测臂 2000 样本正是这条论断的对照实验。）

## 决策问题

**在总注水量相同的前提下**，怎么把水分给 9 口注水井，使目标最大。
🔴 总量约束必不可少：不约束的话"短期最优"和"长期最优"都会收敛到"全部拉满",
   三条策略变成同一个答案, 对比就没了。约束住之后比的才是**怎么分配**。

三种策略：
  short  只让前段产量最大          → 预期前期领先、后期崩
  long   让整条曲线积分最大        → 预期前期落后、后期反超
  switch 前 K 段用 short 的解, 之后用 long 的解（"急需油时短期"）

裁判是模拟器：把方案写进 deck 重跑，比最终累计产油。**不是比网络的预测值。**

用法:
    python decision_net.py train --gpu 6
    python decision_net.py optimize --horizon-short 8
    python decision_net.py verify          # 把优化出的方案送去真跑模拟器
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import norne_bulk as NB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "decision_net"
BULK = NB.OUT                                   # norne_v2
N_TIMES = NB.N_TIMES
DAY_GRID = NB.DAY_GRID
PRODUCERS = NB.PRODUCERS
CTRL = [f"{w}_{ph}" for w, ph in NB.CONTROLS]
WATER_IDX = [i for i, c in enumerate(CTRL) if c.endswith("_WATER")]


def load(max_n: int = 0):
    """θ (n,17) → 产油/产水曲线 (n,22,2,40)。只取井级量, 不碰三维场(那是另一个问题)。"""
    sh = sorted((BULK / "shards").glob("*.npz"))
    if max_n:
        sh = sh[:max_n]
    TH, Y, IA = [], [], []
    for p in sh:
        d = np.load(p)
        ob = d["obs"]
        TH.append(d["theta"])
        IA.append(d["inj_actual"])
        Y.append(np.stack([[ob[(i * 3 + k) * N_TIMES:(i * 3 + k + 1) * N_TIMES]
                            for k in (0, 1)] for i in range(len(PRODUCERS))]))
    return (np.stack(TH).astype(np.float32), np.stack(Y).astype(np.float32),
            np.stack(IA).astype(np.float32))


def cum_oil(Y: np.ndarray, upto: int | None = None) -> np.ndarray:
    """累计产油(梯形积分)。Y (n,22,2,40) 的通道 0 是产油率 m3/day。"""
    j = upto if upto else N_TIMES
    return np.trapezoid(Y[:, :, 0, :j], DAY_GRID[:j], axis=2).sum(1)


def build_net(d_in: int, d_out: int, hidden: int, depth: int):
    import torch.nn as nn
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


def train(args) -> int:
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    TH, Y, IA = load(args.max_n)
    n = len(TH)
    print(f"样本 {n}   θ {TH.shape[1]} 维   输出 {Y.shape[1]}×{Y.shape[2]}×{Y.shape[3]}")

    # 🔴 留出集按样本序号切, 且训练时绝不看 —— 优化阶段的验证也用它
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_te = max(200, n // 10)
    te, tr = idx[:n_te], idx[n_te:]
    print(f"训练 {len(tr)}   留出 {len(te)}")

    Yf = Y.reshape(n, -1)
    xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
    Xn = (TH - xm) / xs
    Yn = (Yf - ym) / ys

    net = build_net(TH.shape[1], Yf.shape[1], args.hidden, args.depth).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"网络 {args.depth} 层 × {args.hidden}   参数 {npar:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr = torch.tensor(Xn[tr], device=dev); Ytr = torch.tensor(Yn[tr], device=dev)
    Xte = torch.tensor(Xn[te], device=dev); Yte = torch.tensor(Yn[te], device=dev)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xtr[b]), Ytr[b])
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(net(Xte), Yte).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save({"state": net.state_dict(), "xm": xm, "xs": xs, "ym": ym, "ys": ys,
                        "shape": Y.shape[1:], "hidden": args.hidden, "depth": args.depth},
                       OUT / "net.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"早停于 epoch {ep}"); break
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:4d}  train {loss.item():.5f}  val {vl:.5f}  best {best:.5f}", flush=True)

    # ---- 用**物理量**报精度, 不只报归一化 MSE ----
    ck = torch.load(OUT / "net.pt", weights_only=False)
    net.load_state_dict(ck["state"]); net.eval()
    with torch.no_grad():
        P = (net(Xte).cpu().numpy() * ys + ym).reshape(-1, *Y.shape[1:])
    T = Y[te]
    co_p, co_t = cum_oil(P), cum_oil(T)
    err = np.abs(co_p - co_t) / np.maximum(np.abs(co_t), 1e-9)
    # 与"直接用训练集均值"这条平凡基线比 —— 打不过就没有价值
    base = np.abs(cum_oil(Y[tr]).mean() - co_t) / np.maximum(np.abs(co_t), 1e-9)
    r2 = 1 - ((co_p - co_t) ** 2).sum() / ((co_t - co_t.mean()) ** 2).sum()
    print(f"\n留出集 · 十年累计产油:")
    print(f"  相对误差 中位 {np.median(err):.2%}  90 分位 {np.percentile(err,90):.2%}")
    print(f"  R² {r2:.4f}")
    print(f"  平凡基线(训练集均值) 相对误差 中位 {np.median(base):.2%}  ← 必须打赢它")
    print(f"  训练耗时 {(time.time()-t0)/60:.1f} min")
    json.dump({"n": n, "n_train": len(tr), "n_test": len(te), "params": npar,
               "val_mse": best, "cumoil_relerr_median": float(np.median(err)),
               "cumoil_relerr_p90": float(np.percentile(err, 90)), "cumoil_r2": float(r2),
               "baseline_relerr_median": float(np.median(base)),
               "beats_baseline": bool(np.median(err) < np.median(base)),
               "test_idx": te.tolist()},
              open(OUT / "train.json", "w"), indent=1)
    return 0


def _load_net(dev):
    import torch
    ck = torch.load(OUT / "net.pt", weights_only=False)
    net = build_net(len(CTRL) + len(NB.PERM_REGIONS), int(np.prod(ck["shape"])),
                    ck["hidden"], ck["depth"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, ck


def fit_water_head(TH, IA, dev):
    """θ → **实测**总注水量 的小网络。约束必须加在这上面, 不能加在目标率上。

    🔴 为什么不能用目标率:deck 里 1463 条注入记录带 600 bar BHP 上限, 第 2947 天还插了
       `GCONINJE 'FIELD' 'WATER' 'RATE' 42000` —— 实测 p95 与 max 精确卡在 42,000。
       用目标率聚合当约束时, 优化出的"长期方案"实际只注了基准的 **0.43 倍**,
       那不是"同样多的水换个分法", 是"把水砍掉 57% 再去比产油"。
       (2026-08-21 Workflow 查出, 主会话独立复现。)
    """
    import torch
    y = np.trapezoid(IA[:, WATER_IDX, :].sum(1), DAY_GRID, axis=1).astype(np.float32)  # 总注水体积
    xm, xs = TH.mean(0), TH.std(0) + 1e-8
    ym, ys = y.mean(), y.std() + 1e-8
    net = build_net(TH.shape[1], 1, 128, 3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    X = torch.tensor((TH - xm) / xs, device=dev)
    Yt = torch.tensor(((y - ym) / ys)[:, None], device=dev)
    n_te = len(X) // 10
    for ep in range(300):
        perm = torch.randperm(len(X) - n_te, device=dev) + n_te
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(X[b]), Yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = (net(X[:n_te]).cpu().numpy().ravel() * ys + ym)
    err = np.abs(pr - y[:n_te]) / np.maximum(y[:n_te], 1e-9)
    print(f"注水量头: 留出相对误差 中位 {np.median(err):.2%}  p90 {np.percentile(err,90):.2%}")
    return net, float(xm.mean()), (xm, xs, ym, ys), float(np.median(err))


def optimize(args) -> int:
    """softmax 重参数化 + 信赖域, 求"短期最大"与"长期最大"的注水配比。

    🔴 三条纪律, 都是踩坑换来的(2026-08-21):
      1. **不用投影梯度**。在 θ=0 处目标梯度与约束法向夹角余弦 0.9615 —— 只有 27.5%
         是切向的。Adam 逐坐标归一化把 9 个相差 8.6 倍的梯度压平, 投影再统一减去常数
         精确抵消, 净位移只剩归一化残差。实测 600 步里 265 步在**往下走**。
         改用 softmax 份额:总量结构性守恒, 无需投影, 梯度不被抹掉。
      2. **必须记录最优迭代**。旧版只返回最后一步 —— 峰值 69.70M(第 225 步)被丢掉,
         返回的是第 600 步的 55M。
      3. **必须有信赖域**。不限制的话解会跑到 θ=-5(等于关井), 到训练集的 5-近邻距离 11.5
         (训练样本自身中位 1.42), 那是代理外推的幻觉区。
    """
    import torch
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    net, ck = _load_net(dev)
    TH, Y, IA = load(args.max_n)
    te = np.asarray(json.load(open(OUT / "train.json"))["test_idx"])
    wnet, _, (wxm, wxs, wym, wys), werr = fit_water_head(TH, IA, dev)

    rng = np.random.default_rng(1)
    tr_pool = np.setdiff1d(np.arange(len(TH)), te)
    geo_opt = TH[rng.choice(tr_pool, args.n_geo_opt, replace=False)][:, len(CTRL):]
    geo_te = TH[rng.choice(te, min(args.n_geo_test, len(te)), replace=False)][:, len(CTRL):]

    xm = torch.tensor(ck["xm"], device=dev); xs = torch.tensor(ck["xs"], device=dev)
    ym = torch.tensor(ck["ym"], device=dev); ys = torch.tensor(ck["ys"], device=dev)
    days = torch.tensor(DAY_GRID, device=dev, dtype=torch.float32)
    wxm_t = torch.tensor(wxm, device=dev); wxs_t = torch.tensor(wxs, device=dev)
    sig = float(TH[:, WATER_IDX].std())

    def full_theta(tw, g):
        gg = torch.tensor(g, device=dev, dtype=torch.float32)
        th = torch.zeros(len(gg), len(CTRL) + len(NB.PERM_REGIONS), device=dev)
        th[:, WATER_IDX] = tw; th[:, len(CTRL):] = gg
        return th

    def cum(tw, g, upto):
        P = (net((full_theta(tw, g) - xm) / xs) * ys + ym).reshape(len(g), *ck["shape"])
        return torch.trapezoid(P[:, :, 0, :upto], days[:upto], dim=2).sum(1).mean()

    def water(tw, g):
        return (wnet((full_theta(tw, g) - wxm_t) / wxs_t).mean() * wys + wym)

    torch.manual_seed(args.seed)          # 🔴 不加种子则 optimize 不可复现(实测两次给出不同方案)
    with torch.no_grad():
        z0 = torch.zeros(len(WATER_IDX), device=dev)
        w0 = float(water(z0, geo_opt))
        base_long = float(cum(z0, geo_te, N_TIMES)); base_short = float(cum(z0, geo_te, args.horizon_short))
    base_ref = base_long                  # 惩罚的量纲基准, 使 water_pen 无量纲可解释
    print(f"基准: 十年 {base_long:,.0f}   短期 {base_short:,.0f}   实测总注水 {w0:,.0f} m3")

    plans, report = {}, {}
    for name, upto in (("short", args.horizon_short), ("long", N_TIMES)):
        best = (-1e30, None)
        for restart in range(args.n_restart):
          g0 = (torch.zeros(len(WATER_IDX), device=dev) if restart == 0
                else torch.randn(len(WATER_IDX), device=dev) * 0.3)
          z = g0.clone().requires_grad_(True)
          opt = torch.optim.Adam([z], lr=args.opt_lr)
          for it in range(args.opt_iters):
              opt.zero_grad()
              th = args.trust_r * sig * torch.tanh(z)                 # 信赖域
              q = torch.softmax(th * np.log(10.0), 0) * len(WATER_IDX)  # 份额 → 总量守恒
              tw = torch.log10(q)
              obj = cum(tw, geo_opt, upto)
            # 🔴 惩罚必须**相对化**。第一版写成绝对量纲:pen = 2000×0.12² = 28.8,
            #    而 obj ≈ 7e7 —— 惩罚只有目标的 4.1e-07, 等于没加。
            #    梯度一路把注水推到 +13%, 600 步里只有第 0 步满足约束,
            #    于是 best 永远停在起点, 长期臂返回全零。那不是"物理上没有更好的解",
            #    是优化器原地没动。(2026-08-21 子智能体定位, 主会话独立复现。)
              pen = args.water_pen * ((water(tw, geo_opt) - w0) / w0) ** 2 * base_ref
              (-(obj - pen)).backward(); opt.step()
              with torch.no_grad():
                # 🔴 只在**满足注水约束**的迭代里挑最优。
                #    第一版这里只看 obj 不看约束, 于是专挑"注水最多因而产油最高"那一步,
                #    惩罚项完全白加 —— 断言连着拒了 3 次才让我发现。
                  dev_w = abs(float(water(tw, geo_opt)) - w0) / w0
                  v = float(obj)
                  if dev_w <= args.water_tol and v > best[0]:
                      best = (v, tw.detach().clone())
        if best[1] is None:
            raise AssertionError(
                f"{name}: {args.opt_iters} 步里没有一步同时满足注水约束(±{args.water_tol:.0%}) —— "
                f"调大 --water-pen 或放宽 --water-tol")
        tw = best[1]
        with torch.no_grad():
            v_l = float(cum(tw, geo_te, N_TIMES)); v_s = float(cum(tw, geo_te, args.horizon_short))
            w = float(water(tw, geo_te)); zn = float(torch.linalg.norm(tw / sig))
        # 🔴 硬断言:优化结果不得劣于起点。旧版就是在这里静默输出了劣解。
        start = base_long if name == "long" else base_short
        got = v_l if name == "long" else v_s
        assert got >= start * 0.999, (
            f"{name} 优化结果 {got:,.0f} 劣于起点 {start:,.0f} —— 优化器坏了, 拒绝输出")
        # 🔴 "同样多的水换个分法"是整个对比的前提。偏离过大就不是同一个实验了。
        assert abs(report_wd := (w - w0) / w0) <= args.water_tol, (
            f"{name} 实测注水偏离基准 {report_wd:+.1%} > 容差 {args.water_tol:.0%} —— "
            f"这不是'同样多的水换分法', 拒绝输出。调大 --water-pen 或放宽 --water-tol")
        plans[name] = tw.cpu().numpy()
        report[name] = {"long": v_l, "short": v_s, "water": w,
                        "water_dev": (w - w0) / w0, "z_norm": zn}
        print(f"\n[{name}] 十年 {v_l:,.0f} ({v_l/base_long-1:+.2%})   "
              f"短期 {v_s:,.0f} ({v_s/base_short-1:+.2%})")
        print(f"    实测注水 {w:,.0f} (偏离基准 {(w-w0)/w0:+.1%})   θ的 z 范数 {zn:.2f}")
        for k, i in enumerate(WATER_IDX):
            print(f"      {CTRL[i]:16s} {plans[name][k]:+.3f}  (×{10**plans[name][k]:.2f})")

    gap = report["long"]["long"] / max(report["short"]["long"], 1e-9) - 1
    print(f"\n=== 网络预测的差距(留出地质) ===")
    print(f"  只顾眼前的代价: 长期方案的十年产油比短期方案高 {gap:+.2%}")
    print(f"\n⚠️ 以上仍是**网络自己的预测**。决策效果以 verify(送模拟器重跑)为准。")

    json.dump({"trust_r_sigma": args.trust_r, "sigma": sig, "horizon_short": args.horizon_short,
               "baseline": {"long": base_long, "short": base_short, "water": w0},
               "plans": {k: v.tolist() for k, v in plans.items()},
               "net_predicted": report, "predicted_gap": float(gap),
               "water_head_relerr_median": werr,
               "caveat": "网络预测值, 非模拟器结果;决策效果以 verify 为准"},
              open(OUT / "plans.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {OUT}/plans.json")
    return 0


def verify(args) -> int:
    """把优化出的方案写进 deck **真跑模拟器** —— 这才是裁判。"""
    import shutil, subprocess, os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pj = json.load(open(OUT / "plans.json"))
    ctrl_full = len(CTRL) + len(NB.PERM_REGIONS)
    rng = np.random.default_rng(2)
    TH = np.load(BULK / "theta_all.npy")
    te = np.asarray(json.load(open(OUT / "train.json"))["test_idx"])
    geos = TH[rng.choice(te, args.n_verify, replace=False)][:, len(CTRL):]

    # 🔴 uniform(均匀分配)是**必打的底线**:若最笨的均分就能拿到同样增益, 网络就白训了。
    #    性质等同于预测那边强制打的"平凡基线"。2026-08-21 之前漏了这条。
    plan_of = {"baseline": np.zeros(len(WATER_IDX)),
               "uniform": np.zeros(len(WATER_IDX)),      # 份额均分 = log10(1) = 0 ... 见下
               "short": np.asarray(pj["plans"]["short"]),
               "long": np.asarray(pj["plans"]["long"])}
    # 均匀分配在"份额"意义上就是每口井分到 1/9 的总量。基准 θ=0 是 deck 原始配比,
    # 两者不同:deck 各井基准注入率本就不等。用实测基准率反推均分所需的 θ。
    TH_all = np.load(BULK / "theta_all.npy")
    _, _, IAf = load(args.max_n_base if hasattr(args, "max_n_base") else 2000)
    br = np.trapezoid(IAf[:, WATER_IDX, :], DAY_GRID, axis=2).mean(0)   # 各井基准累计注水
    plan_of["uniform"] = np.log10(br.mean() / np.maximum(br, 1e-9))     # 拉平到等量
    print(f"均匀分配的 θ: " + "  ".join(f"{CTRL[i].split('_')[0]}={plan_of['uniform'][k]:+.2f}"
                                        for k, i in enumerate(WATER_IDX)))

    cases = []
    for name in ("baseline", "uniform", "short", "long"):
        w = plan_of[name]
        for gi, g in enumerate(geos):
            th = np.zeros(ctrl_full)
            th[WATER_IDX] = w; th[len(CTRL):] = g
            cases.append((f"{name}_g{gi}", th))
    print(f"共 {len(cases)} 个算例 = 3 策略 × {len(geos)} 个留出地质, {args.jobs} 并发")

    vdir = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_verify")
    (vdir / "cases").mkdir(parents=True, exist_ok=True)

    def run(tag, theta):
        sh = vdir / "cases" / f"{tag}.npz"
        if sh.exists():
            return tag, {k: np.load(sh)[k] for k in np.load(sh).files}
        work = vdir / "_work" / tag
        shutil.rmtree(work, ignore_errors=True)
        try:
            NB.build(theta, work)
            cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{work}:/data", "-w", "/data", NB.IMAGE, "flow", NB.DECK,
                   "--output-dir=/data/out", f"--threads-per-process={args.threads}"]
            with open(work / "run.log", "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
            if rc != 0:
                return tag, None
            g = NB.harvest(work)
            if g is None:
                return tag, None
            np.savez_compressed(sh, **g)
            return tag, g
        finally:
            shutil.rmtree(work, ignore_errors=True)

    res = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run, t, th) for t, th in cases]
        for i, f in enumerate(as_completed(futs), 1):
            t, g = f.result(); res[t] = g
            if i % 5 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}", flush=True)

    def oil(g, upto=N_TIMES):
        """累计产油(梯形积分)。🔴 切短期窗口时 y 和 x 必须一起切 ——
        第一版只切了 DAY_GRID 没切产量数组, 报 broadcast (1,7) vs (22,39)。"""
        ob = g["obs"]
        r = np.stack([ob[(i * 3) * N_TIMES:(i * 3 + 1) * N_TIMES] for i in range(len(PRODUCERS))])
        return float(np.trapezoid(r[:, :upto], DAY_GRID[:upto], axis=1).sum())

    print(f"\n=== 🔴 模拟器裁判(留出地质 {len(geos)} 个) ===")
    out = {}
    for name in ("baseline", "uniform", "short", "long"):
        vals = [oil(res[f"{name}_g{gi}"]) for gi in range(len(geos)) if res.get(f"{name}_g{gi}")]
        sv = [oil(res[f"{name}_g{gi}"], args.horizon_short) for gi in range(len(geos))
              if res.get(f"{name}_g{gi}")]
        out[name] = {"n": len(vals), "long_mean": float(np.mean(vals)), "long_std": float(np.std(vals)),
                     "short_mean": float(np.mean(sv))}
        print(f"  {name:9s} n={len(vals):3d}  十年 {np.mean(vals):>14,.0f} ± {np.std(vals):>12,.0f}"
              f"   短期 {np.mean(sv):>12,.0f}")
    if out["baseline"]["n"] and out["long"]["n"]:
        b = out["baseline"]["long_mean"]
        print(f"\n  均匀分配 vs 基准: {out['uniform']['long_mean']/b-1:+.2%}   ← 必打的底线")
        print(f"  长期方案 vs 基准: {out['long']['long_mean']/b-1:+.2%}")
        print(f"  短期方案 vs 基准: {out['short']['long_mean']/b-1:+.2%}")
        beat = out["long"]["long_mean"] > out["uniform"]["long_mean"]
        print(f"  {'✅' if beat else '🔴'} 长期方案{'打赢' if beat else '**没打赢**'}均匀分配"
              f" ({out['long']['long_mean']/out['uniform']['long_mean']-1:+.2%})")
        print(f"  🔴 只顾眼前的代价: {out['long']['long_mean']/max(out['short']['long_mean'],1e-9)-1:+.2%}")
        print(f"  (网络预测的是 {pj['predicted_gap']:+.2%} —— 差距即代理误差)")
    json.dump({"n_geo": len(geos), "simulator": out,
               "net_predicted_gap": pj["predicted_gap"]},
              open(OUT / "verify.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {OUT}/verify.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "optimize", "verify"])
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--max-n", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--horizon-short", type=int, default=8, help="短期窗口=前几个时刻(共40)")
    ap.add_argument("--n-geo-opt", type=int, default=64, help="优化时平均多少个地质实现")
    ap.add_argument("--n-geo-test", type=int, default=200, help="留出地质数")
    ap.add_argument("--opt-iters", type=int, default=600)
    ap.add_argument("--opt-lr", type=float, default=0.05)
    ap.add_argument("--trust-r", type=float, default=2.0, help="信赖域半径(以 θ 的 sigma 为单位)")
    ap.add_argument("--water-pen", type=float, default=300.0, help="实测注水量偏离基准的惩罚权重")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-restart", type=int, default=6, help="多起点, 防停在平台")
    ap.add_argument("--water-tol", type=float, default=0.03, help="实测注水量允许偏离基准多少")
    ap.add_argument("--n-verify", type=int, default=12, help="verify 用几个留出地质")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)
    if args.cmd == "optimize":
        return optimize(args)
    if args.cmd == "verify":
        return verify(args)
    print(f"{args.cmd} 尚未实现"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
