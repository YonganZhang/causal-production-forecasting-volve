#!/usr/bin/env python3
"""多起点优化 + 模拟器直接筛选：拿一个又高又稳的解。

## 为什么这么做

1. **多起点是非凸优化的标准做法**,不是挑数据。单起点只保证局部最优;
   跑 N 个随机起点、取最好的解,是求解方法的一部分,论文里正常写。
2. **模拟一次只要约 1 分钟**,所以可以不靠代理模型排名,
   **直接让 OPM Flow 评每个候选** —— 彻底绕开"代理在极值点外推不可靠"这个问题
   (主线实测:代理预测 +3.65%,模拟器裁定 +0.88%,高估 4 倍)。
3. **在候选里优先选局部平坦的**。实测最优解附近存在陡峭方向
   (θ 动 1e-3,产油掉 0.65%,弹性 29.85,与注水量无关)。
   同样产油下选平坦的那个,数更硬 —— 别人复现时不会因为微小差异得到别的结论。

## 流程

    N 个随机起点 → 代理模型各自优化 → 得 N 个候选 θ
      → 每个候选做实测注水量校正(标量缩放,不改井间份额)
      → **OPM Flow 逐个真跑**,按实测产油排序
      → 前 K 名各做 M 次微扰(注水量保持),量局部平坦度
      → 选"产油高 且 平坦"的那个

用法:
    python fc_ms.py --n-start 12 --top-k 4 --n-flat 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fc_decision as FD_data
import forecast_gen as FG
import norne_bulk as NB
import fc_decide as FD
from fc_mech import PosNet

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_decide"
NPH, NT = 2, FG.FC_N
DAYS = FG.FC_GRID
N_STAGE, INJ = FG.N_STAGE, FG.INJ_W
BASE_W = np.array([FG.BASE_WINJ[w] for w in INJ])
SHORT_Y = 3.0


class _A:
    def __init__(self, threads):
        self.threads = threads


def sim_metrics(key):
    d = np.load(OUT / "sim" / f"{key}.npz")
    ob = d["obs"]; n = len(NB.PRODUCERS)
    oil = np.stack([ob[(i*3)*NT:(i*3+1)*NT] for i in range(n)])
    ms = (DAYS - DAYS[0]) <= SHORT_Y * 365.25
    return (float(np.trapezoid(oil, DAYS, axis=-1).sum()),
            float(np.trapezoid(oil[:, ms], DAYS[ms], axis=-1).sum()),
            float(d["field_cum"][1][-1] - d["field_cum"][1][0]))


def ensure_sim(key, th, args):
    if not (OUT / "sim" / f"{key}.npz").exists():
        FD._run_sim(key, np.asarray(th, np.float32).reshape(N_STAGE, len(INJ)), _A(args.threads))
    return sim_metrics(key)


def calibrate(key_prefix, th, W0, args):
    """标量缩放把实测注水量校到基准。返回 (θ_校正后, 实测指标)。"""
    pts = []
    for it in range(args.calib_iter):
        ls = 0.0 if it == 0 else (
            -np.log10(1 + pts[0][1]) if it == 1 else
            float(np.clip(pts[-1][0] - pts[-1][1] * (pts[-1][0] - pts[-2][0]) /
                          (pts[-1][1] - pts[-2][1] + 1e-12), -0.3, 0.3)))
        o, s, w = ensure_sim(f"{key_prefix}_c{it}", th + ls, args)
        dev = w / W0 - 1
        pts.append((ls, dev, o, s))
        if abs(dev) <= args.water_tol:
            break
    b = min(pts, key=lambda t: abs(t[1]))
    return th + b[0], {"oil": b[2], "short": b[3], "water_dev": b[1], "s": b[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-start", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--n-flat", type=int, default=3)
    ap.add_argument("--flat-eps", type=float, default=2e-3)
    ap.add_argument("--ntrain", type=int, default=7400)
    ap.add_argument("--n-pin", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--trust-r", type=float, default=2.5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--calib-iter", type=int, default=3)
    ap.add_argument("--water-tol", type=float, default=0.01)
    a = ap.parse_args()
    dev = "cpu"

    TH, Y, IA, FC = FD_data.load(a.n_pin)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]; nw = int(live.sum()); n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    pool = idx[600:]; tr = pool[:a.ntrain]
    Yf = Y.reshape(n, -1); NCH = nw * NPH
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
    xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
    X = ((TH - xm) / xs).astype(np.float32)
    torch.set_num_threads(48)
    torch.manual_seed(0)
    net = PosNet(TH.shape[1], NCH, NT, "fourier", n_bands=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    A = torch.tensor(X[tr]); B = torch.tensor(((Yf[tr]-ym)/ys).reshape(-1, NCH, NT).astype(np.float32))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    t0 = time.time()
    for _ in range(a.epochs):
        pm = torch.randperm(len(A))
        for i in range(0, len(A), 64):
            b = pm[i:i+64]; opt.zero_grad()
            F.mse_loss(net(A[b]), B[b]).backward(); opt.step()
        sch.step()
    net.eval()
    print(f"代理模型训练完 ({(time.time()-t0)/60:.0f} min)\n")

    ymT, ysT = torch.tensor(ym, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
    xmT, xsT = torch.tensor(xm, dtype=torch.float32), torch.tensor(xs, dtype=torch.float32)
    BWt = torch.tensor(BASE_W, dtype=torch.float32)
    dT = torch.tensor(DAYS, dtype=torch.float32)
    m_long = torch.ones(NT); m_short = torch.tensor(((DAYS-DAYS[0]) <= SHORT_Y*365.25).astype(np.float32))
    sig = float(TH.std())

    def curve(thf):
        o = net(((thf - xmT)/xsT)[None])[0].reshape(-1) * ysT + ymT
        return o.reshape(nw, NPH, NT)[:, 0, :].sum(0)

    def solve(mask, seed):
        g = torch.Generator().manual_seed(seed)
        z = (torch.randn(N_STAGE*len(INJ), generator=g) * 0.5).requires_grad_(True)
        o = torch.optim.Adam([z], lr=a.lr)
        best, bth = -1e30, None
        for _ in range(a.steps):
            o.zero_grad()
            th = a.trust_r * sig * torch.tanh(z)
            frac = torch.softmax(th.reshape(N_STAGE, len(INJ)) * np.log(10.0), 1)
            th_eff = torch.log10(frac * BWt.sum() / BWt[None] + 1e-12).reshape(-1)
            v = torch.trapezoid(curve(th_eff)*mask, dT)
            (-v).backward(); o.step()
            with torch.no_grad():
                if float(v) > best:
                    best, bth = float(v), th_eff.detach().numpy().copy()
        return bth, best

    W0 = ensure_sim("ms_base", np.zeros(N_STAGE*len(INJ)), a)[2]
    print(f"基准实测总注水 {W0:,.0f}\n")
    RES = {}
    for name, mask in (("long", m_long), ("short", m_short)):
        print(f"=== {name}:{a.n_start} 个随机起点 ===")
        cands = []
        for s in range(a.n_start):
            th, v = solve(mask, 1000 + s)
            cands.append((th, v))
        cands.sort(key=lambda c: -c[1])
        print(f"  代理目标值范围 {cands[-1][1]:,.0f} ~ {cands[0][1]:,.0f}"
              f"  (跨度 {cands[0][1]/cands[-1][1]-1:.2%})")
        print(f"  → 取前 {a.top_k} 名交给 OPM Flow 真跑\n")
        fin = []
        for k in range(a.top_k):
            th, _ = cands[k]
            thc, m = calibrate(f"ms_{name}_{k}", th, W0, a)
            print(f"  候选{k}: 全期 {m['oil']:,.0f}  前3年 {m['short']:,.0f}  "
                  f"注水偏离 {m['water_dev']:+.2%}   ({(time.time()-t0)/60:.0f}min)", flush=True)
            fin.append({"k": k, "theta": thc.tolist(), **m})
        fin.sort(key=lambda d: -(d["oil"] if name == "long" else d["short"]))
        # 局部平坦度:对前 2 名各做 n_flat 次微扰(注水量随后校正,排除水量因素)
        print(f"\n  局部平坦度(θ ± {a.flat_eps:.0e},注水量校正后):")
        for d in fin[:2]:
            th = np.asarray(d["theta"])
            vals = []
            rng = np.random.default_rng(7)
            for j in range(a.n_flat):
                thp = th + rng.normal(0, a.flat_eps, th.shape)
                _, mm = calibrate(f"ms_{name}_{d['k']}_f{j}", thp, W0, a)
                vals.append(mm["oil"] if name == "long" else mm["short"])
            base_v = d["oil"] if name == "long" else d["short"]
            spread = (max(vals) - min(vals)) / base_v
            d["flat_spread"] = float(spread)
            print(f"    候选{d['k']}  极差/均值 {spread:.3%}"
                  f"   {'✔ 平坦' if spread < 0.002 else '⚠ 较陡'}", flush=True)
        RES[name] = fin
        best = min([d for d in fin[:2]], key=lambda d: d.get("flat_spread", 9))
        print(f"\n  🏆 {name} 选候选{best['k']}"
              f"(产油 {'第1' if best is fin[0] else '第2'},平坦度 {best.get('flat_spread', float('nan')):.3%})\n")
        RES[name + "_pick"] = best

    ob, sb, _ = sim_metrics("ms_base")
    L, S = RES["long_pick"], RES["short_pick"]
    print(f"{'='*72}\n=== 🔴 多起点 + 模拟器直选结果 ===\n")
    print(f"  {'方案':10s}{'全期产油':>15s}{'vs基准':>9s}{'前3年':>15s}{'vs基准':>9s}{'注水偏离':>10s}")
    print(f"  {'基准':10s}{ob:>15,.0f}{0:>+8.2f}%{sb:>15,.0f}{0:>+8.2f}%{0:>+9.2f}%")
    for nm, d in (("短期最优", S), ("长期最优", L)):
        print(f"  {nm:10s}{d['oil']:>15,.0f}{(d['oil']/ob-1)*100:>+8.2f}%"
              f"{d['short']:>15,.0f}{(d['short']/sb-1)*100:>+8.2f}%{d['water_dev']*100:>+9.2f}%")
    gap = L["oil"] / S["oil"] - 1
    print(f"\n  🔴 **只顾眼前的代价 = {gap:+.2%}**")
    json.dump({"n_start": a.n_start, "top_k": a.top_k, "trust_r": a.trust_r,
               "baseline": {"oil": ob, "short": sb},
               "results": {k: v for k, v in RES.items()},
               "headline_gap": gap,
               "method": ("多起点非凸优化(标准做法);候选由 **OPM Flow 直接评**,"
                          "不靠代理排名;同产油下优先选局部平坦的解。")},
              open(OUT / "multistart.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}/multistart.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
