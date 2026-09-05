"""无 Agent 对照臂 —— 不含任何 LLM 的平凡基线。

🔴 设计依据(2026-09-04 用户拍板 + petro-knowledge/05-生产优化.md 的「平凡基线」节):
   此前的「无 Agent」臂是"一个什么都不知道的 LLM 在提方案"——既不是 Agent，
   也不是正经基线，是个四不像;而且我还把已知最优解的实例表发给了它，
   等于把答案交给对照组。

   正确做法是用领域内公认的平凡基线。本文件实现其中三条:
     B2  一维均匀缩放  —— 所有注水井同乘 alpha，扫描取最优。1 个决策变量。
                          知识库原话:"这是最狠的那条"——高维优化若打不赢它，
                          "高维井控优化"的整个卖点就不成立。
     B4  等预算随机搜索 —— 与 Agent 臂**完全相同**的模拟预算内均匀抽样取最优。
                          检验"你的智能体到底有没有用"。
     BG  贪心/短期最优  —— 逐时段选当下产油最优，不看长期。
                          对应用户说的"短期最优/贪心算法"。

   三条都不含 LLM，不看任何实例表、配方或边际价值表。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_gen as FG                                    # noqa: E402
from fc_team_loop import adjudicate, SIM, INJ, N_STAGE       # noqa: E402
from fc_water_econ import econ                               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_pipelines" / "fc_baselines"


def _run(th: np.ndarray, key: str, a) -> dict | None:
    return adjudicate(np.asarray(th, float).ravel(), key, a.threads,
                      a.c_inj, a.c_prod, a.w_max_dev)


def b2_uniform(a) -> dict:
    """B2:一维均匀缩放。所有井所有时段同乘 10^alpha。

    🔴 alpha 范围必须收窄:实测注水量对 alpha 极其敏感 ——
       alpha=+0.3 即 +73.4%、-0.3 即 -48.5%，双双越过 ±25% 预算。
       首版用 ±0.3 扫 3 点，两端全部越界，只剩 alpha=0(即基准本身)，
       于是"B2 提升 +0.0M"——那是扫描范围的问题，不是方法本身弱。
       改为 ±0.12(对应约 ±25% 注水量)，并按方法固有需求给点数。
    """
    n = max(a.budget, a.b2_points)
    alphas = np.linspace(-0.12, 0.12, n)
    best, rows = None, []
    for i, al in enumerate(alphas):
        th = np.full((N_STAGE, len(INJ)), al)
        m = _run(th, f"bl_B2{a.tag}_{i}", a)
        if m is None:
            continue
        rows.append({"alpha": float(al), "npv8": m["npv8"], "oil": m["oil"],
                     "water_dev": m["water_dev"], "ok": bool(m["budget_ok"])})
        if m["budget_ok"] and (best is None or m["npv8"] > best[1]["npv8"]):
            best = (float(al), m)
    return {"method": "B2_uniform_scaling", "n_sim": len(rows), "trials": rows,
            "best_alpha": None if best is None else best[0],
            "best": None if best is None else best[1]}


def b4_random(a) -> dict:
    """B4:等预算随机搜索。与 Agent 臂相同的模拟次数，均匀抽 θ。"""
    rng = np.random.default_rng(a.seed)
    best, rows = None, []
    for i in range(a.budget):
        th = rng.uniform(-0.8, 0.8, (N_STAGE, len(INJ)))
        m = _run(th, f"bl_B4{a.tag}_{i}", a)
        if m is None:
            continue
        rows.append({"npv8": m["npv8"], "oil": m["oil"],
                     "water_dev": m["water_dev"], "ok": bool(m["budget_ok"])})
        if m["budget_ok"] and (best is None or m["npv8"] > best["npv8"]):
            best = m
    return {"method": "B4_equal_budget_random", "seed": a.seed,
            "n_sim": len(rows), "trials": rows, "best": best}


def bg_greedy(a) -> dict:
    """BG:贪心 / 短期最优。逐时段扫一维增量，选当下产油最高者后冻结，进入下一段。

    与全局优化的区别在于它**只看当下**:每段定完就不再回头，
    也不考虑本段决定对后续时段的影响。这正是"短期最优"的操作化定义。
    """
    th = np.zeros((N_STAGE, len(INJ)))
    grid = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    budget = max(a.budget, a.bg_budget)
    used, log = 0, []
    for s in range(N_STAGE):
        if used >= budget:
            break
        cand = []
        for g in grid:
            if used >= budget:
                break
            t = th.copy(); t[s] = g
            m = _run(t, f"bl_BG{a.tag}_s{s}_{g:+.1f}".replace(".", "p"), a)
            used += 1
            if m is None:
                continue
            # 🔴 贪心的判据是**当下产油**，不是全期 NPV —— 这是"短期最优"的定义
            cand.append((float(g), m["oil_3y"] if s < 2 else m["oil"], m))
        # 🔴 贪心只看当下产油，天然会一路加水 —— 实测逐段全选最大值
        #    (+0.2 +0.4 +0.4 +0.4 +0.4 +0.4)，注水冲到 +76%。
        #    这正是"短期最优"该有的失败模式，但**必须受同一预算约束**才是合法基线，
        #    否则它是在一个更大的可行域里比赛。故在候选内先滤掉越预算的。
        feas = [c for c in cand if c[2].get("budget_ok")]
        pool = feas if feas else cand
        g, sc, m = max(pool, key=lambda x: x[1])
        th[s] = g
        log.append({"stage": s + 1, "chosen": g, "score": sc, "npv8": m["npv8"],
                    "n_feasible": len(feas), "n_cand": len(cand)})
    fin = _run(th, f"bl_BG{a.tag}_final", a)
    # 🔴 最终解同样要过预算门 —— 首版漏了这一步，导致 water_dev=+76.4% 的解
    #    被当作 BG 的"最佳",而 B2/B4 都是过了门的。口径必须一致。
    ok = bool(fin and fin.get("budget_ok"))
    return {"method": "BG_greedy_short_term", "n_sim": used + 1,
            "stage_log": log, "theta": th.tolist(),
            "final_infeasible": (fin is not None and not ok),
            "final_water_dev": None if fin is None else fin["water_dev"],
            "best": fin if ok else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["B2", "B4", "BG", "all"], default="all")
    ap.add_argument("--budget", type=int, default=3, help="基础模拟预算(与 Agent 臂一致)")
    # 🔴 不同方法的"一次评估"含义不同，故各自声明固有需求，并在结果里如实报告实际用量:
    #    B4 随机搜索 3 次即 3 个样本;B2 一维扫描需要足够分辨率;BG 贪心需 5 值 x 6 段。
    ap.add_argument("--b2-points", type=int, default=7, help="B2 的扫描点数")
    ap.add_argument("--bg-budget", type=int, default=30, help="BG 的模拟预算(5 值 x 6 段)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--c-inj", type=float, default=2.0)
    ap.add_argument("--c-prod", type=float, default=1.0)
    ap.add_argument("--w-max-dev", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = econ(SIM / "baseline.npz", a.c_inj, a.c_prod)
    res = {"baseline_npv8": base["npv8"], "budget": a.budget,
           "water_price": [a.c_inj, a.c_prod], "arms": {}}
    fns = {"B2": b2_uniform, "B4": b4_random, "BG": bg_greedy}
    for k in (fns if a.method == "all" else [a.method]):
        t0 = time.time()
        r = fns[k](a)
        r["sec"] = time.time() - t0
        res["arms"][k] = r
        b = r.get("best")
        print(f"  {k:3s} {r['method']:26s} {r['n_sim']:2d} 次模拟  "
              + (f"ΔNPV8 = {(b['npv8']-base['npv8'])/1e6:+8.1f} M$  "
                 f"油 {100*(b['oil']/base['oil']-1):+6.2f}%  "
                 f"水 {b['water_dev']*100:+6.2f}%" if b else "(无可行解)"))
    (OUT / f"baselines{a.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n→ {OUT/f'baselines{a.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
